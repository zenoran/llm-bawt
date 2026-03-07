"""OpenClaw WS->SSE bridge endpoint."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..dependencies import get_service
from ..logging import get_service_logger

router = APIRouter()
log = get_service_logger(__name__)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.get("/v1/ws", tags=["OpenClaw Bridge"])
async def openclaw_ws_bridge(
    session_key: str = Query("main", description="Session key to stream"),
    since_id: int | None = Query(None, description="Replay events after this DB id before tailing live"),
):
    """Expose OpenClaw bridge events as SSE.

    Supports both Redis subscriber mode (external bridge) and
    in-process fanout mode (legacy).
    """
    service = get_service()

    # Check for Redis subscriber first (external bridge mode)
    redis_sub = getattr(service, "_redis_subscriber", None)

    # Fall back to in-process bridge
    bridge = getattr(service, "_session_bridge", None)
    fanout = getattr(bridge, "_fanout", None) if bridge else None
    store = getattr(bridge, "_store", None) if bridge else None

    if redis_sub is None and (fanout is None or store is None):
        raise HTTPException(status_code=503, detail="OpenClaw bridge not enabled")

    async def event_stream():
        # Initial hello
        cursor = store.get_session_cursor(session_key) if store else None
        connected = bool(getattr(bridge, "connected", False)) if bridge else True
        yield _sse(
            "hello",
            {
                "session_key": session_key,
                "connected": connected,
                "cursor": cursor,
                "ts": datetime.utcnow().isoformat() + "Z",
            },
        )

        last_ping = asyncio.get_running_loop().time()
        try:
            if redis_sub:
                source = redis_sub.subscribe(session_key)
            else:
                source = fanout.subscribe(session_key, since_event_id=since_id)

            async for evt in source:
                payload = {
                    "id": evt.db_id,
                    "event_id": evt.event_id,
                    "session_key": evt.session_key,
                    "run_id": evt.run_id,
                    "kind": evt.kind.value,
                    "origin": evt.origin,
                    "text": evt.text,
                    "tool_name": evt.tool_name,
                    "timestamp": evt.timestamp.isoformat() if evt.timestamp else None,
                    "seq": evt.seq,
                    "raw": evt.raw,
                }
                yield _sse("event", payload)

                now = asyncio.get_running_loop().time()
                if now - last_ping > 20:
                    yield ": ping\n\n"
                    last_ping = now
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("/v1/ws stream error")
            yield _sse("error", {"error": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
