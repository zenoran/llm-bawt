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
):
    """Expose OpenClaw bridge events as SSE via Redis subscriber."""
    service = get_service()

    redis_sub = getattr(service, "_redis_subscriber", None)
    if redis_sub is None:
        raise HTTPException(status_code=503, detail="OpenClaw bridge not enabled (no Redis subscriber)")

    async def event_stream():
        yield _sse(
            "hello",
            {
                "session_key": session_key,
                "connected": True,
                "ts": datetime.utcnow().isoformat() + "Z",
            },
        )

        last_ping = asyncio.get_running_loop().time()
        try:
            async for evt in redis_sub.subscribe(session_key):
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
