"""OpenClaw WS->SSE bridge endpoint."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import List

from ..dependencies import get_service
from ..logging import get_service_logger

router = APIRouter()
log = get_service_logger(__name__)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.get("/v1/ws", tags=["OpenClaw Bridge"])
async def openclaw_ws_bridge(
    session_key: str = Query(None, description="Session key to stream (legacy)"),
    bot_id: List[str] = Query(None, description="Bot ID(s) for unified event stream (repeatable)"),
    user_id: str = Query(None, description="User ID for unified event stream"),
    consumer_id: str = Query(None, description="Consumer ID for durable consumer group"),
):
    """Expose events as SSE via Redis subscriber.

    Two modes:
    - Legacy: ``?session_key=main`` — streams OpenClaw session events (XREAD, no resume)
    - Unified: ``?bot_id=X&bot_id=Y&user_id=Z&consumer_id=UUID`` — durable consumer group
      subscribing to one or more bots. Falls back to legacy if consumer_id is absent.
    """
    service = get_service()

    redis_sub = getattr(service, "_redis_subscriber", None)
    if redis_sub is None:
        raise HTTPException(status_code=503, detail="OpenClaw bridge not enabled (no Redis subscriber)")

    use_unified = bot_id and user_id and consumer_id

    if not use_unified and not session_key:
        raise HTTPException(
            status_code=400,
            detail="Provide either session_key (legacy) or bot_id+user_id+consumer_id (unified)",
        )

    if use_unified:
        # bot_id is a list — pass all of them to subscribe_group
        return StreamingResponse(
            _unified_event_stream(redis_sub, bot_id, user_id, consumer_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Legacy session-based streaming
    return StreamingResponse(
        _legacy_event_stream(redis_sub, session_key),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _unified_event_stream(redis_sub, bot_ids: list[str], user_id: str, consumer_id: str):
    """Durable event stream using Redis consumer groups for one or more bots."""
    yield _sse(
        "hello",
        {
            "bot_ids": bot_ids,
            "user_id": user_id,
            "consumer_id": consumer_id,
            "mode": "unified",
            "connected": True,
            "ts": datetime.now(timezone.utc).isoformat(),
        },
    )

    last_ping = asyncio.get_running_loop().time()
    try:
        async for event_data in redis_sub.subscribe_group(
            bot_ids, user_id, consumer_id, timeout_s=86400,
        ):
            if event_data is None:
                # Keepalive tick from subscriber — send SSE comment to prevent proxy idle timeout
                now = asyncio.get_running_loop().time()
                if now - last_ping > 25:
                    yield ": ping\n\n"
                    last_ping = now
                continue
            replayed = event_data.pop("_replayed", False)
            yield _sse("event", {**event_data, "replayed": replayed})

            now = asyncio.get_running_loop().time()
            if now - last_ping > 25:
                yield ": ping\n\n"
                last_ping = now
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.exception("/v1/ws unified stream error")
        yield _sse("error", {"error": str(e)})


async def _legacy_event_stream(redis_sub, session_key: str):
    """Legacy session-based event stream (no resume)."""
    yield _sse(
        "hello",
        {
            "session_key": session_key,
            "mode": "legacy",
            "connected": True,
            "ts": datetime.now(timezone.utc).isoformat(),
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
            if now - last_ping > 25:
                yield ": ping\n\n"
                last_ping = now
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.exception("/v1/ws legacy stream error")
        yield _sse("error", {"error": str(e)})
