"""WebSocket gateway endpoint — bidirectional alternative to SSE /v1/ws.

Provides a single WebSocket connection for chat, events, and history,
replacing multiple SSE + fetch paths with a unified protocol.

Protocol frames:
  Client → Server (request):
    { "type": "req", "id": "<uuid>", "method": "<method>", "params": { ... } }

  Server → Client (response):
    { "type": "res", "id": "<uuid>", "ok": true, "payload": { ... } }

  Server → Client (event):
    { "type": "evt", "event": "<event>", "seq": <int>, "payload": { ... } }
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..dependencies import get_service
from ..logging import get_service_logger
from ..schemas import ChatCompletionRequest, ChatMessage
from .history import get_history

router = APIRouter()
log = get_service_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_res(req_id: str, *, ok: bool = True, payload: dict | None = None, error: str | None = None) -> dict:
    frame: dict[str, Any] = {"type": "res", "id": req_id, "ok": ok}
    if payload is not None:
        frame["payload"] = payload
    if error is not None:
        frame["ok"] = False
        frame["error"] = error
    return frame


def _make_evt(event: str, seq: int, payload: dict) -> dict:
    return {"type": "evt", "event": event, "seq": seq, "payload": payload}


class _SeqCounter:
    """Monotonically increasing per-connection sequence counter."""
    __slots__ = ("_value",)

    def __init__(self, start: int = 0) -> None:
        self._value = start

    def next(self) -> int:
        self._value += 1
        return self._value

    @property
    def current(self) -> int:
        return self._value


# ---------------------------------------------------------------------------
# Connection state
# ---------------------------------------------------------------------------

class GatewayConnection:
    """Per-WebSocket connection state."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.seq = _SeqCounter()
        self.consumer_id: str | None = None
        self.user_id: str | None = None
        self.bot_ids: list[str] = []
        self._subscription_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._active_runs: dict[str, asyncio.Task] = {}
        self._send_lock = asyncio.Lock()

    async def send(self, frame: dict) -> None:
        async with self._send_lock:
            await self.ws.send_json(frame)

    async def send_evt(self, event: str, payload: dict) -> None:
        seq = self.seq.next()
        await self.send(_make_evt(event, seq, payload))

    async def send_res(self, req_id: str, *, ok: bool = True, payload: dict | None = None, error: str | None = None) -> None:
        await self.send(_make_res(req_id, ok=ok, payload=payload, error=error))

    async def cleanup(self) -> None:
        for t in (self._subscription_task, self._keepalive_task):
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        for task in self._active_runs.values():
            if not task.done():
                task.cancel()
        for task in self._active_runs.values():
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._active_runs.clear()


# ---------------------------------------------------------------------------
# Method handlers
# ---------------------------------------------------------------------------

async def _handle_ping(conn: GatewayConnection, req_id: str, params: dict) -> None:
    await conn.send_res(req_id, payload={"ts": _now_iso()})


async def _handle_subscribe(conn: GatewayConnection, req_id: str, params: dict) -> None:
    """Subscribe to event streams for one or more bots.

    Reuses existing Redis consumer groups from subscriber.py.
    Supports seq-based reconnect via ``lastSeq`` param.
    """
    bot_ids = params.get("botIds") or params.get("bot_ids") or []
    if isinstance(bot_ids, str):
        bot_ids = [bot_ids]
    user_id = params.get("userId") or params.get("user_id")
    consumer_id = params.get("consumerId") or params.get("consumer_id") or str(uuid.uuid4())
    last_seq = params.get("lastSeq") or params.get("last_seq") or 0

    if not bot_ids or not user_id:
        await conn.send_res(req_id, error="botIds and userId are required")
        return

    service = get_service()
    redis_sub = getattr(service, "_redis_subscriber", None)
    if redis_sub is None:
        await conn.send_res(req_id, error="Event streaming not available (no Redis)")
        return

    # Cancel previous subscription if any
    if conn._subscription_task and not conn._subscription_task.done():
        conn._subscription_task.cancel()
        try:
            await conn._subscription_task
        except (asyncio.CancelledError, Exception):
            pass

    conn.consumer_id = consumer_id
    conn.user_id = user_id
    conn.bot_ids = list(bot_ids)

    # Reset seq counter if client is reconnecting with a lastSeq
    if last_seq:
        conn.seq = _SeqCounter(int(last_seq))

    # Send hello event + response
    await conn.send_evt("hello", {
        "consumerId": consumer_id,
        "botIds": bot_ids,
        "userId": user_id,
        "ts": _now_iso(),
    })
    await conn.send_res(req_id, payload={
        "consumerId": consumer_id,
        "botIds": bot_ids,
    })

    # Start background subscription
    async def _run_subscription():
        try:
            async for event_data in redis_sub.subscribe_group(
                bot_ids, user_id, consumer_id, timeout_s=3600,
            ):
                if event_data is None:
                    # Keepalive — no-op, the ping/pong on the WS layer handles liveness
                    continue
                replayed = event_data.pop("_replayed", False)
                payload = {**event_data}
                if replayed:
                    payload["replayed"] = True

                # Map event type to protocol event name
                event_type = event_data.get("event") or event_data.get("_type") or "event"
                evt_name = _map_redis_event(event_type, event_data)
                await conn.send_evt(evt_name, payload)
        except asyncio.CancelledError:
            pass
        except WebSocketDisconnect:
            pass
        except Exception as e:
            log.exception("WS gateway subscription error")
            try:
                await conn.send_evt("chat.warning", {"warnings": [str(e)]})
            except Exception:
                pass

    conn._subscription_task = asyncio.create_task(_run_subscription())


def _map_redis_event(event_type: str, data: dict) -> str:
    """Map internal event types to protocol event names."""
    # For compound _type like "tool_event", resolve via the inner "event" field
    if event_type == "tool_event":
        inner = data.get("event", "")
        if inner == "tool_start":
            return "chat.tool.start"
        if inner == "tool_end":
            return "chat.tool.end"
        return "event"

    mapping = {
        "tool_start": "chat.tool.start",
        "tool_end": "chat.tool.end",
        "turn_complete": "chat.turn_complete",
        "assistant_delta": "chat.delta",
        "assistant_done": "chat.final",
        "error": "chat.error",
    }
    return mapping.get(event_type, "event")


async def _handle_chat_send(conn: GatewayConnection, req_id: str, params: dict) -> None:
    """Send a chat message and stream responses as events.

    Proxies to the existing chat completion pipeline. Returns runId
    synchronously in the response, then streams delta/tool/final events.
    """
    bot_id = params.get("botId") or params.get("bot_id")
    message = params.get("message")
    user_id = params.get("userId") or params.get("user_id") or conn.user_id or "system"
    model = params.get("model")
    tts_mode = params.get("ttsMode") or params.get("tts_mode") or False

    if not bot_id or not message:
        await conn.send_res(req_id, error="botId and message are required")
        return

    run_id = f"run-{uuid.uuid4().hex[:12]}"

    # Respond immediately with runId
    await conn.send_res(req_id, payload={"runId": run_id, "botId": bot_id})

    # Stream in background
    async def _run_chat():
        service = get_service()
        try:
            request = ChatCompletionRequest(
                model=model or "default",
                messages=[ChatMessage(role="user", content=message)],
                stream=True,
                bot_id=bot_id,
                user=user_id,
                tts_mode=tts_mode,
            )

            full_text_parts: list[str] = []

            async for chunk_str in service.chat_completion_stream(request):
                # Parse SSE chunks from the existing pipeline
                if not chunk_str.startswith("data: "):
                    continue
                data_str = chunk_str[6:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Handle service warnings
                if data.get("object") == "service.warning":
                    await conn.send_evt("chat.warning", {
                        "runId": run_id,
                        "botId": bot_id,
                        "warnings": data.get("warnings", []),
                    })
                    continue

                # Process chat completion chunks — text content only.
                # Tool events arrive via the Redis subscription to avoid
                # duplicates (the streaming pipeline publishes them there).
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                content = delta.get("content")
                if content:
                    full_text_parts.append(content)
                    await conn.send_evt("chat.delta", {
                        "runId": run_id,
                        "botId": bot_id,
                        "content": content,
                    })

            # Final message — full message object with server-generated ID
            full_text = "".join(full_text_parts)
            if full_text:
                msg_id = str(uuid.uuid4())
                await conn.send_evt("chat.final", {
                    "runId": run_id,
                    "botId": bot_id,
                    "message": {
                        "id": msg_id,
                        "role": "assistant",
                        "content": full_text,
                        "timestamp": _now_iso(),
                    },
                })

            await conn.send_evt("chat.turn_complete", {
                "runId": run_id,
                "botId": bot_id,
                "status": "ok",
            })

        except asyncio.CancelledError:
            await conn.send_evt("chat.turn_complete", {
                "runId": run_id,
                "botId": bot_id,
                "status": "cancelled",
            })
        except Exception as e:
            log.exception("WS gateway chat.send failed for run %s", run_id)
            try:
                await conn.send_evt("chat.error", {
                    "runId": run_id,
                    "botId": bot_id,
                    "error": str(e),
                })
                await conn.send_evt("chat.turn_complete", {
                    "runId": run_id,
                    "botId": bot_id,
                    "status": "error",
                })
            except Exception:
                pass

    task = asyncio.create_task(_run_chat())
    conn._active_runs[run_id] = task


async def _handle_chat_abort(conn: GatewayConnection, req_id: str, params: dict) -> None:
    """Abort an in-flight chat turn."""
    bot_id = params.get("botId") or params.get("bot_id")
    turn_id = params.get("turnId") or params.get("turn_id")
    run_id = params.get("runId") or params.get("run_id")

    service = get_service()

    # Cancel the local run task if we have one
    if run_id and run_id in conn._active_runs:
        task = conn._active_runs.pop(run_id)
        if not task.done():
            task.cancel()

    # If a turn_id was provided, use the existing abort logic
    if turn_id:
        store = service._turn_log_store
        turn = store.get_turn(turn_id)
        if not turn:
            await conn.send_res(req_id, error="Turn not found")
            return

        if turn.status not in ("streaming", "pending"):
            await conn.send_res(req_id, payload={"detail": "already_completed", "turnId": turn_id})
            return

        # Send abort RPC to gateway if OpenClaw turn
        if turn.agent_session_key:
            from ...agent_backends.openclaw import get_openclaw_subscriber
            subscriber = get_openclaw_subscriber()
            if subscriber:
                abort_req_id = f"abort_{uuid.uuid4().hex}"
                try:
                    await subscriber.send_rpc("chat.abort", {"sessionKey": turn.agent_session_key}, abort_req_id, timeout_s=10)
                except Exception as e:
                    log.warning("chat.abort RPC failed for turn %s: %s", turn_id, e)

        store.update_turn(turn_id=turn_id, status="aborted", error_text="Aborted via ws.gateway")
        await conn.send_res(req_id, payload={"detail": "aborted", "turnId": turn_id})
    elif bot_id:
        # Abort by bot — cancel any active runs for this bot
        cancelled = []
        for rid, task in list(conn._active_runs.items()):
            if not task.done():
                task.cancel()
                cancelled.append(rid)
        for rid in cancelled:
            conn._active_runs.pop(rid, None)
        await conn.send_res(req_id, payload={"detail": "cancelled", "cancelledRuns": cancelled})
    else:
        await conn.send_res(req_id, error="turnId or botId required")


async def _handle_chat_history(conn: GatewayConnection, req_id: str, params: dict) -> None:
    """Retrieve conversation history for a bot.

    Proxies to the existing ``get_history`` route handler so pagination,
    filtering, and cursor logic stay in one place.
    """
    bot_id = params.get("botId") or params.get("bot_id")
    limit = params.get("limit", 50)
    before = params.get("before")

    try:
        from fastapi import HTTPException
        try:
            result = await get_history(
                bot_id=bot_id,
                limit=limit,
                before=before,
            )
        except HTTPException as exc:
            await conn.send_res(req_id, error=exc.detail)
            return

        await conn.send_res(req_id, payload={
            "botId": result.bot_id,
            "messages": [m.model_dump() for m in result.messages],
            "totalCount": result.total_count,
            "hasMore": result.has_more,
        })
    except Exception as e:
        log.error("WS gateway chat.history failed: %s", e)
        await conn.send_res(req_id, error=str(e))


# ---------------------------------------------------------------------------
# Method dispatch table
# ---------------------------------------------------------------------------

_METHODS: dict[str, Any] = {
    "ping": _handle_ping,
    "subscribe": _handle_subscribe,
    "chat.send": _handle_chat_send,
    "chat.abort": _handle_chat_abort,
    "chat.history": _handle_chat_history,
}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/v1/ws/gateway")
async def ws_gateway(ws: WebSocket):
    """Bidirectional WebSocket gateway for chat, events, and history.

    Replaces the SSE /v1/ws endpoint with a unified connection that supports
    both sending requests and receiving events over a single channel.
    """
    await ws.accept()
    conn = GatewayConnection(ws)
    log.info("WS gateway connected")

    # Server-side keepalive — survives proxies (Traefik) that don't
    # forward WebSocket protocol-level pings.
    async def _keepalive():
        try:
            while True:
                await asyncio.sleep(25)
                await conn.send_evt("ping", {"ts": _now_iso()})
        except (asyncio.CancelledError, WebSocketDisconnect, Exception):
            pass

    conn._keepalive_task = asyncio.create_task(_keepalive())

    try:
        while True:
            raw = await ws.receive_text()
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                await conn.send({"type": "res", "id": None, "ok": False, "error": "Invalid JSON"})
                continue

            frame_type = frame.get("type")
            req_id = frame.get("id")
            method = frame.get("method", "")
            params = frame.get("params") or {}

            if frame_type != "req":
                await conn.send({"type": "res", "id": req_id, "ok": False, "error": f"Unknown frame type: {frame_type}"})
                continue

            if not req_id:
                await conn.send({"type": "res", "id": None, "ok": False, "error": "Missing request id"})
                continue

            handler = _METHODS.get(method)
            if not handler:
                await conn.send_res(req_id, error=f"Unknown method: {method}")
                continue

            try:
                await handler(conn, req_id, params)
            except Exception as e:
                log.exception("WS gateway handler error for method=%s", method)
                await conn.send_res(req_id, error=str(e))

    except WebSocketDisconnect:
        log.info("WS gateway disconnected")
    except Exception as e:
        log.exception("WS gateway error")
    finally:
        await conn.cleanup()
