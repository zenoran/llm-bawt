from __future__ import annotations
import asyncio
import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class OpenClawWsConfig:
    url: str = ""
    token: str = ""
    session_keys: list[str] = field(default_factory=list)
    reconnect_max_delay: int = 60


class OpenClawWsClient:
    def __init__(self, config: OpenClawWsConfig) -> None:
        self._config = config
        self._ws = None
        self._connected = False
        self._subscribed_sessions: set[str] = set(config.session_keys)
        self._event_callback: Callable[[dict], Awaitable[None]] | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._receive_task: asyncio.Task | None = None
        self._closing = False
        self._stream_seq = 0
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._run_queues: dict[str, asyncio.Queue[dict | None]] = {}
        # Per-session cancel events — set by chat.abort to unblock send_and_stream
        self._session_cancel_events: dict[str, asyncio.Event] = {}

    async def connect(self) -> None:
        if not self._config.url:
            logger.warning("OpenClaw WS URL not configured, skipping connect")
            return
        self._closing = False
        await self._do_connect()

    async def _do_connect(self) -> None:
        if websockets is None:
            logger.error("websockets package not installed. Install with: pip install websockets>=13.0")
            return

        try:
            self._ws = await websockets.connect(
                self._config.url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            )

            raw_first = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            first = json.loads(raw_first) if isinstance(raw_first, str) else json.loads(raw_first.decode())
            nonce = (((first or {}).get("payload") or {}).get("nonce") if first.get("event") == "connect.challenge" else None)
            if not nonce:
                raise RuntimeError("gateway connect challenge missing nonce")

            req_id = uuid.uuid4().hex
            caps = ["tool-events"]
            connect_req = {
                "type": "req",
                "id": req_id,
                "method": "connect",
                "params": {
                    "minProtocol": 3,
                    "maxProtocol": 3,
                    "client": {
                        "id": "cli",
                        "version": "0.1.0",
                        "platform": "linux",
                        "mode": "cli",
                    },
                    "role": "operator",
                    "caps": caps,
                    "scopes": ["operator.read", "operator.write", "operator.admin", "chat.read", "chat.write", "session.read", "session.write"],
                    "auth": {"token": self._config.token} if self._config.token else {},
                },
            }
            await self._ws.send(json.dumps(connect_req))

            raw_res = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            res = json.loads(raw_res) if isinstance(raw_res, str) else json.loads(raw_res.decode())
            if res.get("type") != "res" or res.get("id") != req_id or not res.get("ok"):
                err = ((res or {}).get("error") or {}).get("message") or str(res)
                raise RuntimeError(f"gateway connect failed: {err}")

            self._connected = True
            logger.info("OpenClaw WS connected to %s (caps=%s)", self._config.url, ",".join(caps))
            from .metrics import get_metrics
            get_metrics().incr("openclaw.ws_connects")
            self._receive_task = asyncio.create_task(self._receive_loop())

        except Exception as e:
            self._connected = False
            logger.warning("OpenClaw WS connect failed: %s", e)
            from .metrics import get_metrics
            get_metrics().incr("openclaw.ws_connect_failures")
            if not self._closing:
                self._schedule_reconnect()

    async def _receive_loop(self) -> None:
        try:
            async for raw_msg in self._ws:
                if self._closing:
                    break
                self._stream_seq += 1
                try:
                    data = json.loads(raw_msg) if isinstance(raw_msg, str) else json.loads(raw_msg.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.debug("OpenClaw WS non-JSON message")
                    continue

                if data.get("type") == "res":
                    req_id = data.get("id", "")
                    fut = self._pending_requests.pop(req_id, None)
                    if fut and not fut.done():
                        fut.set_result(data)
                    continue

                # Route to per-run listeners if this is an event with a runId
                if data.get("type") == "event":
                    payload = data.get("payload") or {}
                    run_id = payload.get("runId") or payload.get("run_id")
                    if run_id and run_id in self._run_queues:
                        try:
                            self._run_queues[run_id].put_nowait(data)
                        except asyncio.QueueFull:
                            logger.warning("Run queue full for %s, dropping event", run_id)

                if self._event_callback:
                    try:
                        await self._event_callback(data)
                    except Exception:
                        logger.exception("Error in WS event callback")

        except Exception as e:
            if not self._closing:
                logger.warning("OpenClaw WS receive loop error: %s", e)
                from .metrics import get_metrics
                get_metrics().incr("openclaw.ws_disconnects", reason="error")
        finally:
            self._connected = False
            if not self._closing:
                self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        if self._reconnect_task and not self._reconnect_task.done():
            return
        self._reconnect_task = asyncio.create_task(self._reconnect_with_backoff())

    async def _reconnect_with_backoff(self) -> None:
        base_delay = 1.0
        attempt = 0
        while not self._closing:
            delay = min(base_delay * (2 ** attempt), self._config.reconnect_max_delay)
            jitter = delay * 0.3 * (2 * random.random() - 1)
            actual_delay = max(0.1, delay + jitter)
            logger.info("OpenClaw WS reconnecting in %.1fs (attempt %d)", actual_delay, attempt + 1)
            await asyncio.sleep(actual_delay)
            if self._closing:
                break
            try:
                await self._do_connect()
                if self._connected:
                    logger.info("OpenClaw WS reconnected after %d attempts", attempt + 1)
                    return
            except Exception as e:
                logger.warning("OpenClaw WS reconnect attempt %d failed: %s", attempt + 1, e)
            attempt += 1

    async def disconnect(self) -> None:
        self._closing = True
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._connected = False
        logger.info("OpenClaw WS disconnected")

    async def _request(self, method: str, params: dict) -> dict:
        if not self._ws or not self._connected:
            raise RuntimeError("OpenClaw WS not connected")

        req_id = uuid.uuid4().hex
        msg = {"type": "req", "id": req_id, "method": method, "params": params}

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        self._pending_requests[req_id] = fut

        await self._ws.send(json.dumps(msg))
        try:
            res = await asyncio.wait_for(fut, timeout=30.0)
            if not res.get("ok"):
                err = ((res or {}).get("error") or {}).get("message") or str(res)
                raise RuntimeError(err)
            return res
        except asyncio.TimeoutError:
            self._pending_requests.pop(req_id, None)
            raise RuntimeError(f"Timed out waiting for gateway response: {method}")

    async def send_user_message(self, session_key: str, text: str, attachments: list | None = None) -> str:
        params: dict = {
            "sessionKey": session_key,
            "message": text,
            "idempotencyKey": f"idem_{uuid.uuid4().hex}",
        }
        if attachments:
            params["attachments"] = attachments

        res = await self._request("chat.send", params)
        payload = res.get("payload") or {}
        run_id = str(payload.get("runId") or payload.get("run_id") or params["idempotencyKey"])
        return run_id

    async def send_and_stream(
        self,
        session_key: str,
        text: str,
        *,
        attachments: list | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[dict]:
        """Send a message via chat.send and yield raw WS events for the resulting run.

        Events are yielded until a lifecycle end/error event is received,
        WS disconnects, or optional timeout. The caller gets the full stream
        of agent events (assistant deltas, tool events, lifecycle, errors).
        """
        run_id = await self.send_user_message(session_key, text, attachments=attachments)
        queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=2000)
        self._run_queues[run_id] = queue

        # Ensure a cancel event exists for this session so cancel_session()
        # can signal it.  We grab the reference once; cancel_session() sets
        # the same Event object.
        if session_key not in self._session_cancel_events:
            self._session_cancel_events[session_key] = asyncio.Event()
        cancel_event = self._session_cancel_events[session_key]

        try:
            deadline = (asyncio.get_event_loop().time() + timeout) if timeout else None
            while True:
                # Check if session was cancelled (chat.abort)
                if cancel_event and cancel_event.is_set():
                    logger.info("send_and_stream cancelled via chat.abort for run %s session=%s", run_id, session_key)
                    return

                if deadline is not None:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        logger.warning("send_and_stream timeout for run %s", run_id)
                        return
                    wait = min(remaining, 5.0)
                else:
                    wait = 5.0

                try:
                    raw = await asyncio.wait_for(queue.get(), timeout=wait)
                except asyncio.TimeoutError:
                    if not self._connected:
                        logger.warning("WS disconnected during send_and_stream for run %s", run_id)
                        return
                    continue

                if raw is None:
                    return

                yield raw

                # Check if this is a lifecycle end/error -> run done
                payload = raw.get("payload") or {}
                if raw.get("type") == "event":
                    if raw.get("event") == "agent":
                        stream = payload.get("stream")
                        data = payload.get("data") or {}
                        if stream == "lifecycle" and data.get("phase") in ("end", "error"):
                            return
                        if stream == "error":
                            return
                    if raw.get("event") == "chat":
                        state = str(payload.get("state") or "").lower()
                        if state == "final":
                            return
        finally:
            self._run_queues.pop(run_id, None)

    async def get_chat_history(self, session_key: str, *, limit: int = 50) -> list[dict]:
        """Fetch chat history for a session via the gateway."""
        res = await self._request("chat.history", {"sessionKey": session_key, "limit": limit})
        payload = res.get("payload") or {}
        return payload.get("messages") or payload.get("history") or []

    def on_event(self, callback: Callable[[dict], Awaitable[None]]) -> None:
        self._event_callback = callback

    def cancel_session(self, session_key: str) -> None:
        """Signal any active send_and_stream for this session to stop.

        Called when chat.abort is received so the bridge releases the
        session lock and allows the next send to proceed.
        """
        event = self._session_cancel_events.get(session_key)
        if event is None:
            event = asyncio.Event()
            self._session_cancel_events[session_key] = event
        event.set()
        logger.info("Session cancel signalled: %s", session_key)

    def clear_session_cancel(self, session_key: str) -> None:
        """Reset the cancel event for a session before a new send."""
        event = self._session_cancel_events.get(session_key)
        if event is not None:
            event.clear()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def subscribed_sessions(self) -> set[str]:
        return set(self._subscribed_sessions)

    @property
    def stream_seq(self) -> int:
        return self._stream_seq
