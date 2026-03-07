from __future__ import annotations
import asyncio
import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class OpenClawWsConfig:
    url: str = ""
    token: str = ""
    session_keys: list[str] = field(default_factory=lambda: ["main"])
    reconnect_max_delay: int = 60


class OpenClawWsClient:
    def __init__(self, config: OpenClawWsConfig) -> None:
        self._config = config
        self._ws = None
        self._connected = False
        self._subscribed_sessions: set[str] = set(config.session_keys or ["main"])
        self._event_callback: Callable[[dict], Awaitable[None]] | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._receive_task: asyncio.Task | None = None
        self._closing = False
        self._stream_seq = 0
        self._pending_requests: dict[str, asyncio.Future] = {}

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
            logger.info("OpenClaw WS connected to %s", self._config.url)
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

    async def send_user_message(self, session_key: str, text: str) -> str:
        params = {
            "sessionKey": session_key,
            "message": text,
            "idempotencyKey": f"idem_{uuid.uuid4().hex}",
        }

        res = await self._request("chat.send", params)
        payload = res.get("payload") or {}
        run_id = str(payload.get("runId") or payload.get("run_id") or params["idempotencyKey"])
        return run_id

    async def get_chat_history(self, session_key: str, *, limit: int = 50) -> list[dict]:
        """Fetch chat history for a session via the gateway."""
        res = await self._request("chat.history", {"sessionKey": session_key, "limit": limit})
        payload = res.get("payload") or {}
        return payload.get("messages") or payload.get("history") or []

    def on_event(self, callback: Callable[[dict], Awaitable[None]]) -> None:
        self._event_callback = callback

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def subscribed_sessions(self) -> set[str]:
        return set(self._subscribed_sessions)

    @property
    def stream_seq(self) -> int:
        return self._stream_seq
