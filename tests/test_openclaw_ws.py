"""Unit tests for OpenClaw WebSocket client.

These tests target the real production handshake: an Ed25519 device-identity
challenge/connect flow over the websocket.  On connect the client:

  1. ``recv()``s a ``connect.challenge`` frame and requires ``payload.nonce``.
  2. Sends a ``{"type":"req","method":"connect",...}`` frame carrying a signed
     device block and an ``auth`` block (the token lives here, NOT in HTTP
     headers).
  3. ``recv()``s a ``{"type":"res","id":<same id>,"ok":true,...}`` frame.

After connect, ``_receive_loop`` iterates the same socket.  ``chat.send`` is
issued via ``_request`` with camelCase params and tracked in
``_pending_requests`` keyed by req id; the matching ``res`` frame carries
``payload.runId``.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

from llm_bawt.integrations.openclaw_ws import OpenClawWsClient, OpenClawWsConfig


class _FakeWebSocket:
    """Mock websocket connection for testing.

    ``recv()`` and async iteration share a single cursor so frames consumed
    during the handshake (via ``recv()``) are not re-delivered to the receive
    loop (via ``async for``).  Once queued frames are exhausted ``recv()``
    blocks forever like a real idle socket, and async iteration stops.
    """

    def __init__(self, messages: list[str | dict] | None = None):
        self._messages = [
            json.dumps(m) if isinstance(m, dict) else m
            for m in (messages or [])
        ]
        self._sent: list[str] = []
        self._closed = False
        self._idx = 0
        # Signalled whenever a new frame is queued so a blocked recv()/iter
        # can wake up; lets tests inject frames after the handshake starts.
        self._has_frame = asyncio.Event()
        if self._messages:
            self._has_frame.set()

    def queue(self, message: str | dict) -> None:
        """Append a frame for later delivery (e.g. a res echoing a sent id)."""
        self._messages.append(
            json.dumps(message) if isinstance(message, dict) else message
        )
        self._has_frame.set()

    async def send(self, data: str) -> None:
        self._sent.append(data)

    async def _next(self) -> str:
        # Block (like a real idle, still-open socket) until a frame is
        # available, then advance the shared cursor.
        while self._idx >= len(self._messages):
            self._has_frame.clear()
            await self._has_frame.wait()
        msg = self._messages[self._idx]
        self._idx += 1
        return msg

    async def recv(self) -> str:
        return await self._next()

    async def close(self) -> None:
        self._closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._next()


def _make_client(**kwargs) -> OpenClawWsClient:
    config = OpenClawWsConfig(
        url=kwargs.get("url", "ws://localhost:18789/v1/ws"),
        token=kwargs.get("token", "test-token"),
        session_keys=kwargs.get("session_keys", ["main"]),
        reconnect_max_delay=kwargs.get("reconnect_max_delay", 1),
    )
    return OpenClawWsClient(config)


def _challenge_frame(nonce: str = "test-nonce-123") -> dict:
    return {"event": "connect.challenge", "payload": {"nonce": nonce}}


def _last_sent_connect_id(ws: _FakeWebSocket) -> str:
    """Read the req id from the connect frame the client just sent."""
    assert ws._sent, "client sent no frames"
    req = json.loads(ws._sent[-1])
    assert req["type"] == "req"
    assert req["method"] == "connect"
    return req["id"]


async def _serve_handshake(client: OpenClawWsClient, ws: _FakeWebSocket,
                           res_payload: dict | None = None) -> None:
    """Drive a full challenge -> connect -> res handshake against ``ws``.

    The client reads the challenge, sends its connect req, then reads the res.
    Because ``recv()`` blocks once frames are exhausted, we run ``_do_connect``
    as a task and queue the ``res`` frame (echoing the client's req id) as soon
    as the connect frame has been sent.
    """
    ws._messages.insert(0, json.dumps(_challenge_frame()))
    ws._has_frame.set()

    connect_task = asyncio.create_task(client._do_connect())

    # Wait until the client has sent its connect req, then queue the matching
    # res so its second recv() unblocks.
    for _ in range(200):
        if ws._sent:
            break
        await asyncio.sleep(0.005)
    req_id = _last_sent_connect_id(ws)
    ws.queue({
        "type": "res",
        "id": req_id,
        "ok": True,
        "payload": res_payload or {},
    })

    await connect_task


class TestOpenClawWsClient:
    def test_connect_and_subscribe(self):
        """Challenge -> connect -> res handshake leaves client connected."""
        ws = _FakeWebSocket()

        async def run():
            client = _make_client()
            with patch("openclaw_bridge.ws_client.websockets") as mock_ws:
                mock_ws.connect = AsyncMock(return_value=ws)
                await _serve_handshake(client, ws)

                assert client.connected
                # Subscriptions come straight from config (no subscribe frame).
                assert "main" in client.subscribed_sessions

                # The client sent a connect req (not a legacy subscribe frame).
                connect_reqs = [
                    json.loads(s) for s in ws._sent
                    if json.loads(s).get("method") == "connect"
                ]
                assert len(connect_reqs) == 1
                req = connect_reqs[0]
                assert req["type"] == "req"
                assert req["params"]["minProtocol"] == 3
                # No subscribe frames are part of the real protocol.
                assert all(
                    json.loads(s).get("type") != "subscribe" for s in ws._sent
                )

                client._closing = True  # prevent reconnect

        asyncio.run(run())

    def test_auth_header_sent(self):
        """Token is transmitted in the connect req's auth block (not HTTP)."""

        async def run():
            client = _make_client(token="my-secret-token")
            ws = _FakeWebSocket()
            with patch("openclaw_bridge.ws_client.websockets") as mock_ws:
                mock_ws.connect = AsyncMock(return_value=ws)
                await _serve_handshake(client, ws)

                # Production no longer passes HTTP auth headers.
                call = mock_ws.connect.call_args
                assert "additional_headers" not in call.kwargs

                # The token must travel inside the signed connect req's auth.
                req = json.loads(ws._sent[-1])
                assert req["method"] == "connect"
                assert req["params"]["auth"]["token"] == "my-secret-token"

                client._closing = True

        asyncio.run(run())

    def test_send_user_message(self):
        """chat.send goes through _request with camelCase params + runId res."""
        ws = _FakeWebSocket()

        async def run():
            client = _make_client()
            with patch("openclaw_bridge.ws_client.websockets") as mock_ws:
                mock_ws.connect = AsyncMock(return_value=ws)
                await _serve_handshake(client, ws)
                assert client.connected

                # Respond to the chat.send req via the _pending_requests flow:
                # watch for the sent frame, then resolve its future with a res
                # frame carrying payload.runId.
                async def confirm():
                    for _ in range(200):
                        for msg_str in ws._sent:
                            msg = json.loads(msg_str)
                            if msg.get("method") == "chat.send":
                                req_id = msg["id"]
                                fut = client._pending_requests.get(req_id)
                                if fut and not fut.done():
                                    fut.set_result({
                                        "type": "res",
                                        "id": req_id,
                                        "ok": True,
                                        "payload": {"runId": "run_xyz"},
                                    })
                                    return
                        await asyncio.sleep(0.005)

                task = asyncio.create_task(confirm())
                run_id = await client.send_user_message("main", "hello")
                await task

                assert run_id == "run_xyz"

                # Verify the sent chat.send frame shape (camelCase params).
                send_frames = [
                    json.loads(s) for s in ws._sent
                    if json.loads(s).get("method") == "chat.send"
                ]
                assert len(send_frames) == 1
                req = send_frames[0]
                assert req["type"] == "req"
                params = req["params"]
                assert params["sessionKey"] == "main"
                assert params["message"] == "hello"
                assert "idempotencyKey" in params

                client._closing = True

        asyncio.run(run())

    def test_graceful_disconnect(self):
        """disconnect() closes the ws and clears connected state."""
        ws = _FakeWebSocket()

        async def run():
            client = _make_client()
            client._ws = ws
            client._connected = True
            await client.disconnect()

            assert ws._closed
            assert not client.connected

        asyncio.run(run())

    def test_event_callback_called(self):
        """Trailing event frames reach the registered callback via _receive_loop."""
        ws = _FakeWebSocket()
        received = []

        async def run():
            client = _make_client()

            async def on_event(data):
                received.append(data)

            client.on_event(on_event)

            with patch("openclaw_bridge.ws_client.websockets") as mock_ws:
                mock_ws.connect = AsyncMock(return_value=ws)
                await _serve_handshake(client, ws)
                assert client.connected

                # Queue an event frame for the receive loop to deliver.
                ws.queue({
                    "type": "event",
                    "event": "agent",
                    "payload": {
                        "sessionKey": "main",
                        "stream": "assistant",
                        "data": {"delta": "hi"},
                    },
                })

                # Let the receive loop pick up the queued event.
                await asyncio.sleep(0.1)

                client._closing = True
                if client._receive_task:
                    client._receive_task.cancel()
                    try:
                        await client._receive_task
                    except (asyncio.CancelledError, Exception):
                        pass

        asyncio.run(run())

        # The trailing event must have been forwarded to the callback.
        event_msgs = [r for r in received if r.get("type") == "event"]
        assert len(event_msgs) == 1
        assert event_msgs[0]["event"] == "agent"

    def test_empty_url_skips_connect(self):
        """No connection attempted when URL is empty."""

        async def run():
            client = _make_client(url="")
            await client.connect()
            assert not client.connected

        asyncio.run(run())
