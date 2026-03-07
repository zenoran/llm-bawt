"""Unit tests for OpenClaw WebSocket client."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_bawt.integrations.openclaw_ws import OpenClawWsClient, OpenClawWsConfig


class _FakeWebSocket:
    """Mock websocket connection for testing."""

    def __init__(self, messages: list[str | dict] | None = None):
        self._messages = [
            json.dumps(m) if isinstance(m, dict) else m
            for m in (messages or [])
        ]
        self._sent: list[str] = []
        self._closed = False
        self._idx = 0

    async def send(self, data: str) -> None:
        self._sent.append(data)

    async def close(self) -> None:
        self._closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._messages):
            raise StopAsyncIteration
        msg = self._messages[self._idx]
        self._idx += 1
        return msg


def _make_client(**kwargs) -> OpenClawWsClient:
    config = OpenClawWsConfig(
        url=kwargs.get("url", "ws://localhost:18789/v1/ws"),
        token=kwargs.get("token", "test-token"),
        session_keys=kwargs.get("session_keys", ["main"]),
        reconnect_max_delay=kwargs.get("reconnect_max_delay", 1),
    )
    return OpenClawWsClient(config)


class TestOpenClawWsClient:
    def test_connect_and_subscribe(self):
        """WS handshake + subscribe message sent."""
        ws = _FakeWebSocket([
            {"type": "subscribed", "session_keys": ["main"]},
        ])

        async def run():
            client = _make_client()
            with patch("openclaw_bridge.ws_client.websockets") as mock_ws:
                mock_ws.connect = AsyncMock(return_value=ws)
                await client._do_connect()

                assert client.connected
                assert "main" in client.subscribed_sessions

                # Verify subscribe message was sent
                assert len(ws._sent) == 1
                sub_msg = json.loads(ws._sent[0])
                assert sub_msg["type"] == "subscribe"
                assert sub_msg["session_keys"] == ["main"]

                client._closing = True  # prevent reconnect

        asyncio.run(run())

    def test_auth_header_sent(self):
        """Bearer token included in WS headers."""

        async def run():
            client = _make_client(token="my-secret-token")
            with patch("openclaw_bridge.ws_client.websockets") as mock_ws:
                mock_ws.connect = AsyncMock(return_value=_FakeWebSocket())

                await client._do_connect()

                # Check that connect was called with auth headers
                call_kwargs = mock_ws.connect.call_args
                headers = call_kwargs.kwargs.get("additional_headers", {})
                assert headers.get("Authorization") == "Bearer my-secret-token"

                client._closing = True

        asyncio.run(run())

    def test_send_user_message(self):
        """Message sent as JSON via WS."""
        # Pre-load a chat.sent response
        ws = _FakeWebSocket([
            {"type": "subscribed", "session_keys": ["main"]},
        ])

        async def run():
            client = _make_client()
            with patch("openclaw_bridge.ws_client.websockets") as mock_ws:
                mock_ws.connect = AsyncMock(return_value=ws)
                await client._do_connect()

                # Simulate sending and receiving confirmation
                async def send_and_confirm():
                    # Wait for the send to happen, then inject confirmation
                    await asyncio.sleep(0.05)
                    # Find the idempotency_key from sent messages
                    for msg_str in ws._sent:
                        msg = json.loads(msg_str)
                        if msg.get("type") == "chat.send":
                            idem_key = msg.get("idempotency_key", "")
                            if idem_key and idem_key in client._pending_sends:
                                fut = client._pending_sends.pop(idem_key)
                                if not fut.done():
                                    fut.set_result("run_xyz")
                                return

                task = asyncio.create_task(send_and_confirm())
                run_id = await client.send_user_message("main", "hello")
                await task

                assert run_id == "run_xyz"

                # Verify the sent message format
                sent_msgs = [json.loads(s) for s in ws._sent if "chat.send" in s]
                assert len(sent_msgs) == 1
                assert sent_msgs[0]["type"] == "chat.send"
                assert sent_msgs[0]["session_key"] == "main"
                assert sent_msgs[0]["text"] == "hello"
                assert "idempotency_key" in sent_msgs[0]

                client._closing = True

        asyncio.run(run())

    def test_graceful_disconnect(self):
        """disconnect() sends close frame."""
        ws = _FakeWebSocket()

        async def run():
            client = _make_client()
            client._ws = ws
            client._connected = True
            await client.disconnect()

            assert ws._closed
            assert not client.connected
            assert len(client.subscribed_sessions) == 0

        asyncio.run(run())

    def test_event_callback_called(self):
        """Registered callback receives parsed WS messages."""
        messages = [
            {"type": "subscribed", "session_keys": ["main"]},
            {
                "type": "event",
                "session_key": "main",
                "event_id": "evt_1",
                "event_type": "response.output_text.delta",
                "data": {"delta": "hi"},
            },
        ]
        ws = _FakeWebSocket(messages)

        received = []

        async def run():
            client = _make_client()

            async def on_event(data):
                received.append(data)

            client.on_event(on_event)

            with patch("openclaw_bridge.ws_client.websockets") as mock_ws:
                mock_ws.connect = AsyncMock(return_value=ws)
                await client._do_connect()

                # Wait for receive loop to process messages
                await asyncio.sleep(0.1)

                client._closing = True
                if client._receive_task:
                    client._receive_task.cancel()
                    try:
                        await client._receive_task
                    except (asyncio.CancelledError, Exception):
                        pass

        asyncio.run(run())

        # subscribed ack is not forwarded, but the event is
        assert len(received) >= 1
        event_msgs = [r for r in received if r.get("type") == "event"]
        assert len(event_msgs) == 1
        assert event_msgs[0]["event_type"] == "response.output_text.delta"

    def test_empty_url_skips_connect(self):
        """No connection attempted when URL is empty."""

        async def run():
            client = _make_client(url="")
            await client.connect()
            assert not client.connected

        asyncio.run(run())
