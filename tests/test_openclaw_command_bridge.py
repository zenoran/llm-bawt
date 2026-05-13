from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from openclaw_bridge.bridge import SessionBridge
from agent_bridge.events import AgentEventKind
from openclaw_bridge.ingest import EventIngestPipeline


class _StubWsClient:
    def __init__(self) -> None:
        self.history_calls = 0

    async def send_and_stream(self, session_key: str, text: str, *, attachments=None):
        yield {
            "type": "event",
            "event": "agent",
            "payload": {
                "runId": "run_err",
                "stream": "lifecycle",
                "data": {"phase": "error", "error": "rate limited"},
                "sessionKey": session_key,
                "seq": 1,
            },
        }

    async def get_chat_history(self, session_key: str, *, limit: int = 20) -> list[dict]:
        self.history_calls += 1
        if self.history_calls == 1:
            return [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hi"}],
                    "timestamp": "2026-03-19T02:45:02Z",
                },
            ]
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hi"}],
                "timestamp": "2026-03-19T02:45:02Z",
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hey Nick"}],
                "timestamp": "2026-03-19T02:45:38Z",
            },
        ]


def test_handle_send_command_recovers_reply_from_history_after_empty_stream():
    ws_client = _StubWsClient()
    ingest = EventIngestPipeline()
    store = MagicMock()
    publisher = MagicMock()
    bridge = SessionBridge(ws_client, ingest, store, publisher)
    bridge._history_reply_timeout_s = 0.05
    bridge._history_reply_poll_interval_s = 0.0

    async_redis = MagicMock()
    async_redis.xack = AsyncMock()

    asyncio.run(
        bridge._handle_send_command(
            {
                "request_id": "req_123",
                "session_key": "agent:main:main",
                "message": "Hi",
            },
            "1-0",
            async_redis,
        )
    )

    published_events = [
        call.args[1]
        for call in publisher.publish_run_event.call_args_list
        if len(call.args) == 2
    ]
    recovered = [
        event for event in published_events
        if getattr(event, "kind", None) == AgentEventKind.ASSISTANT_DELTA
        and getattr(event, "text", None) == "Hey Nick"
    ]

    assert recovered, "expected synthetic assistant delta recovered from chat history"
    publisher.publish_run_done.assert_called_once_with("req_123")
    async_redis.xack.assert_awaited_once()
