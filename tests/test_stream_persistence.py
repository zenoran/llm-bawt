"""Durability tests for streaming persistence and shared turn finalization."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from llm_bawt.service.background_service import BackgroundService
from llm_bawt.service.chat_stream_worker import consume_stream_chunks


class _ClosedLoop:
    """Event-loop stub that simulates a disconnected client loop."""

    def call_soon_threadsafe(self, *args, **kwargs):
        raise RuntimeError("Event loop is closed")


class _CapturingLoop:
    """Event-loop stub that captures thread-safe queue puts."""

    def call_soon_threadsafe(self, callback, *args):
        callback(*args)


def _build_service_for_finalize() -> BackgroundService:
    service = BackgroundService.__new__(BackgroundService)
    service.config = SimpleNamespace(DEBUG_TURN_LOG=False)
    service._update_turn_log = Mock()
    return service


def test_finalize_turn_saves_history() -> None:
    service = _build_service_for_finalize()
    service._extract_agent_backend_tool_calls = Mock(return_value=[])

    llm_bawt = SimpleNamespace(finalize_response=Mock(), adapter=None)

    service._finalize_turn(
        llm_bawt=llm_bawt,
        turn_id="turn-1",
        response_text="hello world",
        tool_context="",
        tool_call_details=[],
        prepared_messages=[{"role": "user", "content": "hi"}],
        user_prompt="hi",
        model="test-model",
        bot_id="proto",
        user_id="nick",
        elapsed_ms=12.5,
        stream=False,
    )

    llm_bawt.finalize_response.assert_called_once_with("hello world", "")
    service._update_turn_log.assert_called_once()
    kwargs = service._update_turn_log.call_args.kwargs
    assert kwargs["turn_id"] == "turn-1"
    assert kwargs["status"] == "ok"
    assert kwargs["response_text"] == "hello world"


def test_finalize_turn_extracts_agent_backend_tools(monkeypatch) -> None:
    service = _build_service_for_finalize()

    class FakeAgentBackendClient:
        def get_tool_calls(self):
            return [
                {
                    "name": "search_web",
                    "display_name": "search_web",
                    "arguments": {"query": "ai"},
                    "result": "ok",
                }
            ]

    from llm_bawt.service import background_service as bg_module

    monkeypatch.setattr(bg_module, "AgentBackendClient", FakeAgentBackendClient)
    llm_bawt = SimpleNamespace(finalize_response=Mock(), adapter=None, client=FakeAgentBackendClient())

    tool_calls: list[dict] = []
    service._finalize_turn(
        llm_bawt=llm_bawt,
        turn_id="turn-2",
        response_text="done",
        tool_context="",
        tool_call_details=tool_calls,
        prepared_messages=[],
        user_prompt="prompt",
        model="test-model",
        bot_id="proto",
        user_id="nick",
        elapsed_ms=20.0,
        stream=True,
    )

    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "search_web"
    assert tool_calls[0]["parameters"] == {"query": "ai"}
    assert tool_calls[0]["result"] == "ok"


def test_stream_persists_on_client_disconnect() -> None:
    cancel_event = threading.Event()
    full_response_holder = [""]

    cancelled = consume_stream_chunks(
        iter(["hello", " ", "world"]),
        cancel_event=cancel_event,
        loop=_ClosedLoop(),
        chunk_queue=asyncio.Queue(),
        full_response_holder=full_response_holder,
    )

    assert cancelled is False
    assert full_response_holder[0] == "hello world"


def test_stream_persists_on_cancellation() -> None:
    cancel_event = threading.Event()
    full_response_holder = [""]

    def _iter_chunks():
        yield "a"
        cancel_event.set()
        yield "b"
        yield "c"

    cancelled = consume_stream_chunks(
        _iter_chunks(),
        cancel_event=cancel_event,
        loop=_CapturingLoop(),
        chunk_queue=asyncio.Queue(),
        full_response_holder=full_response_holder,
    )

    assert cancelled is True
    assert full_response_holder[0] == "abc"


def test_stream_accumulates_only_str_chunks() -> None:
    cancel_event = threading.Event()
    full_response_holder = [""]

    cancelled = consume_stream_chunks(
        iter(["hello", {"event": "tool_call", "name": "search"}, " world"]),
        cancel_event=cancel_event,
        loop=_CapturingLoop(),
        chunk_queue=asyncio.Queue(),
        full_response_holder=full_response_holder,
    )

    assert cancelled is False
    assert full_response_holder[0] == "hello world"
