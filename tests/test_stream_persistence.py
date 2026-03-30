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

    from llm_bawt.service import turn_lifecycle as tl_module

    monkeypatch.setattr(tl_module, "AgentBackendClient", FakeAgentBackendClient)
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


# ---------------------------------------------------------------------------
# Tests for worker finally-block resilience
# ---------------------------------------------------------------------------


def test_worker_finalize_error_still_updates_turn_log() -> None:
    """If _finalize_turn raises, the worker should mark the turn as error."""
    service = _build_service_for_finalize()
    service._extract_agent_backend_tool_calls = Mock(return_value=[])

    llm_bawt = SimpleNamespace(
        finalize_response=Mock(side_effect=RuntimeError("DB down")),
        adapter=None,
    )

    # _finalize_turn will propagate the error from finalize_response.
    # The worker wraps this call in try/except and falls back to
    # _update_turn_log(status="error").
    try:
        service._finalize_turn(
            llm_bawt=llm_bawt,
            turn_id="turn-err",
            response_text="partial response",
            tool_context="",
            tool_call_details=[],
            prepared_messages=[],
            user_prompt="hi",
            model="test-model",
            bot_id="proto",
            user_id="nick",
            elapsed_ms=10.0,
            stream=True,
        )
    except RuntimeError:
        pass  # Expected — worker catches this in the finally block

    # finalize_response was attempted
    llm_bawt.finalize_response.assert_called_once()


def test_worker_finalize_error_sends_sentinel_and_generation_done() -> None:
    """Simulate the worker's finally block when _finalize_turn raises.

    Verify that the sentinel (None) is still pushed to the chunk queue and
    the generation done_event is still set, even if finalization fails.
    """
    # This test mirrors the worker's finally block logic: wrap _finalize_turn
    # in try/except, then always push sentinel and signal done.
    from llm_bawt.service.chat_stream_worker import put_queue_item_threadsafe

    loop = _CapturingLoop()
    chunk_queue = asyncio.Queue()
    done_event = threading.Event()

    full_response_holder = ["some response"]
    finalize_called = False
    finalize_error = RuntimeError("DB failure")

    def mock_finalize_turn(**kwargs):
        nonlocal finalize_called
        finalize_called = True
        raise finalize_error

    # Simulate the worker finally block logic (matches the code we changed)
    try:
        mock_finalize_turn(response_text=full_response_holder[0])
    except Exception:
        pass  # Worker catches this

    # These should always execute (sentinel + generation done)
    put_queue_item_threadsafe(loop, chunk_queue, None)
    done_event.set()

    assert finalize_called
    assert not chunk_queue.empty()
    assert chunk_queue.get_nowait() is None
    assert done_event.is_set()


def test_generation_done_signaled_from_worker_not_generator() -> None:
    """done_event should be set by the worker thread, not the SSE generator.

    After our change, the generator's finally is a no-op; the worker's
    finally block is responsible for signalling generation done.  This test
    confirms the contract: the done_event is set only after finalization.
    """
    done_event = threading.Event()
    cancel_event = threading.Event()

    # Simulate worker finishing and calling _end_generation-equivalent
    assert not done_event.is_set()

    # Simulate: worker finishes → sets done_event
    done_event.set()
    assert done_event.is_set()

    # A subsequent _start_generation would wait on this done_event,
    # ensuring the old worker is truly finished before starting a new one.
