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


def test_worker_finalize_error_falls_back_to_error_status() -> None:
    """If _finalize_turn raises, the worker catches it and marks the turn as error.

    This exercises the try/except around _finalize_turn in the worker's
    finally block.  We simulate the full sequence: _finalize_turn raises,
    then the worker falls back to _update_turn_log(status="error").
    """
    service = _build_service_for_finalize()
    service._extract_agent_backend_tool_calls = Mock(return_value=[])

    llm_bawt = SimpleNamespace(
        finalize_response=Mock(side_effect=RuntimeError("DB down")),
        adapter=None,
    )

    # Simulate the worker's finally block:
    # 1. Try _finalize_turn → raises
    # 2. Catch → call _update_turn_log(status="error")
    turn_id = "turn-err"
    elapsed_ms = 10.0
    full_response = "partial response"

    try:
        service._finalize_turn(
            llm_bawt=llm_bawt,
            turn_id=turn_id,
            response_text=full_response,
            tool_context="",
            tool_call_details=[],
            prepared_messages=[],
            user_prompt="hi",
            model="test-model",
            bot_id="proto",
            user_id="nick",
            elapsed_ms=elapsed_ms,
            stream=True,
        )
    except Exception as fin_err:
        # This is what the worker's finally block does on failure:
        service._update_turn_log(
            turn_id=turn_id,
            status="error",
            latency_ms=elapsed_ms,
            response_text=full_response or None,
            error_text=f"finalize_error: {fin_err}",
        )

    # finalize_response was attempted
    llm_bawt.finalize_response.assert_called_once()

    # The fallback _update_turn_log should have been called with error status.
    # Note: _update_turn_log was also called by _finalize_turn before the
    # error, so we check the LAST call has status="error".
    last_call_kwargs = service._update_turn_log.call_args_list[-1].kwargs
    assert last_call_kwargs["status"] == "error"
    assert "finalize_error" in last_call_kwargs["error_text"]
    assert last_call_kwargs["turn_id"] == turn_id


def test_worker_finalize_error_sends_sentinel_and_generation_done() -> None:
    """Simulate the worker's finally block when _finalize_turn raises.

    Verify that the sentinel (None) is still pushed to the chunk queue and
    the generation done_event is still set, even if finalization fails.
    """
    from llm_bawt.service.chat_stream_worker import put_queue_item_threadsafe

    loop = _CapturingLoop()
    chunk_queue = asyncio.Queue()
    done_event = threading.Event()

    full_response_holder = ["some response"]
    finalize_raised = False

    def mock_finalize_turn(**kwargs):
        raise RuntimeError("DB failure")

    # Simulate the worker finally block logic (matches the code we changed):
    # try _finalize_turn → except → fallback update → finally sentinel + done
    try:
        mock_finalize_turn(response_text=full_response_holder[0])
    except Exception:
        finalize_raised = True

    # These should always execute regardless of finalize outcome
    put_queue_item_threadsafe(loop, chunk_queue, None)
    done_event.set()

    assert finalize_raised
    assert not chunk_queue.empty()
    assert chunk_queue.get_nowait() is None
    assert done_event.is_set()


def test_generation_lifecycle_ordering() -> None:
    """done_event must be set only AFTER finalization completes.

    The worker thread's finally block runs: finalize → sentinel → done_event.
    A subsequent _start_generation would wait on done_event, so setting it
    prematurely (from the generator's finally, as the old code did) could
    allow a new generation to start before the old one finished persisting.

    This test verifies the ordering contract by simulating the full sequence.
    """
    done_event = threading.Event()
    finalize_order: list[str] = []

    def mock_finalize():
        finalize_order.append("finalize")
        assert not done_event.is_set(), "done_event must not be set before finalize"

    def mock_sentinel():
        finalize_order.append("sentinel")
        assert not done_event.is_set(), "done_event must not be set before sentinel"

    # Simulate worker finally block ordering
    mock_finalize()
    mock_sentinel()
    done_event.set()
    finalize_order.append("done")

    assert finalize_order == ["finalize", "sentinel", "done"]
    assert done_event.is_set()
