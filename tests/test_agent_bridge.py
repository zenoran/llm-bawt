"""Unit tests for shared agent-bridge transport components.

Tests EventIngestPipeline (openclaw-flavored), EventStore, FanoutHub,
and SessionBridge — the parts that all bridge backends rely on.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_bawt.integrations.agent_bridge_events import (
    AgentEvent,
    AgentEventKind,
    synthesize_event_id,
)
from llm_bawt.integrations.openclaw_ingest import EventIngestPipeline


# ---------------------------------------------------------------------------
# EventIngestPipeline tests
# ---------------------------------------------------------------------------


class TestEventIngestPipeline:
    def setup_method(self):
        self.pipeline = EventIngestPipeline()

    def test_event_ingest_maps_delta(self):
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_001",
            "event_type": "response.output_text.delta",
            "data": {"delta": "Hello "},
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.ASSISTANT_DELTA
        assert event.text == "Hello "
        assert event.event_id == "evt_001"
        assert event.session_key == "main"

    def test_event_ingest_maps_tool_start(self):
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_002",
            "event_type": "response.output_item.added",
            "data": {
                "item": {
                    "type": "function_call",
                    "name": "exec",
                    "arguments": {"command": "ls -la"},
                }
            },
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.TOOL_START
        assert event.tool_name == "exec"
        assert event.tool_arguments == {"command": "ls -la"}

    def test_event_ingest_maps_tool_end(self):
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_003",
            "event_type": "response.output_item.added",
            "data": {
                "item": {
                    "type": "function_call_output",
                    "name": "exec",
                    "output": "file1.txt\nfile2.txt",
                }
            },
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.TOOL_END
        assert event.tool_name == "exec"
        assert event.tool_result == "file1.txt\nfile2.txt"

    def test_event_ingest_maps_run_started(self):
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_004",
            "event_type": "response.created",
            "data": {"response": {"id": "run_xyz", "model": "gpt-5"}},
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.RUN_STARTED
        assert event.run_id == "run_xyz"
        assert event.model == "gpt-5"

    def test_event_ingest_maps_run_completed(self):
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_005",
            "event_type": "response.completed",
            "data": {"response": {"id": "run_xyz", "model": "gpt-5"}},
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.RUN_COMPLETED
        assert event.run_id == "run_xyz"

    def test_event_ingest_drops_pings(self):
        for msg_type in ("ping", "pong", "heartbeat"):
            raw = {"type": msg_type}
            event = self.pipeline.parse(raw, "main")
            assert event is None, f"Expected None for {msg_type}"

    def test_event_ingest_drops_subscribe_acks(self):
        raw = {"type": "subscribed", "session_keys": ["main"]}
        event = self.pipeline.parse(raw, "main")
        assert event is None

    def test_event_ingest_unknown_event_passthrough(self):
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_006",
            "event_type": "some.unknown.type",
            "data": {"foo": "bar"},
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.SYSTEM_NOTE

    def test_event_ingest_chat_sent(self):
        raw = {
            "type": "chat.sent",
            "session_key": "main",
            "run_id": "run_abc",
            "message_id": "msg_001",
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.USER_MESSAGE
        assert event.origin == "user"
        assert event.run_id == "run_abc"

    def test_event_ingest_error_event(self):
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_err",
            "event_type": "error",
            "data": {"message": "something went wrong"},
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.ERROR
        assert event.text == "something went wrong"

    def test_event_ingest_chat_message_user(self):
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_msg",
            "event_type": "chat.message",
            "data": {"role": "user", "content": "hello"},
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.USER_MESSAGE
        assert event.text == "hello"

    def test_event_ingest_chat_message_assistant(self):
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_msg2",
            "event_type": "message",
            "data": {"role": "assistant", "content": "Hi there!"},
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.ASSISTANT_DONE
        assert event.text == "Hi there!"

    def test_event_ingest_seq_preserved(self):
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_seq",
            "seq": 42,
            "event_type": "response.output_text.delta",
            "data": {"delta": "x"},
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.seq == 42

    def test_event_ingest_tool_args_string_parsed(self):
        """Tool arguments provided as JSON string should be parsed."""
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_tc",
            "event_type": "response.output_item.added",
            "data": {
                "item": {
                    "type": "tool_call",
                    "name": "search",
                    "arguments": '{"query": "test"}',
                }
            },
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.TOOL_START
        assert event.tool_arguments == {"query": "test"}

    def test_synthesize_event_id_deterministic(self):
        id1 = synthesize_event_id("main", "delta", {"text": "hi"}, 1)
        id2 = synthesize_event_id("main", "delta", {"text": "hi"}, 1)
        assert id1 == id2

    def test_synthesize_event_id_differs_with_seq(self):
        id1 = synthesize_event_id("main", "delta", {"text": "hi"}, 1)
        id2 = synthesize_event_id("main", "delta", {"text": "hi"}, 2)
        assert id1 != id2


# ---------------------------------------------------------------------------
# EventStore tests (require PostgreSQL)
# ---------------------------------------------------------------------------

def _get_test_engine():
    """Create a test DB engine. Skip if no PG credentials."""
    pg_pass = os.environ.get("LLM_BAWT_POSTGRES_PASSWORD", "")
    if not pg_pass:
        pytest.skip("No PostgreSQL credentials (LLM_BAWT_POSTGRES_PASSWORD)")

    from sqlalchemy import create_engine

    pg_host = os.environ.get("LLM_BAWT_POSTGRES_HOST", "localhost")
    pg_port = os.environ.get("LLM_BAWT_POSTGRES_PORT", "5432")
    pg_user = os.environ.get("LLM_BAWT_POSTGRES_USER", "llm_bawt")
    pg_db = os.environ.get("LLM_BAWT_POSTGRES_DATABASE", "llm_bawt_test")

    url = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
    return create_engine(url)


@pytest.fixture
def event_store():
    """Create EventStore with fresh tables. Clean up after test."""
    engine = _get_test_engine()
    from llm_bawt.integrations.agent_bridge_store import EventStore, create_agent_event_tables
    from sqlalchemy import text

    create_agent_event_tables(engine)

    # Clean tables before test
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM agent_events"))
        conn.execute(text("DELETE FROM agent_session_state"))
        conn.execute(text("DELETE FROM agent_runs"))
        conn.commit()

    store = EventStore(engine)
    yield store

    # Clean up after test
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM agent_events"))
        conn.execute(text("DELETE FROM agent_session_state"))
        conn.execute(text("DELETE FROM agent_runs"))
        conn.commit()


def _make_event(
    event_id: str = "evt_test",
    session_key: str = "main",
    kind: AgentEventKind = AgentEventKind.ASSISTANT_DELTA,
    text: str | None = "hello",
    run_id: str | None = "run_001",
    **kwargs,
) -> AgentEvent:
    return AgentEvent(
        event_id=event_id,
        session_key=session_key,
        run_id=run_id,
        kind=kind,
        origin=kwargs.get("origin", "system"),
        text=text,
        raw=kwargs.get("raw", {"test": True}),
        seq=kwargs.get("seq"),
    )


class TestEventStore:
    def test_event_store_idempotent(self, event_store):
        """Storing same event_dedupe_key twice returns False second time."""
        event = _make_event(event_id="evt_idem_1")
        assert event_store.store(event) is True
        assert event.db_id is not None

        # Same dedupe key should return False
        event2 = _make_event(event_id="evt_idem_1", text="different")
        assert event_store.store(event2) is False

    def test_event_store_replay(self, event_store):
        """get_events(since_id=N) returns only events after N."""
        e1 = _make_event(event_id="evt_r1", text="first")
        e2 = _make_event(event_id="evt_r2", text="second")
        e3 = _make_event(event_id="evt_r3", text="third")

        event_store.store(e1)
        event_store.store(e2)
        event_store.store(e3)

        # Get all events after e1
        events = event_store.get_events("main", since_id=e1.db_id)
        assert len(events) == 2
        assert events[0].text == "second"
        assert events[1].text == "third"

    def test_event_store_assemble_run_text(self, event_store):
        """Deltas for a run reassemble into full text."""
        for i, chunk in enumerate(["Hello ", "world", "!"]):
            event_store.store(_make_event(
                event_id=f"evt_asm_{i}",
                run_id="run_assemble",
                kind=AgentEventKind.ASSISTANT_DELTA,
                text=chunk,
                seq=i,
            ))

        full = event_store.assemble_run_text("run_assemble")
        assert full == "Hello world!"

    def test_event_store_kind_filter(self, event_store):
        """get_events with kinds filter returns only matching kinds."""
        event_store.store(_make_event(event_id="evt_kf1", kind=AgentEventKind.ASSISTANT_DELTA))
        event_store.store(_make_event(event_id="evt_kf2", kind=AgentEventKind.RUN_STARTED, text=None))
        event_store.store(_make_event(event_id="evt_kf3", kind=AgentEventKind.ASSISTANT_DELTA))

        events = event_store.get_events(
            "main", kinds=[AgentEventKind.ASSISTANT_DELTA]
        )
        assert len(events) == 2
        assert all(e.kind == AgentEventKind.ASSISTANT_DELTA for e in events)

    def test_event_store_session_cursor(self, event_store):
        """Session cursor tracks last processed event."""
        assert event_store.get_session_cursor("main") is None

        event_store.update_session_cursor("main", 42)
        assert event_store.get_session_cursor("main") == 42

        event_store.update_session_cursor("main", 99)
        assert event_store.get_session_cursor("main") == 99

    def test_event_store_create_and_complete_run(self, event_store):
        """Run lifecycle: create → store deltas → complete."""
        event_store.create_run("run_lc", "main", model="gpt-5", origin="user")

        # Store some deltas
        for i, chunk in enumerate(["Hi ", "there"]):
            event_store.store(_make_event(
                event_id=f"evt_lc_{i}",
                run_id="run_lc",
                kind=AgentEventKind.ASSISTANT_DELTA,
                text=chunk,
                seq=i,
            ))

        full_text = event_store.assemble_run_text("run_lc")
        event_store.complete_run("run_lc", full_text, [{"name": "exec", "arguments": {}}])

        # Verify via direct query
        from sqlalchemy import text as sql_text
        with event_store._engine.connect() as conn:
            row = conn.execute(
                sql_text("SELECT status, full_text FROM agent_runs WHERE run_id = :rid"),
                {"rid": "run_lc"},
            ).fetchone()
        assert row is not None
        assert row[0] == "completed"
        assert row[1] == "Hi there"

    def test_event_store_ws_state(self, event_store):
        """WS connection state tracking."""
        event_store.update_session_ws_state("main", True)
        from sqlalchemy import text as sql_text
        with event_store._engine.connect() as conn:
            row = conn.execute(
                sql_text("SELECT ws_connected FROM agent_session_state WHERE session_key = :sk"),
                {"sk": "main"},
            ).fetchone()
        assert row is not None
        assert row[0] is True

        event_store.update_session_ws_state("main", False)
        with event_store._engine.connect() as conn:
            row = conn.execute(
                sql_text("SELECT ws_connected FROM agent_session_state WHERE session_key = :sk"),
                {"sk": "main"},
            ).fetchone()
        assert row[0] is False


# ---------------------------------------------------------------------------
# FanoutHub tests
# ---------------------------------------------------------------------------


class TestFanoutHub:
    def test_fanout_live_broadcast(self):
        """Subscriber receives events pushed via broadcast()."""
        from llm_bawt.integrations.agent_bridge_fanout import FanoutHub

        store = MagicMock()
        hub = FanoutHub(store)

        received = []

        async def run():
            sub = hub.subscribe("main")
            # Start consuming in background
            async def consume():
                async for event in sub:
                    received.append(event)
                    if len(received) >= 2:
                        break

            task = asyncio.create_task(consume())

            # Give subscriber time to register
            await asyncio.sleep(0.05)

            # Broadcast events
            e1 = _make_event(event_id="evt_f1", text="one")
            e2 = _make_event(event_id="evt_f2", text="two")
            hub.broadcast(e1)
            hub.broadcast(e2)

            await asyncio.wait_for(task, timeout=2.0)

        asyncio.run(run())
        assert len(received) == 2
        assert received[0].text == "one"
        assert received[1].text == "two"

    def test_fanout_replay_then_live(self):
        """Subscriber with since_event_id gets gap replay then live events."""
        from llm_bawt.integrations.agent_bridge_fanout import FanoutHub

        # Mock store that returns gap events
        store = MagicMock()
        gap_event = _make_event(event_id="evt_gap", text="from-db")
        store.get_events.return_value = [gap_event]

        hub = FanoutHub(store)
        received = []

        async def run():
            sub = hub.subscribe("main", since_event_id=1)

            async def consume():
                async for event in sub:
                    received.append(event)
                    if len(received) >= 2:
                        break

            task = asyncio.create_task(consume())
            await asyncio.sleep(0.05)

            # Broadcast a live event
            live = _make_event(event_id="evt_live", text="live-event")
            hub.broadcast(live)

            await asyncio.wait_for(task, timeout=2.0)

        asyncio.run(run())
        assert len(received) == 2
        assert received[0].text == "from-db"  # Gap replay
        assert received[1].text == "live-event"  # Live
        store.get_events.assert_called_once_with("main", since_id=1)


# ---------------------------------------------------------------------------
# SessionBridge tests
# ---------------------------------------------------------------------------


class TestSessionBridge:
    def test_bridge_full_flow(self):
        """Full flow: events arrive → run state tracked → fanout."""
        from llm_bawt.integrations.openclaw_bridge import SessionBridge
        from llm_bawt.integrations.agent_bridge_fanout import FanoutHub
        from llm_bawt.integrations.openclaw_ingest import EventIngestPipeline

        store = MagicMock()
        store.store.return_value = True
        store.assemble_run_text.return_value = "Hello world"

        # Track db_id assignment
        _next_id = [0]
        def mock_store(event):
            _next_id[0] += 1
            event.db_id = _next_id[0]
            return True
        store.store.side_effect = mock_store

        fanout = FanoutHub(store)
        ingest = EventIngestPipeline()
        ws_client = MagicMock()
        ws_client.subscribed_sessions = {"main"}

        bridge = SessionBridge(ws_client, ingest, store, fanout)

        broadcasted = []
        original_broadcast = fanout.broadcast
        def track_broadcast(event):
            broadcasted.append(event)
            original_broadcast(event)
        fanout.broadcast = track_broadcast

        async def run():
            # Simulate run lifecycle
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_1",
                "event_type": "response.created",
                "data": {"response": {"id": "run_1", "model": "gpt-5"}},
            })
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_2",
                "event_type": "response.output_text.delta",
                "data": {"delta": "Hello "},
            })
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_3",
                "event_type": "response.output_text.delta",
                "data": {"delta": "world"},
            })
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_4",
                "event_type": "response.completed",
                "data": {"response": {"id": "run_1"}},
            })

        asyncio.run(run())

        # Verify run was created and completed
        store.create_run.assert_called_once_with("run_1", "main", "gpt-5", "system")
        store.complete_run.assert_called_once()
        call_args = store.complete_run.call_args
        assert call_args[0][0] == "run_1"
        assert call_args[0][1] == "Hello world"

        # Verify all events were broadcasted
        assert len(broadcasted) == 4

    def test_bridge_background_event_persisted(self):
        """Event with no matching user send still stored in EventStore."""
        from llm_bawt.integrations.openclaw_bridge import SessionBridge
        from llm_bawt.integrations.agent_bridge_fanout import FanoutHub
        from llm_bawt.integrations.openclaw_ingest import EventIngestPipeline

        store = MagicMock()
        _next_id = [0]
        def mock_store(event):
            _next_id[0] += 1
            event.db_id = _next_id[0]
            return True
        store.store.side_effect = mock_store

        fanout = FanoutHub(store)
        ingest = EventIngestPipeline()
        ws_client = MagicMock()
        ws_client.subscribed_sessions = {"main"}

        bridge = SessionBridge(ws_client, ingest, store, fanout)

        async def run():
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_bg",
                "event_type": "response.output_text.delta",
                "data": {"delta": "background message"},
            })

        asyncio.run(run())

        # Event was stored
        store.store.assert_called_once()
        stored_event = store.store.call_args[0][0]
        assert stored_event.kind == AgentEventKind.ASSISTANT_DELTA
        assert stored_event.text == "background message"

    def test_bridge_duplicate_dropped(self):
        """Duplicate events (store returns False) are not broadcasted."""
        from llm_bawt.integrations.openclaw_bridge import SessionBridge
        from llm_bawt.integrations.agent_bridge_fanout import FanoutHub
        from llm_bawt.integrations.openclaw_ingest import EventIngestPipeline

        store = MagicMock()
        store.store.return_value = False  # duplicate

        fanout = FanoutHub(store)
        broadcasted = []
        fanout.broadcast = lambda e: broadcasted.append(e)

        ingest = EventIngestPipeline()
        ws_client = MagicMock()
        ws_client.subscribed_sessions = {"main"}

        bridge = SessionBridge(ws_client, ingest, store, fanout)

        async def run():
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_dup",
                "event_type": "response.output_text.delta",
                "data": {"delta": "dup"},
            })

        asyncio.run(run())

        assert len(broadcasted) == 0  # not broadcasted

    def test_bridge_tool_calls_tracked(self):
        """Tool start/end events are tracked in run state."""
        from llm_bawt.integrations.openclaw_bridge import SessionBridge
        from llm_bawt.integrations.agent_bridge_fanout import FanoutHub
        from llm_bawt.integrations.openclaw_ingest import EventIngestPipeline

        store = MagicMock()
        _next_id = [0]
        def mock_store(event):
            _next_id[0] += 1
            event.db_id = _next_id[0]
            return True
        store.store.side_effect = mock_store
        store.assemble_run_text.return_value = ""

        fanout = FanoutHub(store)
        ingest = EventIngestPipeline()
        ws_client = MagicMock()
        ws_client.subscribed_sessions = {"main"}

        bridge = SessionBridge(ws_client, ingest, store, fanout)

        async def run():
            # Run started
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_t1",
                "event_type": "response.created",
                "data": {"response": {"id": "run_tools"}},
            })
            # Tool call
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_t2",
                "event_type": "response.output_item.added",
                "run_id": "run_tools",
                "data": {"item": {"type": "function_call", "name": "exec", "arguments": {"command": "ls"}}},
            })
            # Tool result
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_t3",
                "event_type": "response.output_item.added",
                "run_id": "run_tools",
                "data": {"item": {"type": "function_call_output", "name": "exec", "output": "file.txt"}},
            })
            # Run completed
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_t4",
                "event_type": "response.completed",
                "data": {"response": {"id": "run_tools"}},
            })

        asyncio.run(run())

        store.complete_run.assert_called_once()
        call_args = store.complete_run.call_args
        tool_calls = call_args[0][2]
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "exec"
        assert tool_calls[0]["result"] == "file.txt"
