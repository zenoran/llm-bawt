"""TASK-641: session-level summaries.

Hermetic tests cover the thread-keyed windowing (``group_messages_by_thread``,
``find_budget_overflow_sessions``, chunk propagation), the ``/new``
pre-seed summarize gate (``_maybe_summarize_on_new``), and source-level
guards that (a) both dispatch paths summarize BEFORE building the seed and
(b) the background job routes thread-keyed windows through THE common
per-thread unit (``summarize_thread``) — the same function the /new path
calls.

Live-DB tests (marked ``integration``) prove ``summarize_thread`` end to
end: summary row stamped with the thread's session_id, sources flagged,
protected tail kept raw.

    uv run pytest tests/test_session_summaries.py                 # hermetic
    uv run pytest -m integration tests/test_session_summaries.py  # live-DB too
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_bawt.memory.summarization import (
    Session,
    detect_sessions,
    find_budget_overflow_sessions,
    group_messages_by_thread,
    split_session_into_chunks,
)
from llm_bawt.service.chat_streaming_bridge import ChatStreamingBridgeMixin

SRC = Path(__file__).resolve().parents[1] / "src"


def _msg(mid, ts, sid=None, role="user", content="hello world"):
    return {
        "id": mid, "role": role, "content": content,
        "timestamp": ts, "session_id": sid,
    }


# ──────────────────────────────────────────────────────────────────────────
# group_messages_by_thread
# ──────────────────────────────────────────────────────────────────────────

class TestGroupMessagesByThread:
    def test_groups_by_session_id_not_gaps(self):
        """Messages 5h apart in ONE thread stay one Session; interleaved
        threads split — the exact opposite of gap detection."""
        msgs = [
            _msg("a1", 100.0, "t1"),
            _msg("b1", 200.0, "t2"),
            _msg("a2", 100.0 + 5 * 3600, "t1"),  # 5h gap, same thread
            _msg("b2", 300.0, "t2"),
        ]
        sessions = group_messages_by_thread(msgs)
        assert len(sessions) == 2
        by_id = {s.session_id: s for s in sessions}
        assert by_id["t1"].message_ids == ["a1", "a2"]
        assert by_id["t2"].message_ids == ["b1", "b2"]

    def test_every_session_carries_its_thread_id(self):
        sessions = group_messages_by_thread(
            [_msg("a", 1.0, "t1"), _msg("b", 2.0, "t2")]
        )
        assert all(s.session_id for s in sessions)

    def test_null_session_rows_fall_back_to_gap_detection(self):
        msgs = [
            _msg("x1", 100.0),
            _msg("x2", 110.0),
            _msg("x3", 110.0 + 2 * 3600),  # >1h gap -> second window
        ]
        sessions = group_messages_by_thread(msgs, session_gap_seconds=3600)
        assert len(sessions) == 2
        assert all(s.session_id is None for s in sessions)

    def test_summary_rows_excluded(self):
        msgs = [
            _msg("a", 1.0, "t1"),
            _msg("s", 2.0, "t1", role="summary"),
        ]
        sessions = group_messages_by_thread(msgs)
        assert len(sessions) == 1
        assert sessions[0].message_ids == ["a"]

    def test_sorted_by_start_timestamp(self):
        msgs = [_msg("b", 500.0, "t2"), _msg("a", 100.0, "t1")]
        sessions = group_messages_by_thread(msgs)
        assert [s.session_id for s in sessions] == ["t1", "t2"]

    def test_empty(self):
        assert group_messages_by_thread([]) == []


# ──────────────────────────────────────────────────────────────────────────
# find_budget_overflow_sessions is thread-keyed now
# ──────────────────────────────────────────────────────────────────────────

class TestOverflowThreadKeyed:
    def test_overflow_windows_carry_session_id(self):
        # Tiny budget forces old messages into overflow; they span two
        # threads with NO time gap — gap detection would have merged them.
        big = "x" * 400
        msgs = [
            _msg(f"m{i}", 100.0 + i, "t1" if i < 6 else "t2", content=big)
            for i in range(12)
        ]
        sessions = find_budget_overflow_sessions(
            msgs, max_context_tokens=700, protected_recent_turns=1,
            min_messages_per_session=2,
        )
        assert sessions, "expected overflow with a tiny budget"
        assert all(s.session_id in ("t1", "t2") for s in sessions)
        for s in sessions:
            assert all(m.get("session_id") == s.session_id for m in s.messages)


# ──────────────────────────────────────────────────────────────────────────
# chunk splitting preserves the thread id
# ──────────────────────────────────────────────────────────────────────────

class TestChunkPropagation:
    def test_chunks_inherit_session_id(self):
        big = "y" * 2000
        msgs = [_msg(f"c{i}", float(i), "t9", content=big) for i in range(10)]
        session = Session(
            start_timestamp=0.0, end_timestamp=9.0, messages=msgs,
            message_ids=[m["id"] for m in msgs], session_id="t9",
        )
        chunks = split_session_into_chunks(session, max_tokens_per_chunk=1000)
        assert len(chunks) > 1
        assert all(c.session_id == "t9" for c in chunks)

    def test_gap_detected_sessions_have_no_session_id(self):
        sessions = detect_sessions([_msg("a", 1.0), _msg("b", 2.0)])
        assert len(sessions) == 1
        assert sessions[0].session_id is None


# ──────────────────────────────────────────────────────────────────────────
# /new pre-seed summarize gate
# ──────────────────────────────────────────────────────────────────────────

class _Bridge(ChatStreamingBridgeMixin):
    def __init__(self):
        self.config = SimpleNamespace(
            DEFAULT_USER="nick", REDIS_URL="redis://localhost:6379",
        )


def _llm(scope="inline+summaries", backend=...):
    if backend is ...:
        backend = SimpleNamespace(
            user_id="nick", _current_session_id="thread-1",
        )

    def _resolve(key):
        assert key == "history_scope"
        return SimpleNamespace(value=scope)

    return SimpleNamespace(
        history_manager=SimpleNamespace(_db_backend=backend),
        config_resolver=SimpleNamespace(resolve_config_setting=_resolve),
    )


class TestMaybeSummarizeOnNew:
    def test_non_new_prompt_is_noop(self):
        assert _Bridge()._maybe_summarize_on_new(_llm(), "byte", "hello") is False

    def test_thread_bound_turn_is_noop(self):
        assert _Bridge()._maybe_summarize_on_new(
            _llm(), "byte", "/new",
            thread_binding={"thread_session_id": "old-thread"},
        ) is False

    @pytest.mark.parametrize("scope", ["inline", "none"])
    def test_summaryless_scope_is_noop(self, scope):
        """With summaries off, summarizing would only starve the seed."""
        assert _Bridge()._maybe_summarize_on_new(
            _llm(scope=scope), "byte", "/new"
        ) is False

    def test_missing_backend_is_noop(self):
        assert _Bridge()._maybe_summarize_on_new(
            _llm(backend=None), "byte", "/new"
        ) is False

    def test_new_with_summaries_scope_runs_common_unit(self, monkeypatch):
        """/new + summary scope reaches summarize_thread with the ACTIVE
        thread id and the protected tail kept raw."""
        calls = {}

        class _FakeSummarizer:
            def __init__(self, config, bot_id, summarize_fn=None, **kw):
                calls["bot_id"] = bot_id

            def summarize_thread(self, session_id, protect_recent_turns=False, **kw):
                calls["session_id"] = session_id
                calls["protect"] = protect_recent_turns
                return {"summaries_created": 1, "messages_summarized": 4,
                        "errors": []}

        import llm_bawt.memory.summarization as summod
        monkeypatch.setattr(summod, "HistorySummarizer", _FakeSummarizer)

        assert _Bridge()._maybe_summarize_on_new(_llm(), "byte", "/new") is True
        assert calls == {
            "bot_id": "byte", "session_id": "thread-1", "protect": True,
        }

    def test_mcp_proxy_backend_resolves_active_session(self, monkeypatch):
        """Deployed (MCP server) mode: _db_backend is the _MCPShortTermManager
        proxy — NO _current_session_id attribute. The gate must resolve the
        active thread through the proxy's memory client instead of silently
        no-oping (the live-smoke miss that shipped in ea0876b)."""
        calls = {}

        class _FakeSummarizer:
            def __init__(self, config, bot_id, summarize_fn=None, **kw):
                pass

            def summarize_thread(self, session_id, protect_recent_turns=False, **kw):
                calls["session_id"] = session_id
                return {"summaries_created": 1, "messages_summarized": 4,
                        "errors": []}

        import llm_bawt.memory.summarization as summod
        monkeypatch.setattr(summod, "HistorySummarizer", _FakeSummarizer)

        proxy = SimpleNamespace(  # no _current_session_id, like the real proxy
            user_id="nick",
            _memory_client=SimpleNamespace(
                get_active_session=lambda: {"id": "active-thread-9"},
            ),
        )
        assert _Bridge()._maybe_summarize_on_new(
            _llm(backend=proxy), "byte", "/new"
        ) is True
        assert calls == {"session_id": "active-thread-9"}

    def test_proxy_without_active_session_is_noop(self):
        proxy = SimpleNamespace(
            user_id="nick",
            _memory_client=SimpleNamespace(get_active_session=lambda: None),
        )
        assert _Bridge()._maybe_summarize_on_new(
            _llm(backend=proxy), "byte", "/new"
        ) is False


# ──────────────────────────────────────────────────────────────────────────
# Source-level drift guards
# ──────────────────────────────────────────────────────────────────────────

class TestCallSiteGuards:
    @pytest.mark.parametrize("path", [
        "llm_bawt/service/chat_streaming.py",
        "llm_bawt/service/background_service.py",
    ])
    def test_summarize_before_seed_in_both_dispatch_paths(self, path):
        """TASK-641 ordering: _maybe_summarize_on_new must run BEFORE
        maybe_build_session_seed in each dispatch path."""
        text = (SRC / path).read_text()
        summarize_idx = text.index("_maybe_summarize_on_new(")
        seed_idx = text.index("inject_seed_messages = maybe_build_session_seed(")
        assert summarize_idx < seed_idx, f"{path}: summarize must precede seed"

    def test_chat_new_summarizes_before_rotation(self):
        text = (SRC / "llm_bawt/service/chat_streaming_bridge.py").read_text()
        # The shared chat-bot /new helper owns summarize-before-rotation ordering.
        branch = text[text.index("def _maybe_handle_chat_new_command"):]
        branch = branch[:branch.index("\n    def ", 10)]
        assert branch.index("_maybe_summarize_on_new") < branch.index(
            "_rotate_chat_session"
        )

    def test_job_routes_thread_windows_through_common_unit(self):
        """summarize_eligible_sessions must call summarize_thread for
        thread-keyed windows — the shared unit with the /new path."""
        text = (SRC / "llm_bawt/memory/summarization.py").read_text()
        body = text[text.index("def summarize_eligible_sessions"):]
        body = body[:body.index("\n    def ")]
        assert "self.summarize_thread(" in body

    def test_live_window_cutter_is_thread_keyed(self):
        """find_budget_overflow_sessions must group by thread, not call
        detect_sessions directly."""
        text = (SRC / "llm_bawt/memory/summarization.py").read_text()
        body = text[text.index("def find_budget_overflow_sessions"):]
        body = body[:body.index("\ndef ", 10)]
        assert "group_messages_by_thread(" in body
        assert "detect_sessions(" not in body


# ──────────────────────────────────────────────────────────────────────────
# Live-DB integration: summarize_thread end to end
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestSummarizeThreadLive:
    @pytest.fixture()
    def env(self):
        from llm_bawt.utils.config import Config
        from llm_bawt.memory.summarization import HistorySummarizer
        from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend
        from sqlalchemy import text as sql_text

        config = Config()
        bot_id = "byte"  # partition must exist; rows are cleaned up below
        thread_id = str(uuid.uuid4())
        backend = PostgreSQLMemoryBackend(config, bot_id)
        table = backend._messages_table_name

        msg_ids = [str(uuid.uuid4()) for _ in range(16)]
        with backend.engine.connect() as conn:
            conn.execute(sql_text(
                "INSERT INTO sessions (id, bot_id, user_id, status, started_at)"
                " VALUES (:id, :bot, 'task641-test', 'archived', NOW())"
            ), {"id": thread_id, "bot": bot_id})
            for i, mid in enumerate(msg_ids):
                conn.execute(sql_text(f"""
                    INSERT INTO {table}
                        (id, role, content, timestamp, session_id)
                    VALUES (:id, :role, :content, :ts, :sid)
                """), {
                    "id": mid,
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"task641 probe message {i} about testing summaries",
                    "ts": 1000000.0 + i * 60,
                    "sid": thread_id,
                })
            conn.commit()

        summarizer = HistorySummarizer(
            config, bot_id=bot_id,
            summarize_fn=lambda s: None,  # force heuristic — no LLM in tests
        )
        yield SimpleNamespace(
            summarizer=summarizer, backend=backend, table=table,
            thread_id=thread_id, msg_ids=msg_ids, sql_text=sql_text,
        )
        with backend.engine.connect() as conn:
            conn.execute(sql_text(
                f"DELETE FROM {table} WHERE session_id = :sid"
            ), {"sid": thread_id})
            conn.execute(sql_text(
                "DELETE FROM sessions WHERE id = :sid"
            ), {"sid": thread_id})
            conn.commit()

    def test_summary_row_stamped_with_thread_id(self, env):
        result = env.summarizer.summarize_thread(env.thread_id)
        assert result["success"] and result["summaries_created"] >= 1

        with env.backend.engine.connect() as conn:
            rows = conn.execute(env.sql_text(f"""
                SELECT id, session_id FROM {env.table}
                WHERE role = 'summary' AND session_id = :sid
            """), {"sid": env.thread_id}).fetchall()
            assert rows, "summary row must carry the thread's session_id"

            flagged = conn.execute(env.sql_text(f"""
                SELECT count(*) FROM {env.table}
                WHERE id = ANY(:ids) AND summarized = TRUE
            """), {"ids": env.msg_ids}).scalar()
            assert flagged == len(env.msg_ids)

    def test_protected_tail_stays_raw(self, env):
        protected = env.summarizer.protected_recent_turns * 2
        # Fixture seeds enough rows that the pre-tail candidates clear
        # min_messages_per_session — otherwise this is a legitimate no-op.
        assert len(env.msg_ids) - protected >= env.summarizer.min_messages_per_session
        result = env.summarizer.summarize_thread(
            env.thread_id, protect_recent_turns=True,
        )
        assert result["success"] and result["summaries_created"] >= 1
        with env.backend.engine.connect() as conn:
            raw = conn.execute(env.sql_text(f"""
                SELECT count(*) FROM {env.table}
                WHERE session_id = :sid AND role IN ('user','assistant')
                  AND COALESCE(summarized, FALSE) = FALSE
            """), {"sid": env.thread_id}).scalar()
            assert raw == protected

    def test_below_minimum_is_noop(self, env):
        # Summarize everything first, then a second call has nothing left.
        env.summarizer.summarize_thread(env.thread_id)
        result = env.summarizer.summarize_thread(env.thread_id)
        assert result["success"] and result["summaries_created"] == 0
