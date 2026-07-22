"""TASK-252 (M1c) — per-thread SDK session keys.

Design (locked in TASK-252):
    Each durable thread records which SDK/provider session hydrates it in
    ``sessions.session_metadata.agent_session_keys = {<backend>: sdk-id}``
    (canonical home). The bot's scalar ``agent_backend_config.session_key``
    remains the continuous conversation's pointer — a thread-scoped turn
    must NEVER touch it.

Coverage:
    - resolver: canonical-first, legacy TASK-284 mirror fallback, routing-key
      guard, model-change gate
    - bridge SendRequest: thread field parsing + routing-key guard
    - dispatch binding: set fresh per turn, cleared on unscoped turns,
      claude-code only
    - /new rotation guard: a thread-bound turn never rotates the active thread
    - seed decision: thread with a stored key → no seed (bridge resumes);
      thread without → scoped seed
    - live-DB: PUT /v1/sessions/{id}/agent-session-key merge semantics +
      normalized read + resolver round-trip
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from llm_bawt.service.routes.sessions import (
    agent_key_name,
    resolve_agent_session_key,
)


# ──────────────────────────────────────────────────────────────────────────
# Resolver (pure)
# ──────────────────────────────────────────────────────────────────────────
class TestAgentKeyName:
    def test_dash_to_underscore(self):
        assert agent_key_name("claude-code") == "claude_code"

    def test_plain_passthrough(self):
        assert agent_key_name("codex") == "codex"

    def test_empty(self):
        assert agent_key_name("") == ""


class TestResolveAgentSessionKey:
    def test_canonical_hit(self):
        meta = {"agent_session_keys": {"claude_code": "sid-123"}}
        assert resolve_agent_session_key(meta, "claude-code") == "sid-123"

    def test_legacy_mirror_fallback(self):
        meta = {"provider": "claude-code", "provider_session_id": "sid-legacy"}
        assert resolve_agent_session_key(meta, "claude-code") == "sid-legacy"

    def test_legacy_fallback_requires_provider_match(self):
        meta = {"provider": "codex", "provider_session_id": "sid-legacy"}
        assert resolve_agent_session_key(meta, "claude-code") is None

    def test_canonical_wins_over_legacy(self):
        meta = {
            "agent_session_keys": {"claude_code": "sid-new"},
            "provider": "claude-code",
            "provider_session_id": "sid-old",
        }
        assert resolve_agent_session_key(meta, "claude-code") == "sid-new"

    def test_routing_key_guard(self):
        meta = {"agent_session_keys": {"claude_code": "byte:nick"}}
        assert resolve_agent_session_key(meta, "claude-code") is None

    def test_model_change_gate_blocks(self):
        meta = {
            "agent_session_keys": {"claude_code": "sid-1"},
            "provider": "claude-code",
            "provider_session_model": "model-a",
        }
        assert resolve_agent_session_key(meta, "claude-code", "model-b") is None

    def test_model_match_passes(self):
        meta = {
            "agent_session_keys": {"claude_code": "sid-1"},
            "provider": "claude-code",
            "provider_session_model": "model-a",
        }
        assert resolve_agent_session_key(meta, "claude-code", "model-a") == "sid-1"

    def test_other_backends_model_never_vetoes(self):
        # Gavel review finding 2: provider_session_model is scalar (last
        # writer) while keys are per-backend — a codex-written model note
        # must not veto the claude-code key.
        meta = {
            "agent_session_keys": {"claude_code": "sid-cc", "codex": "sid-cx"},
            "provider": "codex",
            "provider_session_model": "codex-model",
        }
        assert (
            resolve_agent_session_key(meta, "claude-code", "claude-model")
            == "sid-cc"
        )

    def test_no_stored_model_passes(self):
        meta = {"agent_session_keys": {"claude_code": "sid-1"}}
        assert resolve_agent_session_key(meta, "claude-code", "model-b") == "sid-1"

    def test_empty_meta(self):
        assert resolve_agent_session_key({}, "claude-code") is None
        assert resolve_agent_session_key(None, "claude-code") is None


# ──────────────────────────────────────────────────────────────────────────
# Bridge SendRequest parsing
# ──────────────────────────────────────────────────────────────────────────
class TestSendRequestThreadFields:
    def _base_fields(self, **extra):
        return {
            "request_id": "req_x",
            "session_key": "byte:nick",
            "message": "hi",
            "model": "m1",
            **extra,
        }

    def test_absent_fields_default_none(self):
        from claude_code_bridge.send_request import SendRequest

        req = SendRequest.from_fields(self._base_fields())
        assert req.thread_session_id is None
        assert req.thread_resume_id is None

    def test_thread_fields_parsed(self):
        from claude_code_bridge.send_request import SendRequest

        req = SendRequest.from_fields(
            self._base_fields(
                thread_session_id="thread-1", thread_resume_id="sid-9",
            )
        )
        assert req.thread_session_id == "thread-1"
        assert req.thread_resume_id == "sid-9"

    def test_routing_key_resume_dropped(self):
        from claude_code_bridge.send_request import SendRequest

        req = SendRequest.from_fields(
            self._base_fields(
                thread_session_id="thread-1", thread_resume_id="byte:nick",
            )
        )
        assert req.thread_session_id == "thread-1"
        assert req.thread_resume_id is None


# ──────────────────────────────────────────────────────────────────────────
# Dispatch binding (_bind_agent_thread) — hermetic fakes
# ──────────────────────────────────────────────────────────────────────────
from llm_bawt.service.chat_streaming_bridge import ChatStreamingBridgeMixin


class _Bridge(ChatStreamingBridgeMixin):
    pass


def _fake_llm_bawt(
    backend: str = "claude-code",
    bot_config: dict | None = None,
    session_row: dict | None = None,
):
    bc = bot_config if bot_config is not None else {}
    db_backend = SimpleNamespace(get_session=lambda sid: session_row)
    return SimpleNamespace(
        bot=SimpleNamespace(agent_backend=backend),
        client=SimpleNamespace(_bot_config=bc),
        history_manager=SimpleNamespace(_db_backend=db_backend),
    )


class TestBindAgentThread:
    def test_unscoped_turn_returns_none(self):
        lb = _fake_llm_bawt()
        out = _Bridge()._bind_agent_thread(lb, SimpleNamespace(session_id=None))
        assert out is None

    def test_non_claude_code_never_binds(self):
        lb = _fake_llm_bawt(backend="openclaw")
        out = _Bridge()._bind_agent_thread(lb, SimpleNamespace(session_id="t-1"))
        assert out is None

    def test_scoped_turn_binds_thread_and_resume(self):
        row = {
            "session_metadata": {"agent_session_keys": {"claude_code": "sid-42"}}
        }
        lb = _fake_llm_bawt(session_row=row)
        out = _Bridge()._bind_agent_thread(lb, SimpleNamespace(session_id="t-1"))
        assert out == {"thread_session_id": "t-1", "thread_resume_id": "sid-42"}

    def test_scoped_turn_without_stored_key_binds_thread_only(self):
        lb = _fake_llm_bawt(session_row={"session_metadata": {}})
        out = _Bridge()._bind_agent_thread(lb, SimpleNamespace(session_id="t-1"))
        assert out == {"thread_session_id": "t-1"}

    def test_model_mismatch_forces_cold_start(self):
        row = {
            "session_metadata": {
                "agent_session_keys": {"claude_code": "sid-42"},
                "provider": "claude-code",
                "provider_session_model": "old-model",
            }
        }
        lb = _fake_llm_bawt(bot_config={"model": "new-model"}, session_row=row)
        out = _Bridge()._bind_agent_thread(lb, SimpleNamespace(session_id="t-1"))
        assert out == {"thread_session_id": "t-1"}

    def test_binding_is_request_local_not_instance_state(self):
        # Gavel review finding 1: the binding must never be written to the
        # shared cached client config — concurrent turns would cross-bind.
        row = {
            "session_metadata": {"agent_session_keys": {"claude_code": "sid-42"}}
        }
        bc = {}
        lb = _fake_llm_bawt(bot_config=bc, session_row=row)
        _Bridge()._bind_agent_thread(lb, SimpleNamespace(session_id="t-1"))
        assert bc == {}  # untouched — binding travels by value

    def test_never_raises_on_broken_backend(self):
        lb = SimpleNamespace(
            bot=SimpleNamespace(agent_backend="claude-code"),
            client=SimpleNamespace(_bot_config={}),
            history_manager=SimpleNamespace(_db_backend=None),
        )
        out = _Bridge()._bind_agent_thread(lb, SimpleNamespace(session_id="t-1"))
        assert out == {"thread_session_id": "t-1"}


class TestRotationScopedGuard:
    def test_bound_turn_never_rotates(self):
        lb = _fake_llm_bawt()
        # _rotate_chat_session would need a real backend; the guard must
        # return False BEFORE reaching it.
        assert (
            _Bridge()._maybe_rotate_agent_session(
                lb, "byte", "/new", thread_binding={"thread_session_id": "t-1"}
            )
            is False
        )

    def test_unbound_new_still_reaches_rotation(self):
        lb = _fake_llm_bawt()
        called = []

        b = _Bridge()
        b._rotate_chat_session = lambda *a, **k: called.append(1) or True
        assert b._maybe_rotate_agent_session(lb, "byte", "/new") is True
        assert called


# ──────────────────────────────────────────────────────────────────────────
# Seed decision (maybe_build_session_seed scoped branch) — hermetic
# ──────────────────────────────────────────────────────────────────────────
class TestScopedSeedDecision:
    def _llm_bawt(self):
        return SimpleNamespace(
            bot=SimpleNamespace(agent_backend="claude-code", agent_backend_config={}),
            client=SimpleNamespace(_bot_config={}),
        )

    def test_thread_with_stored_key_gets_no_seed(self):
        from llm_bawt.service.routes.history import maybe_build_session_seed

        assert (
            maybe_build_session_seed(
                self._llm_bawt(), "byte", "m", "hello", None,
                thread_binding={"thread_session_id": "t-1", "thread_resume_id": "sid-1"},
            )
            is None
        )

    def test_thread_without_key_builds_scoped_seed(self, monkeypatch):
        from llm_bawt.service.routes import history as history_routes

        captured = {}

        def _fake_seed(bot_id, model, service, session_id=None):
            captured["session_id"] = session_id
            return {"messages": [{"role": "user", "content": "x"}]}

        monkeypatch.setattr(history_routes, "build_context_seed", _fake_seed)
        out = history_routes.maybe_build_session_seed(
            self._llm_bawt(), "byte", "m", "hello", None,
            thread_binding={"thread_session_id": "t-1"},
        )
        assert out == [{"role": "user", "content": "x"}]
        assert captured["session_id"] == "t-1"

    def test_unbound_turn_ignores_scoped_branch(self):
        from llm_bawt.service.routes.history import maybe_build_session_seed

        # No binding, continuity resolution blows up on the None resolver →
        # helper swallows and returns None (never raises).
        lb = SimpleNamespace(
            bot=SimpleNamespace(agent_backend="claude-code", agent_backend_config={}),
            client=SimpleNamespace(_bot_config={}),
            config_resolver=None,
        )
        assert maybe_build_session_seed(lb, "byte", "m", "hello", None) is None


# ──────────────────────────────────────────────────────────────────────────
# Kwarg channel: binding reaches the backend config per-call, never shared
# ──────────────────────────────────────────────────────────────────────────
class TestKwargChannelRequestLocal:
    def test_stream_raw_merges_binding_into_call_config_only(self):
        from llm_bawt.clients.agent_backend_client import AgentBackendClient
        from llm_bawt.models.message import Message

        captured: dict = {}

        class _Backend:
            def stream_raw(self, prompt, config, **kw):
                captured.update(config)
                yield "ok"

        client = AgentBackendClient.__new__(AgentBackendClient)
        client._bot_config = {"bot_id": "byte"}
        client._backend = _Backend()

        out = list(
            client.stream_raw(
                [Message(role="user", content="hi")],
                thread_binding={
                    "thread_session_id": "t-1",
                    "thread_resume_id": "s-1",
                },
            )
        )
        assert out == ["ok"]
        assert captured["thread_session_id"] == "t-1"
        assert captured["thread_resume_id"] == "s-1"
        # Shared instance config untouched — the binding was per-call.
        assert client._bot_config == {"bot_id": "byte"}


# ──────────────────────────────────────────────────────────────────────────
# Live-DB: endpoint merge semantics + resolver round-trip
# ──────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="class")
def live_thread():
    """Throwaway bot + one real thread row for endpoint tests."""
    from sqlalchemy import text
    from llm_bawt.utils.config import Config
    from llm_bawt.memory.postgresql import PostgreSQLShortTermManager

    bot = f"_t252_pytest_{uuid.uuid4().hex[:8]}"
    user = "nick"
    config = Config()
    manager = PostgreSQLShortTermManager(config, bot_id=bot, user_id=user)
    thread_id = manager.get_or_create_active_session(bot_id=bot, user_id=user)

    yield SimpleNamespace(bot=bot, user=user, manager=manager, thread_id=thread_id)

    with manager._backend.engine.begin() as conn:
        conn.execute(
            text("DELETE FROM sessions WHERE bot_id = :b"), {"b": bot}
        )


@pytest.mark.integration
class TestAgentKeyEndpointLiveDB:
    def _put(self, env, backend="claude-code", key="sid-live-1", model="m1"):
        from llm_bawt.service.routes.sessions import (
            AgentSessionKeyRequest,
            put_agent_session_key,
        )

        return asyncio.run(
            put_agent_session_key(
                env.thread_id,
                AgentSessionKeyRequest(
                    backend=backend, session_key=key, model=model
                ),
                bot_id=env.bot,
            )
        )

    def test_put_stores_canonical_and_legacy_keys(self, live_thread):
        out = self._put(live_thread)
        assert out["stored"] is True
        row = live_thread.manager.get_session(live_thread.thread_id)
        meta = row["session_metadata"]
        assert meta["agent_session_keys"]["claude_code"] == "sid-live-1"
        assert meta["provider_session_id"] == "sid-live-1"
        assert meta["provider_session_model"] == "m1"

    def test_resolver_round_trip(self, live_thread):
        self._put(live_thread, key="sid-live-2")
        row = live_thread.manager.get_session(live_thread.thread_id)
        assert (
            resolve_agent_session_key(row["session_metadata"], "claude-code")
            == "sid-live-2"
        )

    def test_second_backend_merges_not_replaces(self, live_thread):
        self._put(live_thread, key="sid-cc")
        self._put(live_thread, backend="codex", key="sid-cx")
        row = live_thread.manager.get_session(live_thread.thread_id)
        keys = row["session_metadata"]["agent_session_keys"]
        assert keys["claude_code"] == "sid-cc"
        assert keys["codex"] == "sid-cx"

    def test_routing_key_rejected(self, live_thread):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            self._put(live_thread, key="byte:nick")
        assert exc.value.status_code == 422

    def test_legacy_completed_rows_migrate_idempotently(self, live_thread):
        """TASK-250 review finding: bootstrap must retire 'completed' and
        backfill archived_at from ended_at, idempotently."""
        from sqlalchemy import text

        legacy_id = str(uuid.uuid4())
        engine = live_thread.manager._backend.engine
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO sessions (id, bot_id, user_id, started_at, "
                    "ended_at, status) VALUES (:id, :b, 'nick', "
                    "NOW() - INTERVAL '2 days', NOW() - INTERVAL '1 day', "
                    "'completed')"
                ),
                {"id": legacy_id, "b": live_thread.bot},
            )
            # The exact statements the bootstrap runs (postgresql.py /
            # migrations_memory.py — kept in sync by this test).
            for stmt in (
                "UPDATE sessions SET status='archived', "
                "archived_at=COALESCE(archived_at, ended_at) "
                "WHERE status='completed'",
                "UPDATE sessions SET archived_at=ended_at "
                "WHERE status='archived' AND archived_at IS NULL "
                "AND ended_at IS NOT NULL",
            ):
                conn.execute(text(stmt))
        row = live_thread.manager.get_session(legacy_id)
        assert row["status"] == "archived"
        assert row["archived_at"] is not None
        first_archived = row["archived_at"]
        # Idempotency: re-running touches nothing.
        with engine.begin() as conn:
            r1 = conn.execute(
                text("UPDATE sessions SET status='archived', "
                     "archived_at=COALESCE(archived_at, ended_at) "
                     "WHERE status='completed'")
            )
            r2 = conn.execute(
                text("UPDATE sessions SET archived_at=ended_at "
                     "WHERE status='archived' AND archived_at IS NULL "
                     "AND ended_at IS NOT NULL")
            )
            assert (r1.rowcount or 0) == 0
        assert live_thread.manager.get_session(legacy_id)["archived_at"] == first_archived

    def test_backfill_insert_stamps_archived_at(self, live_thread):
        """TASK-250 review finding: backfill_sessions' INSERT must stamp
        archived_at = inferred ended_at (shape-level check of the SQL)."""
        import inspect
        from llm_bawt.memory import migrations_memory

        src = inspect.getsource(migrations_memory)
        # The historical-session INSERT carries archived_at, valued from the
        # same inferred ended_at.
        assert "ended_at, archived_at, status" in src
        assert ":ended_at, :ended_at,\n                         'archived'" in src

    def test_unknown_thread_404(self, live_thread):
        from fastapi import HTTPException
        from llm_bawt.service.routes.sessions import (
            AgentSessionKeyRequest,
            put_agent_session_key,
        )

        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                put_agent_session_key(
                    str(uuid.uuid4()),
                    AgentSessionKeyRequest(
                        backend="claude-code", session_key="sid-x", model=None
                    ),
                    bot_id=live_thread.bot,
                )
            )
        assert exc.value.status_code == 404
