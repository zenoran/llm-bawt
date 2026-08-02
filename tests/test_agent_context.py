"""Agent context policy, health projection, and manual-reset contracts."""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.agent_context import AgentContextStore, SessionPolicy
from llm_bawt.memory.postgresql import MESSAGES_PARENT, partition_name
from llm_bawt.utils.config import Config

pytestmark = pytest.mark.integration


@pytest.fixture
def context_store():
    store = AgentContextStore(Config())
    if store.engine is None:
        pytest.skip("PostgreSQL unavailable")
    bot = f"testcontext{uuid.uuid4().hex}"
    user = f"user-{uuid.uuid4().hex}"
    table = partition_name(MESSAGES_PARENT, bot)
    from llm_bawt.memory.postgresql import ensure_bot_partitions
    with store.engine.begin() as conn:
        ensure_bot_partitions(conn, bot)
    yield store, bot, user, table
    with store.engine.begin() as conn:
        conn.execute(sa_text("DELETE FROM turn_logs WHERE bot_id=:bot"), {"bot": bot})
        conn.execute(sa_text(f"DELETE FROM {table}"))
        conn.execute(sa_text("DELETE FROM sessions WHERE bot_id=:bot"), {"bot": bot})
        conn.execute(sa_text(f"DROP TABLE IF EXISTS {table}"))


def test_health_prefers_native_resident_usage_and_scopes_active_thread(context_store):
    store, bot, user, table = context_store
    session_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    usage = {
        "input_tokens": 999999,
        "cache_read_tokens": 999999,
        "resident_tokens": 90,
        "resident_source": "claude_sdk_context",
        "context_window": 100,
    }
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO sessions (id, bot_id, user_id, status, started_at)
            VALUES (:id, :bot, :user, 'active', now())
        """), {"id": session_id, "bot": bot, "user": user})
        conn.execute(sa_text(f"""
            INSERT INTO {table} (bot_id, id, role, content, timestamp, session_id)
            VALUES (:bot, :id, 'user', 'x', 1, :session)
        """), {"bot": bot, "id": message_id, "session": session_id})
        conn.execute(sa_text("""
            INSERT INTO turn_logs (
                id, created_at, path, stream, bot_id, user_id, status,
                user_prompt, response_text, trigger_message_id,
                token_usage_json, ended_at
            ) VALUES (
                :id, now(), '/test', false, :bot, :user, 'ok',
                'x', 'y', :message, :usage, now()
            )
        """), {
            "id": f"turn-{uuid.uuid4().hex}", "bot": bot, "user": user,
            "message": message_id, "usage": json.dumps(usage),
        })

    health = store.health(
        bot_id=bot,
        user_id=user,
        backend="claude-code",
        configured_ceiling=372000,
        warning_ratio=0.75,
        critical_ratio=0.90,
    )
    assert health["resident_prompt_tokens"] == 90
    assert health["resident_source"] == "claude_sdk_context"
    assert health["effective_ceiling_tokens"] == 100
    assert health["ceiling_disagreement"] is True
    assert health["state"] == "critical"

    reset = store.reset_idle_session(
        bot_id=bot,
        user_id=user,
        backend="claude-code",
        policy=SessionPolicy.RESET_WITHOUT_HISTORY,
        reason="test reset",
    )
    assert reset["old_session_id"] == session_id
    after = store.health(
        bot_id=bot,
        user_id=user,
        backend="claude-code",
        configured_ceiling=372000,
    )
    assert after["session_id"] == reset["new_session_id"]
    assert after["resident_prompt_tokens"] is None
    assert after["state"] == "unknown"
    assert after["last_lifecycle_action"]["action"] == "reset_without_history"


def test_health_fallback_ignores_cumulative_cached_token_accounting(context_store):
    store, bot, user, table = context_store
    session_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    usage = {
        "input_tokens": 623465,
        "cache_read_tokens": 18359296,
        "cache_creation_tokens": 999999,
        "context_window": 372000,
    }
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO sessions (id, bot_id, user_id, status, started_at)
            VALUES (:id, :bot, :user, 'active', now())
        """), {"id": session_id, "bot": bot, "user": user})
        conn.execute(sa_text(f"""
            INSERT INTO {table} (bot_id, id, role, content, timestamp, session_id)
            VALUES (:bot, :id, 'user', 'x', 1, :session)
        """), {"bot": bot, "id": message_id, "session": session_id})
        conn.execute(sa_text("""
            INSERT INTO turn_logs (
                id, created_at, path, stream, bot_id, user_id, status,
                user_prompt, response_text, trigger_message_id,
                token_usage_json, ended_at
            ) VALUES (
                :id, now(), '/test', false, :bot, :user, 'ok',
                'x', 'y', :message, :usage, now()
            )
        """), {
            "id": f"turn-{uuid.uuid4().hex}", "bot": bot, "user": user,
            "message": message_id, "usage": json.dumps(usage),
        })

    health = store.health(
        bot_id=bot,
        user_id=user,
        backend="codex",
        configured_ceiling=372000,
    )

    assert health["resident_prompt_tokens"] is None
    assert health["resident_source"] is None
    assert health["state"] == "unknown"
    assert health["usage_turn_id"] is not None


def test_manual_reset_refuses_active_turn(context_store):
    store, bot, user, _table = context_store
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO sessions (id, bot_id, user_id, status, started_at)
            VALUES (:id, :bot, :user, 'active', now())
        """), {"id": str(uuid.uuid4()), "bot": bot, "user": user})
        conn.execute(sa_text("""
            INSERT INTO turn_logs (
                id, created_at, path, stream, bot_id, user_id, status,
                user_prompt, response_text, ended_at
            ) VALUES (:id, now(), '/test', true, :bot, :user, 'streaming', '', '', NULL)
        """), {"id": f"turn-{uuid.uuid4().hex}", "bot": bot, "user": user})
    with pytest.raises(RuntimeError, match="active in turn"):
        store.reset_idle_session(
            bot_id=bot,
            user_id=user,
            backend="claude-code",
            policy=SessionPolicy.RESET_RETAIN_HISTORY,
            reason="must wait",
        )
