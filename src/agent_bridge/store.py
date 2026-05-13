from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .events import AgentEvent, AgentEventKind

logger = logging.getLogger(__name__)


def create_agent_event_tables(engine: Engine) -> None:
    """Create agent-bridge tables if they don't exist.

    Also performs a one-time, idempotent rename of the legacy
    ``openclaw_*`` tables (and their indexes) into the new
    ``agent_*`` names. Safe to run on fresh DBs and on already-migrated
    DBs — both branches are guarded with ``IF EXISTS`` / ``IF NOT EXISTS``.
    """
    with engine.connect() as conn:
        # ---- One-time rename of legacy openclaw_* tables --------------
        # Each statement is independent and idempotent.
        conn.execute(text("ALTER TABLE IF EXISTS openclaw_events RENAME TO agent_events"))
        conn.execute(text("ALTER TABLE IF EXISTS openclaw_session_state RENAME TO agent_session_state"))
        conn.execute(text("ALTER TABLE IF EXISTS openclaw_runs RENAME TO agent_runs"))
        conn.execute(text("ALTER INDEX IF EXISTS idx_openclaw_events_session_created RENAME TO idx_agent_events_session_created"))
        conn.execute(text("ALTER INDEX IF EXISTS idx_openclaw_events_session_run RENAME TO idx_agent_events_session_run"))
        conn.execute(text("ALTER INDEX IF EXISTS idx_openclaw_events_session_seq RENAME TO idx_agent_events_session_seq"))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS agent_events (
                id BIGSERIAL PRIMARY KEY,
                event_dedupe_key VARCHAR(255) NOT NULL UNIQUE,
                session_key VARCHAR(255) NOT NULL,
                run_id VARCHAR(255),
                seq BIGINT,
                kind VARCHAR(50) NOT NULL,
                origin VARCHAR(50) NOT NULL DEFAULT 'unknown',
                text TEXT,
                tool_name VARCHAR(255),
                payload_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_agent_events_session_created
            ON agent_events (session_key, created_at)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_agent_events_session_run
            ON agent_events (session_key, run_id)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_agent_events_session_seq
            ON agent_events (session_key, seq)
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS agent_session_state (
                session_key VARCHAR(255) PRIMARY KEY,
                last_event_id BIGINT,
                last_cursor VARCHAR(255),
                ws_connected BOOLEAN NOT NULL DEFAULT FALSE,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS agent_runs (
                run_id VARCHAR(255) PRIMARY KEY,
                session_key VARCHAR(255) NOT NULL,
                status VARCHAR(50) NOT NULL DEFAULT 'running',
                model VARCHAR(255),
                origin VARCHAR(50),
                full_text TEXT,
                tool_calls_json JSONB,
                started_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        conn.commit()
    logger.info("Agent-bridge tables ensured (renamed legacy openclaw_* if present)")


class EventStore:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def store(self, event: AgentEvent) -> bool:
        """Append event. Returns False if dedupe_key already exists (idempotent)."""
        payload = json.dumps(event.raw, ensure_ascii=False, default=str)
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text("""
                    INSERT INTO agent_events
                        (event_dedupe_key, session_key, run_id, seq, kind, origin, text, tool_name, payload_json)
                    VALUES
                        (:dedupe_key, :session_key, :run_id, :seq, :kind, :origin, :text, :tool_name, :payload)
                    ON CONFLICT (event_dedupe_key) DO NOTHING
                    RETURNING id
                """), {
                    "dedupe_key": event.event_id,
                    "session_key": event.session_key,
                    "run_id": event.run_id,
                    "seq": event.seq,
                    "kind": event.kind.value,
                    "origin": event.origin,
                    "text": event.text,
                    "tool_name": event.tool_name,
                    "payload": payload,
                })
                row = result.fetchone()
                conn.commit()
                if row:
                    event.db_id = row[0]
                    return True
                return False
        except Exception:
            logger.exception("Failed to store event %s", event.event_id)
            return False

    def get_events(
        self,
        session_key: str,
        *,
        since_id: int | None = None,
        since_ts: datetime | None = None,
        kinds: list[AgentEventKind] | None = None,
        limit: int = 100,
    ) -> list[AgentEvent]:
        """Query events for a session with optional filters."""
        conditions = ["session_key = :session_key"]
        params: dict[str, Any] = {"session_key": session_key, "limit": limit}

        if since_id is not None:
            conditions.append("id > :since_id")
            params["since_id"] = since_id
        if since_ts is not None:
            conditions.append("created_at > :since_ts")
            params["since_ts"] = since_ts
        if kinds:
            kind_values = [k.value for k in kinds]
            conditions.append("kind = ANY(:kinds)")
            params["kinds"] = kind_values

        where = " AND ".join(conditions)
        query = f"SELECT id, event_dedupe_key, session_key, run_id, seq, kind, origin, text, tool_name, payload_json, created_at FROM agent_events WHERE {where} ORDER BY id ASC LIMIT :limit"

        with self._engine.connect() as conn:
            rows = conn.execute(text(query), params).fetchall()

        events = []
        for row in rows:
            raw = row[9] if isinstance(row[9], dict) else json.loads(row[9]) if row[9] else {}
            events.append(AgentEvent(
                event_id=row[1],
                session_key=row[2],
                run_id=row[3],
                seq=row[4],
                kind=AgentEventKind(row[5]),
                origin=row[6],
                text=row[7],
                tool_name=row[8],
                timestamp=row[10],
                raw=raw,
                db_id=row[0],
            ))
        return events

    def get_session_cursor(self, session_key: str) -> int | None:
        with self._engine.connect() as conn:
            row = conn.execute(text(
                "SELECT last_event_id FROM agent_session_state WHERE session_key = :sk"
            ), {"sk": session_key}).fetchone()
        return row[0] if row else None

    def update_session_cursor(self, session_key: str, event_id: int) -> None:
        with self._engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO agent_session_state (session_key, last_event_id, updated_at)
                VALUES (:sk, :eid, NOW())
                ON CONFLICT (session_key) DO UPDATE SET last_event_id = :eid, updated_at = NOW()
            """), {"sk": session_key, "eid": event_id})
            conn.commit()

    def assemble_run_text(self, run_id: str) -> str:
        """Reassemble full text from ASSISTANT_DELTA events for a run."""
        with self._engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT text FROM agent_events
                WHERE run_id = :run_id AND kind = :kind
                ORDER BY COALESCE(seq, id) ASC
            """), {"run_id": run_id, "kind": AgentEventKind.ASSISTANT_DELTA.value}).fetchall()
        return "".join(row[0] for row in rows if row[0])

    def create_run(self, run_id: str, session_key: str, model: str | None = None, origin: str | None = None) -> None:
        try:
            with self._engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO agent_runs (run_id, session_key, status, model, origin, started_at)
                    VALUES (:run_id, :sk, 'running', :model, :origin, NOW())
                    ON CONFLICT (run_id) DO NOTHING
                """), {"run_id": run_id, "sk": session_key, "model": model, "origin": origin})
                conn.commit()
        except Exception:
            logger.exception("Failed to create run %s", run_id)

    def complete_run(self, run_id: str, full_text: str, tool_calls: list[dict] | None = None) -> None:
        try:
            tc_json = json.dumps(tool_calls, ensure_ascii=False, default=str) if tool_calls else None
            with self._engine.connect() as conn:
                conn.execute(text("""
                    UPDATE agent_runs
                    SET status = 'completed', full_text = :text, tool_calls_json = :tc, completed_at = NOW()
                    WHERE run_id = :run_id
                """), {"run_id": run_id, "text": full_text, "tc": tc_json})
                conn.commit()
        except Exception:
            logger.exception("Failed to complete run %s", run_id)

    def update_session_ws_state(self, session_key: str, connected: bool) -> None:
        with self._engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO agent_session_state (session_key, ws_connected, updated_at)
                VALUES (:sk, :connected, NOW())
                ON CONFLICT (session_key) DO UPDATE SET ws_connected = :connected, updated_at = NOW()
            """), {"sk": session_key, "connected": connected})
            conn.commit()
