"""Persistent pending-question registry for interactive SDK tool calls.

The claude-code bridge pauses on the Claude Agent SDK's built-in
``AskUserQuestion`` tool by holding an asyncio.Future in-process (see
``claude_code_bridge.bridge._make_can_use_tool``).  That covers the in-flight
case, but if the user navigates away and comes back — or if the bridge
restarts mid-pause — there's no way for the UI to know a question is open.

This store mirrors every active pause into Postgres so the UI can hydrate
on page load (``GET /v1/chat/pending-questions``) and so the answer endpoint
(``POST /v1/chat/tool-result``) can detect "the bridge forgot about this
question" via the originating turn's status.

State machine:

    awaiting   ─┬─→  answered    (user picked / typed an answer)
                ├─→  skipped     (user explicitly declined to answer)
                └─→  abandoned   (turn ended without an answer — bridge
                                  restart, turn aborted, model bailed out)

Rows are kept indefinitely for audit; callers filter by status.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Text, text as sa_text
from sqlmodel import Field, SQLModel, Session, select

from ..utils.config import Config

logger = logging.getLogger(__name__)


class PendingQuestion(SQLModel, table=True):
    """Persistent record of an in-flight AskUserQuestion pause."""

    __tablename__ = "chat_pending_questions"

    # SDK-supplied tool_use id — unique across all sessions, doubles as the
    # PK so re-posting an event for the same tool_use_id is idempotent.
    tool_use_id: str = Field(primary_key=True, max_length=128)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
    bot_id: str = Field(index=True, max_length=128)
    user_id: str = Field(index=True, max_length=128)
    turn_id: str = Field(index=True, max_length=128)
    trigger_message_id: str | None = Field(default=None, index=True, max_length=128)
    session_key: str | None = Field(default=None, max_length=128)
    tool_name: str = Field(default="AskUserQuestion", max_length=128)
    # Original tool input (the AskUserQuestion `{questions: [...]}` payload).
    # Stored as JSON text so we don't need a JSONB column for portability.
    arguments_json: str = Field(sa_column=Column(Text, nullable=False))
    status: str = Field(default="awaiting", index=True, max_length=32)
    answer: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    answered_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )


class PendingQuestionStore:
    """DB access helper for ``chat_pending_questions``.

    Mirrors the patterns of :class:`TurnLogStore` — additive ALTERs on init,
    shared SQLAlchemy engine, graceful degradation when Postgres is missing.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.engine = None
        try:
            from ..utils.db import get_shared_engine
            self.engine = get_shared_engine(config)
            if self.engine is None:
                return
            self._ensure_tables_exist()
        except Exception as e:
            self.engine = None
            logger.warning("Pending-question store unavailable: %s", e)

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return
        SQLModel.metadata.create_all(
            self.engine, tables=[PendingQuestion.__table__],
        )
        # Additive index — safe to re-apply.
        with self.engine.connect() as conn:
            try:
                conn.execute(sa_text(
                    "CREATE INDEX IF NOT EXISTS ix_pending_questions_bot_status_created"
                    " ON chat_pending_questions (bot_id, user_id, status, created_at DESC)"
                ))
                conn.commit()
            except Exception:
                pass

    # ----- writes -----

    def upsert_awaiting(
        self,
        *,
        tool_use_id: str,
        bot_id: str,
        user_id: str,
        turn_id: str,
        arguments: dict,
        tool_name: str = "AskUserQuestion",
        trigger_message_id: str | None = None,
        session_key: str | None = None,
    ) -> None:
        """Insert (or no-op-replay) a fresh awaiting question.

        Called from the chat_streaming layer when it sees the bridge emit an
        ``AWAIT_TOOL_RESULT`` AgentEvent.  Idempotent on tool_use_id — a
        duplicate event (Redis replay, multi-window race) does not advance
        the row state, even if it was already answered.
        """
        if self.engine is None:
            return
        try:
            with Session(self.engine) as session:
                existing = session.get(PendingQuestion, tool_use_id)
                if existing is not None:
                    # Already recorded — don't clobber the answer if the row
                    # advanced past awaiting.
                    return
                row = PendingQuestion(
                    tool_use_id=tool_use_id,
                    bot_id=(bot_id or "").strip() or "unknown",
                    user_id=(user_id or "").strip() or "unknown",
                    turn_id=(turn_id or "").strip() or "unknown",
                    trigger_message_id=(trigger_message_id or None) or None,
                    session_key=(session_key or None) or None,
                    tool_name=tool_name or "AskUserQuestion",
                    arguments_json=json.dumps(
                        arguments if isinstance(arguments, dict) else {"value": arguments},
                        ensure_ascii=False, default=str,
                    ),
                )
                session.add(row)
                session.commit()
        except Exception:
            logger.exception(
                "Failed to upsert pending question tool_use_id=%s", tool_use_id,
            )

    def mark(
        self,
        *,
        tool_use_id: str,
        status: str,
        answer: str | None = None,
    ) -> PendingQuestion | None:
        """Transition a row to a terminal state.

        ``status`` must be one of: ``answered``, ``skipped``, ``abandoned``.
        Returns the updated row (or None if not found).  No-ops if the row
        is already in a terminal state — the first answer wins.
        """
        if self.engine is None:
            return None
        if status not in {"answered", "skipped", "abandoned"}:
            raise ValueError(f"invalid pending-question status: {status}")
        try:
            with Session(self.engine) as session:
                row = session.get(PendingQuestion, tool_use_id)
                if row is None:
                    return None
                if row.status != "awaiting":
                    return row
                row.status = status
                if answer is not None:
                    row.answer = answer
                row.answered_at = datetime.now(timezone.utc)
                session.add(row)
                session.commit()
                session.refresh(row)
                return row
        except Exception:
            logger.exception(
                "Failed to mark pending question tool_use_id=%s status=%s",
                tool_use_id, status,
            )
            return None

    def abandon_for_turn(self, turn_id: str) -> int:
        """Mark every awaiting question for a turn as abandoned.

        Called from the turn-completion path when the originating turn ends
        without an answer (timeout, abort, model bailed out, bridge crashed
        and the app noticed via the turn-log status flip).  Returns the
        number of rows changed.
        """
        if self.engine is None or not turn_id:
            return 0
        try:
            with Session(self.engine) as session:
                rows = session.exec(
                    select(PendingQuestion)
                    .where(PendingQuestion.turn_id == turn_id)
                    .where(PendingQuestion.status == "awaiting")
                ).all()
                count = 0
                now = datetime.now(timezone.utc)
                for row in rows:
                    row.status = "abandoned"
                    row.answered_at = now
                    session.add(row)
                    count += 1
                if count:
                    session.commit()
                return count
        except Exception:
            logger.exception("Failed to abandon questions for turn_id=%s", turn_id)
            return 0

    # ----- reads -----

    def get(self, tool_use_id: str) -> PendingQuestion | None:
        if self.engine is None:
            return None
        try:
            with Session(self.engine) as session:
                return session.get(PendingQuestion, tool_use_id)
        except Exception:
            logger.exception("Failed to fetch pending question tool_use_id=%s", tool_use_id)
            return None

    def list_awaiting(
        self,
        *,
        bot_id: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[PendingQuestion]:
        """Return all currently-awaiting questions for a bot/user scope.

        Used by ``GET /v1/chat/pending-questions`` for UI hydration on page
        load and by other tabs that missed the live AWAIT_TOOL_RESULT event.
        """
        if self.engine is None:
            return []
        try:
            with Session(self.engine) as session:
                stmt = select(PendingQuestion).where(
                    PendingQuestion.status == "awaiting"
                )
                if bot_id:
                    stmt = stmt.where(PendingQuestion.bot_id == bot_id.strip())
                if user_id:
                    stmt = stmt.where(PendingQuestion.user_id == user_id.strip())
                stmt = stmt.order_by(PendingQuestion.created_at.desc()).limit(limit)
                return list(session.exec(stmt).all())
        except Exception:
            logger.exception(
                "Failed to list awaiting questions bot=%s user=%s",
                bot_id, user_id,
            )
            return []

    @staticmethod
    def row_to_dict(row: PendingQuestion) -> dict:
        """Serialise a row to the wire shape the UI consumes.

        Mirrors the per-stream ``tool.await_result`` SSE chunk so the same
        client-side store reducer can ingest either source.
        """
        try:
            arguments = json.loads(row.arguments_json) if row.arguments_json else {}
        except Exception:
            arguments = {}
        return {
            "tool_use_id": row.tool_use_id,
            "tool_name": row.tool_name,
            "bot_id": row.bot_id,
            "user_id": row.user_id,
            "turn_id": row.turn_id,
            "trigger_message_id": row.trigger_message_id,
            "session_key": row.session_key,
            "arguments": arguments,
            "status": row.status,
            "answer": row.answer,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "answered_at": row.answered_at.isoformat() if row.answered_at else None,
        }
