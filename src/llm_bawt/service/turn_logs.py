"""Persistent turn logs with short TTL for debugging and UI inspection."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from urllib.parse import quote_plus

from sqlalchemy import Column, DateTime, Text, text as sa_text
from sqlmodel import Field, SQLModel, Session, create_engine, delete, select

from ..utils.config import Config, has_database_credentials

logger = logging.getLogger(__name__)


def _extract_trigger_id(request_payload: dict) -> str | None:
    """Extract the last user message ID from a request payload."""
    messages = request_payload.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict) or str(message.get("role") or "") != "user":
            continue
        for key in ("id", "db_id", "message_id"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


class TurnLog(SQLModel, table=True):
    """Persistent log for one chat turn."""

    __tablename__ = "turn_logs"

    id: str = Field(primary_key=True)
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
    request_id: str | None = Field(default=None, index=True)
    path: str = Field(default="/v1/chat/completions", max_length=128)
    stream: bool = Field(default=False, index=True)
    model: str | None = Field(default=None, index=True)
    bot_id: str | None = Field(default=None, index=True)
    user_id: str | None = Field(default=None, index=True)
    status: str = Field(default="ok", max_length=32)
    latency_ms: float | None = Field(default=None)
    user_prompt: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    request_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    response_text: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    tool_calls_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    error_text: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    trigger_message_id: str | None = Field(default=None, index=True)


class TurnLogStore:
    """DB access helper for persistent turn logs."""

    _last_cleanup_at: float = 0.0
    _cleanup_interval_seconds: float = 300.0
    _backfill_done: bool = False

    def __init__(self, config: Config, ttl_hours: int = 168):
        self.config = config
        self.ttl_hours = max(1, int(ttl_hours))
        self.engine = None
        try:
            host = getattr(config, "POSTGRES_HOST", "localhost")
            port = int(getattr(config, "POSTGRES_PORT", 5432))
            user = getattr(config, "POSTGRES_USER", "llm_bawt")
            password = getattr(config, "POSTGRES_PASSWORD", "")
            database = getattr(config, "POSTGRES_DATABASE", "llm_bawt")
            encoded_password = quote_plus(password)
            url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{database}"
            self.engine = create_engine(url, echo=False)
            self._ensure_tables_exist()
            self._backfill_trigger_message_ids()
        except Exception as e:
            self.engine = None
            logger.warning("Turn logs DB unavailable: %s", e)

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return
        SQLModel.metadata.create_all(self.engine, tables=[TurnLog.__table__])
        # Add columns that may not exist on older tables.
        with self.engine.connect() as conn:
            try:
                conn.execute(sa_text(
                    "ALTER TABLE turn_logs ADD COLUMN IF NOT EXISTS"
                    " trigger_message_id VARCHAR"
                ))
                conn.execute(sa_text(
                    "CREATE INDEX IF NOT EXISTS ix_turn_logs_trigger_message_id"
                    " ON turn_logs (trigger_message_id)"
                ))
                conn.commit()
            except Exception:
                pass

    def _backfill_trigger_message_ids(self) -> None:
        """One-time backfill: populate trigger_message_id for existing rows."""
        if self.engine is None or TurnLogStore._backfill_done:
            return
        TurnLogStore._backfill_done = True
        try:
            with Session(self.engine) as session:
                rows = session.exec(
                    select(TurnLog)
                    .where(TurnLog.trigger_message_id.is_(None))
                    .where(TurnLog.request_json.is_not(None))
                    .limit(5000)
                ).all()
                if not rows:
                    return
                updated = 0
                for row in rows:
                    try:
                        payload = json.loads(row.request_json) if row.request_json else None
                    except Exception:
                        continue
                    if payload:
                        tid = _extract_trigger_id(payload)
                        if tid:
                            row.trigger_message_id = tid
                            session.add(row)
                            updated += 1
                if updated:
                    session.commit()
                    logger.info("Backfilled trigger_message_id for %d turn logs", updated)
        except Exception as e:
            logger.debug("trigger_message_id backfill skipped: %s", e)

    def _cleanup_expired_if_due(self, force: bool = False) -> None:
        if self.engine is None:
            return
        now = time.time()
        if not force and (now - self.__class__._last_cleanup_at) < self.__class__._cleanup_interval_seconds:
            return

        cutoff = datetime.utcnow() - timedelta(hours=self.ttl_hours)
        try:
            with Session(self.engine) as session:
                session.exec(delete(TurnLog).where(TurnLog.created_at < cutoff))
                session.commit()
            self.__class__._last_cleanup_at = now
        except Exception as e:
            logger.debug("Turn log TTL cleanup skipped: %s", e)

    def save_turn(
        self,
        *,
        turn_id: str,
        request_id: str | None,
        path: str,
        stream: bool,
        model: str | None,
        bot_id: str | None,
        user_id: str | None,
        status: str,
        latency_ms: float | None,
        user_prompt: str | None,
        request_payload: dict | None,
        response_text: str | None,
        tool_calls: list[dict] | None,
        error_text: str | None = None,
        trigger_message_id: str | None = None,
    ) -> None:
        """Persist one turn entry and enforce short TTL cleanup."""
        if self.engine is None:
            return

        self._cleanup_expired_if_due()

        # Auto-extract trigger message ID from request if not provided.
        if trigger_message_id is None and request_payload:
            trigger_message_id = _extract_trigger_id(request_payload)

        row = TurnLog(
            id=turn_id,
            request_id=request_id,
            path=path,
            stream=stream,
            model=model,
            bot_id=(bot_id or None),
            user_id=(user_id or None),
            status=status,
            latency_ms=latency_ms,
            user_prompt=user_prompt,
            request_json=json.dumps(request_payload, ensure_ascii=False, default=str) if request_payload else None,
            response_text=response_text,
            tool_calls_json=json.dumps(tool_calls or [], ensure_ascii=False, default=str),
            error_text=error_text,
            trigger_message_id=trigger_message_id,
        )

        with Session(self.engine) as session:
            session.add(row)
            session.commit()

    def update_turn(
        self,
        *,
        turn_id: str,
        status: str | None = None,
        latency_ms: float | None = None,
        response_text: str | None = None,
        request_payload: dict | None = None,
        tool_calls: list[dict] | None = None,
        error_text: str | None = None,
    ) -> None:
        """Update an existing turn log row with new data."""
        if self.engine is None:
            return
        with Session(self.engine) as session:
            row = session.get(TurnLog, turn_id)
            if row is None:
                logger.debug("update_turn: no row with id=%s", turn_id)
                return
            if status is not None:
                row.status = status
            if latency_ms is not None:
                row.latency_ms = latency_ms
            if response_text is not None:
                row.response_text = response_text
            if request_payload is not None:
                row.request_json = json.dumps(request_payload, ensure_ascii=False, default=str)
            if tool_calls is not None:
                row.tool_calls_json = json.dumps(tool_calls, ensure_ascii=False, default=str)
            if error_text is not None:
                row.error_text = error_text
            session.add(row)
            session.commit()

    def get_turn(self, turn_id: str) -> TurnLog | None:
        """Get one turn by id."""
        if self.engine is None:
            return None
        self._cleanup_expired_if_due()
        with Session(self.engine) as session:
            return session.exec(select(TurnLog).where(TurnLog.id == turn_id)).first()

    def list_turns(
        self,
        *,
        bot_id: str | None = None,
        user_id: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
        status: str | None = None,
        stream: bool | None = None,
        has_tools: bool | None = None,
        trigger_message_ids: set[str] | None = None,
        after: float | None = None,
        before: float | None = None,
        since_hours: int = 24,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[TurnLog], int]:
        """List turns with filters and pagination."""
        if self.engine is None:
            return [], 0
        self._cleanup_expired_if_due()

        conditions: list = []
        if after is not None:
            conditions.append(TurnLog.created_at >= datetime.utcfromtimestamp(after))
        elif before is None:
            # Only apply since_hours if no explicit time range given.
            since_cutoff = datetime.utcnow() - timedelta(hours=max(1, int(since_hours)))
            conditions.append(TurnLog.created_at >= since_cutoff)
        if before is not None:
            conditions.append(TurnLog.created_at <= datetime.utcfromtimestamp(before))
        if bot_id:
            conditions.append(TurnLog.bot_id == bot_id.strip().lower())
        if user_id:
            conditions.append(TurnLog.user_id == user_id.strip().lower())
        if model:
            conditions.append(TurnLog.model == model.strip())
        if request_id:
            conditions.append(TurnLog.request_id == request_id.strip())
        if status:
            conditions.append(TurnLog.status == status.strip().lower())
        if stream is not None:
            conditions.append(TurnLog.stream.is_(stream))
        if has_tools is True:
            conditions.append(TurnLog.tool_calls_json.is_not(None))
            conditions.append(TurnLog.tool_calls_json != "[]")
        elif has_tools is False:
            conditions.append((TurnLog.tool_calls_json.is_(None)) | (TurnLog.tool_calls_json == "[]"))
        if trigger_message_ids:
            conditions.append(TurnLog.trigger_message_id.in_(trigger_message_ids))

        statement = select(TurnLog).where(*conditions).order_by(TurnLog.created_at.desc())
        count_statement = select(TurnLog).where(*conditions)

        with Session(self.engine) as session:
            total_count = len(session.exec(count_statement).all())
            rows = session.exec(statement.offset(offset).limit(limit)).all()

        return rows, total_count
