"""Persistent turn logs with short TTL for debugging and UI inspection."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from urllib.parse import quote_plus

from sqlalchemy import Column, DateTime, Text
from sqlmodel import Field, SQLModel, Session, create_engine, delete, select

from ..utils.config import Config, has_database_credentials

logger = logging.getLogger(__name__)


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


class TurnLogStore:
    """DB access helper for persistent turn logs."""

    _last_cleanup_at: float = 0.0
    _cleanup_interval_seconds: float = 300.0

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
        except Exception as e:
            self.engine = None
            logger.warning("Turn logs DB unavailable: %s", e)

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return
        SQLModel.metadata.create_all(self.engine, tables=[TurnLog.__table__])

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
    ) -> None:
        """Persist one turn entry and enforce short TTL cleanup."""
        if self.engine is None:
            return

        self._cleanup_expired_if_due()

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
        )

        with Session(self.engine) as session:
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
        since_hours: int = 24,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[TurnLog], int]:
        """List turns with filters and pagination."""
        if self.engine is None:
            return [], 0
        self._cleanup_expired_if_due()

        since_cutoff = datetime.utcnow() - timedelta(hours=max(1, int(since_hours)))
        conditions: list = [TurnLog.created_at >= since_cutoff]
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

        statement = select(TurnLog).where(*conditions).order_by(TurnLog.created_at.desc())
        count_statement = select(TurnLog).where(*conditions)

        with Session(self.engine) as session:
            total_count = len(session.exec(count_statement).all())
            rows = session.exec(statement.offset(offset).limit(limit)).all()

        return rows, total_count
