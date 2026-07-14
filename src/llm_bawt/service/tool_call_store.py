"""Canonical tool-call and full-result persistence.

The normal tool_call_records row intentionally keeps only a small preview. The
one-to-one payload table is queried only by explicit full-result requests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Integer, String, Text, text as sa_text
from sqlmodel import Field, SQLModel, Session, select

from agent_bridge.tool_results import ToolResultPayload

logger = logging.getLogger(__name__)


class ToolCallRecord(SQLModel, table=True):
    __tablename__ = "tool_call_records"

    id: int | None = Field(default=None, sa_column=Column(Integer, primary_key=True, autoincrement=True))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
    turn_id: str | None = Field(default=None, index=True)
    bot_id: str | None = Field(default=None, index=True)
    user_id: str | None = Field(default=None, index=True)
    call_id: str | None = Field(default=None, index=True)
    tool_name: str = Field(default="unknown", max_length=128)
    arguments_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    result_text: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    iteration: int = Field(default=1)
    started_at: float | None = Field(default=None, sa_column=Column(Float, nullable=True))
    ended_at: float | None = Field(default=None, sa_column=Column(Float, nullable=True))
    duration_ms: float | None = Field(default=None, sa_column=Column(Float, nullable=True))
    text_offset: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    is_error: bool | None = Field(default=None, sa_column=Column(Boolean, nullable=True))
    tool_use_id: str | None = Field(default=None, index=True)
    parent_tool_use_id: str | None = Field(default=None, index=True)
    approval_request_id: str | None = Field(default=None, index=True)
    approval_status: str | None = Field(default=None)
    preapproved: bool | None = Field(default=None, sa_column=Column(Boolean, nullable=True))
    result_total_chars: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    result_total_bytes: int | None = Field(default=None, sa_column=Column(BigInteger, nullable=True))
    result_sha256: str | None = Field(default=None, sa_column=Column(String(64), nullable=True))
    result_content_type: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    result_complete: bool | None = Field(default=None, sa_column=Column(Boolean, nullable=True))
    result_payload_available: bool = Field(default=False, sa_column=Column(Boolean, nullable=False))


class ToolCallResultPayloadRecord(SQLModel, table=True):
    __tablename__ = "tool_call_result_payloads"

    tool_call_record_id: int = Field(sa_column=Column(Integer, primary_key=True))
    content_text: str = Field(sa_column=Column(Text, nullable=False))
    content_type: str = Field(sa_column=Column(String(128), nullable=False))
    total_chars: int = Field(sa_column=Column(Integer, nullable=False))
    total_bytes: int = Field(sa_column=Column(BigInteger, nullable=False))
    sha256: str = Field(sa_column=Column(String(64), nullable=False))
    complete: bool = Field(sa_column=Column(Boolean, nullable=False))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


@dataclass(frozen=True)
class ToolResultPage:
    record_id: int
    content_type: str
    offset: int
    content: str
    total_chars: int
    total_bytes: int
    sha256: str
    complete: bool
    next_offset: int | None


class ToolCallStore:
    def __init__(self, engine) -> None:
        self.engine = engine

    def ensure_schema(self) -> None:
        if self.engine is None:
            return
        SQLModel.metadata.create_all(
            self.engine,
            tables=[ToolCallRecord.__table__, ToolCallResultPayloadRecord.__table__],
        )
        statements = (
            "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS result_total_chars INTEGER",
            "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS result_total_bytes BIGINT",
            "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS result_sha256 VARCHAR(64)",
            "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS result_content_type VARCHAR(128)",
            "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS result_complete BOOLEAN",
            "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS result_payload_available BOOLEAN NOT NULL DEFAULT FALSE",
            "CREATE INDEX IF NOT EXISTS ix_tool_call_records_turn_call ON tool_call_records (turn_id, call_id)",
        )
        with self.engine.begin() as conn:
            for statement in statements:
                conn.execute(sa_text(statement))

    def save_start(
        self,
        *,
        turn_id: str | None,
        bot_id: str | None,
        user_id: str | None,
        call_id: str | None,
        tool_name: str,
        arguments: dict | None = None,
        iteration: int = 1,
        started_at: float | None = None,
        text_offset: int | None = None,
        tool_use_id: str | None = None,
        parent_tool_use_id: str | None = None,
    ) -> int | None:
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            row = None
            if turn_id and call_id:
                row = session.exec(
                    select(ToolCallRecord)
                    .where(ToolCallRecord.turn_id == turn_id)
                    .where(ToolCallRecord.call_id == call_id)
                ).first()
            if row is None:
                row = ToolCallRecord(turn_id=turn_id, call_id=call_id)
            row.bot_id = bot_id
            row.user_id = user_id
            row.tool_name = tool_name
            row.arguments_json = json.dumps(arguments or {}, ensure_ascii=False, default=str)
            row.iteration = iteration
            row.started_at = started_at
            row.text_offset = text_offset
            row.tool_use_id = tool_use_id
            row.parent_tool_use_id = parent_tool_use_id
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def save_result(
        self,
        *,
        turn_id: str | None,
        call_id: str | None,
        tool_use_id: str | None,
        tool_name: str,
        bot_id: str | None,
        user_id: str | None,
        payload: ToolResultPayload,
        ended_at: float | None,
        is_error: bool | None,
        iteration: int = 1,
        parent_tool_use_id: str | None = None,
    ) -> tuple[int | None, bool]:
        """Persist payload and metadata atomically. Returns (record id, available)."""
        if self.engine is None:
            return None, False
        try:
            with Session(self.engine) as session:
                row = None
                if turn_id and call_id:
                    row = session.exec(
                        select(ToolCallRecord)
                        .where(ToolCallRecord.turn_id == turn_id)
                        .where(ToolCallRecord.call_id == call_id)
                    ).first()
                if row is None and turn_id and tool_use_id:
                    row = session.exec(
                        select(ToolCallRecord)
                        .where(ToolCallRecord.turn_id == turn_id)
                        .where(ToolCallRecord.tool_use_id == tool_use_id)
                    ).first()
                if row is None:
                    row = ToolCallRecord(
                        turn_id=turn_id,
                        bot_id=bot_id,
                        user_id=user_id,
                        call_id=call_id,
                        tool_name=tool_name,
                        iteration=iteration,
                        tool_use_id=tool_use_id,
                        parent_tool_use_id=parent_tool_use_id,
                    )
                    session.add(row)
                    session.flush()
                row.bot_id = bot_id or row.bot_id
                row.user_id = user_id or row.user_id
                row.call_id = call_id or row.call_id
                if tool_name and tool_name != "unknown":
                    row.tool_name = tool_name
                row.result_text = payload.preview
                row.result_total_chars = payload.total_chars
                row.result_total_bytes = payload.total_bytes
                row.result_sha256 = payload.sha256
                row.result_content_type = payload.content_type
                row.result_complete = payload.complete
                row.result_payload_available = True
                row.ended_at = ended_at
                row.is_error = is_error
                if ended_at and row.started_at:
                    row.duration_ms = (ended_at - row.started_at) * 1000
                session.add(row)
                session.flush()
                assert row.id is not None
                stored = session.get(ToolCallResultPayloadRecord, row.id)
                if stored is None:
                    stored = ToolCallResultPayloadRecord(tool_call_record_id=row.id, content_text="", content_type="text/plain", total_chars=0, total_bytes=0, sha256="", complete=False)
                stored.content_text = payload.content
                stored.content_type = payload.content_type
                stored.total_chars = payload.total_chars
                stored.total_bytes = payload.total_bytes
                stored.sha256 = payload.sha256
                stored.complete = payload.complete
                session.add(stored)
                session.commit()
                return row.id, True
        except Exception:
            logger.exception("Failed to persist full tool result turn=%s call=%s", turn_id, call_id)
            return None, False

    def page(self, record_id: int, *, bot_id: str, user_id: str, offset: int, limit: int) -> ToolResultPage | None:
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            row = session.get(ToolCallRecord, record_id)
            if row is None or row.bot_id != bot_id or row.user_id != user_id:
                return None
            payload = session.get(ToolCallResultPayloadRecord, record_id)
            if payload is None:
                return None
            end = min(payload.total_chars, offset + limit)
            return ToolResultPage(
                record_id=record_id,
                content_type=payload.content_type,
                offset=offset,
                content=payload.content_text[offset:end],
                total_chars=payload.total_chars,
                total_bytes=payload.total_bytes,
                sha256=payload.sha256,
                complete=payload.complete,
                next_offset=end if end < payload.total_chars else None,
            )

    def raw(self, record_id: int, *, bot_id: str, user_id: str) -> tuple[ToolCallRecord, ToolCallResultPayloadRecord] | None:
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            row = session.get(ToolCallRecord, record_id)
            if row is None or row.bot_id != bot_id or row.user_id != user_id:
                return None
            payload = session.get(ToolCallResultPayloadRecord, record_id)
            if payload is None:
                return None
            session.expunge(row)
            session.expunge(payload)
            return row, payload
