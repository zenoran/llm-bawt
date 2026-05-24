"""PostgreSQL store for the ``media_assets`` table.

This is the canonical registry for normalized, content-addressed media
blobs that get attached to chat messages, tool outputs, or agent payloads.
The actual bytes live on disk (see :mod:`llm_bawt.media.storage`); only
metadata is persisted here.

TASK-222 foundation — read together with:

- ``media_generations`` table (legacy job-tracker; different concern).
- ``{bot}_messages.attachments`` JSONB column added in the same migration:
  every chat-history row carries a tiny array of ``{"asset_id": "ma_...",
  "kind": "image"}`` references that join back to this table on read.

The table is idempotent: it's created on first ``MediaAssetStore`` init
and reused thereafter — same pattern as ``MediaGenerationStore`` and the
per-bot memory tables. There is intentionally no Alembic in this project.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Column, DateTime, Integer, Text, text
from sqlmodel import Field, SQLModel

from ..utils.config import Config

logger = logging.getLogger(__name__)

TABLE_NAME = "media_assets"

# Allowed values for media_assets.source. Stored as a CHECK constraint so
# typos in producer code surface as a DB error instead of silently writing
# garbage that the GC job can't reason about.
ALLOWED_SOURCES = ("chat_upload", "tool_generated", "agent_attachment")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id                  TEXT PRIMARY KEY,
    sha256              TEXT NOT NULL UNIQUE,
    mime_type           TEXT NOT NULL,
    original_mime_type  TEXT,
    size_bytes          INTEGER NOT NULL,
    width               INTEGER,
    height              INTEGER,
    source              TEXT NOT NULL,
    owner_user_id       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMPTZ,
    CONSTRAINT media_assets_source_check
        CHECK (source IN ('chat_upload', 'tool_generated', 'agent_attachment'))
)
"""

# Listing assets for a given user, newest-first, is the dominant access
# pattern (history rendering + the nightly GC job both scan by owner).
CREATE_INDEXES_SQL = [
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_owner_created "
    f"ON {TABLE_NAME}(owner_user_id, created_at DESC)",
    # expires_at lookups for soft-deletion / TTL cleanup.
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_expires_at "
    f"ON {TABLE_NAME}(expires_at) WHERE expires_at IS NOT NULL",
]


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

# Crockford's base32 alphabet — the ULID spec. Lexicographically ordered
# when encoded as a 26-char string, which means ``id`` sorts the same as
# ``created_at``. Useful for batch fetches that don't want to sort.
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _encode_crockford(value: int, length: int) -> str:
    chars: list[str] = []
    for _ in range(length):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def new_asset_id() -> str:
    """Return a fresh ``ma_<ulid>`` identifier.

    No external ulid library — we generate the canonical 26-char encoding
    from a 48-bit millisecond timestamp + 80 bits of cryptographic entropy.
    Roughly 1.2 * 10^24 ids/ms before a collision becomes likely; good
    enough for chat uploads and tool outputs.
    """
    ms = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    return "ma_" + _encode_crockford(ms, 10) + _encode_crockford(rand, 16)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _build_engine(config: Config):
    """Return the process-wide shared SQLAlchemy engine (TASK-202)."""
    from ..utils.db import get_shared_engine
    return get_shared_engine(config)


class MediaAssetStore:
    """CRUD operations for the ``media_assets`` table.

    Construct once per process; the underlying engine is shared across
    every Store via ``get_shared_engine`` so this is cheap.
    """

    def __init__(self, config: Config):
        self.engine = _build_engine(config)
        if self.engine is not None:
            self._ensure_table()
        else:
            logger.warning(
                "MediaAssetStore initialized with no engine — Postgres credentials missing"
            )

    # ------------------------------------------------------------------
    # Table initialisation
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        """Create the ``media_assets`` table + indexes if missing."""
        with self.engine.connect() as conn:
            try:
                conn.execute(text(CREATE_TABLE_SQL))
                for idx_sql in CREATE_INDEXES_SQL:
                    try:
                        conn.execute(text(idx_sql))
                    except Exception as e:
                        logger.debug("Index creation (may exist): %s", e)
                conn.commit()
                logger.debug("Ensured media_assets table exists")
            except Exception as e:
                logger.error("Failed to create media_assets table: %s", e)
                raise

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(
        self,
        *,
        sha256: str,
        mime_type: str,
        size_bytes: int,
        source: str,
        original_mime_type: str | None = None,
        width: int | None = None,
        height: int | None = None,
        owner_user_id: str | None = None,
        expires_at: datetime | None = None,
        asset_id: str | None = None,
    ) -> dict[str, Any]:
        """Insert a new media asset row.

        Returns the inserted row as a dict. If a row with the same
        ``sha256`` already exists, returns that existing row instead
        (dedup-by-content) — callers should rely on this for idempotent
        uploads rather than catching IntegrityError themselves.
        """
        if source not in ALLOWED_SOURCES:
            raise ValueError(
                f"source must be one of {ALLOWED_SOURCES!r}, got {source!r}"
            )

        existing = self.get_by_sha256(sha256)
        if existing is not None:
            logger.debug(
                "media_assets dedup hit: sha256=%s -> id=%s",
                sha256[:12],
                existing["id"],
            )
            return existing

        row_id = asset_id or new_asset_id()
        insert_sql = text(f"""
            INSERT INTO {TABLE_NAME}
                (id, sha256, mime_type, original_mime_type, size_bytes,
                 width, height, source, owner_user_id, expires_at)
            VALUES
                (:id, :sha256, :mime_type, :original_mime_type, :size_bytes,
                 :width, :height, :source, :owner_user_id, :expires_at)
            RETURNING *
        """)
        params = {
            "id": row_id,
            "sha256": sha256,
            "mime_type": mime_type,
            "original_mime_type": original_mime_type,
            "size_bytes": size_bytes,
            "width": width,
            "height": height,
            "source": source,
            "owner_user_id": owner_user_id,
            "expires_at": expires_at,
        }
        with self.engine.connect() as conn:
            row = conn.execute(insert_sql, params).mappings().first()
            conn.commit()
            return dict(row) if row else {}

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_by_id(self, asset_id: str) -> dict[str, Any] | None:
        sql = text(f"SELECT * FROM {TABLE_NAME} WHERE id = :id")
        with self.engine.connect() as conn:
            row = conn.execute(sql, {"id": asset_id}).mappings().first()
            return dict(row) if row else None

    def get_by_sha256(self, sha256: str) -> dict[str, Any] | None:
        sql = text(f"SELECT * FROM {TABLE_NAME} WHERE sha256 = :sha")
        with self.engine.connect() as conn:
            row = conn.execute(sql, {"sha": sha256}).mappings().first()
            return dict(row) if row else None

    def list_by_owner(
        self,
        owner_user_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        sql = text(
            f"SELECT * FROM {TABLE_NAME} "
            f"WHERE owner_user_id = :owner "
            f"ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
        )
        params = {"owner": owner_user_id, "limit": limit, "offset": offset}
        with self.engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, asset_id: str) -> bool:
        """Hard-delete a row. Caller is responsible for removing the
        on-disk blob first (see :class:`MediaStorage`). Returns True if a
        row was removed."""
        sql = text(f"DELETE FROM {TABLE_NAME} WHERE id = :id")
        with self.engine.connect() as conn:
            result = conn.execute(sql, {"id": asset_id})
            conn.commit()
            return result.rowcount > 0


# ---------------------------------------------------------------------------
# SQLModel mirror
# ---------------------------------------------------------------------------


class MediaAsset(SQLModel, table=True):
    """Declarative mirror of the ``media_assets`` table.

    Provided so other Python code can ``from llm_bawt.media.assets import
    MediaAsset`` and use it with SQLModel / FastAPI response models. The
    actual table is created by :class:`MediaAssetStore._ensure_table`
    using raw DDL (the project pattern); this class is **not** used to
    issue ``CREATE TABLE`` -- ``__table_args__ = {"extend_existing": True}``
    lets it coexist with the raw definition without conflict.

    Field names mirror the SQL schema 1:1. ``created_at`` is timezone-aware
    (TIMESTAMPTZ); ``expires_at`` is nullable.
    """

    __tablename__ = "media_assets"
    __table_args__ = {"extend_existing": True}

    id: str = Field(default_factory=new_asset_id, primary_key=True)
    sha256: str = Field(index=True, unique=True)
    mime_type: str
    original_mime_type: Optional[str] = Field(default=None)
    size_bytes: int = Field(sa_column=Column(Integer, nullable=False))
    width: Optional[int] = Field(default=None)
    height: Optional[int] = Field(default=None)
    source: str = Field(description="chat_upload | tool_generated | agent_attachment")
    owner_user_id: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()")),
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
