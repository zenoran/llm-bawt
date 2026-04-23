"""PostgreSQL database operations for the media_generations table.

Uses SQLAlchemy (same pattern as the memory module) for table creation
and CRUD operations. No binary data is stored -- only metadata.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from ..utils.config import Config
from ..utils.db import set_utc_on_connect

logger = logging.getLogger(__name__)

TABLE_NAME = "media_generations"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id               TEXT PRIMARY KEY,
    status           TEXT NOT NULL DEFAULT 'pending',
    media_type       TEXT NOT NULL DEFAULT 'video',
    prompt           TEXT NOT NULL,
    revised_prompt   TEXT,
    provider         TEXT NOT NULL DEFAULT 'xai',
    model            TEXT NOT NULL DEFAULT 'grok-imagine-video',
    aspect_ratio     TEXT DEFAULT '16:9',
    duration         REAL,
    actual_duration  REAL,
    resolution       TEXT DEFAULT '720p',
    width            INTEGER,
    height           INTEGER,
    file_path        TEXT,
    thumbnail_path   TEXT,
    file_size_bytes  BIGINT,
    mime_type        TEXT,
    provider_job_id  TEXT,
    source_image_hash TEXT,
    error            TEXT,
    progress         INTEGER DEFAULT 0,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    completed_at     TIMESTAMPTZ
)
"""

CREATE_INDEXES_SQL = [
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_status ON {TABLE_NAME}(status)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_media_type ON {TABLE_NAME}(media_type)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_created_at ON {TABLE_NAME}(created_at)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_provider_job_id ON {TABLE_NAME}(provider_job_id)",
]


def _build_engine(config: Config):
    """Build a SQLAlchemy engine from the shared config."""
    host = getattr(config, "POSTGRES_HOST", "localhost")
    port = int(getattr(config, "POSTGRES_PORT", 5432))
    user = getattr(config, "POSTGRES_USER", "llm_bawt")
    password = getattr(config, "POSTGRES_PASSWORD", "")
    database = getattr(config, "POSTGRES_DATABASE", "llm_bawt")

    encoded_password = quote_plus(password)
    url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{database}"

    engine = create_engine(
        url,
        echo=False,
        poolclass=QueuePool,
        pool_size=3,
        max_overflow=5,
    )
    set_utc_on_connect(engine)
    return engine


class MediaGenerationStore:
    """CRUD operations for the media_generations table."""

    def __init__(self, config: Config):
        self.engine = _build_engine(config)
        self._ensure_table()

    # ------------------------------------------------------------------
    # Table initialisation
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        """Create the media_generations table if it doesn't exist."""
        with self.engine.connect() as conn:
            try:
                conn.execute(text(CREATE_TABLE_SQL))
                for idx_sql in CREATE_INDEXES_SQL:
                    try:
                        conn.execute(text(idx_sql))
                    except Exception as e:
                        logger.debug("Index creation (may exist): %s", e)
                conn.commit()
                logger.debug("Ensured media_generations table exists")
            except Exception as e:
                logger.error("Failed to create media_generations table: %s", e)
                raise

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, row: dict[str, Any]) -> None:
        """Insert a new generation record."""
        columns = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row.keys())
        sql = text(f"INSERT INTO {TABLE_NAME} ({columns}) VALUES ({placeholders})")

        with self.engine.connect() as conn:
            conn.execute(sql, row)
            conn.commit()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, gen_id: str, updates: dict[str, Any]) -> bool:
        """Update fields on an existing generation record.

        Returns True if a row was updated.
        """
        if not updates:
            return False

        set_clauses = ", ".join(f"{k} = :{k}" for k in updates.keys())
        sql = text(f"UPDATE {TABLE_NAME} SET {set_clauses} WHERE id = :_id")
        params = {**updates, "_id": gen_id}

        with self.engine.connect() as conn:
            result = conn.execute(sql, params)
            conn.commit()
            return result.rowcount > 0

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_by_id(self, gen_id: str) -> dict[str, Any] | None:
        """Fetch a single generation by ID."""
        sql = text(f"SELECT * FROM {TABLE_NAME} WHERE id = :id")
        with self.engine.connect() as conn:
            row = conn.execute(sql, {"id": gen_id}).mappings().first()
            return dict(row) if row else None

    def get_by_provider_job_id(self, provider_job_id: str) -> dict[str, Any] | None:
        """Fetch a generation by its provider job ID."""
        sql = text(f"SELECT * FROM {TABLE_NAME} WHERE provider_job_id = :pjid")
        with self.engine.connect() as conn:
            row = conn.execute(sql, {"pjid": provider_job_id}).mappings().first()
            return dict(row) if row else None

    def list_generations(
        self,
        media_type: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List generations with optional filtering and pagination.

        Returns (rows, total_count).
        """
        where = ""
        params: dict[str, Any] = {"limit": limit, "offset": offset}

        if media_type:
            where = "WHERE media_type = :media_type"
            params["media_type"] = media_type

        count_sql = text(f"SELECT COUNT(*) FROM {TABLE_NAME} {where}")
        list_sql = text(
            f"SELECT * FROM {TABLE_NAME} {where} "
            f"ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
        )

        with self.engine.connect() as conn:
            total = conn.execute(count_sql, params).scalar() or 0
            rows = conn.execute(list_sql, params).mappings().all()
            return [dict(r) for r in rows], total

    def list_pending_jobs(self) -> list[dict[str, Any]]:
        """Get all jobs with status 'pending' or 'processing' for background polling."""
        sql = text(
            f"SELECT * FROM {TABLE_NAME} "
            f"WHERE status IN ('pending', 'processing') "
            f"ORDER BY created_at ASC"
        )
        with self.engine.connect() as conn:
            rows = conn.execute(sql).mappings().all()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, gen_id: str) -> bool:
        """Delete a generation record. Returns True if deleted."""
        sql = text(f"DELETE FROM {TABLE_NAME} WHERE id = :id")
        with self.engine.connect() as conn:
            result = conn.execute(sql, {"id": gen_id})
            conn.commit()
            return result.rowcount > 0
