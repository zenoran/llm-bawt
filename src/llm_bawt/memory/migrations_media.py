"""Media schema migrations for llm-bawt (TASK-222).

Split out of ``migrations.py`` (TASK-553). These migrations manage the
``media_assets`` table and the ``attachments`` JSONB column on every
``*_messages`` table.

The public ``migrations`` facade re-exports every function here, so
``from llm_bawt.memory.migrations import add_media_assets_table`` keeps working.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def add_media_assets_table(backend: Any, dry_run: bool = False) -> dict:
    """TASK-222: create the ``media_assets`` table + supporting indexes.

    Idempotent (uses CREATE TABLE IF NOT EXISTS). The runtime path also
    creates this table via :class:`MediaAssetStore._ensure_table` on app
    startup; this migration entry exists so the schema can be applied
    explicitly (e.g. before backfill scripts) without booting the full
    service.

    Args:
        backend: Any object exposing ``.engine`` (e.g.
            :class:`PostgreSQLMemoryBackend`).
        dry_run: If True, only report what would be done.

    Returns:
        Dict with action summary.
    """
    from sqlalchemy import text
    from ..media.assets import CREATE_TABLE_SQL, CREATE_INDEXES_SQL, TABLE_NAME

    with backend.engine.connect() as conn:
        exists = conn.execute(text(
            "SELECT 1 FROM information_schema.tables WHERE table_name = :n"
        ), {"n": TABLE_NAME}).fetchone()

        if dry_run:
            return {
                "table": TABLE_NAME,
                "already_exists": bool(exists),
                "would_create": not exists,
                "dry_run": True,
            }

        conn.execute(text(CREATE_TABLE_SQL))
        for idx_sql in CREATE_INDEXES_SQL:
            try:
                conn.execute(text(idx_sql))
            except Exception as e:
                logger.debug("Index create (may exist): %s", e)
        conn.commit()

    return {"table": TABLE_NAME, "created": not exists}


def drop_media_assets_table(backend: Any, dry_run: bool = False) -> dict:
    """TASK-222 downgrade: drop the ``media_assets`` table.

    This is the inverse of :func:`add_media_assets_table`. Also clears
    the ``attachments`` column from every ``*_messages`` table (idempotent
    -- ``DROP COLUMN IF EXISTS``).

    .. warning::
       This permanently removes media asset metadata. Caller is
       responsible for cleaning up on-disk blobs separately.

    Args:
        backend: Any object exposing ``.engine``.
        dry_run: If True, only report what would be done.

    Returns:
        Dict with action summary.
    """
    from sqlalchemy import text
    from ..media.assets import TABLE_NAME

    with backend.engine.connect() as conn:
        # Find every messages table so we strip the attachments column too.
        tables_sql = text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name LIKE '%\\_messages' ESCAPE '\\'
              AND table_name NOT LIKE '%\\_forgotten\\_messages' ESCAPE '\\'
        """)
        tables = [row.table_name for row in conn.execute(tables_sql).fetchall()]

        exists = conn.execute(text(
            "SELECT 1 FROM information_schema.tables WHERE table_name = :n"
        ), {"n": TABLE_NAME}).fetchone()

        if dry_run:
            return {
                "table": TABLE_NAME,
                "would_drop_table": bool(exists),
                "would_drop_attachments_from": tables,
                "dry_run": True,
            }

        for table_name in tables:
            try:
                conn.execute(text(
                    f"ALTER TABLE {table_name} DROP COLUMN IF EXISTS attachments"
                ))
            except Exception as e:
                logger.warning(
                    "Failed dropping attachments from %s: %s", table_name, e
                )
        conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_NAME}"))
        conn.commit()

    return {
        "table": TABLE_NAME,
        "dropped": bool(exists),
        "stripped_attachments_from": tables,
    }


def add_attachments_to_messages(backend: Any, dry_run: bool = False) -> dict:
    """TASK-222: add ``attachments JSONB`` to every ``*_messages`` table.

    Existing rows fill with ``'[]'::jsonb`` via the column default — no
    explicit backfill needed.

    The same ALTER also runs on first boot for each bot's own table inside
    :meth:`PostgreSQLMemoryBackend._ensure_tables_exist`; this migration
    entry catches every existing per-bot table in one pass so a fresh
    upgrade doesn't have to wait for every bot to come online.

    Args:
        backend: Any object exposing ``.engine``.
        dry_run: If True, only report what would be done.

    Returns:
        Dict listing the tables that were altered.
    """
    from sqlalchemy import text

    altered: list[str] = []
    already: list[str] = []

    with backend.engine.connect() as conn:
        tables_sql = text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name LIKE '%\\_messages' ESCAPE '\\'
              AND table_name NOT LIKE '%\\_forgotten\\_messages' ESCAPE '\\'
        """)
        tables = [row.table_name for row in conn.execute(tables_sql).fetchall()]

        for table_name in tables:
            check_sql = text("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = :t AND column_name = 'attachments'
            """)
            present = conn.execute(check_sql, {"t": table_name}).fetchone()
            if present:
                already.append(table_name)
                continue
            if dry_run:
                altered.append(table_name)
                continue
            conn.execute(text(
                f"ALTER TABLE {table_name} "
                f"ADD COLUMN IF NOT EXISTS attachments JSONB NOT NULL DEFAULT '[]'::jsonb"
            ))
            altered.append(table_name)

        if not dry_run:
            conn.commit()

    return {
        "altered": altered,
        "already_present": already,
        "scanned": len(tables) if 'tables' in locals() else 0,
        "dry_run": dry_run,
    }
