"""Migration utilities for llm-bawt memory schema changes.

Run from command line:
    python -m llm_bawt.memory.migrations backfill_meaning --bot nova
    python -m llm_bawt.memory.migrations all --bot nova

Or import and use programmatically:
    from llm_bawt.memory.migrations import run_all_migrations
    run_all_migrations(backend)
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def backfill_empty_tags(
    backend: Any,
    batch_size: int = 100,
    dry_run: bool = False,
) -> dict:
    """Migrate existing memories: populate empty tags with default.
    
    For each memory where tags IS NULL or empty, set tags = ["misc"].
    
    Args:
        backend: PostgreSQLMemoryBackend instance
        batch_size: Number of rows to process per commit
        dry_run: If True, only report what would be done
        
    Returns:
        Dict with migration statistics
    """
    from sqlalchemy import text

    updated = 0

    with backend.engine.connect() as conn:
        # Count total needing migration (NULL or empty array)
        count_sql = text(f"""
            SELECT COUNT(*) FROM {backend._memories_table_name}
            WHERE tags IS NULL OR tags = '[]'::jsonb
        """)
        total = conn.execute(count_sql).scalar() or 0
        logger.debug(f"Found {total} memories needing tag backfill")

        if total == 0:
            return {"updated": 0, "total": 0}

        if dry_run:
            logger.debug(f"[DRY RUN] Would update {total} memories")
            return {"updated": 0, "total": total, "dry_run": True}

        # Process in batches
        while True:
            fetch_sql = text(f"""
                SELECT id FROM {backend._memories_table_name}
                WHERE tags IS NULL OR tags = '[]'::jsonb
                LIMIT :limit
            """)
            rows = conn.execute(fetch_sql, {"limit": batch_size}).fetchall()
            if not rows:
                break

            for row in rows:
                update_sql = text(f"""
                    UPDATE {backend._memories_table_name}
                    SET tags = CAST(:tags AS jsonb), updated_at = CURRENT_TIMESTAMP
                    WHERE id = :id
                """)
                conn.execute(update_sql, {"id": row.id, "tags": json.dumps(["misc"])})
                updated += 1

            conn.commit()
            logger.debug(f"Migrated {updated}/{total} memories")

    logger.debug(f"Tag backfill complete: {updated} updated")
    return {"updated": updated, "total": total}


def backfill_meaning_embeddings(
    backend: Any,
    batch_size: int = 50,
    dry_run: bool = False,
) -> dict:
    """Generate meaning embeddings for memories that have meaning fields but no meaning_embedding.
    
    Args:
        backend: PostgreSQLMemoryBackend instance
        batch_size: Number of rows to process per commit
        dry_run: If True, only report what would be done
        
    Returns:
        Dict with migration statistics
    """
    from sqlalchemy import text
    from .embeddings import generate_embedding

    updated = 0
    failed = 0

    with backend.engine.connect() as conn:
        # Count total needing migration
        count_sql = text(f"""
            SELECT COUNT(*) FROM {backend._memories_table_name}
            WHERE meaning_embedding IS NULL
              AND (intent IS NOT NULL OR stakes IS NOT NULL OR recurrence_keywords IS NOT NULL)
        """)
        total = conn.execute(count_sql).scalar() or 0
        logger.debug(f"Found {total} memories needing meaning embedding generation")

        if total == 0:
            return {"updated": 0, "failed": 0, "total": 0}

        if dry_run:
            logger.debug(f"[DRY RUN] Would generate embeddings for {total} memories")
            return {"updated": 0, "failed": 0, "total": total, "dry_run": True}

        # Process in batches
        while True:
            fetch_sql = text(f"""
                SELECT id, intent, stakes, recurrence_keywords FROM {backend._memories_table_name}
                WHERE meaning_embedding IS NULL
                  AND (intent IS NOT NULL OR stakes IS NOT NULL OR recurrence_keywords IS NOT NULL)
                LIMIT :limit
            """)
            rows = conn.execute(fetch_sql, {"limit": batch_size}).fetchall()
            if not rows:
                break

            for row in rows:
                try:
                    keywords = row.recurrence_keywords or []
                    if isinstance(keywords, str):
                        keywords = json.loads(keywords)
                    parts = [row.intent or "", row.stakes or "", " ".join(keywords)]
                    meaning_text = " | ".join([p for p in parts if p])
                    if not meaning_text:
                        continue

                    emb = generate_embedding(meaning_text, backend.embedding_model)
                    if emb:
                        update_sql = text(f"""
                            UPDATE {backend._memories_table_name}
                            SET meaning_embedding = :embedding,
                                meaning_updated_at = CURRENT_TIMESTAMP,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = :id
                        """)
                        conn.execute(update_sql, {
                            "id": row.id,
                            "embedding": f"[{','.join(str(x) for x in emb)}]",
                        })
                        updated += 1
                except Exception as e:
                    logger.warning(f"Failed to generate meaning embedding for {row.id}: {e}")
                    failed += 1

            conn.commit()
            logger.debug(f"Generated embeddings for {updated}/{total} memories")

    logger.debug(f"Meaning embedding generation complete: {updated} updated, {failed} failed")
    return {"updated": updated, "failed": failed, "total": total}


def run_all_migrations(backend: Any, dry_run: bool = False) -> dict:
    """Run all pending migrations."""
    results = {}
    
    logger.debug("Running tag backfill...")
    results["tags"] = backfill_empty_tags(backend, dry_run=dry_run)
    
    logger.debug("Running meaning embedding generation...")
    results["meaning_embeddings"] = backfill_meaning_embeddings(backend, dry_run=dry_run)
    
    return results


# CLI entry point
def main():
    import argparse
    from ..utils.config import Config
    from .postgresql import PostgreSQLMemoryBackend

    parser = argparse.ArgumentParser(description="Run memory schema migrations")
    parser.add_argument("command", choices=["backfill_tags", "backfill_meaning", "all"])
    parser.add_argument("--bot", default="nova", help="Bot ID to migrate")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be done")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    config = Config()
    backend = PostgreSQLMemoryBackend(config=config, bot_id=args.bot)

    if args.command == "backfill_tags":
        result = backfill_empty_tags(backend, dry_run=args.dry_run)
    elif args.command == "backfill_meaning":
        result = backfill_meaning_embeddings(backend, dry_run=args.dry_run)
    elif args.command == "all":
        result = run_all_migrations(backend, dry_run=args.dry_run)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
