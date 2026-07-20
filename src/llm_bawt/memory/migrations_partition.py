"""TASK-571: consolidate per-bot shard tables into LIST-partitioned parents.

Migrates the legacy ``<bot>_messages`` / ``<bot>_memories`` /
``<bot>_forgotten_messages`` shard tables (33 tables across 11 bots,
~102 MB total) into three LIST-partitioned parents (``messages`` /
``memories`` / ``forgotten_messages``) with one partition per bot —
see ``postgresql.py`` for the runtime model (partition-direct access).

Modes (mutually exclusive):

``--dry-run``
    Runs the full Phase-1 DDL (parents + partitions + parent indexes)
    inside a transaction and ROLLS IT BACK. Verifies that the hnsw and
    gin_trgm parent indexes template onto partitions on the installed
    pgvector build, and prints the copy plan (per-shard row counts).
    Zero persistent impact — safe on the live DB at any time.

``--execute``
    THE cutover. **Stop the app first** (``docker compose stop app``) —
    this script cannot verify that itself. Idempotently creates parents +
    partitions (persisted), then in ONE transaction per bot: copies shard
    rows into the parent (explicit column lists — the live shards have
    column-ORDER drift, so ``SELECT *`` would misalign), asserts count
    parity, and renames the shard to ``zz_old_<shard>`` (kept for
    rollback). Records the cutover timestamp in
    ``partition_migration_meta``. Re-running skips already-cutover bots.

``--verify``
    Post-cutover checks: per-bot count parity (``zz_old_*`` vs
    partition), hash spot-check on the two largest message shards, and
    the per-partition index inventory.

``--finalize``
    After the soak window: drops every ``zz_old_*`` table.

``--rollback``
    Within the soak window: copies rows written AFTER the cutover
    (``created_at > cutover_ts``) from the partitions back into the
    ``zz_old_*`` tables, then renames them back to their original shard
    names. The parents keep their copies; the reverted app code ignores
    them. Pair with reverting the code commit + restarting app.

Usage::

    uv run python -m llm_bawt.memory.migrations_partition --dry-run
    uv run python -m llm_bawt.memory.migrations_partition --execute --yes
"""

import argparse
import json
import logging
import sys
import time
from typing import Any

from sqlalchemy import text

from .postgresql import (
    FORGOTTEN_PARENT,
    MEMORIES_PARENT,
    MESSAGES_PARENT,
    PARENT_TABLES,
    ensure_bot_partitions,
    ensure_parent_tables,
    partition_name,
)

logger = logging.getLogger(__name__)

# Suffix → parent mapping. Longest suffix first so ``x_forgotten_messages``
# never misclassifies as ``x_forgotten`` + ``_messages``.
SHARD_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_forgotten_messages", FORGOTTEN_PARENT),
    ("_memories", MEMORIES_PARENT),
    ("_messages", MESSAGES_PARENT),
)

META_TABLE = "partition_migration_meta"


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def _list_tables(conn) -> set[str]:
    rows = conn.execute(text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
    )).fetchall()
    return {r[0] for r in rows}


def discover_shards(conn) -> dict[str, list[tuple[str, str]]]:
    """Map parent → [(bot_id, shard_table)] for every live legacy shard.

    Excludes the parents themselves (``forgotten_messages`` ends with
    ``_messages``!), partitions (``*_p_*`` names don't carry the suffixes),
    and ``zz_old_*`` rename-kept tables.
    """
    all_tables = _list_tables(conn)
    result: dict[str, list[tuple[str, str]]] = {p: [] for p in PARENT_TABLES}
    for tbl in sorted(all_tables):
        if tbl in PARENT_TABLES or tbl.startswith("zz_old_"):
            continue
        for suffix, parent in SHARD_SUFFIXES:
            if tbl.endswith(suffix):
                bot_id = tbl.removesuffix(suffix)
                if bot_id:
                    result[parent].append((bot_id, tbl))
                break
    return result


def _columns(conn, table: str) -> list[str]:
    rows = conn.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = :t "
        "ORDER BY ordinal_position"
    ), {"t": table}).fetchall()
    return [r[0] for r in rows]


def _copy_columns(conn, shard: str, parent: str) -> list[str]:
    """Ordered intersection of shard and parent columns (minus bot_id).

    Explicit lists are load-bearing: the live message shards have
    column-ORDER drift (attachments/reasoning/recalled_history sit at
    different ordinal positions across bots), so positional ``SELECT *``
    copies would misalign or fail.
    """
    shard_cols = set(_columns(conn, shard))
    parent_cols = [c for c in _columns(conn, parent) if c != "bot_id"]
    return [c for c in parent_cols if c in shard_cols]


def _count(conn, table: str, where: str = "", params: dict | None = None) -> int:
    row = conn.execute(
        text(f"SELECT COUNT(*) FROM {table} {where}"), params or {}
    ).fetchone()
    return int(row[0]) if row else 0


def _id_hash(conn, table: str, where: str = "", params: dict | None = None) -> int:
    """Order-independent content spot-check: sum of hashtext(id)."""
    row = conn.execute(
        text(f"SELECT COALESCE(SUM(hashtext(id)::bigint), 0) FROM {table} {where}"),
        params or {},
    ).fetchone()
    return int(row[0]) if row else 0


def _partition_indexes(conn, partition: str) -> list[str]:
    rows = conn.execute(text(
        "SELECT indexdef FROM pg_indexes "
        "WHERE schemaname = 'public' AND tablename = :t ORDER BY indexname"
    ), {"t": partition}).fetchall()
    return [r[0] for r in rows]


def _ensure_meta(conn) -> None:
    conn.execute(text(
        f"CREATE TABLE IF NOT EXISTS {META_TABLE} "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    ))


def _meta_set(conn, key: str, value: str) -> None:
    conn.execute(text(
        f"INSERT INTO {META_TABLE} (key, value) VALUES (:k, :v) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
    ), {"k": key, "v": value})


def _meta_get(conn, key: str) -> str | None:
    try:
        row = conn.execute(text(
            f"SELECT value FROM {META_TABLE} WHERE key = :k"
        ), {"k": key}).fetchone()
        return row[0] if row else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_dry_run(engine, embedding_dim: int) -> dict[str, Any]:
    """Phase-1 DDL inside a rolled-back transaction + templating check."""
    report: dict[str, Any] = {"mode": "dry-run", "ok": True, "problems": []}

    with engine.connect() as conn:
        shards = discover_shards(conn)
        plan: dict[str, list[dict[str, Any]]] = {}
        for parent, entries in shards.items():
            plan[parent] = [
                {
                    "bot": bot_id,
                    "shard": shard,
                    "rows": _count(conn, shard),
                    "columns": None,  # resolved post-DDL below
                }
                for bot_id, shard in entries
            ]
        report["copy_plan"] = plan

        # End the implicit read transaction so an explicit (rolled-back)
        # one can own the DDL.
        conn.rollback()
        trans = conn.begin()
        try:
            ensure_parent_tables(conn, embedding_dim)
            all_bots = sorted({b for entries in shards.values() for b, _ in entries})
            for bot_id in all_bots:
                ensure_bot_partitions(conn, bot_id)
            sample_bot = all_bots[0] if all_bots else "default"
            if not all_bots:
                ensure_bot_partitions(conn, sample_bot)

            # Column intersection is only computable once the parents exist.
            for parent, entries in shards.items():
                for i, (bot_id, shard) in enumerate(entries):
                    plan[parent][i]["columns"] = _copy_columns(conn, shard, parent)

            # Index-templating verification on a sample partition.
            mem_part = partition_name(MEMORIES_PARENT, sample_bot)
            msg_part = partition_name(MESSAGES_PARENT, sample_bot)
            mem_idx = _partition_indexes(conn, mem_part)
            msg_idx = _partition_indexes(conn, msg_part)
            report["sample_partition_indexes"] = {
                mem_part: mem_idx,
                msg_part: msg_idx,
            }
            hnsw_ok = sum("USING hnsw" in d for d in mem_idx) >= 2
            trgm_ok = any("gin_trgm_ops" in d for d in msg_idx)
            report["hnsw_templated"] = hnsw_ok
            report["gin_trgm_templated"] = trgm_ok
            if not hnsw_ok:
                report["problems"].append(
                    "hnsw parent indexes did NOT template to the memories "
                    "partition — per-partition fallback in "
                    "ensure_bot_partitions will handle it, but verify "
                    "pgvector build"
                )
            if not trgm_ok:
                report["ok"] = False
                report["problems"].append(
                    "gin_trgm parent index did not template — investigate "
                    "before cutover"
                )
        finally:
            trans.rollback()

    return report


def run_execute(engine, embedding_dim: int) -> dict[str, Any]:
    """The cutover: build (persisted) + copy + parity + rename, per bot."""
    report: dict[str, Any] = {"mode": "execute", "ok": True, "bots": {}, "problems": []}
    cutover_ts = time.time()

    # Phase 1 — persisted build (idempotent, additive, no legacy contact).
    with engine.connect() as conn:
        shards = discover_shards(conn)
        ensure_parent_tables(conn, embedding_dim)
        all_bots = sorted({b for entries in shards.values() for b, _ in entries})
        for bot_id in all_bots:
            ensure_bot_partitions(conn, bot_id)
        _ensure_meta(conn)
        conn.commit()

    if not all_bots:
        report["problems"].append(
            "no legacy shard tables found — nothing to copy (already cutover?)"
        )
        return report

    # Phase 2 — copy + parity + rename. One transaction PER BOT: either a
    # bot's three shards fully moved (copied + renamed) or none did — a
    # failure leaves prior bots durably cutover and the failed bot intact.
    for bot_id in all_bots:
        bot_report: dict[str, Any] = {}
        report["bots"][bot_id] = bot_report
        with engine.connect() as conn:
            bot_shards = {
                parent: next((s for b, s in entries if b == bot_id), None)
                for parent, entries in discover_shards(conn).items()
            }
            conn.rollback()  # end the implicit read txn before the explicit one
            trans = conn.begin()
            try:
                for parent in PARENT_TABLES:
                    shard = bot_shards.get(parent)
                    if shard is None:
                        bot_report[parent] = "no shard (skipped)"
                        continue
                    cols = _copy_columns(conn, shard, parent)
                    col_list = ", ".join(cols)
                    conn.execute(text(
                        f"INSERT INTO {parent} (bot_id, {col_list}) "
                        f"SELECT :bot, {col_list} FROM {shard}"
                    ), {"bot": bot_id})

                    src_n = _count(conn, shard)
                    dst_n = _count(
                        conn, parent, "WHERE bot_id = :b", {"b": bot_id}
                    )
                    if src_n != dst_n:
                        raise RuntimeError(
                            f"parity failure {shard}: shard={src_n} "
                            f"partition={dst_n}"
                        )
                    conn.execute(text(
                        f"ALTER TABLE {shard} RENAME TO zz_old_{shard}"
                    ))
                    bot_report[parent] = {"rows": src_n, "renamed": f"zz_old_{shard}"}
                trans.commit()
            except Exception as e:
                trans.rollback()
                report["ok"] = False
                report["problems"].append(f"{bot_id}: {e}")
                logger.error("Cutover failed for %s: %s", bot_id, e)
                break  # stop at first failure; already-done bots stay done

    with engine.connect() as conn:
        _ensure_meta(conn)
        _meta_set(conn, "cutover_ts", repr(cutover_ts))
        _meta_set(conn, "cutover_report", json.dumps(report["bots"], default=str))
        conn.commit()

    return report


def run_verify(engine) -> dict[str, Any]:
    """Post-cutover parity + index inventory."""
    report: dict[str, Any] = {"mode": "verify", "ok": True, "bots": {}, "problems": []}
    with engine.connect() as conn:
        all_tables = _list_tables(conn)
        old_tables = sorted(t for t in all_tables if t.startswith("zz_old_"))
        if not old_tables:
            report["problems"].append("no zz_old_* tables found (finalized already?)")

        sizes: list[tuple[int, str, str, str]] = []
        for old in old_tables:
            legacy = old.removeprefix("zz_old_")
            for suffix, parent in SHARD_SUFFIXES:
                if legacy.endswith(suffix):
                    bot_id = legacy.removesuffix(suffix)
                    part = partition_name(parent, bot_id)
                    old_n = _count(conn, old)
                    new_n = _count(conn, part) if part in all_tables else -1
                    entry = report["bots"].setdefault(bot_id, {})
                    entry[parent] = {"old": old_n, "new": new_n}
                    if old_n > new_n:
                        # new >= old is fine (post-cutover writes); fewer is not.
                        report["ok"] = False
                        report["problems"].append(
                            f"{part}: partition has FEWER rows ({new_n}) "
                            f"than {old} ({old_n})"
                        )
                    if parent == MESSAGES_PARENT:
                        sizes.append((old_n, bot_id, old, part))
                    break

        # Hash spot-check on the two largest message shards. Only meaningful
        # when the partition hasn't accrued post-cutover writes, so compare
        # the shard-id subset: sum hashtext over ids present in the old table.
        for old_n, bot_id, old, part in sorted(sizes, reverse=True)[:2]:
            old_h = _id_hash(conn, old)
            new_h = _id_hash(
                conn, part,
                f"WHERE id IN (SELECT id FROM {old})",
            )
            match = old_h == new_h
            report["bots"][bot_id]["id_hash_match"] = match
            if not match:
                report["ok"] = False
                report["problems"].append(f"id-hash mismatch: {old} vs {part}")

        # Index inventory for one sample partition per parent.
        sample = next(iter(report["bots"]), None)
        if sample:
            report["sample_partition_indexes"] = {
                partition_name(p, sample): _partition_indexes(
                    conn, partition_name(p, sample)
                )
                for p in PARENT_TABLES
            }
    return report


def run_finalize(engine) -> dict[str, Any]:
    """Drop the zz_old_* safety tables after the soak window."""
    report: dict[str, Any] = {"mode": "finalize", "dropped": []}
    with engine.connect() as conn:
        old_tables = sorted(t for t in _list_tables(conn) if t.startswith("zz_old_"))
        for tbl in old_tables:
            conn.execute(text(f'DROP TABLE IF EXISTS "{tbl}" CASCADE'))
            report["dropped"].append(tbl)
        _ensure_meta(conn)
        _meta_set(conn, "finalized_ts", repr(time.time()))
        conn.commit()
    return report


def run_rollback(engine, cutover_ts: float | None) -> dict[str, Any]:
    """Rename zz_old_* back, after copying post-cutover delta rows into them.

    Pair with: revert the code commit, then restart app.
    """
    report: dict[str, Any] = {"mode": "rollback", "ok": True, "bots": {}, "problems": []}
    with engine.connect() as conn:
        if cutover_ts is None:
            raw = _meta_get(conn, "cutover_ts")
            if raw is None:
                report["ok"] = False
                report["problems"].append(
                    "no recorded cutover_ts and none passed via --cutover-ts"
                )
                return report
            cutover_ts = float(raw)

        old_tables = sorted(t for t in _list_tables(conn) if t.startswith("zz_old_"))
        conn.rollback()  # end the implicit read txn before the explicit one
        trans = conn.begin()
        try:
            for old in old_tables:
                legacy = old.removeprefix("zz_old_")
                for suffix, parent in SHARD_SUFFIXES:
                    if legacy.endswith(suffix):
                        bot_id = legacy.removesuffix(suffix)
                        part = partition_name(parent, bot_id)
                        cols = _copy_columns(conn, old, parent)
                        col_list = ", ".join(cols)
                        # Post-cutover delta only; shard PK is (id).
                        # created_at is naive-UTC (CURRENT_TIMESTAMP on a
                        # UTC server) — convert the epoch the same way.
                        res = conn.execute(text(
                            f"INSERT INTO {old} ({col_list}) "
                            f"SELECT {col_list} FROM {part} "
                            f"WHERE created_at > (to_timestamp(:ts) AT TIME ZONE 'UTC') "
                            f"ON CONFLICT (id) DO NOTHING"
                        ), {"ts": cutover_ts})
                        conn.execute(text(
                            f"ALTER TABLE {old} RENAME TO {legacy}"
                        ))
                        report["bots"].setdefault(bot_id, {})[parent] = {
                            "delta_rows": res.rowcount,
                            "restored": legacy,
                        }
                        break
            trans.commit()
        except Exception as e:
            trans.rollback()
            report["ok"] = False
            report["problems"].append(str(e))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="TASK-571 shard→partition migration (see module docstring)"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--rollback", action="store_true")
    parser.add_argument(
        "--yes", action="store_true",
        help="Confirm destructive modes (--execute / --finalize / --rollback)",
    )
    parser.add_argument(
        "--cutover-ts", type=float, default=None,
        help="Override the recorded cutover timestamp for --rollback",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    from ..utils.config import Config
    from ..utils.db import get_shared_engine

    config = Config()
    engine = get_shared_engine(config, application_name="llm-bawt-migration")
    if engine is None:
        print("ERROR: Postgres credentials not configured", file=sys.stderr)
        return 2
    embedding_dim = int(getattr(config, "MEMORY_EMBEDDING_DIM", 384))

    if args.execute or args.finalize or args.rollback:
        if not args.yes:
            print(
                "Refusing to run a mutating mode without --yes. "
                "For --execute, STOP THE APP FIRST (docker compose stop app).",
                file=sys.stderr,
            )
            return 2

    if args.dry_run:
        result = run_dry_run(engine, embedding_dim)
    elif args.execute:
        result = run_execute(engine, embedding_dim)
    elif args.verify:
        result = run_verify(engine)
    elif args.finalize:
        result = run_finalize(engine)
    else:
        result = run_rollback(engine, args.cutover_ts)

    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
