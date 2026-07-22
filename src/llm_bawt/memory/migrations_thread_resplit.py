"""TASK-253 (M1d): idle-gap re-split of the legacy catch-all threads.

TASK-284's backfill (``migrations_session_backfill``) pointed every
pre-session message at ONE giant ``legacy-<bot>`` catch-all thread per bot,
so day-one thread lists show a single monster "conversation" spanning months.
This migration splits each catch-all into natural conversations by walking
its messages chronologically and cutting a new thread at every idle gap of
``--gap-minutes`` (default 30) or more.

Per resulting group it creates a ``sessions`` row::

    status            = 'archived'
    started_at        = first message timestamp
    ended_at          = archived_at = last message timestamp
    session_metadata  = {"title": "Conversation from <YYYY-MM-DD>",
                         "title_source": "auto",
                         "resplit_from": "<catch-all id>",
                         "task": "TASK-253"}

and re-points the group's messages at it (batched UPDATEs by primary key).
Once a catch-all is verifiably empty it is soft-deleted (``status='deleted'``)
so it stops cluttering the thread list; its row survives for provenance.

Guardrails / design facts:

- **Scope**: ONLY threads flagged ``session_metadata.legacy_backfill = true``
  are touched. Live/active threads and post-284 organic threads are never
  candidates. Re-running after commit is a no-op (empty catch-alls yield no
  groups).
- **Provenance only**: this rewrites ``messages.session_id`` VALUES — the
  continuous default read path does not filter on session, so context
  assembly and scroll-back are unaffected. Summary provenance
  (``source_session_id``) is deliberately left alone.
- **Validation**: per catch-all, moved-row total must equal the pre-count and
  zero rows may remain on the catch-all before it is soft-deleted; each
  group's UPDATE rowcount must equal its planned size.
- **Transactions**: one per group (session INSERT + its message UPDATEs
  commit atomically), so an interrupted run leaves whole conversations either
  moved or untouched — never half-moved — and a re-run finishes the rest.

Modes (mutually exclusive):

``--dry-run`` (default)
    Read-only. Prints a per-bot summary table (thread count, message counts,
    date ranges, largest/smallest conversation) plus a JSON tail. No writes.

``--commit``
    Requires env ``THREADS_BACKFILL_ENABLED=true`` AND ``--yes``. Performs
    the split live, then soft-deletes each emptied catch-all.

``--verify``
    Reports catch-all residue (target: 0 messages on every catch-all),
    split-thread counts, and soft-delete status.

Usage::

    uv run python -m llm_bawt.memory.migrations_thread_resplit --dry-run
    THREADS_BACKFILL_ENABLED=true uv run python -m llm_bawt.memory.migrations_thread_resplit --commit --yes
    uv run python -m llm_bawt.memory.migrations_thread_resplit --verify
"""

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

DEFAULT_GAP_MINUTES = 30
UPDATE_BATCH_SIZE = 1000
RESPLIT_TASK = "TASK-253"


# ---------------------------------------------------------------------------
# Discovery + planning (pure reads)
# ---------------------------------------------------------------------------

def _catchall_threads(conn) -> list[dict[str, Any]]:
    """All TASK-284 legacy catch-all threads, regardless of message count."""
    rows = conn.execute(
        text(
            "SELECT id, bot_id, user_id, status FROM sessions "
            "WHERE session_metadata->>'legacy_backfill' = 'true' "
            "ORDER BY bot_id"
        )
    ).fetchall()
    return [
        {"id": r.id, "bot_id": r.bot_id, "user_id": r.user_id, "status": r.status}
        for r in rows
    ]


def _load_message_index(conn, bot_id: str, session_id: str) -> list[tuple[str, float]]:
    """(id, timestamp) for every message on the catch-all, chronological."""
    rows = conn.execute(
        text(
            "SELECT id, timestamp FROM messages "
            "WHERE bot_id = :bot_id AND session_id = :sid "
            "ORDER BY timestamp ASC, id ASC"
        ),
        {"bot_id": bot_id, "sid": session_id},
    ).fetchall()
    return [(r.id, float(r.timestamp)) for r in rows]


def _split_groups(
    index: list[tuple[str, float]], gap_seconds: float
) -> list[dict[str, Any]]:
    """Cut the chronological index into conversations at idle gaps."""
    groups: list[dict[str, Any]] = []
    current: list[str] = []
    first_ts = last_ts = None
    for msg_id, ts in index:
        if last_ts is not None and (ts - last_ts) >= gap_seconds:
            groups.append({"ids": current, "first_ts": first_ts, "last_ts": last_ts})
            current, first_ts = [], None
        if first_ts is None:
            first_ts = ts
        current.append(msg_id)
        last_ts = ts
    if current:
        groups.append({"ids": current, "first_ts": first_ts, "last_ts": last_ts})
    return groups


def _title_for(first_ts: float) -> str:
    day = datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    return f"Conversation from {day}"


def _plan_bot(conn, thread: dict[str, Any], gap_seconds: float) -> dict[str, Any]:
    index = _load_message_index(conn, thread["bot_id"], thread["id"])
    groups = _split_groups(index, gap_seconds)
    sizes = [len(g["ids"]) for g in groups]
    return {
        "catchall": thread["id"],
        "bot_id": thread["bot_id"],
        "user_id": thread["user_id"],
        "messages": len(index),
        "conversations": len(groups),
        "largest": max(sizes) if sizes else 0,
        "smallest": min(sizes) if sizes else 0,
        "date_range": (
            f"{datetime.fromtimestamp(groups[0]['first_ts'], tz=timezone.utc):%Y-%m-%d}"
            f" -> {datetime.fromtimestamp(groups[-1]['last_ts'], tz=timezone.utc):%Y-%m-%d}"
            if groups
            else "-"
        ),
        "groups": groups,
    }


# ---------------------------------------------------------------------------
# Commit path
# ---------------------------------------------------------------------------

def _insert_split_session(
    conn, group: dict[str, Any], plan: dict[str, Any]
) -> str:
    sid = str(uuid.uuid4())
    meta = {
        "title": _title_for(group["first_ts"]),
        "title_source": "auto",
        "resplit_from": plan["catchall"],
        "task": RESPLIT_TASK,
    }
    conn.execute(
        text(
            """
            INSERT INTO sessions
                (id, bot_id, user_id, started_at, ended_at, status,
                 archived_at, session_metadata)
            VALUES
                (:sid, :bot_id, :user_id,
                 to_timestamp(:first_ts) AT TIME ZONE 'UTC',
                 to_timestamp(:last_ts) AT TIME ZONE 'UTC',
                 'archived',
                 to_timestamp(:last_ts) AT TIME ZONE 'UTC',
                 CAST(:meta AS jsonb))
            """
        ),
        {
            "sid": sid,
            "bot_id": plan["bot_id"],
            "user_id": plan["user_id"],
            "first_ts": group["first_ts"],
            "last_ts": group["last_ts"],
            "meta": json.dumps(meta),
        },
    )
    return sid


def _move_group(conn, plan: dict[str, Any], group: dict[str, Any], sid: str) -> int:
    """Re-point the group's messages at the new session, batched by PK."""
    moved = 0
    ids = group["ids"]
    for i in range(0, len(ids), UPDATE_BATCH_SIZE):
        batch = ids[i : i + UPDATE_BATCH_SIZE]
        result = conn.execute(
            text(
                "UPDATE messages SET session_id = :sid "
                "WHERE bot_id = :bot_id AND session_id = :catchall "
                "AND id = ANY(:ids)"
            ),
            {
                "sid": sid,
                "bot_id": plan["bot_id"],
                "catchall": plan["catchall"],
                "ids": batch,
            },
        )
        moved += int(result.rowcount or 0)
    return moved


def _soft_delete_catchall(conn, catchall_id: str) -> None:
    conn.execute(
        text(
            """
            UPDATE sessions SET
                status = 'deleted',
                session_metadata = COALESCE(session_metadata, '{}'::jsonb)
                    || jsonb_build_object('resplit_emptied', true,
                                          'resplit_task', :task)
            WHERE id = :sid
            """
        ),
        {"sid": catchall_id, "task": RESPLIT_TASK},
    )


def run_commit(engine, gap_seconds: float) -> dict[str, Any]:
    with engine.connect() as conn:
        threads = _catchall_threads(conn)
        plans = [_plan_bot(conn, t, gap_seconds) for t in threads]

    class _GroupParityError(RuntimeError):
        pass

    per_bot: dict[str, dict[str, Any]] = {}
    ok = True
    for plan in plans:
        moved_total = 0
        created = 0
        error: str | None = None
        try:
            for group in plan["groups"]:
                with engine.begin() as conn:
                    sid = _insert_split_session(conn, group, plan)
                    moved = _move_group(conn, plan, group, sid)
                    if moved != len(group["ids"]):
                        # Group changed under us (or partial prior run) —
                        # abort this group's txn rather than half-move it.
                        raise _GroupParityError(
                            f"group expected {len(group['ids'])} rows, "
                            f"moved {moved} — group rolled back"
                        )
                    moved_total += moved
                    created += 1
        except _GroupParityError as exc:
            error = str(exc)
            logger.error("resplit %s aborted: %s", plan["catchall"], exc)

        with engine.begin() as conn:
            remaining = int(
                conn.execute(
                    text(
                        "SELECT count(*) FROM messages "
                        "WHERE bot_id = :bot_id AND session_id = :sid"
                    ),
                    {"bot_id": plan["bot_id"], "sid": plan["catchall"]},
                ).scalar()
                or 0
            )
            emptied = remaining == 0
            if emptied and error is None:
                _soft_delete_catchall(conn, plan["catchall"])

        parity = error is None and moved_total == plan["messages"] and emptied
        ok = ok and parity
        per_bot[plan["bot_id"]] = {
            "catchall": plan["catchall"],
            "messages_before": plan["messages"],
            "moved": moved_total,
            "threads_created": created,
            "remaining_on_catchall": remaining,
            "catchall_soft_deleted": emptied and error is None,
            "parity": parity,
            **({"error": error} if error else {}),
        }
        logger.info(
            "resplit %s: %d msgs -> %d threads (remaining=%d, parity=%s)",
            plan["catchall"], moved_total, created, remaining, parity,
        )
    return {"ok": ok, "mode": "commit", "per_bot": per_bot}


# ---------------------------------------------------------------------------
# Dry-run + verify
# ---------------------------------------------------------------------------

def run_dry_run(engine, gap_seconds: float) -> dict[str, Any]:
    with engine.connect() as conn:
        threads = _catchall_threads(conn)
        plans = [_plan_bot(conn, t, gap_seconds) for t in threads]

    header = (
        f"{'bot':<8} {'catch-all':<14} {'msgs':>6} {'threads':>8} "
        f"{'largest':>8} {'smallest':>9}  date range"
    )
    lines = [header, "-" * len(header)]
    for p in plans:
        lines.append(
            f"{p['bot_id']:<8} {p['catchall']:<14} {p['messages']:>6} "
            f"{p['conversations']:>8} {p['largest']:>8} {p['smallest']:>9}  "
            f"{p['date_range']}"
        )
    total_msgs = sum(p["messages"] for p in plans)
    total_threads = sum(p["conversations"] for p in plans)
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<8} {'':<14} {total_msgs:>6} {total_threads:>8}"
    )
    print("\n".join(lines))

    return {
        "ok": True,
        "mode": "dry-run (read-only)",
        "gap_minutes": gap_seconds / 60,
        "total_messages": total_msgs,
        "total_threads_planned": total_threads,
        "per_bot": [
            {k: v for k, v in p.items() if k != "groups"} for p in plans
        ],
    }


def run_verify(engine) -> dict[str, Any]:
    with engine.connect() as conn:
        threads = _catchall_threads(conn)
        residue = {}
        for t in threads:
            residue[t["id"]] = {
                "status": t["status"],
                "remaining_messages": int(
                    conn.execute(
                        text(
                            "SELECT count(*) FROM messages "
                            "WHERE bot_id = :bot_id AND session_id = :sid"
                        ),
                        {"bot_id": t["bot_id"], "sid": t["id"]},
                    ).scalar()
                    or 0
                ),
            }
        split = conn.execute(
            text(
                "SELECT s.bot_id, count(*) AS threads, "
                "COALESCE(sum(m.c), 0) AS msgs "
                "FROM sessions s "
                "LEFT JOIN LATERAL (SELECT count(*) AS c FROM messages m "
                "  WHERE m.session_id = s.id) m ON true "
                "WHERE s.session_metadata->>'task' = :task "
                "GROUP BY s.bot_id ORDER BY s.bot_id"
            ),
            {"task": RESPLIT_TASK},
        ).fetchall()
    all_empty = all(r["remaining_messages"] == 0 for r in residue.values())
    return {
        "ok": all_empty,
        "mode": "verify",
        "catchalls": residue,
        "split_threads": [
            {"bot_id": r.bot_id, "threads": int(r.threads), "messages": int(r.msgs)}
            for r in split
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="TASK-253 idle-gap re-split of legacy catch-all threads."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Plan only (default).")
    mode.add_argument("--commit", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--gap-minutes", type=float, default=DEFAULT_GAP_MINUTES,
        help=f"Idle-gap threshold in minutes (default {DEFAULT_GAP_MINUTES}).",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Confirm a mutating --commit run."
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.commit:
        if os.environ.get("THREADS_BACKFILL_ENABLED", "").lower() != "true":
            print(
                "Refusing --commit: set THREADS_BACKFILL_ENABLED=true.",
                file=sys.stderr,
            )
            return 2
        if not args.yes:
            print("Refusing to --commit without --yes.", file=sys.stderr)
            return 2

    from ..utils.config import Config
    from ..utils.db import get_shared_engine

    config = Config()
    engine = get_shared_engine(config, application_name="llm-bawt-thread-resplit")
    if engine is None:
        print("ERROR: Postgres credentials not configured", file=sys.stderr)
        return 2

    gap_seconds = args.gap_minutes * 60
    if args.commit:
        result = run_commit(engine, gap_seconds)
    elif args.verify:
        result = run_verify(engine)
    else:
        result = run_dry_run(engine, gap_seconds)

    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
