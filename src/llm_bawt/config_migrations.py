"""Config migrations for the Prompt & Config Unification project (TASK-491/492).

Backfills typed runtime_settings rows from the legacy ``agent_backend_config``
JSON blob so promoted tunables have a first-class home. ADDITIVE and REVERSIBLE:
it only writes bot-scoped runtime_settings rows (never mutates the blob), so
rollback is ``delete_value`` on the same keys. The blob stays intact as the
compat source (the bridge still reads it) until a later coordinated cutover.

Idempotent: skips a row that already holds the target value.

Run:
    python -m llm_bawt.config_migrations --dry-run
    python -m llm_bawt.config_migrations            # apply
    python -m llm_bawt.config_migrations --rollback # delete backfilled rows
"""

from __future__ import annotations

import argparse
import logging

from sqlmodel import Session, select

from .runtime_settings import BotProfile, BotProfileStore, RuntimeSettingsStore
from .utils.config import Config

logger = logging.getLogger(__name__)

# Keys promoted out of agent_backend_config into typed runtime_settings.
PROMOTED_KEYS = ("seed_summary_on_new_session", "timeout_seconds", "session_model")

# The unified session-memory-continuity key (TASK-492). For agent bots it
# reflects the current seed flag so behavior is preserved on cutover.
CONTINUITY_KEY = "session_memory_continuity"


def _agent_bots(config: Config) -> list[dict]:
    store = BotProfileStore(config)
    with Session(store.engine) as s:
        rows = s.exec(select(BotProfile)).all()
        # Detach plain dicts of the fields we need (session closes after this).
        return [
            {
                "slug": r.slug,
                "bot_type": r.bot_type,
                "agent_backend": getattr(r, "agent_backend", None),
                "agent_backend_config": dict(r.agent_backend_config or {}),
            }
            for r in rows
            if (r.bot_type == "agent" or getattr(r, "agent_backend", None))
        ]


def backfill_typed_agent_settings(config: Config, dry_run: bool = True) -> dict:
    """Backfill typed runtime_settings rows from agent_backend_config blobs."""
    store = RuntimeSettingsStore(config)
    actions: list[str] = []
    skipped: list[str] = []

    for bot in _agent_bots(config):
        slug = bot["slug"]
        blob = dict(bot["agent_backend_config"] or {})
        existing = store.get_scope_settings("bot", slug)

        targets: dict[str, object] = {}
        for key in PROMOTED_KEYS:
            if key in blob:
                targets[key] = blob[key]
        # Unified continuity: preserve current effective behavior — for agent
        # bots that is the seed flag (default False when absent).
        targets[CONTINUITY_KEY] = bool(blob.get("seed_summary_on_new_session", False))

        for key, value in targets.items():
            if key in existing and existing[key] == value:
                skipped.append(f"{slug}.{key} (already {value!r})")
                continue
            actions.append(f"{slug}.{key} = {value!r} (was {existing.get(key, '<unset>')!r})")
            if not dry_run:
                store.set_value("bot", slug, key, value)

    return {"dry_run": dry_run, "written": actions, "skipped": skipped}


def rollback_typed_agent_settings(config: Config, dry_run: bool = True) -> dict:
    """Delete the backfilled rows (PROMOTED_KEYS + continuity) for all agent bots."""
    store = RuntimeSettingsStore(config)
    removed: list[str] = []
    for bot in _agent_bots(config):
        slug = bot["slug"]
        for key in (*PROMOTED_KEYS, CONTINUITY_KEY):
            if not dry_run:
                if store.delete_value("bot", slug, key):
                    removed.append(f"{slug}.{key}")
            else:
                removed.append(f"{slug}.{key} (would delete)")
    return {"dry_run": dry_run, "removed": removed}


# ── TASK-614: migrate runtime_settings rows to the canonical tier keys ──
#
# One-time data transformation after the Tier-1/2/3 canonical keys landed
# (TASK-610/611). Moves existing operator choices from the old keys into the
# new canonical settings, then deletes the old rows. NO env values are read;
# NO runtime dual-read is introduced (this is a manually-run CLI, not boot
# code). Idempotent: re-running skips already-migrated rows.
#
#   Renames (value preserved, all scopes):
#     max_context_tokens         -> history_tokens   (drop the 0 footgun row)
#     summarization_max_in_context -> summary_count
#   Fold into the global-only Tier-1 summarization_job dict (store only the
#   deviations from SUMMARIZATION_JOB_DEFAULTS; resolve merges the rest), then
#   delete the retired standalone keys (all scopes — the job is global-only):
#     summarization_session_gap_seconds -> session_gap_seconds
#     summarization_min_messages        -> min_messages_per_session
#     memory_protected_recent_turns     -> protected_recent_turns
#
# Run:
#     python -m llm_bawt.config_migrations --context --dry-run
#     python -m llm_bawt.config_migrations --context            # apply
#     python -m llm_bawt.config_migrations --context --rollback

CONTEXT_RENAMES = {
    "max_context_tokens": "history_tokens",
    "summarization_max_in_context": "summary_count",
}
JOB_FOLD = {
    "summarization_session_gap_seconds": "session_gap_seconds",
    "summarization_min_messages": "min_messages_per_session",
    "memory_protected_recent_turns": "protected_recent_turns",
}


def _snapshot_rows(store: RuntimeSettingsStore) -> list[tuple[str, str, str, object]]:
    """(scope_type, scope_id, key, decoded_value) for every runtime_settings row."""
    import json as _json

    from .runtime_settings import RuntimeSetting

    with Session(store.engine) as s:
        rows = s.exec(select(RuntimeSetting)).all()
        out = []
        for r in rows:
            try:
                out.append((r.scope_type, r.scope_id, r.key, _json.loads(r.value_json)))
            except Exception:
                out.append((r.scope_type, r.scope_id, r.key, None))
        return out


def migrate_context_config_keys(config: Config, dry_run: bool = True) -> dict:
    """Move old context/summarization rows onto the canonical tier keys."""
    from .setting_definitions import SUMMARIZATION_JOB_DEFAULTS

    store = RuntimeSettingsStore(config)
    rows = _snapshot_rows(store)
    actions: list[str] = []

    # 1. Renames (value preserved). Drop the max_context_tokens 0-footgun row.
    for scope_type, scope_id, key, value in rows:
        if key not in CONTEXT_RENAMES:
            continue
        new_key = CONTEXT_RENAMES[key]
        if key == "max_context_tokens" and value in (0, "0", None):
            actions.append(f"DROP {scope_type}:{scope_id}:{key}={value!r} (0 footgun; new default is bounded)")
            if not dry_run:
                store.delete_value(scope_type, scope_id, key)
            continue
        existing = store.get_scope_settings(scope_type, scope_id)
        if new_key in existing:
            actions.append(f"SKIP {scope_type}:{scope_id}:{key}->{new_key} (target already {existing[new_key]!r}); drop old")
            if not dry_run:
                store.delete_value(scope_type, scope_id, key)
            continue
        actions.append(f"RENAME {scope_type}:{scope_id}:{key}->{new_key} = {value!r}")
        if not dry_run:
            store.set_value(scope_type, scope_id, new_key, value)
            store.delete_value(scope_type, scope_id, key)

    # 2. Fold Tier-1 job params (GLOBAL scope only) into the summarization_job
    #    dict, storing ONLY deviations from the canonical defaults.
    job_overrides: dict[str, object] = {}
    for scope_type, scope_id, key, value in rows:
        if key in JOB_FOLD and scope_type == "global" and scope_id == "*":
            job_key = JOB_FOLD[key]
            if value != SUMMARIZATION_JOB_DEFAULTS.get(job_key):
                job_overrides[job_key] = value
    existing_global = store.get_scope_settings("global", "*")
    if "summarization_job" in existing_global:
        actions.append(f"SKIP summarization_job (already {existing_global['summarization_job']!r})")
    elif job_overrides:
        actions.append(f"INSERT global:summarization_job = {job_overrides!r}")
        if not dry_run:
            store.set_value("global", "*", "summarization_job", job_overrides)
    else:
        actions.append("summarization_job: no deviations from defaults; nothing stored")

    # 3. Delete the retired standalone Tier-1 keys (all scopes — global-only now).
    for scope_type, scope_id, key, _value in rows:
        if key in JOB_FOLD:
            actions.append(f"DELETE {scope_type}:{scope_id}:{key}")
            if not dry_run:
                store.delete_value(scope_type, scope_id, key)

    return {"dry_run": dry_run, "actions": actions}


def rollback_context_config_keys(config: Config, dry_run: bool = True) -> dict:
    """Best-effort reverse of migrate_context_config_keys (renames back, drop job dict).

    Cannot resurrect the exact old fold rows (their bot-scope values were
    dropped by design); restores only the global standalone keys from the
    stored summarization_job dict. Intended for pre-615 emergency rollback.
    """
    from .setting_definitions import SUMMARIZATION_JOB_DEFAULTS

    store = RuntimeSettingsStore(config)
    rows = _snapshot_rows(store)
    reverse = {v: k for k, v in CONTEXT_RENAMES.items()}
    job_reverse = {v: k for k, v in JOB_FOLD.items()}
    actions: list[str] = []

    for scope_type, scope_id, key, value in rows:
        if key in reverse:
            old_key = reverse[key]
            actions.append(f"REVERT {scope_type}:{scope_id}:{key}->{old_key} = {value!r}")
            if not dry_run:
                store.set_value(scope_type, scope_id, old_key, value)
                store.delete_value(scope_type, scope_id, key)
        elif key == "summarization_job" and scope_type == "global":
            merged = dict(SUMMARIZATION_JOB_DEFAULTS)
            if isinstance(value, dict):
                merged.update({k: v for k, v in value.items() if k in merged})
            for job_key, old_key in job_reverse.items():
                actions.append(f"RESTORE global:{old_key} = {merged[job_key]!r}")
                if not dry_run:
                    store.set_value("global", "*", old_key, merged[job_key])
            actions.append("DELETE global:summarization_job")
            if not dry_run:
                store.delete_value("global", "*", "summarization_job")

    return {"dry_run": dry_run, "actions": actions}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Config migrations (TASK-491/492/614)")
    ap.add_argument("--dry-run", action="store_true", help="Show actions without writing")
    ap.add_argument("--rollback", action="store_true", help="Reverse the migration")
    ap.add_argument(
        "--context",
        action="store_true",
        help="TASK-614: migrate context/summarization rows to canonical tier keys",
    )
    args = ap.parse_args()
    config = Config()
    if args.context:
        if args.rollback:
            result = rollback_context_config_keys(config, dry_run=args.dry_run)
        else:
            result = migrate_context_config_keys(config, dry_run=args.dry_run)
    elif args.rollback:
        result = rollback_typed_agent_settings(config, dry_run=args.dry_run)
    else:
        result = backfill_typed_agent_settings(config, dry_run=args.dry_run)
    import json as _json
    print(_json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
