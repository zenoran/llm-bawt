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


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Backfill typed agent settings (TASK-491/492)")
    ap.add_argument("--dry-run", action="store_true", help="Show actions without writing")
    ap.add_argument("--rollback", action="store_true", help="Delete backfilled rows")
    args = ap.parse_args()
    config = Config()
    if args.rollback:
        result = rollback_typed_agent_settings(config, dry_run=args.dry_run)
    else:
        result = backfill_typed_agent_settings(config, dry_run=args.dry_run)
    import json as _json
    print(_json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
