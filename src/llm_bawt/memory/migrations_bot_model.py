"""Bot/model compatibility schema migrations for llm-bawt.

Split out of ``migrations.py`` (TASK-553). These migrations wire
``bot_profiles`` to ``model_definitions``: the default_model FK + the
agent-backend/model-type compatibility trigger, plus the consolidation of
the legacy ``agent_backend_config.model`` key onto ``default_model``.

The public ``migrations`` facade re-exports every function here, so
``from llm_bawt.memory.migrations import add_bot_model_constraints`` (and
``_derive_model_alias``, used by tests) keeps working.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def add_bot_model_constraints(backend: Any, dry_run: bool = False) -> dict:
    """Add FK + compatibility trigger linking bot_profiles to model_definitions.

    Two guarantees added:
      1. ``bot_profiles.default_model`` (when not NULL) must reference an
         existing ``model_definitions.alias``. ON UPDATE CASCADE so an
         alias rename ripples; ON DELETE SET NULL so a model removal
         doesn't take a bot offline.
      2. A BEFORE INSERT/UPDATE trigger enforces the agent-backend ↔
         model.type compatibility matrix:

            agent_backend      |  model.type required
            -------------------+-----------------------
            'claude-code'      |  'claude-code'
            'codex'            |  'openai'
            'openclaw'         |  any (or NULL)
            NULL               |  any except 'claude-code' (or NULL)

         NULL ``default_model`` is always allowed (lets a backend bridge
         use its own configured fallback).

    Both are idempotent: re-running is a no-op.

    Args:
        backend: PostgreSQLMemoryBackend instance (for engine access)
        dry_run: If True, only report what would be done.

    Returns:
        Dict with action summary.
    """
    from sqlalchemy import text

    actions: list[str] = []

    with backend.engine.connect() as conn:
        # Skip if bot_profiles doesn't exist (fresh DB without bawthub yet).
        has_bots = conn.execute(text("""
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'bot_profiles'
        """)).fetchone()
        has_models = conn.execute(text("""
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'model_definitions'
        """)).fetchone()
        if not (has_bots and has_models):
            logger.debug(
                "bot_profiles or model_definitions missing — skipping constraint migration"
            )
            return {"skipped": True, "reason": "tables not present"}

        # Quarantine any rows that would violate the FK by NULLing them out
        # first (with a warning log). Better than crashing the migration.
        bad_rows = conn.execute(text("""
            SELECT slug, default_model, agent_backend
            FROM bot_profiles
            WHERE default_model IS NOT NULL
              AND default_model NOT IN (SELECT alias FROM model_definitions)
        """)).fetchall()
        for r in bad_rows:
            logger.warning(
                "bot_profiles row will be FK-quarantined: slug=%s default_model=%s "
                "(no matching model_definitions.alias) — clearing default_model",
                r.slug, r.default_model,
            )

        # Quarantine rows that would violate the compat trigger.
        compat_violations = conn.execute(text("""
            SELECT b.slug, b.default_model, b.agent_backend, m.type AS model_type
            FROM bot_profiles b
            LEFT JOIN model_definitions m ON m.alias = b.default_model
            WHERE b.default_model IS NOT NULL
              AND m.alias IS NOT NULL
              AND (
                (b.agent_backend = 'claude-code' AND m.type <> 'claude-code')
                OR (b.agent_backend = 'codex'       AND m.type <> 'openai')
                OR (b.agent_backend IS NULL         AND m.type = 'claude-code')
              )
        """)).fetchall()
        for r in compat_violations:
            logger.warning(
                "bot_profiles row will be compat-quarantined: slug=%s "
                "default_model=%s (type=%s) is incompatible with "
                "agent_backend=%s — clearing default_model",
                r.slug, r.default_model, r.model_type, r.agent_backend,
            )

        if dry_run:
            return {
                "dry_run": True,
                "fk_violations": len(bad_rows),
                "compat_violations": len(compat_violations),
                "would_clear": [r.slug for r in bad_rows]
                                + [r.slug for r in compat_violations],
            }

        # Apply quarantine before adding the constraint.
        for r in bad_rows + compat_violations:
            conn.execute(
                text(
                    "UPDATE bot_profiles SET default_model = NULL, updated_at = NOW() "
                    "WHERE slug = :slug"
                ),
                {"slug": r.slug},
            )
            actions.append(f"cleared default_model on {r.slug}")
        if bad_rows or compat_violations:
            conn.commit()

        # Add FK if missing.
        fk_exists = conn.execute(text("""
            SELECT 1
            FROM pg_constraint
            WHERE conname = 'bot_profiles_default_model_fk'
        """)).fetchone()
        if not fk_exists:
            conn.execute(text("""
                ALTER TABLE bot_profiles
                ADD CONSTRAINT bot_profiles_default_model_fk
                FOREIGN KEY (default_model)
                REFERENCES model_definitions (alias)
                ON UPDATE CASCADE
                ON DELETE SET NULL
                DEFERRABLE INITIALLY DEFERRED
            """))
            conn.commit()
            actions.append("added FK bot_profiles_default_model_fk")
        else:
            actions.append("FK already present")

        # Compatibility trigger function. Replace on every run so updates
        # to the rule set roll out without a manual DROP.
        # NOTE: codex models are stored with type='agent_backend' and
        # extra->>'backend'='codex' (see _normalize_model_definition in
        # service/routes/models.py), not type='codex'. The trigger has to
        # account for that storage normalization.
        conn.execute(text("""
            CREATE OR REPLACE FUNCTION bot_profiles_check_model_backend()
            RETURNS TRIGGER AS $$
            DECLARE
                model_type    TEXT;
                model_backend TEXT;
                is_codex      BOOLEAN;
            BEGIN
                IF NEW.default_model IS NULL THEN
                    RETURN NEW;  -- always allowed; backend uses its fallback
                END IF;

                SELECT type, COALESCE(extra->>'backend', '')
                INTO model_type, model_backend
                FROM model_definitions
                WHERE alias = NEW.default_model;

                IF model_type IS NULL THEN
                    -- FK should catch this first, but be defensive.
                    RAISE EXCEPTION
                        'bot %: default_model % does not exist in model_definitions',
                        NEW.slug, NEW.default_model
                    USING ERRCODE = 'foreign_key_violation';
                END IF;

                -- "codex-flavored" model: either type=codex (legacy/clean shape)
                -- or type=agent_backend with extra.backend=codex (normalized
                -- storage shape used by the model definitions API).
                is_codex := model_type = 'codex'
                    OR (model_type = 'agent_backend' AND model_backend = 'codex');

                IF NEW.agent_backend = 'claude-code' AND model_type <> 'claude-code' THEN
                    RAISE EXCEPTION
                        'bot %: agent_backend=claude-code requires a model of '
                        'type=claude-code, got % (type=%)',
                        NEW.slug, NEW.default_model, model_type
                    USING ERRCODE = 'check_violation';
                END IF;

                IF NEW.agent_backend = 'codex' AND NOT (is_codex OR model_type = 'openai') THEN
                    RAISE EXCEPTION
                        'bot %: agent_backend=codex requires a Codex model '
                        '(type=codex / type=agent_backend+backend=codex) or an '
                        'OpenAI GPT model (type=openai); got % (type=%, backend=%)',
                        NEW.slug, NEW.default_model, model_type, model_backend
                    USING ERRCODE = 'check_violation';
                END IF;

                IF NEW.agent_backend IS NULL AND model_type = 'claude-code' THEN
                    RAISE EXCEPTION
                        'bot %: chat-only bots (agent_backend=NULL) cannot use a '
                        'claude-code model directly — set agent_backend=claude-code '
                        'or pick a chat-completion model',
                        NEW.slug
                    USING ERRCODE = 'check_violation';
                END IF;

                -- 'openclaw' accepts anything; no further checks.

                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
        """))

        trigger_exists = conn.execute(text("""
            SELECT 1 FROM pg_trigger
            WHERE tgname = 'bot_profiles_model_backend_check'
        """)).fetchone()
        if not trigger_exists:
            conn.execute(text("""
                CREATE TRIGGER bot_profiles_model_backend_check
                BEFORE INSERT OR UPDATE ON bot_profiles
                FOR EACH ROW
                EXECUTE FUNCTION bot_profiles_check_model_backend()
            """))
            actions.append("added trigger bot_profiles_model_backend_check")
        else:
            actions.append("trigger already present")

        conn.commit()

    return {
        "actions": actions,
        "fk_quarantined": [r.slug for r in bad_rows],
        "compat_quarantined": [r.slug for r in compat_violations],
    }


def _derive_model_alias(model_id: str) -> str:
    """Derive a catalog alias from an SDK model id.

    ``claude-opus-4-20250514`` → ``opus-4-20250514``; dots/spaces/slashes
    become dashes; result is lowercased.
    """
    alias = (model_id or "").strip().lower()
    if alias.startswith("claude-"):
        alias = alias[len("claude-"):]
    for ch in (".", " ", "/", ":"):
        alias = alias.replace(ch, "-")
    while "--" in alias:
        alias = alias.replace("--", "-")
    return alias.strip("-") or "model"


def migrate_agent_backend_config_model(backend: Any, dry_run: bool = False) -> dict:
    """Consolidate ``agent_backend_config.model`` onto ``default_model``.

    ``default_model`` is the single canonical model reference for every
    bot. The legacy ``agent_backend_config.model`` key (raw SDK model id
    read by the bridges) is replaced by catalog-driven injection (see
    ``ServiceLLMBawt._init_bot``).

    For each agent bot still carrying the legacy key (and no
    ``session_model`` yet):
      * openclaw (and unknown backends): the key is only COPIED to
        ``session_model`` — the openclaw gateway owns its model, and the
        bridges use the stored value for session resume-vs-reset
        comparison, so deleting it would reset live SDK sessions.
      * claude-code / codex:
          1. find-or-create a ``model_definitions`` entry for the legacy
             SDK model id (claude-code → ``type='claude-code'`` as required
             by the ``bot_profiles_check_model_backend`` trigger; codex →
             ``type='agent_backend'`` + ``extra.backend='codex'``)
          2. set ``default_model`` to that alias ONLY if the current value
             doesn't already resolve to a backend-compatible entry (an
             existing compatible value is user intent — the legacy key was
             typically stale)
          3. copy the key to ``session_model`` (the legacy ``model`` key
             is KEPT so un-restarted bridges keep resuming sessions; new
             bridge code drops it on its first persist)

    Idempotent: rows that already carry ``session_model`` are skipped, so
    re-runs — and the lingering compat ``model`` key — are no-ops.
    """
    from sqlalchemy import text

    actions: list[str] = []
    with backend.engine.connect() as conn:
        has_tables = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name IN ('bot_profiles', 'model_definitions')
        """)).scalar()
        if has_tables != 2:
            return {"skipped": True, "reason": "tables not present"}

        rows = conn.execute(text("""
            SELECT slug, agent_backend, default_model,
                   agent_backend_config->>'model' AS legacy_model
            FROM bot_profiles
            WHERE agent_backend IS NOT NULL
              AND agent_backend_config ? 'model'
              AND NOT (agent_backend_config ? 'session_model')
        """)).fetchall()

        if not rows:
            return {"migrated": [], "actions": ["no legacy agent_backend_config.model keys"]}

        def _compatible_alias(alias: str | None, agent_backend: str) -> bool:
            """Does ``alias`` resolve to an entry compatible with the backend?"""
            if not alias:
                return False
            row = conn.execute(
                text(
                    "SELECT type, COALESCE(extra->>'backend','') AS b, model_id "
                    "FROM model_definitions WHERE alias = :a"
                ),
                {"a": alias},
            ).fetchone()
            if row is None:
                return False
            mtype = (row.type or "").lower()
            if agent_backend == "claude-code":
                return mtype == "claude-code" and bool(row.model_id)
            if agent_backend == "codex":
                return bool(row.model_id) and (
                    mtype == "codex"
                    or (mtype == "agent_backend" and row.b == "codex")
                    or mtype == "openai"
                )
            return False

        def _find_or_create_entry(legacy: str, agent_backend: str) -> str | None:
            """Return alias of a catalog entry for ``legacy`` model id."""
            if agent_backend == "claude-code":
                found = conn.execute(text(
                    "SELECT alias FROM model_definitions "
                    "WHERE model_id = :m AND type = 'claude-code' LIMIT 1"
                ), {"m": legacy}).fetchone()
            else:  # codex
                found = conn.execute(text(
                    "SELECT alias FROM model_definitions WHERE model_id = :m AND ("
                    "  type = 'codex'"
                    "  OR (type = 'agent_backend' AND extra->>'backend' = 'codex')"
                    "  OR type = 'openai'"
                    ") LIMIT 1"
                ), {"m": legacy}).fetchone()
            if found:
                return found.alias

            base = _derive_model_alias(legacy)
            alias = base
            n = 1
            while conn.execute(
                text("SELECT 1 FROM model_definitions WHERE alias = :a"),
                {"a": alias},
            ).fetchone():
                n += 1
                alias = f"{base}-{n}"
            if dry_run:
                return alias
            if agent_backend == "claude-code":
                conn.execute(text(
                    "INSERT INTO model_definitions "
                    "(alias, type, model_id, description, created_at, updated_at) VALUES "
                    "(:a, 'claude-code', :m, :d, NOW(), NOW())"
                ), {"a": alias, "m": legacy,
                    "d": "Auto-created by agent_backend_config.model migration"})
            else:
                conn.execute(text(
                    "INSERT INTO model_definitions "
                    "(alias, type, model_id, description, extra, created_at, updated_at) VALUES "
                    "(:a, 'agent_backend', :m, :d, '{\"backend\": \"codex\"}'::jsonb, NOW(), NOW())"
                ), {"a": alias, "m": legacy,
                    "d": "Auto-created by agent_backend_config.model migration"})
            actions.append(f"created model_definitions entry '{alias}' for {legacy}")
            return alias

        migrated: list[str] = []
        for r in rows:
            legacy = (r.legacy_model or "").strip()
            agent_backend = (r.agent_backend or "").strip().lower()

            if agent_backend in ("claude-code", "codex") and legacy:
                if _compatible_alias(r.default_model, agent_backend):
                    actions.append(
                        f"{r.slug}: kept existing default_model={r.default_model!r}"
                    )
                else:
                    alias = _find_or_create_entry(legacy, agent_backend)
                    if alias and not dry_run:
                        conn.execute(text(
                            "UPDATE bot_profiles SET default_model = :a, "
                            "updated_at = NOW() WHERE slug = :s"
                        ), {"a": alias, "s": r.slug})
                    actions.append(f"{r.slug}: set default_model={alias!r}")

            # Copy model → session_model for ALL backends (bridge
            # session-resume comparison must survive the migration). The
            # legacy "model" key is deliberately KEPT: un-restarted bridges
            # still read it for resume-vs-reset, and new bridge code pops it
            # on its first _set_session persist, so it self-cleans.
            if not dry_run:
                conn.execute(text(
                    "UPDATE bot_profiles SET agent_backend_config = "
                    "  agent_backend_config "
                    "  || jsonb_build_object('session_model', "
                    "       agent_backend_config->>'model'), "
                    "  updated_at = NOW() "
                    "WHERE slug = :s AND agent_backend_config ? 'model' "
                    "  AND NOT (agent_backend_config ? 'session_model')"
                ), {"s": r.slug})
            actions.append(f"{r.slug}: copied agent_backend_config.model → session_model")
            migrated.append(r.slug)

        if not dry_run:
            conn.commit()

    return {"migrated": migrated, "actions": actions, "dry_run": dry_run}
