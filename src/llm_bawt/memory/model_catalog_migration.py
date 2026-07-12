"""TASK-548: normalize models, access paths, endpoints, and bot harnesses.

This is deliberately an additive cutover.  ``model_definitions`` remains the
writable legacy table until the resolver and CRUD API tasks move all callers to
the normalized catalog.  ``model_definitions_compat`` proves that the new
catalog can reproduce the legacy read shape without making the current runtime
depend on a multi-table view prematurely.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping

from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)

PROTOCOLS = ("chat-completions", "responses", "anthropic-messages")
HARNESSES = ("chat", "claude-code", "codex", "claude-proxy", "openclaw")


@dataclass(frozen=True)
class AccessPathSpec:
    key: str
    vendor: str
    protocol: str
    base_url: str | None
    auth_mechanism: str
    engine_kind: str | None = None


@dataclass(frozen=True)
class CatalogMapping:
    model_key: str
    model_vendor: str
    display_name: str
    description: str | None
    access_path: AccessPathSpec
    upstream_model_id: str | None
    serving_config: dict[str, Any]
    context_window_override: int | None
    tool_support_override: str | None
    pricing: dict[str, Any] | None
    legacy_type: str
    created_at: Any
    updated_at: Any


ACCESS_PATHS: dict[str, AccessPathSpec] = {
    "openai-api": AccessPathSpec(
        "openai-api",
        "openai",
        "chat-completions",
        "https://api.openai.com/v1",
        "api-key",
    ),
    "openai-oauth": AccessPathSpec(
        "openai-oauth",
        "openai",
        "responses",
        "https://chatgpt.com/backend-api/codex",
        "oauth",
    ),
    "anthropic-api": AccessPathSpec(
        "anthropic-api",
        "anthropic",
        "anthropic-messages",
        "https://api.anthropic.com",
        "api-key",
    ),
    "anthropic-oauth": AccessPathSpec(
        "anthropic-oauth",
        "anthropic",
        "anthropic-messages",
        "https://api.anthropic.com",
        "oauth",
    ),
    "xai-chat": AccessPathSpec(
        "xai-chat",
        "xai",
        "chat-completions",
        "https://api.x.ai/v1",
        "api-key",
    ),
    "xai-responses": AccessPathSpec(
        "xai-responses",
        "xai",
        "responses",
        "https://api.x.ai/v1",
        "api-key",
    ),
    "zai-anthropic": AccessPathSpec(
        "zai-anthropic",
        "zai",
        "anthropic-messages",
        None,
        "api-key",
    ),
    "local-llamacpp": AccessPathSpec(
        "local-llamacpp",
        "local",
        "chat-completions",
        None,
        "none",
        "llama-cpp",
    ),
    "local-vllm": AccessPathSpec(
        "local-vllm",
        "local",
        "chat-completions",
        None,
        "none",
        "vllm",
    ),
    "ollama": AccessPathSpec(
        "ollama",
        "local",
        "chat-completions",
        None,
        "none",
        "ollama",
    ),
    "openai-compatible": AccessPathSpec(
        "openai-compatible",
        "custom",
        "chat-completions",
        None,
        "configured",
    ),
}


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value)
        return dict(parsed) if isinstance(parsed, dict) else {}
    return dict(value)


def _strip_provider_prefix(model_id: str | None) -> tuple[str | None, str | None]:
    if not model_id:
        return None, None
    for prefix, provider in (
        ("openai_chatgpt/", "openai_chatgpt"),
        ("xai/", "xai"),
        ("zai/", "zai"),
    ):
        if model_id.startswith(prefix):
            return model_id[len(prefix) :], provider
    return model_id, None


def _access_path_for(
    legacy_type: str, model_id: str | None, extra: Mapping[str, Any]
) -> AccessPathSpec:
    kind = (legacy_type or "").strip().lower()
    upstream, prefix_provider = _strip_provider_prefix(model_id)
    provider = str(extra.get("provider") or prefix_provider or "").lower()
    backend = str(extra.get("backend") or "").lower()

    if kind in {"codex"} or (kind == "agent_backend" and backend == "codex"):
        return ACCESS_PATHS["openai-oauth"]
    if kind == "claude-code":
        if provider == "openai_chatgpt":
            return ACCESS_PATHS["openai-oauth"]
        if provider == "xai":
            return ACCESS_PATHS["xai-responses"]
        if provider == "zai":
            return ACCESS_PATHS["zai-anthropic"]
        return ACCESS_PATHS["anthropic-oauth"]
    if kind == "openai":
        return ACCESS_PATHS["openai-api"]
    if kind in {"grok", "xai"}:
        return ACCESS_PATHS["xai-chat"]
    if kind == "gguf":
        return ACCESS_PATHS["local-llamacpp"]
    if kind == "vllm":
        return ACCESS_PATHS["local-vllm"]
    if kind == "ollama":
        return ACCESS_PATHS["ollama"]
    if kind in {"openai-compatible", "openai_compatible"}:
        return ACCESS_PATHS["openai-compatible"]
    # Unknown legacy types stay reachable through the generic access path.
    return ACCESS_PATHS["openai-compatible"]


def _model_vendor(alias: str, access: AccessPathSpec, model_id: str | None) -> str:
    probe = f"{alias} {model_id or ''}".lower()
    if "claude" in probe or access.vendor == "anthropic":
        return "anthropic"
    if "gpt" in probe or access.vendor == "openai":
        return "openai"
    if "grok" in probe or access.vendor == "xai":
        return "xai"
    if access.vendor == "zai" or "glm" in probe:
        return "zai"
    if access.vendor == "local":
        return "community"
    return access.vendor


def map_legacy_definition(row: Mapping[str, Any]) -> CatalogMapping:
    """Pure mapping used by both the migration and focused unit tests."""
    alias = str(row["alias"]).strip()
    legacy_type = str(row["type"]).strip().lower()
    model_id = row.get("model_id")
    extra = _as_dict(row.get("extra"))
    access = _access_path_for(legacy_type, model_id, extra)
    stripped_model_id, _ = _strip_provider_prefix(model_id)
    upstream = extra.get("upstream_model") or stripped_model_id

    context_window = extra.get("context_window")
    if context_window is not None:
        context_window = int(context_window)
    tool_support = extra.get("tool_support")
    pricing = extra.get("pricing") if isinstance(extra.get("pricing"), dict) else None

    serving_config: dict[str, Any] = {}
    # psycopg decodes both SQL NULL and JSONB ``null`` as Python None.  The
    # explicit flag from the migration query preserves that distinction so the
    # compatibility view is byte-for-byte faithful to the legacy JSON shape.
    extra_is_sql_null = row.get("extra_is_sql_null", row.get("extra") is None)
    if not extra_is_sql_null:
        serving_config["compat_extra"] = None if row.get("extra") is None else extra
    for key in ("repo_id", "filename"):
        if row.get(key) is not None:
            serving_config[key] = row[key]
    for key in ("chat_format", "n_gpu_layers", "tensor_split", "gpu_layers"):
        if key in extra:
            serving_config[key] = extra[key]

    description = row.get("description")
    return CatalogMapping(
        model_key=alias,
        model_vendor=_model_vendor(alias, access, model_id),
        display_name=str(description or alias),
        description=description,
        access_path=access,
        upstream_model_id=upstream,
        serving_config=serving_config,
        context_window_override=context_window,
        tool_support_override=str(tool_support) if tool_support is not None else None,
        pricing=pricing,
        legacy_type=legacy_type,
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


_CREATE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS models (
    id BIGSERIAL PRIMARY KEY,
    key VARCHAR(128) NOT NULL UNIQUE,
    vendor VARCHAR(64) NOT NULL,
    display_name VARCHAR(256) NOT NULL,
    description TEXT,
    default_context_window INTEGER,
    default_tool_support VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS access_paths (
    id BIGSERIAL PRIMARY KEY,
    key VARCHAR(128) NOT NULL UNIQUE,
    vendor VARCHAR(64) NOT NULL,
    protocol VARCHAR(32) NOT NULL,
    base_url TEXT,
    auth_mechanism VARCHAR(64) NOT NULL,
    engine_kind VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT access_paths_protocol_check
        CHECK (protocol IN ('chat-completions', 'responses', 'anthropic-messages'))
);

CREATE TABLE IF NOT EXISTS model_endpoints (
    id BIGSERIAL PRIMARY KEY,
    model_id BIGINT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    access_path_id BIGINT NOT NULL REFERENCES access_paths(id) ON DELETE RESTRICT,
    upstream_model_id VARCHAR(512),
    serving_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    context_window_override INTEGER,
    tool_support_override VARCHAR(64),
    pricing JSONB,
    legacy_type VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT model_endpoints_model_access_key UNIQUE (model_id, access_path_id)
);

CREATE INDEX IF NOT EXISTS idx_model_endpoints_model_id ON model_endpoints(model_id);
CREATE INDEX IF NOT EXISTS idx_model_endpoints_access_path_id ON model_endpoints(access_path_id);
"""


_BOT_TRIGGER_SQL = """
CREATE OR REPLACE FUNCTION bot_profiles_check_model_backend()
RETURNS TRIGGER AS $$
DECLARE
    endpoint_protocol TEXT;
    endpoint_vendor TEXT;
    endpoint_model_key TEXT;
BEGIN
    -- Legacy callers still write (agent_backend, default_model). Resolve those
    -- writes onto the canonical (harness, endpoint_id) pair during cutover.
    IF TG_OP = 'INSERT'
       OR NEW.default_model IS DISTINCT FROM OLD.default_model
       OR NEW.agent_backend IS DISTINCT FROM OLD.agent_backend THEN
        IF NEW.default_model IS NOT NULL THEN
            SELECT e.id
              INTO NEW.endpoint_id
              FROM model_endpoints e
              JOIN models m ON m.id = e.model_id
              JOIN access_paths a ON a.id = e.access_path_id
             WHERE m.key = NEW.default_model
             ORDER BY
               CASE
                 WHEN NEW.agent_backend = 'codex' AND a.protocol = 'responses' THEN 0
                 WHEN NEW.agent_backend = 'claude-code' AND a.protocol = 'anthropic-messages' THEN 0
                 WHEN NEW.agent_backend = 'claude-code' AND a.protocol IN ('responses', 'chat-completions') THEN 1
                 WHEN NEW.agent_backend IS NULL AND a.protocol = 'chat-completions' THEN 0
                 ELSE 2
               END,
               e.id
             LIMIT 1;
            IF NEW.endpoint_id IS NULL THEN
                RAISE EXCEPTION 'bot %: default_model % has no normalized endpoint',
                    NEW.slug, NEW.default_model
                USING ERRCODE = 'foreign_key_violation';
            END IF;
        ELSE
            NEW.endpoint_id := NULL;
        END IF;

        IF NEW.agent_backend = 'codex' THEN
            NEW.harness := 'codex';
        ELSIF NEW.agent_backend = 'openclaw' THEN
            NEW.harness := 'openclaw';
        ELSIF NEW.agent_backend = 'claude-code' THEN
            SELECT a.protocol, a.vendor
              INTO endpoint_protocol, endpoint_vendor
              FROM model_endpoints e JOIN access_paths a ON a.id = e.access_path_id
             WHERE e.id = NEW.endpoint_id;
            IF endpoint_protocol = 'anthropic-messages' THEN
                NEW.harness := 'claude-code';
            ELSE
                NEW.harness := 'claude-proxy';
            END IF;
        ELSE
            NEW.harness := 'chat';
        END IF;
    ELSE
        -- New callers write the normalized pair. Mirror it into the legacy
        -- fields until TASK-549/550 remove their runtime use.
        IF NEW.endpoint_id IS DISTINCT FROM OLD.endpoint_id
           OR NEW.harness IS DISTINCT FROM OLD.harness THEN
            IF NEW.endpoint_id IS NOT NULL THEN
                SELECT m.key INTO endpoint_model_key
                  FROM model_endpoints e JOIN models m ON m.id = e.model_id
                 WHERE e.id = NEW.endpoint_id;
                NEW.default_model := endpoint_model_key;
            ELSE
                NEW.default_model := NULL;
            END IF;
            NEW.agent_backend := CASE NEW.harness
                WHEN 'chat' THEN NULL
                WHEN 'claude-proxy' THEN 'claude-code'
                ELSE NEW.harness
            END;
        END IF;
    END IF;

    IF NEW.harness = 'openclaw' AND NEW.endpoint_id IS NULL THEN
        RETURN NEW;
    END IF;
    IF NEW.endpoint_id IS NULL THEN
        RAISE EXCEPTION 'bot %: harness % requires endpoint_id', NEW.slug, NEW.harness
        USING ERRCODE = 'not_null_violation';
    END IF;

    SELECT a.protocol, a.vendor, m.key
      INTO endpoint_protocol, endpoint_vendor, endpoint_model_key
      FROM model_endpoints e
      JOIN access_paths a ON a.id = e.access_path_id
      JOIN models m ON m.id = e.model_id
     WHERE e.id = NEW.endpoint_id;
    IF endpoint_protocol IS NULL THEN
        RAISE EXCEPTION 'bot %: endpoint_id % does not exist', NEW.slug, NEW.endpoint_id
        USING ERRCODE = 'foreign_key_violation';
    END IF;

    IF NEW.harness = 'chat' AND endpoint_protocol <> 'chat-completions' THEN
        RAISE EXCEPTION 'bot %: chat harness requires chat-completions endpoint, got %',
            NEW.slug, endpoint_protocol USING ERRCODE = 'check_violation';
    ELSIF NEW.harness = 'codex' AND endpoint_protocol <> 'responses' THEN
        RAISE EXCEPTION 'bot %: codex harness requires responses endpoint, got %',
            NEW.slug, endpoint_protocol USING ERRCODE = 'check_violation';
    ELSIF NEW.harness = 'claude-code' AND endpoint_protocol <> 'anthropic-messages' THEN
        RAISE EXCEPTION 'bot %: claude-code harness requires anthropic-messages endpoint, got %',
            NEW.slug, endpoint_protocol USING ERRCODE = 'check_violation';
    ELSIF NEW.harness = 'claude-proxy'
          AND (endpoint_protocol NOT IN ('responses', 'chat-completions')
               OR endpoint_vendor = 'anthropic') THEN
        RAISE EXCEPTION 'bot %: claude-proxy requires a non-Anthropic responses/chat endpoint',
            NEW.slug USING ERRCODE = 'check_violation';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS bot_profiles_model_backend_check ON bot_profiles;
CREATE TRIGGER bot_profiles_model_backend_check
BEFORE INSERT OR UPDATE ON bot_profiles
FOR EACH ROW EXECUTE FUNCTION bot_profiles_check_model_backend();
"""


_COMPAT_VIEW_SQL = """
CREATE OR REPLACE VIEW model_definitions_compat AS
SELECT
    e.id::INTEGER AS id,
    m.key::VARCHAR(128) AS alias,
    e.legacy_type::VARCHAR(64) AS type,
    CASE
      WHEN e.legacy_type = 'claude-code' AND a.key = 'openai-oauth'
        THEN 'openai_chatgpt/' || e.upstream_model_id
      WHEN e.legacy_type = 'claude-code' AND a.key = 'xai-responses'
        THEN 'xai/' || e.upstream_model_id
      WHEN e.legacy_type = 'claude-code' AND a.key = 'zai-anthropic'
        THEN 'zai/' || e.upstream_model_id
      ELSE e.upstream_model_id
    END::VARCHAR(512) AS model_id,
    (e.serving_config->>'repo_id')::VARCHAR(512) AS repo_id,
    (e.serving_config->>'filename')::VARCHAR(512) AS filename,
    m.description,
    e.serving_config->'compat_extra' AS extra,
    e.created_at,
    e.updated_at
FROM model_endpoints e
JOIN models m ON m.id = e.model_id
JOIN access_paths a ON a.id = e.access_path_id;
"""


def _row_mapping(row: Any) -> dict[str, Any]:
    return dict(row._mapping if hasattr(row, "_mapping") else row)


def _derive_harness(agent_backend: str | None, access: AccessPathSpec | None) -> str:
    backend = (agent_backend or "").strip().lower()
    if backend == "codex":
        return "codex"
    if backend == "openclaw":
        return "openclaw"
    if backend == "claude-code":
        if access and access.protocol == "anthropic-messages":
            return "claude-code"
        return "claude-proxy"
    return "chat"


def migrate_model_catalog(engine: Engine, dry_run: bool = False) -> dict[str, Any]:
    """Create and backfill the normalized model catalog in one transaction."""
    with engine.connect() as conn:
        legacy_exists = bool(
            conn.execute(
                text("""
            SELECT 1 FROM information_schema.tables
             WHERE table_schema = 'public' AND table_name = 'model_definitions'
               AND table_type = 'BASE TABLE'
        """)
            ).fetchone()
        )
        bots_exist = bool(
            conn.execute(
                text("""
            SELECT 1 FROM information_schema.tables
             WHERE table_schema = 'public' AND table_name = 'bot_profiles'
               AND table_type = 'BASE TABLE'
        """)
            ).fetchone()
        )
        if not legacy_exists:
            return {
                "skipped": True,
                "reason": "model_definitions base table not present",
            }

        legacy_rows = [
            _row_mapping(row)
            for row in conn.execute(
                text(
                    "SELECT id, alias, type, model_id, repo_id, filename, description, "
                    "extra, extra IS NULL AS extra_is_sql_null, created_at, updated_at "
                    "FROM model_definitions ORDER BY id"
                )
            ).fetchall()
        ]
        mappings = [map_legacy_definition(row) for row in legacy_rows]
        bot_rows = []
        if bots_exist:
            bot_rows = [
                _row_mapping(row)
                for row in conn.execute(
                    text(
                        "SELECT slug, agent_backend, default_model FROM bot_profiles ORDER BY slug"
                    )
                ).fetchall()
            ]

        preview = {
            "legacy_definitions": len(legacy_rows),
            "bots": len(bot_rows),
            "models": len({item.model_key for item in mappings}),
            "endpoints": len(
                {(item.model_key, item.access_path.key) for item in mappings}
            ),
            "access_paths": sorted({item.access_path.key for item in mappings}),
        }
        if dry_run:
            return {"dry_run": True, **preview}

        # Serialize startup/manual invocations without holding a global lock.
        conn.execute(
            text("SELECT pg_advisory_xact_lock(hashtext('task-548-model-catalog'))")
        )
        conn.execute(text(_CREATE_SCHEMA_SQL))

        endpoint_by_alias: dict[str, tuple[int, AccessPathSpec]] = {}
        for item in mappings:
            access_id = conn.execute(
                text("""
                INSERT INTO access_paths
                    (key, vendor, protocol, base_url, auth_mechanism, engine_kind)
                VALUES (:key, :vendor, :protocol, :base_url, :auth, :engine)
                ON CONFLICT (key) DO UPDATE SET
                    vendor = EXCLUDED.vendor,
                    protocol = EXCLUDED.protocol,
                    base_url = EXCLUDED.base_url,
                    auth_mechanism = EXCLUDED.auth_mechanism,
                    engine_kind = EXCLUDED.engine_kind,
                    updated_at = NOW()
                RETURNING id
            """),
                {
                    "key": item.access_path.key,
                    "vendor": item.access_path.vendor,
                    "protocol": item.access_path.protocol,
                    "base_url": item.access_path.base_url,
                    "auth": item.access_path.auth_mechanism,
                    "engine": item.access_path.engine_kind,
                },
            ).scalar_one()
            model_id = conn.execute(
                text("""
                INSERT INTO models
                    (key, vendor, display_name, description, default_context_window,
                     default_tool_support, created_at, updated_at)
                VALUES (:key, :vendor, :display, :description, :context_window,
                        :tool_support, COALESCE(:created_at, NOW()), COALESCE(:updated_at, NOW()))
                ON CONFLICT (key) DO UPDATE SET
                    vendor = EXCLUDED.vendor,
                    display_name = EXCLUDED.display_name,
                    description = EXCLUDED.description,
                    default_context_window = COALESCE(models.default_context_window, EXCLUDED.default_context_window),
                    default_tool_support = COALESCE(models.default_tool_support, EXCLUDED.default_tool_support),
                    updated_at = EXCLUDED.updated_at
                RETURNING id
            """),
                {
                    "key": item.model_key,
                    "vendor": item.model_vendor,
                    "display": item.display_name,
                    "description": item.description,
                    "context_window": item.context_window_override,
                    "tool_support": item.tool_support_override,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                },
            ).scalar_one()
            endpoint_id = conn.execute(
                text("""
                INSERT INTO model_endpoints
                    (model_id, access_path_id, upstream_model_id, serving_config,
                     context_window_override, tool_support_override, pricing,
                     legacy_type, created_at, updated_at)
                VALUES (:model_id, :access_id, :upstream, CAST(:serving AS JSONB),
                        :context_window, :tool_support, CAST(:pricing AS JSONB),
                        :legacy_type, COALESCE(:created_at, NOW()), COALESCE(:updated_at, NOW()))
                ON CONFLICT (model_id, access_path_id) DO UPDATE SET
                    upstream_model_id = EXCLUDED.upstream_model_id,
                    serving_config = EXCLUDED.serving_config,
                    context_window_override = EXCLUDED.context_window_override,
                    tool_support_override = EXCLUDED.tool_support_override,
                    pricing = EXCLUDED.pricing,
                    legacy_type = EXCLUDED.legacy_type,
                    updated_at = EXCLUDED.updated_at
                RETURNING id
            """),
                {
                    "model_id": model_id,
                    "access_id": access_id,
                    "upstream": item.upstream_model_id,
                    "serving": json.dumps(item.serving_config),
                    "context_window": item.context_window_override,
                    "tool_support": item.tool_support_override,
                    "pricing": json.dumps(item.pricing)
                    if item.pricing is not None
                    else None,
                    "legacy_type": item.legacy_type,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                },
            ).scalar_one()
            endpoint_by_alias[item.model_key] = (endpoint_id, item.access_path)

        if bots_exist:
            conn.execute(
                text(
                    "ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS harness VARCHAR(32)"
                )
            )
            conn.execute(
                text(
                    "ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS endpoint_id BIGINT"
                )
            )
            conn.execute(
                text(
                    "ALTER TABLE bot_profiles DROP CONSTRAINT IF EXISTS bot_profiles_default_model_fk"
                )
            )
            conn.execute(
                text(
                    "ALTER TABLE bot_profiles DROP CONSTRAINT IF EXISTS bot_profiles_harness_check"
                )
            )
            conn.execute(
                text(
                    "ALTER TABLE bot_profiles DROP CONSTRAINT IF EXISTS bot_profiles_endpoint_fk"
                )
            )
            # The legacy trigger rejects precisely the claude-proxy/codex alias
            # collision this migration is fixing, so remove it before backfill.
            conn.execute(
                text(
                    "DROP TRIGGER IF EXISTS bot_profiles_model_backend_check ON bot_profiles"
                )
            )

            for bot in bot_rows:
                endpoint = endpoint_by_alias.get(bot["default_model"])
                endpoint_id = endpoint[0] if endpoint else None
                access = endpoint[1] if endpoint else None
                harness = _derive_harness(bot["agent_backend"], access)
                conn.execute(
                    text("""
                    UPDATE bot_profiles
                       SET harness = :harness, endpoint_id = :endpoint_id
                     WHERE slug = :slug
                """),
                    {
                        "harness": harness,
                        "endpoint_id": endpoint_id,
                        "slug": bot["slug"],
                    },
                )

            conn.execute(
                text("""
                ALTER TABLE bot_profiles
                    ALTER COLUMN harness SET DEFAULT 'chat',
                    ALTER COLUMN harness SET NOT NULL,
                    ADD CONSTRAINT bot_profiles_harness_check
                        CHECK (harness IN ('chat','claude-code','codex','claude-proxy','openclaw')),
                    ADD CONSTRAINT bot_profiles_endpoint_fk
                        FOREIGN KEY (endpoint_id) REFERENCES model_endpoints(id)
                        ON UPDATE CASCADE ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED
            """)
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_bot_profiles_endpoint_id ON bot_profiles(endpoint_id)"
                )
            )
            conn.execute(text(_BOT_TRIGGER_SQL))

        conn.execute(text(_COMPAT_VIEW_SQL))
        conn.commit()

    with engine.connect() as verify:
        compat_count = verify.execute(
            text("SELECT COUNT(*) FROM model_definitions_compat")
        ).scalar_one()
        endpoint_count = verify.execute(
            text("SELECT COUNT(*) FROM model_endpoints")
        ).scalar_one()
        shared_gpt_endpoint_count = verify.execute(
            text("""
            SELECT COUNT(*) FROM model_endpoints e
            JOIN models m ON m.id = e.model_id
            JOIN access_paths a ON a.id = e.access_path_id
            WHERE m.key = 'gpt-5.6-sol' AND a.key = 'openai-oauth'
        """)
        ).scalar_one()
        compat_mismatches = verify.execute(
            text("""
            SELECT COUNT(*)
              FROM model_definitions legacy
              FULL JOIN model_definitions_compat compat USING (alias)
             WHERE legacy.alias IS NULL OR compat.alias IS NULL
                OR legacy.type IS DISTINCT FROM compat.type
                OR legacy.model_id IS DISTINCT FROM compat.model_id
                OR legacy.repo_id IS DISTINCT FROM compat.repo_id
                OR legacy.filename IS DISTINCT FROM compat.filename
                OR legacy.description IS DISTINCT FROM compat.description
                OR legacy.extra IS DISTINCT FROM compat.extra
        """)
        ).scalar_one()
        unresolved_bots = 0
        if bots_exist:
            unresolved_bots = verify.execute(
                text("""
                SELECT COUNT(*) FROM bot_profiles
                 WHERE harness <> 'openclaw' AND endpoint_id IS NULL
            """)
            ).scalar_one()
    return {
        **preview,
        "compat_rows": compat_count,
        "endpoint_rows": endpoint_count,
        "compat_mismatches": compat_mismatches,
        "gpt_5_6_sol_openai_oauth_endpoints": shared_gpt_endpoint_count,
        "unresolved_bots": unresolved_bots,
    }


def main() -> None:
    from ..utils.config import Config
    from ..utils.db import get_shared_engine

    parser = argparse.ArgumentParser(
        description="Normalize the model catalog (TASK-548)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    engine = get_shared_engine(Config())
    if engine is None:
        raise SystemExit("database unavailable")
    print(
        json.dumps(
            migrate_model_catalog(engine, dry_run=args.dry_run), indent=2, default=str
        )
    )


if __name__ == "__main__":
    main()
