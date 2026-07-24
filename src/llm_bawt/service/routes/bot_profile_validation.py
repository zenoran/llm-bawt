"""Bot-profile payload building, normalization, and config-time validation.

Split out of ``routes/settings.py`` (TASK-554) as a cohesive unit: turning an
upsert/create request into a profile payload, normalizing the legacy
``agent_backend_config.model`` key, validating that an agent bot's
``default_model`` resolves to a backend-compatible catalog entry, and
humanizing the ``bot_profiles`` DB constraint errors.

``routes/settings.py`` re-imports every name here so existing attribute access
(``settings_routes._normalize_agent_backend_config_model`` etc., used by tests)
and the route handlers keep working unchanged.
"""

import logging

from fastapi import HTTPException

from ...bot_types import normalize_bot_type
from ..dependencies import get_bot_profile_store, get_service
from ..schemas import BotCreateRequest, BotProfileUpsertRequest

logger = logging.getLogger(__name__)


def _request_to_profile_payload(
    slug: str,
    request: BotProfileUpsertRequest | BotCreateRequest,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "slug": slug,
        "name": request.name,
        "description": request.description,
        "system_prompt": request.system_prompt,
        "requires_memory": request.requires_memory,
        "voice_optimized": request.voice_optimized,
        "tts_mode": request.tts_mode,
        "include_summaries": request.include_summaries,
        "include_in_global_search": request.include_in_global_search,
        "uses_tools": request.uses_tools,
        "uses_search": request.uses_search,
        "uses_home_assistant": request.uses_home_assistant,
    }
    for field_name in (
        "prompt_override_id",
        "default_model",
        "color",
        "avatar",
        "default_voice",
        "nextcloud_config",
        "bot_type",
        "agent_backend",
        "agent_backend_config",
    ):
        if field_name in request.model_fields_set:
            payload[field_name] = getattr(request, field_name)
    return payload


def _validate_profile_payload(payload: dict[str, object]) -> None:
    resolved_type = normalize_bot_type(
        str(payload.get("bot_type")) if payload.get("bot_type") is not None else None,
        str(payload.get("agent_backend")) if payload.get("agent_backend") is not None else None,
    )
    agent_backend = str(payload.get("agent_backend")).strip() if payload.get("agent_backend") is not None else ""
    if resolved_type == "agent" and not agent_backend:
        raise HTTPException(status_code=400, detail="Agent bots require agent_backend")
    _normalize_agent_backend_config_model(agent_backend, payload)
    _validate_agent_default_model(agent_backend, payload)


#: Agent backends whose model is canonically configured via ``default_model``
#: (a catalog alias). Openclaw is exempt — its gateway owns the model, and
#: its ``agent_backend_config`` keys are session-path strings.
_CATALOG_MODEL_BACKENDS = ("claude-code", "codex")


def _normalize_agent_backend_config_model(
    agent_backend: str,
    payload: dict[str, object],
) -> None:
    """Mirror the legacy ``agent_backend_config.model`` key on save.

    ``default_model`` is the single canonical model reference; the bridges
    persist their session metadata under ``session_model``. Any ``model``
    key arriving here is either a stale UI payload or an old (pre-rename)
    bridge ``_set_session`` PATCH — in both cases the value is session
    metadata, so it's MIRRORED to ``session_model`` rather than rejected.

    The legacy ``model`` key is deliberately KEPT (not popped): bridges that
    haven't been restarted onto the renamed key still read it to decide
    session resume-vs-reset, so stripping it here would reset every agent
    session on the next turn. New (post-rename) bridge code pops ``model``
    on its first ``_set_session`` persist, so the key self-cleans once each
    bridge restarts. Empty/non-string values ARE dropped.
    """
    if agent_backend not in _CATALOG_MODEL_BACKENDS:
        return
    config = payload.get("agent_backend_config")
    if not isinstance(config, dict) or "model" not in config:
        return
    config = dict(config)
    legacy = config.get("model")
    if isinstance(legacy, str) and legacy.strip():
        config["session_model"] = legacy.strip()
        logger.warning(
            "bot %s: mirrored legacy agent_backend_config.model=%r to "
            "session_model (kept for un-restarted bridges) — the bot's "
            "model is configured via default_model",
            payload.get("slug"), legacy,
        )
    else:
        config.pop("model")
    payload["agent_backend_config"] = config


def _validate_agent_default_model(
    agent_backend: str,
    payload: dict[str, object],
) -> None:
    """Config-time guarantee that an agent bot's model is resolvable.

    For claude-code/codex bots, ``default_model`` must reference a
    ``model_definitions`` entry whose shape matches the backend (see
    ``agent_backend_for_model_def``) and which carries a non-empty
    ``model_id``. This fails the SAVE with a 422 instead of letting the
    bot 500 at request time (claude_code.py's hard-require remains as the
    final safety net). Openclaw is exempt; chat bots are covered by the
    DB trigger.
    """
    if agent_backend not in _CATALOG_MODEL_BACKENDS:
        return

    raw = payload.get("default_model")
    default_model = raw.strip() if isinstance(raw, str) else ""
    if not default_model:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Agent bots with agent_backend={agent_backend} require "
                f"default_model: set it to a {agent_backend} model catalog "
                f"entry (add one via /v1/models/definitions if needed)."
            ),
        )

    service = get_service()
    from ...bot_types import agent_backend_for_model_def

    resolver = getattr(service.config, "resolve_model", None)
    if not callable(resolver):
        store = get_bot_profile_store(service.config)
        engine = getattr(store, "engine", None)
        if engine is None:
            return
        from sqlalchemy import text as sa_text

        try:
            with engine.connect() as conn:
                row = conn.execute(
                    sa_text(
                        "SELECT type, model_id, COALESCE(extra->>'backend', '') AS extra_backend "
                        "FROM model_definitions WHERE alias = :v LIMIT 1"
                    ),
                    {"v": default_model},
                ).fetchone()
        except Exception:
            return
        if row is None:
            model_def = None
        else:
            model_def = {
                "type": (row[0] or "").lower(),
                "backend": (row[2] or "").lower() or None,
                "model_id": row[1],
            }
    else:
        model_def = None
    try:
        from ...model_catalog import ModelNotFoundError

        if callable(resolver):
            harness = payload.get("harness")
            if not harness and agent_backend == "codex":
                harness = "codex"
            if not harness and agent_backend == "claude-code":
                endpoint = service.config.ensure_model_catalog().resolve_endpoint(default_model)
                harness = (
                    "claude-code"
                    if endpoint.access_path.vendor == "anthropic"
                    and endpoint.access_path.protocol == "anthropic-messages"
                    else "claude-proxy"
                )
            model_def = resolver(
                default_model,
                harness=str(harness) if harness else None,
            )
    except ModelNotFoundError:
        model_def = None
    except Exception:
        # Lookup failure must not gate the save — the DB trigger still
        # enforces backend↔type compatibility on default_model.
        return

    if not model_def:
        raise HTTPException(
            status_code=422,
            detail=(
                f"default_model={default_model!r} is not a known model "
                f"catalog entry. Register it via /v1/models/definitions first."
            ),
        )

    if agent_backend_for_model_def(model_def) != agent_backend:
        raise HTTPException(
            status_code=422,
            detail=(
                f"default_model={default_model!r} (type={model_def['type'] or 'unknown'}, "
                f"backend={model_def.get('backend') or 'n/a'}) is not compatible with "
                f"agent_backend={agent_backend}. Pick a {agent_backend} catalog entry."
            ),
        )
    if not (isinstance(model_def.get("model_id"), str) and model_def["model_id"].strip()):
        raise HTTPException(
            status_code=422,
            detail=(
                f"default_model={default_model!r} has no model_id — the "
                f"{agent_backend} bridge needs the SDK model id. Fix the "
                f"catalog entry via /v1/models/definitions."
            ),
        )


def _humanize_bot_constraint_error(exc: Exception) -> str | None:
    """Extract a clean message from a bot_profiles DB constraint failure.

    The migration ``add_bot_model_constraints`` installs:
      • a foreign key on ``default_model``
      • a BEFORE trigger that enforces backend↔model.type compatibility

    Both surface through SQLAlchemy as ``IntegrityError`` wrapping a
    psycopg ``ForeignKeyViolation`` / ``RaiseException``. The trigger's
    own ``RAISE EXCEPTION`` text is already user-friendly (e.g.
    ``bot loopy: agent_backend=codex requires a model of type=openai...``)
    so we just lift it out of the SQLAlchemy wrapper.

    Returns the cleaned message, or None if this isn't a bot-constraint
    error (in which case the caller re-raises).
    """
    text = str(getattr(exc, "orig", exc) or exc)
    markers = (
        "bot_profiles_default_model_fk",
        "bot_profiles_check_model_backend",
        "bot_profiles_model_backend_check",
        "agent_backend=",
        "bot_type=",
        "default_model",
    )
    if not any(m in text for m in markers):
        return None
    # Strip psycopg/SQLAlchemy line prefixes and extra context.
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("bot ") and ":" in line:
            return line
    return text.split("CONTEXT:")[0].strip() or None
