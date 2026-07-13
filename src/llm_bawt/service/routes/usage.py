"""Canonical subscription-usage endpoint.

``GET /v1/usage``
    All registered providers (the "across all backends" view). Returns
    ``AllUsage`` — one canonical ``ProviderUsage`` per provider, including
    not-yet-implemented ones (clearly flagged) so the UI can list every
    configured subscription.

``GET /v1/usage?provider=claude``
    A single provider. One upstream call (cached). What the per-chat usage
    popup uses.

``GET /v1/usage?bot_id=byte``
    Resolves the bot's provider server-side (from its model definition), then
    returns that provider's usage. Lets the UI ask "usage for the bot I'm
    talking to" without knowing the provider mapping itself.

``?force=true`` bypasses the cache (use sparingly — these endpoints are
rate-limited).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from ...model_catalog import bot_model_ref
from ..dependencies import get_service
from ..usage import AllUsage, ProviderUsage, get_all, get_usage, has_provider, list_providers

log = logging.getLogger(__name__)

router = APIRouter()


def _provider_for_bot(service, bot_id: str) -> str:
    """Best-effort map a bot slug -> canonical usage provider.

    A claude-code bot's model_id carries the provider as a ``<provider>/<model>``
    prefix (e.g. ``xai/grok-4.5``, ``zai/glm-5.2``, ``openai_chatgpt/gpt-5.4``);
    native Claude bots have no prefix. Falls back to ``claude`` only when the
    model has no provider prefix or the prefix isn't a registered usage
    provider.

    Aliases: model namespaces sometimes differ from usage provider ids
    (``grok`` → ``xai``). Keep the map tiny and explicit.
    """
    # model_id prefix → usage registry key (only when they differ)
    _PREFIX_ALIASES = {
        "grok": "xai",
        "x-ai": "xai",
        "openai": "openai_chatgpt",
        "chatgpt": "openai_chatgpt",
        "codex": "openai_chatgpt",
    }

    try:
        from ...bots import BotManager

        bot = BotManager(service.config).get_bot(bot_id)
    except Exception:  # noqa: BLE001
        bot = None
    if bot is None:
        return "claude"

    # Normalized profiles bind to a canonical endpoint. ``default_model`` is
    # only the shared model key and may not uniquely identify its access path.
    # Usage attribution needs that access path (not the harness name) to tell an
    # OpenAI-backed Claude proxy from a native Anthropic subscription.
    try:
        alias = bot_model_ref(service.config, bot)
    except Exception:  # noqa: BLE001
        alias = getattr(bot, "default_model", None)
    model_id = None
    model_type = None
    access_path = None
    if alias:
        try:
            model_def = service.config.resolve_model(
                alias,
                harness=getattr(bot, "harness", None),
                default={},
            )
            model_id = model_def.get("model_id")
            model_type = model_def.get("type")
            access_path = model_def.get("access_path")
        except Exception:  # noqa: BLE001
            model_id = None
            model_type = None
            access_path = None

    # ChatGPT/Codex subscription usage is defined by the OAuth access path.
    # Do not map every OpenAI vendor/API model here: API-key endpoints do not
    # consume the ChatGPT subscription quota shown by this endpoint.
    if str(access_path or "").strip().lower() == "openai-oauth":
        return "openai_chatgpt"

    if model_id and "/" in str(model_id):
        prefix = str(model_id).split("/", 1)[0].strip().lower()
        prefix = _PREFIX_ALIASES.get(prefix, prefix)
        if has_provider(prefix):
            return prefix
        # Known non-Claude bridge prefixes must NOT fall through to Claude
        # (that put the Anthropic sunburst + Max plan limits on Grok bots).
        # Prefer a registered alias; otherwise still return the prefix so the
        # route 404s with a clear "unknown provider" rather than lying.
        if prefix in {"xai", "grok", "zai", "openai_chatgpt", "openai", "codex"}:
            return prefix

    # Native (non-namespaced) model types — e.g. type=grok with bare model_id.
    if model_type:
        t = str(model_type).strip().lower()
        t = _PREFIX_ALIASES.get(t, t)
        if has_provider(t):
            return t
        if t in {"grok", "xai"}:
            return "xai"

    # Last-resort: alias/name heuristics so non-Anthropic proxy aliases never
    # map to Claude just because their model_id is bare.
    hint = " ".join(
        str(x) for x in (alias, model_id, model_type) if x
    ).lower()
    if "xai/" in hint or hint.startswith("grok") or " grok" in f" {hint}":
        return "xai" if has_provider("xai") else "xai"
    if (
        "openai_chatgpt/" in hint
        or hint.startswith("gpt-")
        or hint.startswith("o1")
        or hint.startswith("o3")
        or hint.startswith("o4")
        or " gpt-" in f" {hint}"
        or " codex" in f" {hint}"
        or " chatgpt" in f" {hint}"
    ):
        return "openai_chatgpt"

    return "claude"


@router.get("/v1/usage", tags=["Usage"])
async def get_usage_endpoint(
    provider: str | None = Query(None, description="Canonical provider id (e.g. 'claude')."),
    bot_id: str | None = Query(None, description="Resolve the provider from this bot slug."),
    force: bool = Query(False, description="Bypass the cache (rate-limited; use sparingly)."),
) -> ProviderUsage | AllUsage:
    service = get_service()

    # Resolve a bot_id to its provider when no explicit provider was given.
    if provider is None and bot_id:
        provider = _provider_for_bot(service, bot_id)

    if provider is not None:
        if not has_provider(provider):
            raise HTTPException(
                status_code=404,
                detail=f"Unknown provider '{provider}'. Registered: {list_providers()}",
            )
        snap = await get_usage(provider, force=force)
        if snap is None:  # registered-check above makes this unreachable
            raise HTTPException(status_code=404, detail=f"Unknown provider '{provider}'.")
        return snap

    # No filter -> all providers.
    return await get_all(force=force)
