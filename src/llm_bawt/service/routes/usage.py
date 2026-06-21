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

from ..dependencies import get_service
from ..usage import AllUsage, ProviderUsage, get_all, get_usage, has_provider, list_providers

log = logging.getLogger(__name__)

router = APIRouter()


def _provider_for_bot(service, bot_id: str) -> str:
    """Best-effort map a bot slug -> canonical usage provider.

    A claude-code bot's model_id carries the provider as a ``<provider>/<model>``
    prefix (e.g. ``zai/glm-5.2``, ``openai_chatgpt/gpt-5.4``); native Claude
    bots have no prefix. Falls back to ``claude``.
    """
    try:
        from ...bots import BotManager

        bot = BotManager(service.config).get_bot(bot_id)
    except Exception:  # noqa: BLE001
        bot = None
    if bot is None:
        return "claude"

    alias = getattr(bot, "default_model", None)
    model_id = None
    if alias:
        try:
            from ...runtime_settings import ModelDefinitionStore

            for row in ModelDefinitionStore(service.config).list_all():
                if getattr(row, "alias", None) == alias:
                    model_id = getattr(row, "model_id", None)
                    break
        except Exception:  # noqa: BLE001
            model_id = None

    if model_id and "/" in model_id:
        prefix = model_id.split("/", 1)[0]
        if has_provider(prefix):
            return prefix
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
