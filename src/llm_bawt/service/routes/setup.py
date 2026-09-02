"""First-run tenant setup surface (TASK-355).

The installer and the BawtHub wizard share one idempotent seed engine; these
routes are thin HTTP adapters around it. Seed policy and data stay in
:mod:`llm_bawt.seeding`.

Flow the wizard drives:

1. ``GET  /v1/setup/providers`` — which connections can back the first bot,
   and whether each is connected (health from the provider adapters).
2. ``GET  /v1/models/upstream?provider=…`` — live model list for a connected
   provider (existing route; discovery reads the stored key).
3. ``POST /v1/setup/first-bot`` — provision infrastructure, register the
   chosen model + endpoint, create the bot. Refuses when the provider isn't
   connected: a bot without a credential is broken on arrival.

``POST /v1/setup/seed`` remains the infrastructure-only apply for the
installer / operator fallback (``tenant.sh seed``).
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from ...seeding import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_FIRST_BOT_PROMPT,
    FIRST_BOT_PROVIDERS,
    TenantSeeder,
    first_bot_provider,
    seed_tenant,
)
from ..dependencies import get_runtime_settings_store, get_service
from ..providers.base import HEALTH_OK, HEALTH_WARNING
from ..providers.registry import get_adapter
from .models import reload_models_catalog
from .settings import _persist_bot_profile

log = logging.getLogger(__name__)
router = APIRouter()


_DOMAIN_RE = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
_SLUG_RE = r"^[a-z0-9][a-z0-9_-]{1,63}$"

#: Health states under which a provider credential is usable right now.
_USABLE_STATES = frozenset({HEALTH_OK, HEALTH_WARNING})


class TenantPreferences(BaseModel):
    exposure: str | None = Field(default=None, pattern=r"^(private|public)$")
    domains: list[str] | None = Field(default=None, max_length=10)
    voice_enabled: bool | None = None

    @field_validator("domains")
    @classmethod
    def validate_domains(cls, domains: list[str] | None) -> list[str] | None:
        if domains is None:
            return None
        normalized = [domain.strip().lower() for domain in domains if domain.strip()]
        invalid = [domain for domain in normalized if not _DOMAIN_RE.fullmatch(domain)]
        if invalid:
            raise ValueError("Invalid domains: " + ", ".join(invalid))
        return list(dict.fromkeys(normalized))


class TenantSeedRequest(TenantPreferences):
    """Infrastructure-only seed (installer / operator fallback)."""


class FirstBotRequest(TenantPreferences):
    """Create the tenant's first bot against a connected provider."""

    provider_id: str = Field(min_length=1, max_length=64)
    model_id: str = Field(min_length=1, max_length=255)
    display_name: str | None = Field(default=None, max_length=255)
    context_window: int = Field(default=DEFAULT_CONTEXT_WINDOW, gt=0)
    slug: str = Field(pattern=_SLUG_RE)
    name: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=2000)
    system_prompt: str = Field(min_length=1)


def _store_setup_preferences(service, body: TenantPreferences | None) -> dict[str, object]:
    if body is None:
        return {}
    values = {
        "tenant.exposure": body.exposure,
        "tenant.domains": body.domains,
        "tenant.voice_enabled": body.voice_enabled,
    }
    selected = {key: value for key, value in values.items() if value is not None}
    if not selected:
        return {}
    store = get_runtime_settings_store(service.config)
    if store.engine is None:
        raise RuntimeError("Runtime settings DB unavailable")
    for key, value in selected.items():
        store.set_value("global", "*", key, value)
    return selected


def _provider_rows(service) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for route in FIRST_BOT_PROVIDERS:
        adapter = get_adapter(service.config, route.provider_id)
        if adapter is None:  # registry drift — surface, don't hide
            log.warning("First-bot provider %s has no adapter", route.provider_id)
            continue
        try:
            health = adapter.health()
        except Exception as exc:  # noqa: BLE001 — one bad adapter must not hide the rest
            log.warning("health probe failed for %s: %s", adapter.id, exc)
            health = {"state": "unknown", "detail": str(exc)[:200]}
        state = str(health.get("state") or "unknown")
        rows.append(
            {
                "id": route.provider_id,
                "label": adapter.label,
                "vendor": route.vendor,
                "access_path": route.access_path_key,
                "discovery": route.discovery,
                "state": state,
                "detail": health.get("detail"),
                "connected": state in _USABLE_STATES,
            }
        )
    return rows


@router.get("/v1/setup/providers", tags=["Setup"])
async def list_first_bot_providers():
    """Providers that can back the first bot, with live connection state."""
    service = get_service()
    providers = await run_in_threadpool(_provider_rows, service)
    return {
        "providers": providers,
        "default_system_prompt": DEFAULT_FIRST_BOT_PROMPT,
        "default_context_window": DEFAULT_CONTEXT_WINDOW,
    }


def _http_error(exc: Exception, *, what: str) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, RuntimeError):
        log.warning("%s unavailable: %s", what, exc)
        return HTTPException(status_code=503, detail=str(exc))
    log.exception("%s failed", what)
    return HTTPException(status_code=500, detail=f"{what} failed")


@router.post("/v1/setup/seed", tags=["Setup"])
async def apply_tenant_seed(body: TenantSeedRequest | None = None):
    """Apply the idempotent infrastructure seed and return the report."""
    service = get_service()
    try:
        report = await run_in_threadpool(seed_tenant, service.config)
        preferences = await run_in_threadpool(_store_setup_preferences, service, body)
    except Exception as exc:  # noqa: BLE001 — mapped to HTTP below
        raise _http_error(exc, what="Tenant seed") from exc
    return {"ok": True, "report": report, "preferences": preferences}


@router.post("/v1/setup/first-bot", tags=["Setup"])
async def create_first_bot(body: FirstBotRequest):
    """Provision infrastructure, register the chosen model, create the bot.

    Order: provider must be connected → infra seed → model + endpoint
    (insert-if-missing) → catalog reload so the new endpoint resolves →
    bot create (409 if the slug exists) → preferences.
    """
    service = get_service()
    route = first_bot_provider(body.provider_id)
    if route is None:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{body.provider_id}' cannot back the first bot",
        )
    adapter = get_adapter(service.config, route.provider_id)
    if adapter is None:
        raise HTTPException(status_code=500, detail=f"Provider '{route.provider_id}' has no adapter")

    try:
        health = await run_in_threadpool(adapter.health)
        if str(health.get("state")) not in _USABLE_STATES:
            raise HTTPException(
                status_code=409,
                detail=f"Connect {adapter.label} before creating the first bot",
            )

        seeder = TenantSeeder(service.config)
        infra = await run_in_threadpool(seeder.apply)
        registered = await run_in_threadpool(
            seeder.register_model,
            vendor=route.vendor,
            model_id=body.model_id,
            access_path_key=route.access_path_key,
            display_name=body.display_name,
            context_window=body.context_window,
        )
        await run_in_threadpool(reload_models_catalog)

        bot = await _persist_bot_profile(
            {
                "slug": body.slug,
                "name": body.name,
                "description": body.description,
                "system_prompt": body.system_prompt,
                "requires_memory": True,
                "voice_optimized": False,
                "tts_mode": False,
                "include_summaries": True,
                "include_in_global_search": True,
                "uses_tools": True,
                "uses_search": True,
                "uses_home_assistant": False,
                "harness": "chat",
                "bot_type": "chat",
                "endpoint_id": registered["endpoint_id"],
            },
            create_only=True,
        )
        preferences = await run_in_threadpool(_store_setup_preferences, service, body)
    except Exception as exc:  # noqa: BLE001 — mapped to HTTP below
        raise _http_error(exc, what="First bot setup") from exc

    return {
        "ok": True,
        "report": {**infra, "models": registered["models"], "endpoints": registered["endpoints"]},
        "bot": bot,
        "preferences": preferences,
    }
