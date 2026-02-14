"""Runtime settings routes."""

from fastapi import APIRouter, HTTPException, Query

from ...runtime_settings import BotProfileStore, RuntimeSettingsStore
from ..dependencies import get_service
from ..schemas import (
    BotProfileResponse,
    BotProfileUpsertRequest,
    RuntimeSettingItem,
    RuntimeSettingBatchUpsertRequest,
    RuntimeSettingsResponse,
    RuntimeSettingUpsertRequest,
)

router = APIRouter()


def _normalize_scope(scope_type: str, scope_id: str | None, default_bot: str) -> tuple[str, str]:
    st = (scope_type or "").strip().lower()
    if st not in {"global", "bot"}:
        raise HTTPException(status_code=400, detail="scope_type must be 'global' or 'bot'")
    if st == "global":
        return "global", "*"
    sid = (scope_id or default_bot or "").strip().lower()
    if not sid:
        raise HTTPException(status_code=400, detail="scope_id is required for bot scope")
    return "bot", sid


def _to_profile_response(profile) -> BotProfileResponse:
    return BotProfileResponse(
        slug=profile.slug,
        name=profile.name,
        description=profile.description,
        system_prompt=profile.system_prompt,
        requires_memory=profile.requires_memory,
        voice_optimized=profile.voice_optimized,
        uses_tools=profile.uses_tools,
        uses_search=profile.uses_search,
        uses_home_assistant=profile.uses_home_assistant,
        default_model=profile.default_model,
        nextcloud_config=profile.nextcloud_config,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


@router.get("/v1/settings", response_model=RuntimeSettingsResponse, tags=["System"])
async def list_runtime_settings(
    scope_type: str = Query("bot", description="global or bot"),
    scope_id: str | None = Query(None, description="bot slug for bot scope"),
):
    """List runtime settings for a scope."""
    service = get_service()
    st, sid = _normalize_scope(scope_type, scope_id, service._default_bot)
    store = RuntimeSettingsStore(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Runtime settings DB unavailable")
    data = store.get_scope_settings(st, sid)
    items = [RuntimeSettingItem(key=k, value=v) for k, v in sorted(data.items())]
    return RuntimeSettingsResponse(scope_type=st, scope_id=sid, settings=items)


@router.put("/v1/settings", tags=["System"])
async def upsert_runtime_setting(request: RuntimeSettingUpsertRequest):
    """Upsert one runtime setting for global or bot scope."""
    service = get_service()
    st, sid = _normalize_scope(request.scope_type, request.scope_id, service._default_bot)
    store = RuntimeSettingsStore(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Runtime settings DB unavailable")
    store.set_value(st, sid, request.key, request.value)
    return {
        "success": True,
        "scope_type": st,
        "scope_id": sid,
        "key": request.key,
        "value": request.value,
    }


@router.delete("/v1/settings", tags=["System"])
async def delete_runtime_setting(
    scope_type: str = Query(..., description="global or bot"),
    key: str = Query(..., description="Setting key"),
    scope_id: str | None = Query(None, description="bot slug for bot scope"),
):
    """Delete one runtime setting."""
    service = get_service()
    st, sid = _normalize_scope(scope_type, scope_id, service._default_bot)
    store = RuntimeSettingsStore(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Runtime settings DB unavailable")
    deleted = store.delete_value(st, sid, key)
    return {
        "success": True,
        "scope_type": st,
        "scope_id": sid,
        "key": key,
        "deleted": bool(deleted),
    }


@router.post("/v1/settings/batch", tags=["System"])
async def batch_upsert_runtime_settings(request: RuntimeSettingBatchUpsertRequest):
    """Batch upsert runtime settings."""
    service = get_service()
    store = RuntimeSettingsStore(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Runtime settings DB unavailable")

    applied: list[dict] = []
    for item in request.items:
        st, sid = _normalize_scope(item.scope_type, item.scope_id, service._default_bot)
        store.set_value(st, sid, item.key, item.value)
        applied.append(
            {
                "scope_type": st,
                "scope_id": sid,
                "key": item.key,
            }
        )
    return {"success": True, "applied_count": len(applied), "applied": applied}


@router.get("/v1/bots/{slug}/profile", response_model=BotProfileResponse, tags=["System"])
async def get_bot_profile(slug: str):
    """Get bot personality profile by slug."""
    service = get_service()
    store = BotProfileStore(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Bot profiles DB unavailable")

    profile = store.get(slug)
    if profile is None:
        raise HTTPException(status_code=404, detail="Bot profile not found")
    return _to_profile_response(profile)


@router.put("/v1/bots/{slug}/profile", response_model=BotProfileResponse, tags=["System"])
async def upsert_bot_profile(slug: str, request: BotProfileUpsertRequest):
    """Create or update a bot personality profile."""
    service = get_service()
    store = BotProfileStore(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Bot profiles DB unavailable")

    profile = store.upsert(
        {
            "slug": slug,
            "name": request.name,
            "description": request.description,
            "system_prompt": request.system_prompt,
            "requires_memory": request.requires_memory,
            "voice_optimized": request.voice_optimized,
            "uses_tools": request.uses_tools,
            "uses_search": request.uses_search,
            "uses_home_assistant": request.uses_home_assistant,
            "default_model": request.default_model,
            "nextcloud_config": request.nextcloud_config,
        }
    )
    return _to_profile_response(profile)


@router.post("/v1/admin/reload-bots", tags=["Admin"])
async def reload_bots():
    """Force reload bot registry from DB + YAML."""
    from ...bots import invalidate_bots_cache, _check_reload

    invalidate_bots_cache()
    _check_reload()
    return {"status": "reloaded"}


@router.delete("/v1/bots/{slug}/profile", tags=["System"])
async def delete_bot_profile(slug: str):
    """Delete a bot personality profile."""
    service = get_service()
    store = BotProfileStore(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Bot profiles DB unavailable")

    deleted = store.delete(slug)
    if not deleted:
        raise HTTPException(status_code=404, detail="Bot profile not found")

    return {"success": True, "slug": slug.strip().lower()}
