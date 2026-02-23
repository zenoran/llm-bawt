"""Runtime settings routes."""

import json

from sqlalchemy import func
from sqlmodel import Session, select

from fastapi import APIRouter, HTTPException, Query

from ...runtime_settings import BotProfileStore, RuntimeSetting, RuntimeSettingsStore
from ..dependencies import get_service
from ..schemas import (
    BotProfileListResponse,
    BotProfileResponse,
    BotProfileUpsertRequest,
    RuntimeSettingRecord,
    RuntimeSettingsListResponse,
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


def _reload_bot_registry() -> None:
    """Reload in-memory bot registry from YAML + DB overrides."""
    from ...bots import invalidate_bots_cache, _check_reload

    invalidate_bots_cache()
    _check_reload()


def _invalidate_bot_instance_cache(service, bot_id: str) -> int:
    """Drop cached ServiceLLMBawt instances for a bot so prompt changes apply."""
    invalidate = getattr(service, "invalidate_bot_instances", None)
    if callable(invalidate):
        return int(invalidate(bot_id))

    normalized = (bot_id or "").strip().lower()
    if not normalized:
        return 0
    keys_to_remove = [key for key in service._llm_bawt_cache if key[1] == normalized]
    for key in keys_to_remove:
        del service._llm_bawt_cache[key]
    return len(keys_to_remove)


def _invalidate_all_instance_cache(service) -> int:
    """Drop all cached ServiceLLMBawt instances."""
    invalidate = getattr(service, "invalidate_all_instances", None)
    if callable(invalidate):
        return int(invalidate())
    cleared = len(service._llm_bawt_cache)
    service._llm_bawt_cache.clear()
    return cleared


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


@router.get("/v1/settings/all", response_model=RuntimeSettingsListResponse, tags=["System"])
async def list_all_runtime_settings(
    scope_type: str | None = Query(None, description="Filter by scope type: global or bot"),
    scope_id: str | None = Query(None, description="Filter by scope id ('*' for global)"),
    key: str | None = Query(None, description="Filter by exact key"),
    key_prefix: str | None = Query(None, description="Filter by key prefix"),
    limit: int = Query(200, ge=1, le=1000, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """List runtime settings across all scopes."""
    service = get_service()
    store = RuntimeSettingsStore(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Runtime settings DB unavailable")

    normalized_scope_type = scope_type.strip().lower() if scope_type else None
    if normalized_scope_type and normalized_scope_type not in {"global", "bot"}:
        raise HTTPException(status_code=400, detail="scope_type must be 'global' or 'bot'")

    normalized_scope_id = scope_id.strip().lower() if scope_id else None
    normalized_key = key.strip() if key else None
    normalized_key_prefix = key_prefix.strip() if key_prefix else None

    conditions: list = []
    if normalized_scope_type:
        conditions.append(RuntimeSetting.scope_type == normalized_scope_type)
    if normalized_scope_id:
        conditions.append(RuntimeSetting.scope_id == normalized_scope_id)
    if normalized_key:
        conditions.append(RuntimeSetting.key == normalized_key)
    if normalized_key_prefix:
        conditions.append(RuntimeSetting.key.startswith(normalized_key_prefix))

    statement = select(RuntimeSetting)
    count_statement = select(func.count()).select_from(RuntimeSetting)
    if conditions:
        statement = statement.where(*conditions)
        count_statement = count_statement.where(*conditions)

    statement = statement.order_by(RuntimeSetting.scope_type, RuntimeSetting.scope_id, RuntimeSetting.key)

    with Session(store.engine) as session:
        total_count = int(session.exec(count_statement).one() or 0)
        rows = session.exec(statement.offset(offset).limit(limit)).all()

    items = []
    for row in rows:
        try:
            value = json.loads(row.value_json)
        except Exception:
            value = row.value_json
        items.append(
            RuntimeSettingRecord(
                scope_type=row.scope_type,
                scope_id=row.scope_id,
                key=row.key,
                value=value,
                updated_at=row.updated_at,
            )
        )

    return RuntimeSettingsListResponse(
        settings=items,
        total_count=total_count,
        filters={
            "scope_type": normalized_scope_type,
            "scope_id": normalized_scope_id,
            "key": normalized_key,
            "key_prefix": normalized_key_prefix,
            "limit": limit,
            "offset": offset,
        },
    )


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


@router.get("/v1/bots/profiles", response_model=BotProfileListResponse, tags=["System"])
async def list_bot_profiles(
    q: str | None = Query(None, description="Text filter against slug, name, and description"),
    uses_tools: bool | None = Query(None, description="Filter by uses_tools"),
    uses_search: bool | None = Query(None, description="Filter by uses_search"),
    requires_memory: bool | None = Query(None, description="Filter by requires_memory"),
    voice_optimized: bool | None = Query(None, description="Filter by voice_optimized"),
    uses_home_assistant: bool | None = Query(None, description="Filter by uses_home_assistant"),
    default_model: str | None = Query(None, description="Filter by default model alias"),
    limit: int = Query(200, ge=1, le=1000, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """List DB-backed bot personality profiles with filters."""
    service = get_service()
    store = BotProfileStore(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Bot profiles DB unavailable")

    query_text = q.strip().lower() if q else None
    model_filter = default_model.strip() if default_model else None
    rows = store.list_all()

    filtered = []
    for row in rows:
        if query_text:
            haystack = " ".join(
                [
                    (row.slug or ""),
                    (row.name or ""),
                    (row.description or ""),
                ]
            ).lower()
            if query_text not in haystack:
                continue
        if uses_tools is not None and row.uses_tools != uses_tools:
            continue
        if uses_search is not None and row.uses_search != uses_search:
            continue
        if requires_memory is not None and row.requires_memory != requires_memory:
            continue
        if voice_optimized is not None and row.voice_optimized != voice_optimized:
            continue
        if uses_home_assistant is not None and row.uses_home_assistant != uses_home_assistant:
            continue
        if model_filter is not None and row.default_model != model_filter:
            continue
        filtered.append(row)

    total_count = len(filtered)
    page = filtered[offset : offset + limit]

    return BotProfileListResponse(
        profiles=[_to_profile_response(profile) for profile in page],
        total_count=total_count,
        filters={
            "q": query_text,
            "uses_tools": uses_tools,
            "uses_search": uses_search,
            "requires_memory": requires_memory,
            "voice_optimized": voice_optimized,
            "uses_home_assistant": uses_home_assistant,
            "default_model": model_filter,
            "limit": limit,
            "offset": offset,
        },
    )


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
    _reload_bot_registry()
    _invalidate_bot_instance_cache(service, profile.slug)
    return _to_profile_response(profile)


@router.post("/v1/admin/reload-bots", tags=["Admin"])
async def reload_bots():
    """Force reload bot registry from DB + YAML."""
    service = get_service()
    _reload_bot_registry()
    cleared_instances = _invalidate_all_instance_cache(service)
    return {"status": "reloaded", "cleared_instances": cleared_instances}


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

    normalized_slug = slug.strip().lower()
    _reload_bot_registry()
    _invalidate_bot_instance_cache(service, normalized_slug)

    return {"success": True, "slug": normalized_slug}
