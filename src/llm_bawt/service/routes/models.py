"""Model and bot listing routes."""

from fastapi import APIRouter, HTTPException

from ...bots import BotManager
from ..dependencies import get_service
from ..schemas import (
    BotInfo,
    BotsResponse,
    ModelDefinitionDeleteResponse,
    ModelDefinitionListResponse,
    ModelDefinitionResponse,
    ModelDefinitionSeedRequest,
    ModelDefinitionSeedResponse,
    ModelDefinitionUpsertRequest,
    ModelDetail,
    ModelInfo,
    ModelsResponse,
    ModelSwitchRequest,
    ModelSwitchResponse,
)

router = APIRouter()

@router.get("/v1/models", response_model=ModelsResponse, tags=["OpenAI Compatible"])
async def list_models():
    """List available models (OpenAI-compatible)."""
    service = get_service()
    defined = service.config.defined_models.get("models", {})
    models = []
    for alias in service._available_models:
        info = defined.get(alias, {})
        models.append(ModelInfo(
            id=alias,
            type=info.get("type"),
            model_id=info.get("model_id"),
            description=info.get("description"),
        ))
    return ModelsResponse(data=models)

@router.get("/v1/models/current", tags=["Models"])
async def get_current_model():
    """Get the currently active model."""
    service = get_service()
    current = service.model_lifecycle.current_model
    if not current:
        return {"model": None, "message": "No model currently loaded"}
    info = service.model_lifecycle.get_model_info(current)
    detail = ModelDetail(
        id=current,
        type=info.get("type") if info else None,
        model_id=info.get("model_id", info.get("repo_id")) if info else None,
        description=info.get("description") if info else None,
        current=True,
    )
    return {"model": detail}

@router.post("/v1/models/switch", response_model=ModelSwitchResponse, tags=["Models"])
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model. Takes effect on the next request."""
    service = get_service()
    previous = service.model_lifecycle.current_model
    success, message = service.model_lifecycle.switch_model(request.model)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return ModelSwitchResponse(
        success=True,
        message=message,
        previous_model=previous,
        new_model=request.model,
    )


@router.post("/v1/models/reload", tags=["Models"])
async def reload_models_catalog():
    """Reload model catalog from DB/YAML and refresh service model availability."""
    service = get_service()
    config = service.config

    # Reset from YAML first so aliases removed from DB don't linger in-memory.
    config._load_models_config()

    from ..dependencies import get_model_definition_store

    db_count = 0
    store = get_model_definition_store(config)
    if store.engine is not None:
        db_models = store.to_config_dict()
        db_count = len(db_models)
        if db_models:
            config.merge_db_models(db_models)

    service._load_available_models()
    cleared = service.invalidate_all_instances()

    return {
        "ok": True,
        "models": list(service._available_models),
        "default_model": service._default_model,
        "db_models_loaded": db_count,
        "cleared_instances": cleared,
    }

# =============================================================================
# Model Definition CRUD (DB-backed)
# =============================================================================

def _get_model_store():
    """Get a ModelDefinitionStore instance."""
    from ..dependencies import get_model_definition_store
    service = get_service()
    store = get_model_definition_store(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Model definitions database unavailable")
    return store


def _row_to_response(row) -> ModelDefinitionResponse:
    return ModelDefinitionResponse(
        alias=row.alias,
        type=row.type,
        model_id=row.model_id,
        repo_id=row.repo_id,
        filename=row.filename,
        description=row.description,
        extra=row.extra,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("/v1/models/definitions", response_model=ModelDefinitionListResponse, tags=["Models"])
async def list_model_definitions():
    """List all DB-backed model definitions."""
    store = _get_model_store()
    rows = store.list_all()
    return ModelDefinitionListResponse(
        models=[_row_to_response(r) for r in rows],
        total_count=len(rows),
    )


@router.get("/v1/models/definitions/{alias}", response_model=ModelDefinitionResponse, tags=["Models"])
async def get_model_definition(alias: str):
    """Get a single model definition by alias."""
    store = _get_model_store()
    row = store.get(alias)
    if not row:
        raise HTTPException(status_code=404, detail=f"Model alias '{alias}' not found")
    return _row_to_response(row)


@router.put("/v1/models/definitions/{alias}", response_model=ModelDefinitionResponse, tags=["Models"])
async def upsert_model_definition(alias: str, request: ModelDefinitionUpsertRequest):
    """Create or update a model definition. Triggers model catalog reload."""
    store = _get_model_store()
    model_data: dict = {"type": request.type}
    if request.model_id is not None:
        model_data["model_id"] = request.model_id
    if request.repo_id is not None:
        model_data["repo_id"] = request.repo_id
    if request.filename is not None:
        model_data["filename"] = request.filename
    if request.description is not None:
        model_data["description"] = request.description
    if request.extra:
        model_data.update(request.extra)

    row = store.upsert(alias, model_data)
    # Reload catalog so the new model is immediately available
    await reload_models_catalog()
    return _row_to_response(row)


@router.delete("/v1/models/definitions/{alias}", response_model=ModelDefinitionDeleteResponse, tags=["Models"])
async def delete_model_definition(alias: str):
    """Delete a model definition by alias. Triggers model catalog reload."""
    store = _get_model_store()
    deleted = store.delete(alias)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Model alias '{alias}' not found")
    await reload_models_catalog()
    return ModelDefinitionDeleteResponse(
        success=True,
        alias=alias,
        message=f"Model '{alias}' deleted",
    )


@router.post("/v1/models/definitions/seed", response_model=ModelDefinitionSeedResponse, tags=["Models"])
async def seed_model_definitions(request: ModelDefinitionSeedRequest | None = None):
    """Seed DB model definitions from the current YAML config."""
    service = get_service()
    yaml_models = service.config.defined_models.get("models", {})
    if not yaml_models:
        return ModelDefinitionSeedResponse(seeded=0, total_yaml=0, message="No models in YAML config")

    store = _get_model_store()
    if request and request.overwrite:
        seeded = 0
        for alias, model_data in yaml_models.items():
            if isinstance(model_data, dict):
                store.upsert(alias, model_data)
                seeded += 1
    else:
        seeded = store.seed_from_yaml(yaml_models)

    await reload_models_catalog()
    return ModelDefinitionSeedResponse(
        seeded=seeded,
        total_yaml=len(yaml_models),
        message=f"Seeded {seeded} model(s) from YAML",
    )


@router.get("/v1/bots", response_model=BotsResponse, tags=["System"])
async def list_bots():
    """List available bots configured on the service."""
    service = get_service()
    bot_manager = BotManager(service.config)
    bots = [
        BotInfo(
            slug=bot.slug,
            name=bot.name,
            description=bot.description,
            system_prompt=bot.system_prompt,
            requires_memory=bot.requires_memory,
            voice_optimized=bot.voice_optimized,
            tts_mode=bot.tts_mode,
            include_summaries=bot.include_summaries,
            default_voice=bot.default_voice,
            uses_tools=bot.uses_tools,
            uses_search=bot.uses_search,
            uses_home_assistant=bot.uses_home_assistant,
            default_model=bot.default_model,
            color=bot.color,
            avatar=bot.avatar,
            bot_type=bot.bot_type,
            agent_backend=bot.agent_backend,
            agent_backend_config=bot.agent_backend_config or {},
            settings=bot.settings,
        )
        for bot in bot_manager.list_bots()
    ]
    return BotsResponse(data=bots)
