"""Model and bot listing routes."""

from fastapi import APIRouter, HTTPException

from ...bots import BotManager
from ..dependencies import get_service
from ..schemas import (
    BotInfo,
    BotsResponse,
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
    models = [
        ModelInfo(id=alias)
        for alias in service._available_models
    ]
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

    from ...runtime_settings import ModelDefinitionStore

    db_count = 0
    store = ModelDefinitionStore(config)
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
            default_voice=bot.default_voice,
            uses_tools=bot.uses_tools,
            uses_search=bot.uses_search,
            uses_home_assistant=bot.uses_home_assistant,
            default_model=bot.default_model,
            color=bot.color,
            settings=bot.settings,
        )
        for bot in bot_manager.list_bots()
    ]
    return BotsResponse(data=bots)
