"""Raw LLM utility completion route."""

from fastapi import APIRouter, HTTPException

from ..dependencies import get_service
from ..logging import get_service_logger
from ..schemas import RawCompletionRequest, RawCompletionResponse

router = APIRouter()
log = get_service_logger(__name__)

@router.post("/v1/llm/complete", response_model=RawCompletionResponse, tags=["LLM"])
async def raw_completion(request: RawCompletionRequest):
    """Raw LLM completion using the currently loaded model.
    
    Use this for utility tasks like:
    - Memory consolidation (merging similar memories)
    - Summarization
    - Classification
    - Any task that needs LLM but not the full chat pipeline
    
    This endpoint uses the already-loaded model in the service.
    It will NOT load a new model - if no model is loaded, it returns 503.
    Only one LLM model can be loaded at a time (embedding model is separate).
    """
    import time
    service = get_service()
    
    # Check if we have any loaded client
    if not service._client_cache:
        raise HTTPException(
            status_code=503, 
            detail="No model loaded. Make a chat request first to load a model."
        )
    
    # Use the currently loaded model (there should only be one)
    loaded_models = list(service._client_cache.keys())
    model_alias = loaded_models[0]  # Use whatever is loaded
    
    # If caller specified a model, warn if it doesn't match
    if request.model and request.model != model_alias:
        log.debug(f"Requested model '{request.model}' but using loaded model '{model_alias}'")
    
    try:
        start = time.perf_counter()
        client = service._client_cache[model_alias]
        
        # Build messages as Message objects (required by client.query)
        from ..models.message import Message
        messages = []
        if request.system:
            messages.append(Message(role="system", content=request.system))
        messages.append(Message(role="user", content=request.prompt))
        
        # Query the model directly
        response = client.query(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Estimate tokens
        tokens = len(response) // 4 if response else 0
        
        return RawCompletionResponse(
            content=response,
            model=model_alias,
            tokens=tokens,
            elapsed_ms=elapsed_ms,
        )
        
    except Exception as e:
        log.error(f"Raw completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================================
# History Summarization Endpoints
# =========================================================================
