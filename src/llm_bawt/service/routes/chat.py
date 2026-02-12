"""OpenAI-compatible chat completion route."""

from fastapi import APIRouter, HTTPException

from ..dependencies import get_service
from ..logging import get_service_logger
from ..schemas import ChatCompletionRequest

router = APIRouter()
log = get_service_logger(__name__)

@router.post("/v1/chat/completions", tags=["OpenAI Compatible"])
async def chat_completions(request: ChatCompletionRequest):
    """
    Create a chat completion (OpenAI-compatible).
    
    Supports all standard OpenAI parameters plus llm_bawt extensions:
    - `bot_id`: Bot personality to use (default: nova)
    - `augment_memory`: Whether to include memory context (default: true)
    - `extract_memory`: Whether to extract memories from response (default: true)
    """
    from fastapi.responses import StreamingResponse
    
    service = get_service()
    
    # Log request BEFORE validation so we can debug failures
    log.debug(f"Request payload: {request.model_dump(exclude_none=True)}")
    
    if request.stream:
        # Streaming response
        try:
            return StreamingResponse(
                service.chat_completion_stream(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("Streaming chat completion failed")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Non-streaming
    try:
        response = await service.chat_completion(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Chat completion failed")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------------
# Task Management Endpoints
# -------------------------------------------------------------------------
