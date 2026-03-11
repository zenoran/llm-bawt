"""OpenAI-compatible chat completion route."""

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..dependencies import get_service
from ..logging import get_service_logger
from ..schemas import ChatCompletionRequest

router = APIRouter()
log = get_service_logger(__name__)


class ChatAbortRequest(BaseModel):
    """Request to abort an in-flight turn."""
    turn_id: str = Field(..., description="Turn log ID to abort")


class ChatAbortResponse(BaseModel):
    ok: bool
    detail: str | None = None
    turn_id: str | None = None


@router.post("/v1/chat/abort", tags=["OpenAI Compatible"])
async def chat_abort(request: ChatAbortRequest) -> ChatAbortResponse:
    """Abort a turn by ID.

    Looks up the turn's agent_session_key to send chat.abort to the
    OpenClaw gateway, then marks the turn as aborted.  If the turn has
    no agent_session_key (native model), just marks it aborted.
    """
    service = get_service()
    store = service._turn_log_store
    turn = store.get_turn(request.turn_id)
    if not turn:
        raise HTTPException(status_code=404, detail="Turn not found")

    # If it's already terminal, nothing to do
    if turn.status not in ("streaming", "pending"):
        return ChatAbortResponse(
            ok=True,
            detail="already_completed",
            turn_id=turn.id,
        )

    # Send chat.abort to the gateway if this is an OpenClaw turn
    gateway_aborted = False
    if turn.agent_session_key:
        from ...agent_backends.openclaw import get_openclaw_subscriber

        subscriber = get_openclaw_subscriber()
        if subscriber:
            params: dict = {"sessionKey": turn.agent_session_key}
            abort_req_id = f"abort_{uuid.uuid4().hex}"
            try:
                await subscriber.send_rpc("chat.abort", params, abort_req_id, timeout_s=10)
                gateway_aborted = True
            except Exception as e:
                log.warning("chat.abort RPC failed for turn %s: %s", turn.id, e)

    # Mark the turn as aborted regardless
    store.update_turn(
        turn_id=turn.id,
        status="aborted",
        error_text="Aborted via chat.abort",
    )
    log.info("Marked turn %s as aborted (gateway_aborted=%s)", turn.id, gateway_aborted)

    return ChatAbortResponse(
        ok=True,
        detail="aborted",
        turn_id=turn.id,
    )

@router.post("/v1/chat/completions", tags=["OpenAI Compatible"])
async def chat_completions(request: ChatCompletionRequest):
    """
    Create a chat completion (OpenAI-compatible).
    
    Supports all standard OpenAI parameters plus llm_bawt extensions:
    - `bot_id`: Bot personality to use (default: nova)
    - `augment_memory`: Whether to include memory context (default: true)
    - `extract_memory`: Whether to extract memories from response (default: true)
    - `include_summaries`: Whether to inject conversation summary records into context (default: true)
    - `tts_mode`: Whether to append TTS output formatting instructions to the system prompt (default: false)
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
