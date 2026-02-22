"""Bot-specific OpenAI-compatible endpoints.

Provides /v1/botchat/{bot_id}/{user_id}/chat/completions and /models
so clients that can't pass extra params can use a base URL like:
    http://host:8642/v1/botchat/nova/nick
and point their OpenAI-compatible client at it.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..dependencies import get_service
from ..logging import get_service_logger
from ..schemas import ChatCompletionRequest, ModelInfo, ModelsResponse

router = APIRouter()
log = get_service_logger(__name__)


_HA_CONFLICTING_LINES = frozenset(
    {
        # HA's default assistant prompt tells the LLM to describe actions instead
        # of executing them, which fights the tool-calling system.
        "do not execute service without user's confirmation.",
        "use execute_services function only for requested action, not for current states.",
        # Redundant state restatement instruction — Nova handles this via tools.
        "do not restate or appreciate what user says, rather make a quick inquiry.",
    }
)


def _sanitize_client_context(text: str) -> str:
    """Strip HA boilerplate from client context.

    The HA prompt instructions conflict with our tool-calling system,
    so we remove those lines. The device CSV list is kept intact so
    the LLM can identify devices by entity ID and friendly name.
    """
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.strip().lower()

        # Skip known conflicting HA instructions
        if stripped in _HA_CONFLICTING_LINES:
            continue

        # Skip HA prompt boilerplate that conflicts with our tool system
        if any(phrase in stripped for phrase in (
            "i want you to act as smart home manager",
            "i will provide information of smart home",
            "available devices:",
            "the current state of devices",
            "use execute_services function",
        )):
            continue

        cleaned.append(line)

    result = "\n".join(cleaned).strip()
    # Remove excessive blank lines
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    return result


def _extract_client_system_context(request: ChatCompletionRequest) -> None:
    """Extract system messages from the request and set them as client context.

    Pulls out all role='system' messages, concatenates their content into
    ``request.client_system_context``, and removes them from the message list
    so only user/assistant messages remain.

    HA-specific boilerplate that conflicts with tool-calling is stripped.
    """
    system_parts: list[str] = []
    non_system: list = []

    for msg in request.messages:
        if msg.role == "system" and msg.content:
            system_parts.append(msg.content)
        else:
            non_system.append(msg)

    if system_parts:
        combined = "\n\n".join(system_parts)
        request.client_system_context = _sanitize_client_context(combined)
        request.ha_mode = True
        request.messages = non_system


@router.post(
    "/v1/botchat/{bot_id}/{user_id}/chat/completions",
    tags=["Bot-Specific OpenAI Compatible"],
)
async def botchat_completions(bot_id: str, user_id: str, request: ChatCompletionRequest):
    """OpenAI-compatible chat completion with bot_id and user_id baked into the URL.

    Clients set their base URL to /v1/botchat/{bot_id}/{user_id} and call
    /chat/completions as normal — no extra body params needed.

    Any system messages in the request are extracted and injected as a
    "Client Context" section in the prompt builder, alongside the bot's
    own system prompt, user profile, memory, etc.
    """
    # Override from path — ignore whatever the client sent
    request.bot_id = bot_id
    request.user = user_id

    # Pull system messages out of the message list and into client_system_context
    _extract_client_system_context(request)

    service = get_service()
    log.debug(
        f"botchat request: bot={bot_id} user={user_id} model={request.model}"
        f" client_context={'yes' if request.client_system_context else 'no'}"
    )

    if request.stream:
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
            log.exception("Streaming botchat completion failed")
            raise HTTPException(status_code=500, detail=str(e))

    try:
        response = await service.chat_completion(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("botchat completion failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/v1/botchat/{bot_id}/{user_id}/models",
    response_model=ModelsResponse,
    tags=["Bot-Specific OpenAI Compatible"],
)
async def botchat_models(bot_id: str, user_id: str):
    """List available models (OpenAI-compatible) — scoped under the botchat path."""
    service = get_service()
    models = [ModelInfo(id=alias) for alias in service._available_models]
    return ModelsResponse(data=models)
