"""OpenAI-compatible chat completion route."""

import json
import time
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
        from ...agent_backends.agent_bridge import get_agent_subscriber
        from ...bots import BotManager

        # Resolve the bot's agent_backend so the abort RPC can be routed to
        # the right bridge (claude-code, codex, openclaw). Bridges that
        # filter on `backend` will skip RPCs that aren't theirs, preventing
        # cross-bridge RPC races.
        backend_name = None
        if turn.bot_id:
            try:
                bot = BotManager(service.config).get_bot(turn.bot_id)
                backend_name = getattr(bot, "agent_backend", None) if bot else None
            except Exception:
                backend_name = None

        subscriber = get_agent_subscriber()
        if subscriber:
            params: dict = {"sessionKey": turn.agent_session_key}
            abort_req_id = f"abort_{uuid.uuid4().hex}"
            try:
                await subscriber.send_rpc(
                    "chat.abort", params, abort_req_id,
                    timeout_s=10, backend=backend_name,
                )
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

class ToolResultRequest(BaseModel):
    """Answer to a paused AskUserQuestion tool call.

    Routed through Redis (chat.tool_result action) to the claude-code bridge,
    which resolves the pending asyncio.Future keyed by tool_use_id.  The SDK
    turn resumes immediately on its own — this endpoint does not stream the
    continuation; the originating SSE connection (or its resume path) does.
    """
    bot_id: str = Field(..., description="Bot slug whose paused turn this answers")
    user_id: str = Field("nick", description="User id (scopes the routing key)")
    tool_use_id: str = Field(..., description="SDK tool_use id from the AWAIT_TOOL_RESULT event")
    result: str = Field(..., description="User's answer (label, free text, or JSON-encoded multi-pick)")


class ToolResultResponse(BaseModel):
    ok: bool
    detail: str | None = None


@router.post("/v1/chat/tool-result", tags=["Agent Backends"])
async def chat_tool_result(request: ToolResultRequest) -> ToolResultResponse:
    """Deliver a user's AskUserQuestion answer back to the paused SDK turn.

    Sanity checks the pending-question registry first:
    - 404 if the tool_use_id was never recorded (typo / stale UI / race).
    - 410 Gone if the row is in a terminal state (already answered, skipped,
      or marked abandoned because the originating turn ended).
    - 410 Gone if the originating turn's status is no longer streaming
      (covers the bridge-restart case — the SDK Future is gone and the
      command would be a no-op on the bridge side).

    On success: dispatches the Redis command AND flips the registry row to
    ``answered`` / ``skipped`` so other tabs converge to the post-answer UI
    even before the bridge's TOOL_END flows back.
    """
    service = get_service()
    from ...bots import BotManager
    from ...agent_backends.agent_bridge import get_agent_subscriber

    bot = BotManager(service.config).get_bot(request.bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail=f"Bot '{request.bot_id}' not found")

    backend_name = getattr(bot, "agent_backend", None)
    if backend_name != "claude-code":
        # Only claude-code defers AskUserQuestion today.  Other backends would
        # need their own can_use_tool equivalent (or MCP override) before this
        # endpoint could mean anything for them.
        raise HTTPException(
            status_code=400,
            detail=f"Bot '{request.bot_id}' backend '{backend_name}' does not support tool-result",
        )

    # Look up the row before touching Redis so we can fail fast on stale or
    # abandoned questions.
    pq_store = getattr(service, "_pending_question_store", None)
    row = pq_store.get(request.tool_use_id) if pq_store else None
    if pq_store and row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No pending question recorded for tool_use_id={request.tool_use_id}",
        )
    if row is not None and row.status != "awaiting":
        raise HTTPException(
            status_code=410,
            detail=f"Question is no longer awaiting (status={row.status})",
        )
    # Cross-check the originating turn — if it's already terminal, the bridge
    # Future is gone and the SDK won't see this answer.  Mark abandoned and
    # bail out so the UI doesn't pretend the answer landed.
    if row is not None:
        try:
            turn = service._turn_log_store.get_turn(row.turn_id)
        except Exception:
            turn = None
        if turn is not None and turn.status not in ("streaming", "pending"):
            if pq_store:
                pq_store.mark(tool_use_id=request.tool_use_id, status="abandoned")
            raise HTTPException(
                status_code=410,
                detail=(
                    f"Originating turn {row.turn_id} ended (status={turn.status}); "
                    "answer cannot be delivered."
                ),
            )

    # claude-code's session_key is f"{bot_id}:{user_id}" — same as in
    # chat_streaming when it computes oc_session_key.
    session_key = f"{request.bot_id}:{request.user_id}"

    subscriber = get_agent_subscriber()
    if not subscriber:
        raise HTTPException(status_code=503, detail="Redis subscriber not available")

    try:
        await subscriber.send_tool_result(
            session_key=session_key,
            tool_use_id=request.tool_use_id,
            result=request.result,
            backend=backend_name,
            request_id=f"toolres_{uuid.uuid4().hex}",
        )
    except Exception as e:
        log.exception("send_tool_result failed for tool_use_id=%s", request.tool_use_id)
        raise HTTPException(status_code=502, detail=f"Failed to publish tool_result: {e}") from e

    # Flip the registry row.  We treat the synthetic "(The user skipped this
    # question…)" sentinel the UI sends on Skip as a distinct status so the
    # audit trail and any future analytics can tell user-answer apart from
    # user-decline.  Any other text counts as a real answer.
    next_status = "answered"
    if pq_store:
        sentinel = "(The user skipped this question"
        if request.result.startswith(sentinel):
            next_status = "skipped"
        pq_store.mark(
            tool_use_id=request.tool_use_id,
            status=next_status,
            answer=request.result,
        )

    # Publish a unified-stream event so every open tab/window converges to
    # the post-answer UI without waiting for the bridge's TOOL_END to flow
    # back through Redis.  Best-effort; failures here don't block the user.
    try:
        from ...agent_backends.agent_bridge import get_agent_subscriber as _get_sub
        sub = _get_sub()
        if sub is not None:
            await sub._redis.xadd(  # type: ignore[attr-defined]
                f"events:{request.bot_id}:{request.user_id}",
                {"payload": json.dumps({
                    "_type": "tool_await_resolved",
                    "bot_id": request.bot_id,
                    "user_id": request.user_id,
                    "tool_use_id": request.tool_use_id,
                    "turn_id": row.turn_id if row else None,
                    "status": next_status,
                    "answer": request.result,
                    "ts": time.time(),
                }, ensure_ascii=False, default=str)},
                maxlen=5000,
                approximate=True,
            )
    except Exception:
        log.debug(
            "failed to publish tool_await_resolved unified event for tool_use_id=%s",
            request.tool_use_id,
        )

    log.info(
        "chat.tool_result dispatched: bot=%s session=%s tool_use_id=%s",
        request.bot_id, session_key, request.tool_use_id,
    )
    return ToolResultResponse(ok=True, detail="dispatched")


class PendingQuestionItem(BaseModel):
    """Single row of the pending-questions API."""
    tool_use_id: str
    tool_name: str
    bot_id: str
    user_id: str
    turn_id: str
    trigger_message_id: str | None = None
    session_key: str | None = None
    arguments: dict
    status: str
    answer: str | None = None
    created_at: str | None = None
    answered_at: str | None = None


class PendingQuestionList(BaseModel):
    data: list[PendingQuestionItem]


@router.get("/v1/chat/pending-questions", tags=["Agent Backends"])
async def list_pending_questions(
    bot_id: str | None = None,
    user_id: str | None = None,
    limit: int = 50,
) -> PendingQuestionList:
    """Return every currently-awaiting AskUserQuestion for a bot/user scope.

    Used by the chat UI on page load to hydrate any open pickers — the
    inline tool-call card reconciles each row against its activity entry
    and renders the active picker without waiting for an SSE replay.
    """
    service = get_service()
    pq_store = getattr(service, "_pending_question_store", None)
    if pq_store is None:
        return PendingQuestionList(data=[])
    rows = pq_store.list_awaiting(
        bot_id=bot_id, user_id=user_id, limit=max(1, min(int(limit or 50), 200)),
    )
    return PendingQuestionList(
        data=[PendingQuestionItem(**pq_store.row_to_dict(r)) for r in rows],
    )


class SessionResetRequest(BaseModel):
    """Request to reset an agent backend session."""
    bot_id: str = Field(..., description="Bot slug to reset session for")


class SessionResetResponse(BaseModel):
    ok: bool
    bot_id: str
    session_key: str | None = None
    detail: str | None = None


@router.post("/v1/chat/session/reset", tags=["Agent Backends"])
async def session_reset(request: SessionResetRequest) -> SessionResetResponse:
    """Reset the agent backend session for a bot.

    Sends a session.reset RPC to the bridge, which clears the stored
    conversation and starts fresh on the next message. Works for any
    agent backend that supports the RPC.
    """
    service = get_service()
    from ...bots import BotManager

    bot = BotManager(service.config).get_bot(request.bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail=f"Bot '{request.bot_id}' not found")

    backend_name = getattr(bot, "agent_backend", None)
    if not backend_name:
        raise HTTPException(status_code=400, detail=f"Bot '{request.bot_id}' is not an agent backend bot")

    bc = getattr(bot, "agent_backend_config", {}) or {}
    # claude-code and codex bridges route session.reset by parsing the bot
    # slug out of the sessionKey param (split on ':' or use as-is). Their
    # persisted ``session_key`` in agent_backend_config is the SDK-internal
    # thread id, not a routing key — sending it here means the bridge tries
    # to PATCH /v1/bots/<thread_id>/profile and silently fails. Always send
    # the bot slug for these backends.
    if backend_name in ("claude-code", "codex"):
        session_key = request.bot_id
    else:
        session_key = bc.get("session_key", "")

    from ...agent_backends.agent_bridge import get_agent_subscriber

    subscriber = get_agent_subscriber()
    if not subscriber:
        raise HTTPException(status_code=503, detail="Redis subscriber not available")

    rpc_req_id = f"reset_{uuid.uuid4().hex}"
    try:
        await subscriber.send_rpc(
            "session.reset",
            {"sessionKey": session_key},
            rpc_req_id,
            timeout_s=10,
            backend=backend_name,
        )
    except Exception as e:
        log.warning("session.reset RPC failed for bot %s: %s", request.bot_id, e)
        # Even if the bridge fails to clear gateway-side context, we still
        # return ok — every bridge (claude-code, codex, openclaw) now
        # intercepts session.reset and emits a SESSION_RESET unified event
        # on a best-effort path, so the frontend-visible reset still
        # happens.  See TASK-249.

    log.info("Session reset requested: bot=%s session=%s", request.bot_id, session_key)
    return SessionResetResponse(
        ok=True,
        bot_id=request.bot_id,
        session_key=session_key,
        detail="reset",
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
