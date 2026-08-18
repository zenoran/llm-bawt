"""Approval-resolve continuation helpers (TASK-639 slice A).

Extracted from ``service/routes/approval_policies.py`` so the resolve route
stays focused on HTTP-shape concerns and the continuation dispatch surface
has a stable home for TASK-639 Slice F (MCP-kind result-envelope
continuations and the durable outbox).

Public API (renamed from route-local underscore-prefixed helpers):

- ``spawn_continuation``      — fire-and-forget continuation turn dispatch
- ``build_continuation_prompt`` — canned prompt for approve/deny outcomes
- ``build_respond_prompt``    — prompt when the user chose 'respond'

Behavior is intentionally identical to the pre-extraction helpers. Slice F
will layer an MCP-kind branch and a durable outbox on top; keep the
signatures byte-stable until that lands so cross-slice diffs stay small.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .dependencies import get_service


MCP_RESULT_ENVELOPE_PREFIX = "[LLM_BAWT_MCP_TOOL_RESULT]"


def build_mcp_result_prompt(payload: dict[str, Any]) -> str:
    """Protocol-safe fallback carrying the actual persisted MCP result.

    The original tool already returned a non-blocking pending result, so some
    Claude CLI versions reject a second native ToolResultBlock for that tool-use
    id. This deterministic user-message envelope delivers the real normalized
    result without asking the model to execute anything again.
    """
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return (
        f"{MCP_RESULT_ENVELOPE_PREFIX}\n{canonical}\n"
        "The MCP tool has already been executed or refused server-side. Do not "
        "retry or re-issue the tool. Treat the result above as authoritative and "
        "continue from it."
    )


def continuation_payload_from_row(row) -> dict[str, Any]:
    try:
        result = json.loads(row.result_json or "null")
    except json.JSONDecodeError:
        result = row.result_json
    return {
        "kind": "mcp_tool_result",
        "approval_request_id": row.id,
        "original_tool_use_id": row.tool_use_id,
        "tool_name": row.tool_name,
        "result": result,
        "is_error": bool(row.result_is_error),
    }


async def dispatch_mcp_result_continuation(service, store, row) -> None:
    """Drive one claimed result continuation through the real chat pipeline."""
    from .schemas import ChatCompletionRequest, ChatMessage, McpToolResultContinuation

    payload = continuation_payload_from_row(row)
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content=build_mcp_result_prompt(payload))],
        bot_id=row.bot_id,
        user=row.user_id,
        stream=True,
        parent_turn_id=row.turn_id,
        continuation_payload=McpToolResultContinuation(**payload),
    )
    try:
        async for _ in service.chat_completion_stream(request):
            pass
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001
        store.mark_continuation_failed(row.id, error=str(error))
        raise
    else:
        store.mark_continuation_delivered(row.id)


async def run_mcp_continuation_outbox(
    service,
    store,
    *,
    idle_seconds: float = 2.0,
) -> None:
    """Lifespan worker: claim due persisted results and deliver with retry."""
    while True:
        due = await asyncio.to_thread(store.find_pending_continuations, limit=20)
        if not due:
            await asyncio.sleep(idle_seconds)
            continue
        for pending in due:
            claimed = await asyncio.to_thread(store.claim_continuation, pending.id)
            if claimed is None:
                continue
            try:
                await dispatch_mcp_result_continuation(service, store, claimed)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("MCP result continuation failed id=%s", claimed.id)


log = logging.getLogger(__name__)


def spawn_continuation(
    *,
    bot_id: str,
    user_id: str,
    prompt: str,
    parent_turn_id: str | None,
    grant_settle_s: float = 0.0,
) -> None:
    """Dispatch the post-resolution continuation turn SERVER-SIDE.

    This is the fix for the "approving did nothing" gap: the continuation that
    re-issues the approved call (or tells the agent it was denied) used to
    depend on the *client* making a second /v1/chat/completions call. Surfaces
    that only resolve — the admin page, the API, a script — never messaged the
    bot, so the approved command silently never ran and the grant expired.

    We drive the real streaming pipeline detached (fire-and-forget) so all the
    normal side effects happen — turn log, persistence, and the unified Redis
    SSE events the chat UI renders. The HTTP resolve response returns
    immediately; the turn streams into the chat on its own.

    ``grant_settle_s`` gives the bridge a beat to store the one-shot approval
    grant (sent on a different Redis stream than chat.send) before the re-issued
    tool call arrives, so it isn't re-gated by a race.
    """
    try:
        from .schemas import ChatCompletionRequest, ChatMessage

        service = get_service()
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content=prompt)],
            bot_id=bot_id,
            user=user_id,
            stream=True,
            parent_turn_id=parent_turn_id,
        )

        async def _drive() -> None:
            try:
                if grant_settle_s:
                    await asyncio.sleep(grant_settle_s)
                async for _ in service.chat_completion_stream(req):
                    pass
            except Exception:  # noqa: BLE001
                log.exception(
                    "server-side continuation dispatch failed (bot=%s parent_turn=%s)",
                    bot_id, parent_turn_id,
                )

        asyncio.create_task(_drive())
    except Exception:  # noqa: BLE001
        # Never let a dispatch failure break the resolve response itself.
        log.exception(
            "failed to spawn continuation (bot=%s parent_turn=%s)",
            bot_id, parent_turn_id,
        )


def build_continuation_prompt(approved: bool, subject: str, tool_name: str) -> str:
    shown = subject if len(subject) <= 400 else subject[:397] + "…"
    if approved:
        return (
            f"[The user APPROVED the {tool_name} action you requested. Re-issue "
            f"exactly this call now and continue:\n\n{shown}]"
        )
    return (
        f"[The user DENIED the {tool_name} action you requested:\n\n{shown}\n\n"
        f"Do not attempt it again. Continue without it, or explain what you need.]"
    )


def build_respond_prompt(message: str, subject: str, tool_name: str) -> str:
    """Continuation for a 'respond' resolution: the tool is NOT run; the user's
    own guidance steers the agent. Empty message falls back to a neutral note so
    the turn still closes cleanly."""
    shown = subject if len(subject) <= 400 else subject[:397] + "…"
    msg = (message or "").strip()
    if msg:
        return (
            f"[The user did NOT run the {tool_name} action you requested:\n\n{shown}\n\n"
            f"They responded with this guidance — follow it and do not re-issue the "
            f"original call unless it tells you to:\n\n{msg}]"
        )
    return (
        f"[The user reviewed the {tool_name} action you requested and chose not to "
        f"run it, with no further instruction. Continue without it.]"
    )


__all__ = [
    "MCP_RESULT_ENVELOPE_PREFIX",
    "spawn_continuation",
    "build_continuation_prompt",
    "build_respond_prompt",
    "build_mcp_result_prompt",
    "continuation_payload_from_row",
    "dispatch_mcp_result_continuation",
    "run_mcp_continuation_outbox",
]
