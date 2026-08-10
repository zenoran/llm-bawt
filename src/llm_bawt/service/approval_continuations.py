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
import logging

from .dependencies import get_service

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
    "spawn_continuation",
    "build_continuation_prompt",
    "build_respond_prompt",
]
