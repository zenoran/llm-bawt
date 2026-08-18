"""Trusted BawtHub task association operations for MCP task tools."""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar, Token
from typing import Any

import httpx

from ..task_turn_context import (
    TASK_TURN_CONTEXT_HEADER,
    TaskTurnContext,
    TaskTurnContextError,
    is_delivery_turn_id,
    open_task_turn_context,
)

logger = logging.getLogger(__name__)

_HEADER_BYTES = TASK_TURN_CONTEXT_HEADER.lower().encode("ascii")
_INTERNAL_URL = os.getenv(
    "BAWTHUB_TASK_ASSOCIATION_INTERNAL_URL",
    "http://frontend-prod:3002",
).rstrip("/")
_INTERNAL_TOKEN_ENV = "TASK_ASSOCIATION_INTERNAL_TOKEN"
_TIMEOUT = 10.0
_current_capability: ContextVar[str | None] = ContextVar(
    "task_turn_context_capability",
    default=None,
)


def set_current_task_turn_capability(value: str | None) -> Token:
    """Bind one HTTP request's opaque capability to its async task context."""
    return _current_capability.set(str(value or "").strip() or None)


def reset_current_task_turn_capability(token: Token) -> None:
    _current_capability.reset(token)


def current_task_turn_capability() -> str | None:
    """Return the opaque capability bound to this HTTP request, if any."""
    return _current_capability.get()


def current_task_turn_context() -> TaskTurnContext:
    return open_task_turn_context(current_task_turn_capability())


def task_turn_context_header_name() -> str:
    return TASK_TURN_CONTEXT_HEADER


def capability_from_asgi_scope(scope: dict[str, Any]) -> str | None:
    """Extract the single opaque capability header from an ASGI HTTP scope."""
    for raw_name, raw_value in scope.get("headers") or []:
        if raw_name.lower() == _HEADER_BYTES:
            try:
                return raw_value.decode("ascii").strip() or None
            except UnicodeDecodeError:
                return None
    return None


class TaskTurnCapabilityMiddleware:
    """Bind the MCP HTTP header to a ContextVar for one ASGI request only."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        binding = set_current_task_turn_capability(capability_from_asgi_scope(scope))
        try:
            await self.app(scope, receive, send)
        finally:
            reset_current_task_turn_capability(binding)


async def associate_current_task(task_ref: str) -> dict[str, Any]:
    """Associate the trusted current session+turn to ``task_ref`` as AGENT."""
    context = current_task_turn_context()
    token = os.getenv(_INTERNAL_TOKEN_ENV, "").strip()
    if not token:
        raise TaskTurnContextError(
            "Task association is unavailable because the internal service credential is not configured"
        )

    # Synthetic inter-bot-delivery turn ids (``turn-delivery-<hex>``) are
    # signed into the capability for audit but are NOT valid ``AgentTaskTurn.turnId``
    # values on the BawtHub side. Per the TASK-706 design, delivery-turn
    # attribution rides on ``triggerMessageId`` alone; the resolve batch and
    # PUT bodies both filter delivery ids so BawtHub's strict validator can
    # stay strict (``turn-<32 hex>`` only). Null the turnId here instead of
    # letting the PUT 400 the whole association.
    turn_id_for_put = None if is_delivery_turn_id(context.turn_id) else context.turn_id
    body = {
        "sessionId": context.session_id,
        "botId": context.bot_id,
        "userId": context.user_id,
        "source": "AGENT",
        "turn": {
            "triggerMessageId": context.trigger_message_id,
            "turnId": turn_id_for_put,
            "source": "AGENT",
        },
    }
    async with httpx.AsyncClient(
        base_url=_INTERNAL_URL,
        timeout=_TIMEOUT,
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        response = await client.put(
            f"/internal/tasks/{task_ref}/associations",
            json=body,
        )
        response.raise_for_status()
        payload = response.json()
    association = payload.get("association") if isinstance(payload, dict) else None
    return {
        "ok": True,
        "taskRef": task_ref,
        "source": "AGENT",
        "association": association,
    }
