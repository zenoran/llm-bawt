"""Approval-aware FastMCP dispatch for the first-party BawtHub server.

Every registered tool passes through the shared pure policy evaluator. Claude
turns carry a signed reserved input field that binds the exact SDK tool-use id,
name, and original arguments to the opaque server-minted turn capability. The
field is stripped before FastMCP schema validation/function binding.
"""

from __future__ import annotations

import inspect
import json
import logging
import uuid
from typing import Any, Awaitable, Callable

from agent_bridge.approval import PolicyAction, evaluate
from agent_bridge.mcp_call_context import (
    MCP_CALL_CONTEXT_KEY,
    McpCallContext,
    canonical_invocation_hash,
    verify_mcp_call_context,
)
from mcp.server.fastmcp import FastMCP

from ..approval_policies import ApprovalPersistError
from ..task_turn_context import TaskTurnContext, open_task_turn_context
from .task_association import current_task_turn_capability

logger = logging.getLogger(__name__)

PolicyProvider = Callable[[], Any]
ApprovalPublisher = Callable[[dict[str, Any]], Awaitable[None] | None]


def _new_request_id() -> str:
    return "mcp-appr-" + uuid.uuid4().hex


class ApprovalAwareFastMCP(FastMCP):
    """FastMCP whose ``call_tool`` performs app-owned policy interception."""

    def __init__(
        self,
        *args,
        approval_store_provider: PolicyProvider | None = None,
        approval_publisher: ApprovalPublisher | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._approval_store_provider = approval_store_provider
        self._approval_publisher = approval_publisher

    def _approval_store(self):
        if self._approval_store_provider is not None:
            return self._approval_store_provider()
        from ..service.dependencies import get_tool_approval_policy_store
        from ..utils.config import Config

        config = getattr(self, "_approval_config", None)
        if config is None:
            config = Config()
            self._approval_config = config
        return get_tool_approval_policy_store(config)

    async def _publish_approval_required(self, payload: dict[str, Any]) -> None:
        if self._approval_publisher is not None:
            result = self._approval_publisher(payload)
            if inspect.isawaitable(result):
                await result
            return
        # The DB row is authoritative. Redis is a live fanout optimization and
        # may be unavailable during exactly the restart scenarios ops exists for.
        try:
            import redis.asyncio as aioredis

            from ..utils.config import Config

            config = getattr(self, "_approval_config", None) or Config()
            self._approval_config = config
            if not config.REDIS_URL:
                return
            client = aioredis.from_url(config.REDIS_URL, decode_responses=True)
            try:
                await client.xadd(
                    f"events:{payload['bot_id']}:{payload['user_id']}",
                    {"payload": json.dumps(payload, ensure_ascii=False, default=str)},
                    maxlen=5000,
                    approximate=True,
                )
            finally:
                await client.aclose()
        except Exception:
            logger.warning(
                "MCP approval committed but live publish failed id=%s",
                payload.get("request_id"),
                exc_info=True,
            )

    async def call_approved_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        expected_invocation_hash: str,
        trusted_argument_overrides: dict[str, Any] | None = None,
    ):
        """Invoke an already-approved exact stored call without policy recursion.

        The persisted public arguments are hash-verified *before* optional
        server-owned transport overrides (currently the ops idempotency key) are
        added. A caller cannot use the override seam through public MCP input.
        """
        received = dict(arguments or {})
        actual_hash = canonical_invocation_hash(name, received)
        if actual_hash != str(expected_invocation_hash or ""):
            raise ValueError("approved MCP invocation hash mismatch")
        if trusted_argument_overrides:
            received.update(trusted_argument_overrides)
        return await super().call_tool(name, received)

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        received = dict(arguments or {})
        raw_call_context = received.pop(MCP_CALL_CONTEXT_KEY, None)
        capability = current_task_turn_capability()
        call_context: McpCallContext | None = None
        turn_context: TaskTurnContext | None = None

        if raw_call_context is not None:
            call_context = verify_mcp_call_context(
                capability=capability or "",
                tool_name=name,
                tool_input=received,
                raw_context=raw_call_context,
            )
            turn_context = open_task_turn_context(capability)

        store = self._approval_store()
        decision = evaluate(
            store.compile_bundle().policies,
            call_context.backend if call_context else "mcp",
            name,
            received,
        )
        if decision.action is PolicyAction.ALLOW:
            return await super().call_tool(name, received)
        if decision.action is PolicyAction.DENY:
            return {
                "status": "denied",
                "tool": name,
                "subject": decision.subject,
                "message": "This MCP call is blocked by policy. Do not retry it.",
                "is_error": True,
            }

        request_id = _new_request_id()
        invocation_hash = canonical_invocation_hash(name, received)
        caller_context = None
        if turn_context is not None and call_context is not None:
            caller_context = {
                "session_id": turn_context.session_id,
                "turn_id": turn_context.turn_id,
                "trigger_message_id": turn_context.trigger_message_id,
                "bot_id": turn_context.bot_id,
                "user_id": turn_context.user_id,
                "issued_at": turn_context.issued_at,
                "agent_request_id": call_context.agent_request_id,
                "session_key": call_context.session_key,
                "backend": call_context.backend,
                "tool_use_id": call_context.tool_use_id,
            }
        try:
            row = store.record_mcp_request(
                request_id=request_id,
                tool_use_id=call_context.tool_use_id if call_context else None,
                mcp_server="bawthub",
                bot_id=turn_context.bot_id if turn_context else "unknown",
                user_id=turn_context.user_id if turn_context else "unknown",
                turn_id=turn_context.turn_id if turn_context else "unknown",
                backend=call_context.backend if call_context else "mcp",
                tool_name=name,
                tool_arguments=received,
                subject=decision.subject,
                grant_key=decision.grant_key,
                policy_id=getattr(decision.policy, "id", None),
                severity=decision.severity.value,
                prompt=decision.prompt,
                invocation_hash=invocation_hash,
                caller_context_json=(
                    json.dumps(caller_context, ensure_ascii=False, sort_keys=True)
                    if caller_context else None
                ),
                continuation_capable=caller_context is not None,
                trigger_message_id=(
                    turn_context.trigger_message_id if turn_context else None
                ),
                session_key=call_context.session_key if call_context else None,
            )
        except ApprovalPersistError as error:
            logger.error("Could not persist MCP approval for %s: %s", name, error)
            return {
                "status": "approval_persist_failed",
                "tool": name,
                "message": (
                    "The approval request could not be saved, so the tool was not "
                    "executed. Do not retry automatically."
                ),
                "is_error": True,
            }

        await self._publish_approval_required({
            "_type": "tool_approval_required",
            "request_id": row.id,
            "turn_id": row.turn_id,
            "trigger_message_id": row.trigger_message_id,
            "bot_id": row.bot_id,
            "user_id": row.user_id,
            "tool_name": row.tool_name,
            "arguments": received,
            "subject": row.subject,
            "label": decision.label,
            "prompt": row.prompt,
            "severity": row.severity,
            "policy_id": row.policy_id,
            "session_key": row.session_key or "",
            "provider": row.backend,
            "request_kind": "mcp",
            "continuation_capable": bool(row.continuation_capable),
        })
        return {
            "status": "approval_required",
            "approval_request_id": row.id,
            "tool": name,
            "subject": row.subject,
            "continuation_capable": bool(row.continuation_capable),
            "message": (
                "Approval is required. The call was not executed. Do not retry or "
                "re-issue this tool; wait for the persisted approval result."
            ),
        }


__all__ = ["ApprovalAwareFastMCP"]
