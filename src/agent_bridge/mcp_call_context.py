"""Signed per-call context for first-party BawtHub MCP invocations.

The app sends an opaque, server-minted turn capability to the Claude bridge and
BawtHub MCP server as a request-local HTTP header. The Claude PreToolUse hook
uses that same opaque value as HMAC key material to stamp the exact SDK tool
call into a reserved input field. The MCP server strips the field before tool
schema validation and verifies that neither the tool name nor original args
were changed.

Pure stdlib only: both bridge and app/MCP processes import this module.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any

MCP_CALL_CONTEXT_KEY = "_llm_bawt_call_context"
MCP_CALL_CONTEXT_VERSION = 1


class McpCallContextError(ValueError):
    """The per-call stamp is missing, malformed, forged, or mismatched."""


@dataclass(frozen=True)
class McpCallContext:
    tool_use_id: str
    invocation_hash: str
    agent_request_id: str
    session_key: str
    backend: str
    signature: str
    version: int = MCP_CALL_CONTEXT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tool_use_id": self.tool_use_id,
            "invocation_hash": self.invocation_hash,
            "agent_request_id": self.agent_request_id,
            "session_key": self.session_key,
            "backend": self.backend,
            "signature": self.signature,
        }


def canonical_invocation_hash(tool_name: str, tool_input: Any) -> str:
    """Hash the exact public tool name + original arguments deterministically."""
    clean_input = dict(tool_input) if isinstance(tool_input, dict) else {}
    clean_input.pop(MCP_CALL_CONTEXT_KEY, None)
    canonical = json.dumps(
        {"tool": str(tool_name or ""), "arguments": clean_input},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8", errors="surrogateescape")).hexdigest()


def _unsigned_payload(context: McpCallContext) -> dict[str, Any]:
    return {
        "version": context.version,
        "tool_use_id": context.tool_use_id,
        "invocation_hash": context.invocation_hash,
        "agent_request_id": context.agent_request_id,
        "session_key": context.session_key,
        "backend": context.backend,
    }


def _signature(capability: str, payload: dict[str, Any]) -> str:
    key = str(capability or "").strip().encode("utf-8", errors="surrogateescape")
    if not key:
        raise McpCallContextError("trusted turn capability is unavailable")
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8", errors="surrogateescape")
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()


def mint_mcp_call_context(
    *,
    capability: str,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_use_id: str,
    agent_request_id: str,
    session_key: str,
    backend: str,
) -> dict[str, Any]:
    """Create the reserved signed input field for one exact MCP invocation."""
    tool_use_id = str(tool_use_id or "").strip()
    if not tool_use_id:
        raise McpCallContextError("tool_use_id is required")
    context = McpCallContext(
        tool_use_id=tool_use_id,
        invocation_hash=canonical_invocation_hash(tool_name, tool_input),
        agent_request_id=str(agent_request_id or "").strip(),
        session_key=str(session_key or "").strip(),
        backend=str(backend or "").strip() or "claude-code",
        signature="",
    )
    signature = _signature(capability, _unsigned_payload(context))
    return McpCallContext(**{**_unsigned_payload(context), "signature": signature}).to_dict()


def verify_mcp_call_context(
    *,
    capability: str,
    tool_name: str,
    tool_input: dict[str, Any],
    raw_context: Any,
) -> McpCallContext:
    """Verify one stamp and bind it to the received tool name + clean args."""
    if not isinstance(raw_context, dict):
        raise McpCallContextError("MCP call context is malformed")
    try:
        context = McpCallContext(
            version=int(raw_context.get("version", 0)),
            tool_use_id=str(raw_context.get("tool_use_id") or "").strip(),
            invocation_hash=str(raw_context.get("invocation_hash") or "").strip(),
            agent_request_id=str(raw_context.get("agent_request_id") or "").strip(),
            session_key=str(raw_context.get("session_key") or "").strip(),
            backend=str(raw_context.get("backend") or "").strip(),
            signature=str(raw_context.get("signature") or "").strip(),
        )
    except (TypeError, ValueError) as error:
        raise McpCallContextError("MCP call context is malformed") from error
    if context.version != MCP_CALL_CONTEXT_VERSION:
        raise McpCallContextError("MCP call context version is unsupported")
    if not context.tool_use_id or not context.signature:
        raise McpCallContextError("MCP call context is incomplete")
    expected_hash = canonical_invocation_hash(tool_name, tool_input)
    if not hmac.compare_digest(context.invocation_hash, expected_hash):
        raise McpCallContextError("MCP call context does not match tool arguments")
    expected_signature = _signature(capability, _unsigned_payload(context))
    if not hmac.compare_digest(context.signature, expected_signature):
        raise McpCallContextError("MCP call context signature is invalid")
    return context


__all__ = [
    "MCP_CALL_CONTEXT_KEY",
    "MCP_CALL_CONTEXT_VERSION",
    "McpCallContext",
    "McpCallContextError",
    "canonical_invocation_hash",
    "mint_mcp_call_context",
    "verify_mcp_call_context",
]
