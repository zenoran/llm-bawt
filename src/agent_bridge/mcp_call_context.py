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

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any

MCP_CALL_CONTEXT_KEY = "_llm_bawt_call_context"
MCP_CALL_CONTEXT_VERSION = 1
MCP_REQUEST_CONTEXT_VERSION = 1
MCP_REQUEST_CONTEXT_HEADER = "X-LLM-Bawt-MCP-Request-Context"
MCP_TASK_TURN_CONTEXT_ENV = "LLM_BAWT_TASK_TURN_CONTEXT"
MCP_REQUEST_CONTEXT_ENV = "LLM_BAWT_MCP_REQUEST_CONTEXT"
_SENTINEL_IDENTITIES = {"unknown", "none", "null", "undefined"}


class McpCallContextError(ValueError):
    """The per-call stamp is missing, malformed, forged, or mismatched."""


def _required_identity(value: Any, field: str, *, max_length: int = 256) -> str:
    normalized = str(value or "").strip()
    if (
        not normalized
        or len(normalized) > max_length
        or normalized.lower() in _SENTINEL_IDENTITIES
        or any(ord(char) < 32 or ord(char) == 127 for char in normalized)
    ):
        raise McpCallContextError(f"{field} is missing or invalid")
    return normalized


@dataclass(frozen=True)
class McpRequestContext:
    """Signed identity for one bridge request, carried outside model input."""

    agent_request_id: str
    session_key: str
    backend: str
    signature: str
    version: int = MCP_REQUEST_CONTEXT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "agent_request_id": self.agent_request_id,
            "session_key": self.session_key,
            "backend": self.backend,
            "signature": self.signature,
        }


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
    return hashlib.sha256(
        canonical.encode("utf-8", errors="surrogateescape")
    ).hexdigest()


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


def _request_unsigned_payload(context: McpRequestContext) -> dict[str, Any]:
    return {
        "version": context.version,
        "agent_request_id": context.agent_request_id,
        "session_key": context.session_key,
        "backend": context.backend,
    }


def mint_mcp_request_context(
    *, capability: str, agent_request_id: str, session_key: str, backend: str
) -> str:
    """Mint a compact signed HTTP-header value for one agent request."""
    context = McpRequestContext(
        agent_request_id=_required_identity(agent_request_id, "agent_request_id"),
        session_key=_required_identity(session_key, "session_key"),
        backend=_required_identity(backend, "backend", max_length=64),
        signature="",
    )
    signed = McpRequestContext(
        **_request_unsigned_payload(context),
        signature=_signature(capability, _request_unsigned_payload(context)),
    )
    raw = json.dumps(signed.to_dict(), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def verify_mcp_request_context(
    *, capability: str, raw_context: Any
) -> McpRequestContext:
    """Open and verify the request-local bridge identity header."""
    encoded = str(raw_context or "").strip()
    if not encoded:
        raise McpCallContextError("MCP request context is missing")
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        body = json.loads(base64.b64decode(padded, altchars=b"-_", validate=True))
        context = McpRequestContext(
            version=int(body.get("version", 0)),
            agent_request_id=_required_identity(
                body.get("agent_request_id"), "agent_request_id"
            ),
            session_key=_required_identity(body.get("session_key"), "session_key"),
            backend=_required_identity(body.get("backend"), "backend", max_length=64),
            signature=str(body.get("signature") or "").strip(),
        )
    except (
        AttributeError,
        ValueError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise McpCallContextError("MCP request context is malformed") from error
    if context.version != MCP_REQUEST_CONTEXT_VERSION:
        raise McpCallContextError("MCP request context version is unsupported")
    if not context.signature:
        raise McpCallContextError("MCP request context is incomplete")
    expected = _signature(capability, _request_unsigned_payload(context))
    if not hmac.compare_digest(context.signature, expected):
        raise McpCallContextError("MCP request context signature is invalid")
    return context


def derive_mcp_call_context(
    *,
    capability: str,
    raw_request_context: Any,
    protocol_request_id: Any,
    tool_name: str,
    tool_input: dict[str, Any],
) -> McpCallContext:
    """Bind a signed bridge request to one exact MCP protocol invocation."""
    request = verify_mcp_request_context(
        capability=capability, raw_context=raw_request_context
    )
    protocol_id = _required_identity(protocol_request_id, "MCP protocol request_id")
    identity = json.dumps(
        [request.backend, request.agent_request_id, request.session_key, protocol_id],
        separators=(",", ":"),
    )
    tool_use_id = "mcp-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:48]
    raw_call = mint_mcp_call_context(
        capability=capability,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_use_id=tool_use_id,
        agent_request_id=request.agent_request_id,
        session_key=request.session_key,
        backend=request.backend,
    )
    return verify_mcp_call_context(
        capability=capability,
        tool_name=tool_name,
        tool_input=tool_input,
        raw_context=raw_call,
    )


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
    tool_use_id = _required_identity(tool_use_id, "tool_use_id", max_length=128)
    context = McpCallContext(
        tool_use_id=tool_use_id,
        invocation_hash=canonical_invocation_hash(tool_name, tool_input),
        agent_request_id=_required_identity(agent_request_id, "agent_request_id"),
        session_key=_required_identity(session_key, "session_key"),
        backend=_required_identity(backend, "backend", max_length=64),
        signature="",
    )
    signature = _signature(capability, _unsigned_payload(context))
    return McpCallContext(
        **{**_unsigned_payload(context), "signature": signature}
    ).to_dict()


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
    _required_identity(context.tool_use_id, "tool_use_id", max_length=128)
    _required_identity(context.agent_request_id, "agent_request_id")
    _required_identity(context.session_key, "session_key")
    _required_identity(context.backend, "backend", max_length=64)
    if not context.signature:
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
    "MCP_REQUEST_CONTEXT_ENV",
    "MCP_REQUEST_CONTEXT_HEADER",
    "MCP_REQUEST_CONTEXT_VERSION",
    "MCP_TASK_TURN_CONTEXT_ENV",
    "McpCallContext",
    "McpCallContextError",
    "McpRequestContext",
    "canonical_invocation_hash",
    "derive_mcp_call_context",
    "mint_mcp_call_context",
    "mint_mcp_request_context",
    "verify_mcp_call_context",
    "verify_mcp_request_context",
]
