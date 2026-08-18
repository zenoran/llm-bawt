from __future__ import annotations

import copy

import pytest

from agent_bridge.mcp_call_context import (
    MCP_CALL_CONTEXT_KEY,
    McpCallContextError,
    canonical_invocation_hash,
    mint_mcp_call_context,
    verify_mcp_call_context,
)


def _stamp(**over):
    values = {
        "capability": "opaque-trusted-turn-token",
        "tool_name": "ops_run",
        "tool_input": {"operation": "llm-bawt.restart-app", "args": {}},
        "tool_use_id": "toolu_123",
        "agent_request_id": "req_abc",
        "session_key": "snark:nick",
        "backend": "claude-code",
    }
    values.update(over)
    return values, mint_mcp_call_context(**values)


def test_round_trip_binds_exact_tool_and_args():
    values, stamp = _stamp()
    opened = verify_mcp_call_context(
        capability=values["capability"],
        tool_name=values["tool_name"],
        tool_input=values["tool_input"],
        raw_context=stamp,
    )
    assert opened.tool_use_id == "toolu_123"
    assert opened.agent_request_id == "req_abc"
    assert opened.session_key == "snark:nick"
    assert opened.invocation_hash == canonical_invocation_hash(
        "ops_run", values["tool_input"]
    )


def test_reserved_context_field_is_excluded_from_invocation_hash():
    base = {"operation": "x", "args": {"b": 2, "a": 1}}
    stamped = {**base, MCP_CALL_CONTEXT_KEY: {"forged": True}}
    assert canonical_invocation_hash("ops_run", base) == canonical_invocation_hash(
        "ops_run", stamped
    )


def test_tampered_args_or_tool_fail_closed():
    values, stamp = _stamp()
    with pytest.raises(McpCallContextError, match="does not match"):
        verify_mcp_call_context(
            capability=values["capability"],
            tool_name="ops_run",
            tool_input={"operation": "llm-bawt.restart-redis", "args": {}},
            raw_context=stamp,
        )
    with pytest.raises(McpCallContextError, match="does not match"):
        verify_mcp_call_context(
            capability=values["capability"],
            tool_name="ops_job_status",
            tool_input=values["tool_input"],
            raw_context=stamp,
        )


def test_forged_signature_or_wrong_capability_fails_closed():
    values, stamp = _stamp()
    forged = copy.deepcopy(stamp)
    forged["signature"] = "0" * 64
    with pytest.raises(McpCallContextError, match="signature is invalid"):
        verify_mcp_call_context(
            capability=values["capability"],
            tool_name=values["tool_name"],
            tool_input=values["tool_input"],
            raw_context=forged,
        )
    with pytest.raises(McpCallContextError, match="signature is invalid"):
        verify_mcp_call_context(
            capability="different-turn",
            tool_name=values["tool_name"],
            tool_input=values["tool_input"],
            raw_context=stamp,
        )
