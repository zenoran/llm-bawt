from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet

from agent_bridge.approval import ApprovalPolicy, MatcherType, PolicyAction, PolicyBundle
from agent_bridge.mcp_call_context import MCP_CALL_CONTEXT_KEY, mint_mcp_call_context
from llm_bawt import task_turn_context as turn_codec
from llm_bawt.mcp_server import task_association
from llm_bawt.mcp_server.approval_interceptor import ApprovalAwareFastMCP


def _turn_values():
    return {
        "session_id": str(uuid4()),
        "turn_id": "turn-" + "a" * 32,
        "trigger_message_id": str(uuid4()),
        "bot_id": "snark",
        "user_id": "nick",
    }


class FakeStore:
    def __init__(self, policy, order):
        self.policy = policy
        self.order = order
        self.rows = []

    def compile_bundle(self):
        policies = [self.policy] if self.policy else []
        return PolicyBundle(version=1, etag="test", policies=policies)

    def record_mcp_request(self, **kwargs):
        self.order.append("persist")
        row = SimpleNamespace(
            id=kwargs["request_id"],
            turn_id=kwargs["turn_id"],
            trigger_message_id=kwargs.get("trigger_message_id"),
            bot_id=kwargs["bot_id"],
            user_id=kwargs["user_id"],
            tool_name=kwargs["tool_name"],
            subject=kwargs["subject"],
            prompt=kwargs["prompt"],
            severity=kwargs["severity"],
            policy_id=kwargs["policy_id"],
            session_key=kwargs.get("session_key"),
            backend=kwargs["backend"],
            continuation_capable=kwargs["continuation_capable"],
        )
        self.rows.append((row, kwargs))
        return row


def _policy(action: PolicyAction):
    return ApprovalPolicy(
        id="p1",
        tool_name="ops_run",
        matcher_type=MatcherType.ALWAYS,
        action=action,
    )


def _server(store, order, executed):
    async def publish(_payload):
        order.append("publish")

    mcp = ApprovalAwareFastMCP(
        "test",
        json_response=True,
        approval_store_provider=lambda: store,
        approval_publisher=publish,
    )

    @mcp.tool(name="ops_run")
    async def ops_run(operation: str, args: dict | None = None):
        executed.append((operation, args or {}))
        return {"ran": operation}

    return mcp


def _stamped_args(capability):
    clean = {"operation": "llm-bawt.restart-app", "args": {}}
    stamp = mint_mcp_call_context(
        capability=capability,
        tool_name="ops_run",
        tool_input=clean,
        tool_use_id="toolu_123",
        agent_request_id="req_123",
        session_key="snark:nick",
        backend="claude-code",
    )
    return {**clean, MCP_CALL_CONTEXT_KEY: stamp}


@pytest.fixture
def capability(monkeypatch):
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(turn_codec, "_get_fernet", lambda: fernet)
    return turn_codec.mint_task_turn_context(**_turn_values())


def _call(mcp, capability, args):
    binding = task_association.set_current_task_turn_capability(capability)
    try:
        return asyncio.run(mcp.call_tool("ops_run", args))
    finally:
        task_association.reset_current_task_turn_capability(binding)


def test_allow_strips_reserved_context_before_binding_and_executes(capability):
    order, executed = [], []
    store = FakeStore(_policy(PolicyAction.ALLOW), order)
    result = _call(_server(store, order, executed), capability, _stamped_args(capability))
    assert executed == [("llm-bawt.restart-app", {})]
    assert store.rows == []
    assert len(result) == 1
    assert '"ran": "llm-bawt.restart-app"' in result[0].text


def test_deny_never_executes_or_persists(capability):
    order, executed = [], []
    store = FakeStore(_policy(PolicyAction.DENY), order)
    result = _call(_server(store, order, executed), capability, _stamped_args(capability))
    assert result["status"] == "denied"
    assert executed == []
    assert store.rows == []


def test_require_commits_before_publish_and_returns_pending(capability):
    order, executed = [], []
    store = FakeStore(_policy(PolicyAction.REQUIRE_APPROVAL), order)
    result = _call(_server(store, order, executed), capability, _stamped_args(capability))
    assert result["status"] == "approval_required"
    assert "Do not retry" in result["message"]
    assert executed == []
    assert order == ["persist", "publish"]
    row, persisted = store.rows[0]
    assert persisted["tool_arguments"] == {
        "operation": "llm-bawt.restart-app",
        "args": {},
    }
    assert persisted["tool_use_id"] == "toolu_123"
    assert persisted["continuation_capable"] is True
    assert row.bot_id == "snark"


def test_tampered_args_fail_before_policy_or_execution(capability):
    order, executed = [], []
    store = FakeStore(_policy(PolicyAction.ALLOW), order)
    args = _stamped_args(capability)
    args["operation"] = "llm-bawt.restart-redis"
    with pytest.raises(ValueError, match="does not match"):
        _call(_server(store, order, executed), capability, args)
    assert executed == []
    assert store.rows == []
