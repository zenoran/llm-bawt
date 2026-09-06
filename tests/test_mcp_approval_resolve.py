from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from agent_bridge.mcp_call_context import canonical_invocation_hash
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

from llm_bawt.approval_policies import (
    EXEC_FAILED,
    EXEC_SKIPPED,
    EXEC_SUCCEEDED,
    REQ_APPROVED,
    REQ_DENIED,
    ToolApprovalPolicyStore,
)
from llm_bawt.service.routes import approval_policies as routes


def _store():
    store = object.__new__(ToolApprovalPolicyStore)
    store.engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    store._ensure_tables_exist()
    return store


def _mcp_req(**over):
    base = {
        "request_id": "req-mcp-1",
        "tool_use_id": "toolu_01ABC",
        "mcp_server": "bawthub",
        "bot_id": "snark",
        "user_id": "nick",
        "turn_id": "turn-1",
        "backend": "claude-code",
        "tool_name": "ops_run",
        "tool_arguments": {"operation": "llm-bawt.restart-app", "args": {}},
        "subject": "operation=llm-bawt.restart-app args={}",
        "grant_key": "grant-1",
        "policy_id": "pol-mcp-1",
        "severity": "high",
        "prompt": "Approve restart?",
        "invocation_hash": "deadbeef" * 8,
        "continuation_capable": True,
    }
    base.update(over)
    return base


class FakeMcp:
    def __init__(self):
        self.calls = []
        self.caller_contexts = []
        self.fail = None

    async def call_approved_tool(
        self,
        name,
        arguments,
        *,
        expected_invocation_hash,
        trusted_argument_overrides=None,
        caller_context=None,
    ):
        if self.fail:
            raise self.fail
        assert expected_invocation_hash == canonical_invocation_hash(name, arguments)
        effective = {**arguments, **(trusted_argument_overrides or {})}
        self.calls.append((name, arguments, trusted_argument_overrides))
        self.caller_contexts.append(caller_context)
        return [SimpleNamespace(text=json.dumps({
            "job_id": "job-1",
            "operation": effective["operation"],
            "idempotency_key": effective.get("idempotency_key"),
            "state": "queued",
        }))]


def persisted_bot(store, request_id):
    return store.get_request(request_id).bot_id


def _record(store, *, continuation_capable=True, invocation_hash=None):
    args = {"operation": "llm-bawt.restart-app", "args": {}}
    return store.record_mcp_request(**_mcp_req(
        tool_arguments=args,
        continuation_capable=continuation_capable,
        invocation_hash=invocation_hash or canonical_invocation_hash("ops_run", args),
    ))


def _run(store, row, *, outcome="approve", message=""):
    return asyncio.run(routes._resolve_mcp_request(
        store,
        row,
        outcome=outcome,
        message=message,
        resolved_by="nick",
    ))


def test_approve_executes_stored_call_once_and_persists_actual_result(monkeypatch):
    store = _store()
    row = _record(store)
    fake = FakeMcp()
    from llm_bawt.mcp_server import registry
    monkeypatch.setattr(registry, "mcp", fake)

    first = _run(store, row)
    second = _run(store, store.get_request(row.id))

    assert len(fake.calls) == 1
    name, public_args, overrides = fake.calls[0]
    assert name == "ops_run"
    assert public_args == {"operation": "llm-bawt.restart-app", "args": {}}
    assert overrides == {"idempotency_key": row.id}
    # Provenance comes from the persisted row, so the executed job is
    # traceable back to the approval that authorized it (TASK-639).
    caller = fake.caller_contexts[0]
    assert caller is not None
    assert caller.approval_request_id == row.id
    assert caller.bot_id == persisted_bot(store, row.id)
    assert first["execution_state"] == EXEC_SUCCEEDED
    assert first["result"]["job_id"] == "job-1"
    assert first["result"]["idempotency_key"] == row.id
    assert first["continuation_prompt"] is None
    assert first["server_dispatched"] is True
    assert second["already_resolved"] is True
    assert second["result"] == first["result"]
    persisted = store.get_request(row.id)
    assert persisted.status == REQ_APPROVED
    assert persisted.execution_attempts == 1


def test_deny_never_executes_and_stores_refusal(monkeypatch):
    store = _store()
    row = _record(store, continuation_capable=False)
    fake = FakeMcp()
    from llm_bawt.mcp_server import registry
    monkeypatch.setattr(registry, "mcp", fake)

    result = _run(store, row, outcome="deny")

    assert fake.calls == []
    assert result["status"] == REQ_DENIED
    assert result["execution_state"] == EXEC_SKIPPED
    assert result["result"]["status"] == "denied"
    assert result["result_is_error"] is True
    assert result["server_dispatched"] is False


def test_hash_mismatch_fails_without_invoking(monkeypatch):
    store = _store()
    row = _record(store, invocation_hash="0" * 64)
    fake = FakeMcp()
    # Fake the real bypass's hash validation failure.
    fake.fail = ValueError("approved MCP invocation hash mismatch")
    from llm_bawt.mcp_server import registry
    monkeypatch.setattr(registry, "mcp", fake)

    result = _run(store, row)

    assert fake.calls == []
    assert result["execution_state"] == EXEC_FAILED
    assert result["result_is_error"] is True
    assert "hash mismatch" in result["result"]["error"]


def test_live_execution_claim_cannot_be_taken_twice():
    store = _store()
    row = _record(store)
    store.resolve_request(row.id, status=REQ_APPROVED)
    first = store.claim_mcp_execution(row.id, lease_seconds=60)
    second = store.claim_mcp_execution(row.id, lease_seconds=60)
    assert first is not None
    assert second is None
    assert first.execution_attempts == 1


# ---- cross-process tool registration (TASK-639) ----------------------------

def test_ensure_tools_registered_populates_bare_app_process_singleton():
    """The FastAPI process must be able to dispatch an approved MCP call.

    Regression: the resolve route imports the ``mcp`` singleton from
    ``registry`` alone. In the app process nothing else imports the tool
    modules, so the singleton was bare and every approved call died with
    "Unknown tool: ops_run" *after* the user had already approved it.

    Run in a subprocess so an unrelated test importing ``server`` first can't
    mask the bug through ``sys.modules``.
    """
    import subprocess
    import sys

    script = """
import asyncio
from llm_bawt.mcp_server.registry import mcp, ensure_tools_registered

async def names():
    return {t.name for t in await mcp.list_tools()}

before = asyncio.run(names())
assert "ops_run" not in before, "expected a bare singleton before registration"
ensure_tools_registered()
after = asyncio.run(names())
assert "ops_run" in after, "ops_run still missing after ensure_tools_registered()"
assert "ops_job_status" in after
print("OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr[-3000:]}"
    assert "OK" in proc.stdout
