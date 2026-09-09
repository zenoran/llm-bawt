from __future__ import annotations

import asyncio
import json

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

from agent_bridge.mcp_call_context import canonical_invocation_hash
from llm_bawt.approval_policies import (
    CONT_DELIVERED,
    CONT_PENDING,
    REQ_APPROVED,
    ToolApprovalPolicyStore,
)
from llm_bawt.service.approval_continuations import (
    MCP_RESULT_ENVELOPE_PREFIX,
    _terminal_ops_result,
    dispatch_due_continuations_once,
    dispatch_mcp_result_continuation,
)


def _store():
    store = object.__new__(ToolApprovalPolicyStore)
    store.engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    store._ensure_tables_exist()
    return store


def _ready_row(store, *, tool_name="ops_run", result=None):
    args = {"operation": "llm-bawt.restart-app", "args": {}}
    row = store.record_mcp_request(
        request_id="req-mcp-1",
        tool_use_id="toolu-1",
        mcp_server="bawthub",
        bot_id="snark",
        user_id="nick",
        turn_id="turn-1",
        backend="claude-code",
        tool_name=tool_name,
        tool_arguments=args,
        subject="operation=llm-bawt.restart-app args={}",
        grant_key="grant",
        policy_id="policy",
        severity="medium",
        prompt="Approve?",
        invocation_hash=canonical_invocation_hash("ops_run", args),
        continuation_capable=True,
    )
    store.resolve_request(row.id, status=REQ_APPROVED)
    store.claim_mcp_execution(row.id)
    store.complete_mcp_execution(
        row.id,
        result_json=json.dumps(result or {"job_id": "job-1", "state": "queued"}),
        is_error=False,
    )
    store.enqueue_continuation(row.id)
    return store.claim_continuation(row.id)


class FakeService:
    def __init__(self, error=None):
        self.requests = []
        self.error = error
        self.config = None

    async def chat_completion_stream(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        yield "data: [DONE]\n\n"


class FakeOps:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def get_job_status(self, job_id, **kwargs):
        self.calls.append((job_id, kwargs))
        return self.result


class OpsServiceHost(FakeService):
    def __init__(self, result):
        super().__init__()
        self._ops_service = FakeOps(result)


def test_outbox_does_not_claim_ops_continuation_before_job_is_terminal(monkeypatch):
    store = _store()
    row = _ready_row(store, result={"job_id": "job-1", "state": "accepted"})
    # Put the fixture back into a due, unclaimed outbox state.
    store.mark_continuation_failed(
        row.id, error="fixture reset", backoff_seconds=0,
        claim_token=row.continuation_claim_token,
    )
    service = OpsServiceHost({"job_id": "job-1", "state": "running", "terminal": False})
    monkeypatch.setattr(
        "llm_bawt.service.approval_continuations.recover_approval_requests",
        lambda _store: asyncio.sleep(0),
    )

    asyncio.run(dispatch_due_continuations_once(service, store))

    current = store.get_request(row.id)
    assert current.continuation_state == CONT_PENDING
    assert service.requests == []


def test_outbox_dispatches_ops_continuation_once_after_terminal_receipt(monkeypatch):
    store = _store()
    row = _ready_row(store, result={"job_id": "job-1", "state": "accepted"})
    store.mark_continuation_failed(
        row.id, error="fixture reset", backoff_seconds=0,
        claim_token=row.continuation_claim_token,
    )
    terminal = {
        "job_id": "job-1",
        "operation": "llm-bawt.restart-app",
        "state": "succeeded",
        "terminal": True,
        "exit_code": 0,
    }
    service = OpsServiceHost(terminal)
    monkeypatch.setattr(
        "llm_bawt.service.approval_continuations.recover_approval_requests",
        lambda _store: asyncio.sleep(0),
    )

    asyncio.run(dispatch_due_continuations_once(service, store))
    asyncio.run(dispatch_due_continuations_once(service, store))

    current = store.get_request(row.id)
    assert current.continuation_state == CONT_DELIVERED
    assert len(service.requests) == 1
    assert service.requests[0].continuation_payload.result == terminal


def test_ops_continuation_waits_for_terminal_job_receipt():
    store = _store()
    row = _ready_row(store, result={"job_id": "job-1", "state": "accepted"})
    service = OpsServiceHost({"job_id": "job-1", "state": "running", "terminal": False})

    assert _terminal_ops_result(service, row) is None
    assert service._ops_service.calls == [(
        "job-1",
        {"output_tail_bytes": 4096, "reconcile_if_active": True},
    )]


def test_terminal_ops_receipt_replaces_initial_accepted_result():
    store = _store()
    row = _ready_row(store, result={"job_id": "job-1", "state": "accepted"})
    terminal = {
        "job_id": "job-1",
        "operation": "llm-bawt.restart-app",
        "state": "succeeded",
        "terminal": True,
        "exit_code": 0,
    }
    service = OpsServiceHost(terminal)

    final = _terminal_ops_result(service, row)
    assert final == terminal
    asyncio.run(
        dispatch_mcp_result_continuation(
            service, store, row, result_override=final,
        )
    )
    assert service.requests[0].continuation_payload.result == terminal
    assert service.requests[0].continuation_payload.is_error is False
    assert '"state": "succeeded"' in service.requests[0].messages[0].content


def test_failed_terminal_ops_receipt_marks_continuation_result_error():
    store = _store()
    row = _ready_row(store, result={"job_id": "job-1", "state": "accepted"})
    failed = {
        "job_id": "job-1",
        "operation": "llm-bawt.restart-app",
        "state": "failed",
        "terminal": True,
        "error_text": "restart refused",
    }
    service = OpsServiceHost(failed)

    asyncio.run(
        dispatch_mcp_result_continuation(
            service, store, row, result_override=failed,
        )
    )
    assert service.requests[0].continuation_payload.is_error is True


def test_dispatch_delivers_actual_result_envelope_and_marks_done():
    store = _store()
    row = _ready_row(store)
    service = FakeService()

    asyncio.run(dispatch_mcp_result_continuation(service, store, row))

    request = service.requests[0]
    prompt = request.messages[0].content
    assert prompt.startswith(MCP_RESULT_ENVELOPE_PREFIX)
    assert '"job_id": "job-1"' in prompt
    assert "Do not retry or re-issue the tool" in prompt
    assert request.parent_turn_id == "turn-1"
    assert request.continuation_payload.approval_request_id == "req-mcp-1"
    assert request.continuation_payload.result == {"job_id": "job-1", "state": "queued"}
    persisted = store.get_request(row.id)
    assert persisted.continuation_state == CONT_DELIVERED
    assert persisted.continuation_delivered_at is not None


def test_dispatch_requires_persisted_success_before_ack():
    from types import SimpleNamespace
    from llm_bawt.service.approval_continuations import _continuation_identity

    store = _store()
    row = _ready_row(store)
    identity = _continuation_identity(row)

    class PersistedService(FakeService):
        def __init__(self):
            super().__init__()
            self.turn = None
            self._turn_log_store = SimpleNamespace(get_turn=lambda key: self.turn)

        async def chat_completion_stream(self, request):
            assert request.user_message_id == identity["user_message_id"]
            assert request.assistant_message_id == identity["assistant_message_id"]
            assert len(request.user_message_id) == len(request.assistant_message_id) == 36
            self.turn = SimpleNamespace(ended_at=1, status="ok", error_text=None)
            yield "data: [DONE]\n\n"

    service = PersistedService()
    asyncio.run(dispatch_mcp_result_continuation(service, store, row))
    assert store.get_request(row.id).continuation_state == CONT_DELIVERED


def test_existing_failed_turn_is_never_replayed():
    from types import SimpleNamespace

    store = _store()
    row = _ready_row(store)
    service = FakeService()
    service._turn_log_store = SimpleNamespace(get_turn=lambda key: SimpleNamespace(
        ended_at=1, status="error", error_text="prior persistence failure"))
    with pytest.raises(RuntimeError, match="manual reconciliation"):
        asyncio.run(dispatch_mcp_result_continuation(service, store, row))
    assert not service.requests
    assert store.get_request(row.id).continuation_state != CONT_DELIVERED


def test_dispatch_failure_reschedules_for_retry():
    store = _store()
    row = _ready_row(store)
    service = FakeService(RuntimeError("bridge offline"))

    with pytest.raises(RuntimeError, match="bridge offline"):
        asyncio.run(dispatch_mcp_result_continuation(service, store, row))

    persisted = store.get_request(row.id)
    assert persisted.continuation_state == CONT_PENDING
    assert persisted.continuation_last_error == "bridge offline"
    assert persisted.continuation_next_attempt_at is not None
