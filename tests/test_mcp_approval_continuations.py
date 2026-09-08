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


def _ready_row(store):
    args = {"operation": "llm-bawt.restart-app", "args": {}}
    row = store.record_mcp_request(
        request_id="req-mcp-1",
        tool_use_id="toolu-1",
        mcp_server="bawthub",
        bot_id="snark",
        user_id="nick",
        turn_id="turn-1",
        backend="claude-code",
        tool_name="ops_run",
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
        result_json=json.dumps({"job_id": "job-1", "state": "queued"}),
        is_error=False,
    )
    store.enqueue_continuation(row.id)
    return store.claim_continuation(row.id)


class FakeService:
    def __init__(self, error=None):
        self.requests = []
        self.error = error

    async def chat_completion_stream(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        yield "data: [DONE]\n\n"


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
