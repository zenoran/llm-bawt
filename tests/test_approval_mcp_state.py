"""Tests for TASK-639 MCP-kind approval state machine.

Exercises the new ``ToolApprovalPolicyStore`` methods against a real in-memory
SQLite engine:

* ``record_mcp_request`` — KIND_MCP row creation + idempotency
* ``claim_mcp_execution`` — atomic pending→running with lease semantics
* ``complete_mcp_execution`` — exactly-once terminal transitions
* ``enqueue_continuation``, ``claim_continuation``,
  ``mark_continuation_delivered``, ``mark_continuation_failed`` — outbox lifecycle
* ``find_pending_continuations`` — worker query

No Redis, no HTTP, no live SDK. Same _store helper pattern as
test_approval_persist.py.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

try:
    from sqlmodel import Session, create_engine, select
    from sqlalchemy.pool import StaticPool

    from llm_bawt.approval_policies import (
        ApprovalPersistError,
        CONT_DELIVERED,
        CONT_DISPATCHING,
        CONT_FAILED,
        CONT_NOT_NEEDED,
        CONT_PENDING,
        EXEC_FAILED,
        EXEC_PENDING,
        EXEC_RUNNING,
        EXEC_SKIPPED,
        EXEC_SUCCEEDED,
        KIND_HARNESS,
        KIND_MCP,
        REQ_APPROVED,
        REQ_PENDING,
        ToolApprovalPolicyStore,
        ToolApprovalRequest,
    )

    _OK = True
    _SKIP_REASON = ""
except Exception as exc:  # noqa: BLE001
    _OK = False
    _SKIP_REASON = f"llm_bawt deps unavailable ({exc}); run in app container"


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
    base = dict(
        request_id="req-mcp-1",
        tool_use_id="toolu_01ABC",
        mcp_server="bawthub",
        bot_id="snark",
        user_id="nick",
        turn_id="turn-1",
        backend="claude-code",
        tool_name="ops_run",
        tool_arguments={"operation": "llm-bawt.restart-app", "args": {}},
        subject="operation=llm-bawt.restart-app args={}",
        grant_key="grant-1",
        policy_id="pol-mcp-1",
        severity="high",
        prompt="Approve restart?",
        invocation_hash="deadbeef" * 8,
        continuation_capable=True,
    )
    base.update(over)
    return base


def _approve(store, request_id):
    """Flip a request to REQ_APPROVED (simulate the resolve path)."""
    from llm_bawt.approval_policies import REQ_APPROVED as _APPROVED
    with Session(store.engine) as session:
        row = session.get(ToolApprovalRequest, request_id)
        row.status = _APPROVED
        session.add(row)
        session.commit()


# ---- record_mcp_request ----------------------------------------------------

def test_record_mcp_writes_kind_mcp_row_with_pending_execution():
    store = _store()
    row = store.record_mcp_request(**_mcp_req())
    assert row is not None
    assert row.id == "req-mcp-1"
    assert row.request_kind == KIND_MCP
    assert row.tool_use_id == "toolu_01ABC"
    assert row.mcp_server == "bawthub"
    assert row.invocation_hash == "deadbeef" * 8
    assert row.status == REQ_PENDING
    assert row.execution_state == EXEC_PENDING
    assert row.continuation_state == CONT_NOT_NEEDED
    assert row.continuation_capable is True


def test_record_mcp_idempotent_on_request_id():
    store = _store()
    first = store.record_mcp_request(**_mcp_req())
    second = store.record_mcp_request(**_mcp_req(tool_name="ops_list_operations"))
    assert first.id == second.id == "req-mcp-1"
    assert second.tool_name == "ops_run"  # original preserved
    with Session(store.engine) as s:
        assert len(s.exec(select(ToolApprovalRequest)).all()) == 1


def test_record_mcp_none_engine_raises():
    store = object.__new__(ToolApprovalPolicyStore)
    store.engine = None
    raised = False
    try:
        store.record_mcp_request(**_mcp_req())
    except ApprovalPersistError:
        raised = True
    assert raised


# ---- claim_mcp_execution ---------------------------------------------------

def test_claim_execution_fails_when_not_approved():
    store = _store()
    store.record_mcp_request(**_mcp_req())
    # Still REQ_PENDING — must not be claimable.
    assert store.claim_mcp_execution("req-mcp-1") is None


def test_claim_execution_succeeds_on_approved_pending():
    store = _store()
    store.record_mcp_request(**_mcp_req())
    _approve(store, "req-mcp-1")
    row = store.claim_mcp_execution("req-mcp-1")
    assert row is not None
    assert row.execution_state == EXEC_RUNNING
    assert row.execution_started_at is not None
    assert row.execution_attempts == 1


def test_claim_execution_rejects_double_claim_within_lease():
    store = _store()
    store.record_mcp_request(**_mcp_req())
    _approve(store, "req-mcp-1")
    assert store.claim_mcp_execution("req-mcp-1", lease_seconds=60) is not None
    # Second immediate claim must fail — lease held.
    assert store.claim_mcp_execution("req-mcp-1", lease_seconds=60) is None


def test_claim_execution_reclaims_after_stale_lease():
    store = _store()
    store.record_mcp_request(**_mcp_req())
    _approve(store, "req-mcp-1")
    store.claim_mcp_execution("req-mcp-1", lease_seconds=60)
    # Backdate the lease so it's expired, no result yet.
    with Session(store.engine) as s:
        r = s.get(ToolApprovalRequest, "req-mcp-1")
        r.execution_started_at = datetime.now(timezone.utc) - timedelta(seconds=120)
        s.add(r); s.commit()
    reclaimed = store.claim_mcp_execution("req-mcp-1", lease_seconds=60)
    assert reclaimed is not None
    assert reclaimed.execution_attempts == 2


def test_claim_execution_never_reclaims_terminal_row():
    store = _store()
    store.record_mcp_request(**_mcp_req())
    _approve(store, "req-mcp-1")
    store.claim_mcp_execution("req-mcp-1")
    store.complete_mcp_execution(
        "req-mcp-1", result_json='{"ok":true}', is_error=False,
    )
    # Terminal — even with expired lease, cannot reclaim.
    assert store.claim_mcp_execution("req-mcp-1", lease_seconds=0) is None


# ---- complete_mcp_execution ------------------------------------------------

def test_complete_execution_success_sets_result_and_state():
    store = _store()
    store.record_mcp_request(**_mcp_req())
    _approve(store, "req-mcp-1")
    store.claim_mcp_execution("req-mcp-1")
    row = store.complete_mcp_execution(
        "req-mcp-1", result_json='{"job_id":"j-1"}', is_error=False,
    )
    assert row.execution_state == EXEC_SUCCEEDED
    assert row.result_json == '{"job_id":"j-1"}'
    assert row.result_is_error is False
    assert row.execution_finished_at is not None


def test_complete_execution_error_sets_failed():
    store = _store()
    store.record_mcp_request(**_mcp_req())
    _approve(store, "req-mcp-1")
    store.claim_mcp_execution("req-mcp-1")
    row = store.complete_mcp_execution(
        "req-mcp-1", result_json='{"error":"nope"}', is_error=True,
        error="ssh: connection refused",
    )
    assert row.execution_state == EXEC_FAILED
    assert row.result_is_error is True
    assert "ssh: connection refused" in (row.execution_error or "")


def test_complete_execution_is_idempotent_on_terminal():
    store = _store()
    store.record_mcp_request(**_mcp_req())
    _approve(store, "req-mcp-1")
    store.claim_mcp_execution("req-mcp-1")
    first = store.complete_mcp_execution(
        "req-mcp-1", result_json='{"a":1}', is_error=False,
    )
    # Second call with different data must NOT overwrite the stored result.
    second = store.complete_mcp_execution(
        "req-mcp-1", result_json='{"b":2}', is_error=True,
    )
    assert second.result_json == '{"a":1}'
    assert second.execution_state == EXEC_SUCCEEDED
    assert first.execution_finished_at == second.execution_finished_at


# ---- continuation outbox ---------------------------------------------------

def test_enqueue_continuation_noop_when_not_capable():
    store = _store()
    store.record_mcp_request(**_mcp_req(continuation_capable=False))
    _approve(store, "req-mcp-1")
    store.claim_mcp_execution("req-mcp-1")
    store.complete_mcp_execution(
        "req-mcp-1", result_json='{"a":1}', is_error=False,
    )
    row = store.enqueue_continuation("req-mcp-1")
    assert row.continuation_state == CONT_NOT_NEEDED


def test_enqueue_continuation_moves_to_pending_when_capable():
    store = _store()
    store.record_mcp_request(**_mcp_req())
    _approve(store, "req-mcp-1")
    store.claim_mcp_execution("req-mcp-1")
    store.complete_mcp_execution(
        "req-mcp-1", result_json='{"a":1}', is_error=False,
    )
    row = store.enqueue_continuation("req-mcp-1")
    assert row.continuation_state == CONT_PENDING
    assert row.continuation_next_attempt_at is not None


def test_enqueue_continuation_ignores_incomplete_execution():
    store = _store()
    store.record_mcp_request(**_mcp_req())
    _approve(store, "req-mcp-1")
    # execution never claimed/completed
    row = store.enqueue_continuation("req-mcp-1")
    assert row.continuation_state == CONT_NOT_NEEDED


def test_claim_continuation_moves_pending_to_dispatching():
    store = _store()
    store.record_mcp_request(**_mcp_req())
    _approve(store, "req-mcp-1")
    store.claim_mcp_execution("req-mcp-1")
    store.complete_mcp_execution(
        "req-mcp-1", result_json='{"a":1}', is_error=False,
    )
    store.enqueue_continuation("req-mcp-1")
    row = store.claim_continuation("req-mcp-1")
    assert row is not None
    assert row.continuation_state == CONT_DISPATCHING
    assert row.continuation_attempts == 1


def test_claim_continuation_rejects_delivered():
    store = _store()
    store.record_mcp_request(**_mcp_req())
    _approve(store, "req-mcp-1")
    store.claim_mcp_execution("req-mcp-1")
    store.complete_mcp_execution(
        "req-mcp-1", result_json='{"a":1}', is_error=False,
    )
    store.enqueue_continuation("req-mcp-1")
    store.claim_continuation("req-mcp-1")
    store.mark_continuation_delivered("req-mcp-1")
    assert store.claim_continuation("req-mcp-1") is None


def test_mark_continuation_delivered_is_idempotent():
    store = _store()
    store.record_mcp_request(**_mcp_req())
    _approve(store, "req-mcp-1")
    store.claim_mcp_execution("req-mcp-1")
    store.complete_mcp_execution(
        "req-mcp-1", result_json='{"a":1}', is_error=False,
    )
    store.enqueue_continuation("req-mcp-1")
    store.claim_continuation("req-mcp-1")
    first = store.mark_continuation_delivered("req-mcp-1")
    second = store.mark_continuation_delivered("req-mcp-1")
    assert first.continuation_state == second.continuation_state == CONT_DELIVERED
    assert first.continuation_delivered_at == second.continuation_delivered_at


def test_mark_continuation_failed_reschedules_with_backoff():
    store = _store()
    store.record_mcp_request(**_mcp_req())
    _approve(store, "req-mcp-1")
    store.claim_mcp_execution("req-mcp-1")
    store.complete_mcp_execution(
        "req-mcp-1", result_json='{"a":1}', is_error=False,
    )
    store.enqueue_continuation("req-mcp-1")
    store.claim_continuation("req-mcp-1")
    row = store.mark_continuation_failed(
        "req-mcp-1", error="redis offline", backoff_seconds=10,
    )
    assert row.continuation_state == CONT_PENDING  # rescheduled
    assert row.continuation_last_error == "redis offline"
    # SQLite strips tzinfo on round-trip; coerce for the compare.
    next_at = row.continuation_next_attempt_at
    if next_at.tzinfo is None:
        next_at = next_at.replace(tzinfo=timezone.utc)
    assert next_at > datetime.now(timezone.utc)


def test_mark_continuation_failed_terminates_after_max_attempts():
    store = _store()
    store.record_mcp_request(**_mcp_req())
    _approve(store, "req-mcp-1")
    store.claim_mcp_execution("req-mcp-1")
    store.complete_mcp_execution(
        "req-mcp-1", result_json='{"a":1}', is_error=False,
    )
    store.enqueue_continuation("req-mcp-1")
    # Simulate 5 failed dispatches.
    for _ in range(5):
        store.claim_continuation("req-mcp-1", lease_seconds=0)
        store.mark_continuation_failed(
            "req-mcp-1", error="broken", max_attempts=5, backoff_seconds=1,
        )
    row = store.claim_continuation("req-mcp-1", lease_seconds=0)  # last claim
    # After hitting max, the mark on the LAST attempt sets CONT_FAILED
    store.mark_continuation_failed(
        "req-mcp-1", error="broken", max_attempts=5, backoff_seconds=1,
    )
    with Session(store.engine) as s:
        final = s.get(ToolApprovalRequest, "req-mcp-1")
    assert final.continuation_state == CONT_FAILED
    assert final.continuation_next_attempt_at is None


# ---- find_pending_continuations --------------------------------------------

def test_find_pending_returns_due_pending_rows():
    store = _store()
    store.record_mcp_request(**_mcp_req(request_id="req-1"))
    store.record_mcp_request(**_mcp_req(request_id="req-2"))
    for rid in ("req-1", "req-2"):
        _approve(store, rid)
        store.claim_mcp_execution(rid)
        store.complete_mcp_execution(rid, result_json='{"ok":1}', is_error=False)
        store.enqueue_continuation(rid)
    due = store.find_pending_continuations(limit=10)
    assert {r.id for r in due} == {"req-1", "req-2"}


def test_find_pending_excludes_future_attempts():
    store = _store()
    store.record_mcp_request(**_mcp_req(request_id="req-1"))
    _approve(store, "req-1")
    store.claim_mcp_execution("req-1")
    store.complete_mcp_execution("req-1", result_json='{"ok":1}', is_error=False)
    store.enqueue_continuation("req-1")
    # Push next_attempt_at into the future.
    with Session(store.engine) as s:
        r = s.get(ToolApprovalRequest, "req-1")
        r.continuation_next_attempt_at = datetime.now(timezone.utc) + timedelta(minutes=5)
        s.add(r); s.commit()
    assert store.find_pending_continuations(limit=10) == []


def test_find_pending_includes_stale_dispatching():
    store = _store()
    store.record_mcp_request(**_mcp_req(request_id="req-1"))
    _approve(store, "req-1")
    store.claim_mcp_execution("req-1")
    store.complete_mcp_execution("req-1", result_json='{"ok":1}', is_error=False)
    store.enqueue_continuation("req-1")
    store.claim_continuation("req-1")
    # Backdate the dispatching lease.
    with Session(store.engine) as s:
        r = s.get(ToolApprovalRequest, "req-1")
        r.continuation_next_attempt_at = datetime.now(timezone.utc) - timedelta(minutes=10)
        s.add(r); s.commit()
    due = store.find_pending_continuations(
        limit=10, include_dispatching_older_than_s=60,
    )
    assert len(due) == 1
    assert due[0].id == "req-1"


# ---- harness-kind backward compatibility -----------------------------------

def test_existing_record_request_still_defaults_to_kind_harness():
    """TASK-639 must not change legacy harness behavior: record_request() rows
    still land as KIND_HARNESS with EXEC_NOT_APPLICABLE."""
    from llm_bawt.approval_policies import EXEC_NOT_APPLICABLE
    store = _store()
    row = store.record_request(
        request_id="tuid-legacy",
        bot_id="snark",
        user_id="nick",
        turn_id="turn-1",
        backend="claude-code",
        tool_name="Bash",
        tool_arguments={"command": "make rebuild-prod"},
        subject="make rebuild-prod",
        grant_key="g-1",
        policy_id="p-1",
        severity="medium",
        prompt="",
    )
    assert row.request_kind == KIND_HARNESS
    assert row.execution_state == EXEC_NOT_APPLICABLE
    assert row.continuation_state == CONT_NOT_NEEDED
    assert row.tool_use_id is None


if __name__ == "__main__":
    import sys
    import traceback

    if not _OK:
        print(f"SKIP test_approval_mcp_state: {_SKIP_REASON}")
        sys.exit(0)

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed ({len(fns)} total)")
    sys.exit(1 if failed else 0)
