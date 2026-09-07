"""Durable approval request lifecycle and execution claims."""
from __future__ import annotations
import json
import logging
from datetime import timedelta
from typing import Any
from hashlib import sha256
from sqlalchemy import and_, or_, update, func
from sqlmodel import Session, select
from .approval_models import (
    ApprovalPersistError,
    ApprovalStoreUnavailable,
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
    REQ_CANCELLED,
    REQ_DENIED,
    REQ_EXPIRED,
    REQ_PENDING,
    REQ_RESPONDED,
    REQ_SUPERSEDED,
    ToolApprovalRequest,
    _new_id,
    _utcnow,
)


logger = logging.getLogger(__name__)


def _continuation_id(request_id):
    return "approval-cont-" + sha256(request_id.encode()).hexdigest()[:32]

class ApprovalRequestStoreMixin:
    # ---- request lifecycle (audit) -----------------------------------------

    def record_request(
        self,
        *,
        request_id: str,
        bot_id: str,
        user_id: str,
        turn_id: str,
        backend: str,
        tool_name: str,
        tool_arguments: dict[str, Any],
        subject: str,
        grant_key: str,
        policy_id: str | None,
        severity: str,
        prompt: str,
        trigger_message_id: str | None = None,
        session_key: str | None = None,
        session_id: str | None = None,
    ) -> ToolApprovalRequest:
        """Persist a new pending approval. Idempotent on request_id.

        Returns the committed (or pre-existing) row. Raises
        ``ApprovalPersistError`` if the request cannot be durably committed —
        no DB engine, or the insert/commit failed. TASK-306 Section A: the
        single caller treats a raise as a hard, agent-visible failure and must
        NOT swallow it.
        """
        if self.engine is None:
            raise ApprovalPersistError(
                f"approval store has no DB engine; cannot persist request {request_id}"
            )
        try:
            with Session(self.engine) as session:
                existing = session.get(ToolApprovalRequest, request_id)
                if existing is not None:
                    return existing
                row = ToolApprovalRequest(
                    id=request_id,
                    bot_id=(bot_id or "unknown").strip() or "unknown",
                    user_id=(user_id or "unknown").strip() or "unknown",
                    turn_id=(turn_id or "unknown").strip() or "unknown",
                    trigger_message_id=trigger_message_id or None,
                    session_key=session_key or None,
                    caller_context_json=json.dumps({"session_id": session_id}) if session_id else None,
                    backend=backend or "claude-code",
                    tool_name=tool_name or "",
                    tool_arguments_json=json.dumps(
                        tool_arguments if isinstance(tool_arguments, dict) else {"value": tool_arguments},
                        ensure_ascii=False, default=str,
                    ),
                    subject=subject or "",
                    grant_key=grant_key or "",
                    policy_id=policy_id,
                    severity=severity or "medium",
                    prompt=prompt or "",
                    status=REQ_PENDING,
                )
                session.add(row)
                session.commit()
                session.refresh(row)
                return row
        except ApprovalPersistError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to record approval request id=%s", request_id)
            raise ApprovalPersistError(
                f"insert failed for approval request {request_id}: {exc}"
            ) from exc

    def get_request(self, request_id: str) -> ToolApprovalRequest | None:
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            return session.get(ToolApprovalRequest, request_id)

    def resolve_request(
        self, request_id: str, *, status: str, resolved_by: str | None = None,
        resolved_turn_id: str | None = None, message: str = "",
        continuation_owner: str | None = None,
    ) -> ToolApprovalRequest | None:
        """First terminal decision wins, including message and dispatch ownership."""
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        if status not in (REQ_APPROVED, REQ_DENIED, REQ_CANCELLED, REQ_RESPONDED, REQ_EXPIRED, REQ_SUPERSEDED):
            raise ValueError("Invalid approval resolution status")
        with Session(self.engine) as session:
            values = dict(status=status, resolved_at=_utcnow(), resolved_by=resolved_by,
                          resolved_turn_id=resolved_turn_id, resolution_message=message)
            if continuation_owner is not None:
                if continuation_owner not in ("server", "client", "none"):
                    raise ValueError("Invalid continuation owner")
                values["continuation_owner"] = continuation_owner
            session.exec(update(ToolApprovalRequest).where(
                ToolApprovalRequest.id == request_id, ToolApprovalRequest.status == REQ_PENDING,
            ).values(**values))
            session.commit()
            return session.get(ToolApprovalRequest, request_id)

    # ---- MCP-kind execution + continuation state machine (TASK-639) --------

    def record_mcp_request(
        self,
        *,
        request_id: str,
        tool_use_id: str | None,
        mcp_server: str,
        bot_id: str,
        user_id: str,
        turn_id: str,
        backend: str,
        tool_name: str,
        tool_arguments: dict[str, Any],
        subject: str,
        grant_key: str,
        policy_id: str | None,
        severity: str,
        prompt: str,
        invocation_hash: str,
        caller_context_json: str | None = None,
        operations_snapshot: dict[str, Any] | None = None,
        continuation_capable: bool = False,
        trigger_message_id: str | None = None,
        session_key: str | None = None,
    ) -> ToolApprovalRequest:
        """Persist a new MCP-kind gated request. Idempotent on request_id.

        Same durability contract as :meth:`record_request` — raises
        ``ApprovalPersistError`` on missing engine / insert failure so the MCP
        interceptor never returns ``approval_required`` to the caller unless
        the row is committed. Starts in status=REQ_PENDING (approval flow) +
        execution_state=EXEC_PENDING (MCP execution has not run yet).
        """
        if self.engine is None:
            raise ApprovalPersistError(
                f"approval store has no DB engine; cannot persist MCP request {request_id}"
            )
        try:
            with Session(self.engine) as session:
                existing = session.get(ToolApprovalRequest, request_id)
                if existing is not None:
                    return existing
                row = ToolApprovalRequest(
                    id=request_id,
                    bot_id=(bot_id or "unknown").strip() or "unknown",
                    user_id=(user_id or "unknown").strip() or "unknown",
                    turn_id=(turn_id or "unknown").strip() or "unknown",
                    trigger_message_id=trigger_message_id or None,
                    session_key=session_key or None,
                    backend=backend or "claude-code",
                    tool_name=tool_name or "",
                    tool_arguments_json=json.dumps(
                        tool_arguments if isinstance(tool_arguments, dict) else {"value": tool_arguments},
                        ensure_ascii=False, default=str,
                    ),
                    subject=subject or "",
                    grant_key=grant_key or "",
                    policy_id=policy_id,
                    severity=severity or "medium",
                    prompt=prompt or "",
                    status=REQ_PENDING,
                    request_kind=KIND_MCP,
                    tool_use_id=tool_use_id,
                    mcp_server=mcp_server or "",
                    invocation_hash=invocation_hash,
                    caller_context_json=caller_context_json,
                    operations_snapshot_json=json.dumps(operations_snapshot, sort_keys=True) if operations_snapshot is not None else None,
                    continuation_owner="server",
                    continuation_id=_continuation_id(request_id),
                    continuation_capable=bool(continuation_capable),
                    execution_state=EXEC_PENDING,
                    continuation_state=CONT_NOT_NEEDED,
                )
                session.add(row)
                session.commit()
                session.refresh(row)
                return row
        except ApprovalPersistError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to record MCP approval request id=%s", request_id)
            raise ApprovalPersistError(
                f"insert failed for MCP approval request {request_id}: {exc}"
            ) from exc

    def claim_mcp_execution(
        self,
        request_id: str,
        *,
        lease_seconds: int = 300,
    ) -> ToolApprovalRequest | None:
        """Atomically claim an approved MCP request for execution.

        Returns the claimed row (execution_state=EXEC_RUNNING) or None if:
          * request does not exist
          * status is not REQ_APPROVED
          * execution has already completed (EXEC_SUCCEEDED / EXEC_FAILED / EXEC_SKIPPED)
          * another worker holds a fresh RUNNING lease (started_at within lease_seconds)

        Expired generic claims become uncertain failures without replay. Only
        ops_run with an immutable snapshot may re-enter execution with the same
        durable approval idempotency key. Every claimant receives a fencing token.
        """
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        now = _utcnow()
        stale_before = now - timedelta(seconds=max(0, lease_seconds))
        with Session(self.engine) as session:
            claimable = and_(
                ToolApprovalRequest.id == request_id,
                ToolApprovalRequest.status == REQ_APPROVED,
                ToolApprovalRequest.request_kind == KIND_MCP,
                ToolApprovalRequest.result_json.is_(None),
                or_(
                    ToolApprovalRequest.execution_state == EXEC_PENDING,
                    and_(
                        ToolApprovalRequest.execution_state == EXEC_RUNNING,
                        or_(
                            ToolApprovalRequest.execution_started_at.is_(None),
                            ToolApprovalRequest.execution_started_at <= stale_before,
                        ),
                    ),
                ),
            )
            stmt = (
                update(ToolApprovalRequest)
                .where(claimable)
                .values(
                    execution_state=EXEC_RUNNING,
                    execution_claim_token=_new_id(),
                    execution_started_at=now,
                    execution_attempts=ToolApprovalRequest.execution_attempts + 1,
                )
            )
            result = session.exec(stmt)
            session.commit()
            if int(result.rowcount or 0) != 1:
                return None
            row = session.get(ToolApprovalRequest, request_id)
        if row.execution_attempts > 1 and (row.tool_name != "ops_run" or not row.operations_snapshot_json):
            self.complete_mcp_execution(
                request_id, result_json=json.dumps({"status": "failed", "uncertain": True,
                    "error": "Execution lease expired. Side effects may have occurred; manual reconciliation required. Do not replay."}),
                is_error=True, error="Uncertain execution; manual reconciliation required",
                claim_token=row.execution_claim_token,
            )
            return None
        return row

    def complete_mcp_execution(
        self, request_id: str, *, result_json: str, is_error: bool,
        error: str | None = None, skipped: bool = False,
        claim_token: str | None = None,
    ) -> ToolApprovalRequest | None:
        """CAS result and outbox enqueue in ONE transaction; fence stale workers."""
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            row = session.get(ToolApprovalRequest, request_id)
            if row is None:
                return None
            if row.result_json is not None:
                return row
            if skipped and row.status == REQ_APPROVED:
                return row
            now = _utcnow()
            conditions = [ToolApprovalRequest.id == request_id, ToolApprovalRequest.result_json.is_(None)]
            if claim_token is not None:
                conditions.append(ToolApprovalRequest.execution_claim_token == claim_token)
            if skipped:
                conditions.append(ToolApprovalRequest.execution_state == EXEC_PENDING)
            else:
                conditions.append(ToolApprovalRequest.execution_state == EXEC_RUNNING)
            enqueue = bool(row.continuation_capable and row.status != REQ_CANCELLED)
            result = session.exec(update(ToolApprovalRequest).where(*conditions).values(
                execution_state=EXEC_SKIPPED if skipped else (EXEC_FAILED if is_error else EXEC_SUCCEEDED),
                execution_finished_at=now, result_json=result_json, result_is_error=bool(is_error),
                execution_error=error, execution_claim_token=None,
                continuation_state=CONT_PENDING if enqueue else CONT_NOT_NEEDED,
                continuation_next_attempt_at=now if enqueue else None,
                continuation_id=row.continuation_id or _continuation_id(request_id),
            ))
            session.commit()
            session.expire_all()
            if int(result.rowcount or 0) != 1:
                raise ApprovalPersistError("Execution claim lost; result not overwritten")
            return session.get(ToolApprovalRequest, request_id)

    def enqueue_continuation(
        self,
        request_id: str,
    ) -> ToolApprovalRequest | None:
        """Mark a completed MCP execution as CONT_PENDING for the outbox worker.

        No-op if the row is not continuation_capable, has no terminal execution
        result, or is already in a continuation lifecycle. Idempotent.
        """
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            row = session.get(ToolApprovalRequest, request_id)
            if row is None:
                return None
            if not row.continuation_capable or row.status == REQ_CANCELLED:
                return row
            if row.execution_state not in (EXEC_SUCCEEDED, EXEC_FAILED, EXEC_SKIPPED):
                return row
            # Already enqueued / in-flight / delivered — no-op.
            if row.continuation_state != CONT_NOT_NEEDED:
                return row
            row.continuation_id = row.continuation_id or _continuation_id(request_id)
            row.continuation_state = CONT_PENDING
            row.continuation_next_attempt_at = _utcnow()
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def claim_continuation(
        self,
        request_id: str,
        *,
        lease_seconds: int = 60,
    ) -> ToolApprovalRequest | None:
        """Atomically transition CONT_PENDING → CONT_DISPATCHING for the worker.

        Same lease semantics as ``claim_mcp_execution``: a DISPATCHING row with
        an expired lease and no continuation_delivered_at may be reclaimed.
        """
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        now = _utcnow()
        stale_before = now - timedelta(seconds=max(0, lease_seconds))
        with Session(self.engine) as session:
            claimable = and_(
                ToolApprovalRequest.id == request_id,
                ToolApprovalRequest.continuation_delivered_at.is_(None),
                or_(
                    and_(ToolApprovalRequest.continuation_state == CONT_PENDING,
                         or_(ToolApprovalRequest.continuation_next_attempt_at.is_(None),
                             ToolApprovalRequest.continuation_next_attempt_at <= now)),
                    and_(
                        ToolApprovalRequest.continuation_state == CONT_DISPATCHING,
                        or_(
                            ToolApprovalRequest.continuation_next_attempt_at.is_(None),
                            ToolApprovalRequest.continuation_next_attempt_at <= stale_before,
                        ),
                    ),
                ),
            )
            stmt = (
                update(ToolApprovalRequest)
                .where(claimable)
                .values(
                    continuation_state=CONT_DISPATCHING,
                    continuation_claim_token=_new_id(),
                    continuation_id=_continuation_id(request_id),
                    continuation_next_attempt_at=now,
                    continuation_attempts=ToolApprovalRequest.continuation_attempts + 1,
                )
            )
            result = session.exec(stmt)
            session.commit()
            if int(result.rowcount or 0) != 1:
                return None
            return session.get(ToolApprovalRequest, request_id)

    def mark_continuation_delivered(self, request_id: str, *, claim_token: str | None = None):
        return self._finish_continuation(request_id, claim_token=claim_token)

    def mark_continuation_failed(self, request_id: str, *, error: str, max_attempts: int = 5,
                                 backoff_seconds: int = 30, claim_token: str | None = None):
        return self._finish_continuation(request_id, claim_token=claim_token, error=error,
                                        max_attempts=max_attempts, backoff_seconds=backoff_seconds)

    def _finish_continuation(self, request_id, *, claim_token=None, error=None,
                             max_attempts=5, backoff_seconds=30):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            row = session.get(ToolApprovalRequest, request_id)
            if row is None or row.continuation_state != CONT_DISPATCHING:
                return row
            expected = claim_token if claim_token is not None else row.continuation_claim_token
            now = _utcnow()
            failed = error is not None
            terminal_failure = failed and row.continuation_attempts >= max_attempts
            values = dict(
                continuation_state=(CONT_FAILED if terminal_failure else CONT_PENDING) if failed else CONT_DELIVERED,
                continuation_last_error=str(error)[:2000] if failed else None,
                continuation_delivered_at=None if failed else now,
                continuation_claim_token=None,
                continuation_next_attempt_at=(now + timedelta(seconds=min(backoff_seconds * 2 ** max(0, row.continuation_attempts - 1), 600))) if failed and not terminal_failure else None,
            )
            session.exec(update(ToolApprovalRequest).where(
                ToolApprovalRequest.id == request_id,
                ToolApprovalRequest.continuation_state == CONT_DISPATCHING,
                ToolApprovalRequest.continuation_claim_token == expected,
            ).values(**values))
            session.commit()
            session.expire_all()
            return session.get(ToolApprovalRequest, request_id)

    def renew_continuation_claim(self, request_id, claim_token):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            result = session.exec(update(ToolApprovalRequest).where(
                ToolApprovalRequest.id == request_id,
                ToolApprovalRequest.continuation_state == CONT_DISPATCHING,
                ToolApprovalRequest.continuation_claim_token == claim_token,
            ).values(continuation_next_attempt_at=_utcnow()))
            session.commit()
            return int(result.rowcount or 0) == 1

    def find_pending_continuations(
        self,
        *,
        limit: int = 50,
        include_dispatching_older_than_s: int = 120,
    ) -> list[ToolApprovalRequest]:
        """Outbox worker query: continuations ready for dispatch.

        Includes CONT_PENDING rows whose next_attempt_at is due, plus any
        CONT_DISPATCHING rows whose lease expired without delivery (crash /
        app restart mid-dispatch).
        """
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        now = _utcnow()
        stale_before = now - timedelta(seconds=include_dispatching_older_than_s)
        with Session(self.engine) as session:
            stmt = (
                select(ToolApprovalRequest)
                .where(
                    (
                        (ToolApprovalRequest.continuation_state == CONT_PENDING)
                        & (
                            (ToolApprovalRequest.continuation_next_attempt_at.is_(None))
                            | (ToolApprovalRequest.continuation_next_attempt_at <= now)
                        )
                    )
                    | (
                        (ToolApprovalRequest.continuation_state == CONT_DISPATCHING)
                        & (ToolApprovalRequest.continuation_delivered_at.is_(None))
                        & (ToolApprovalRequest.continuation_next_attempt_at <= stale_before)
                    )
                )
                .order_by(ToolApprovalRequest.continuation_next_attempt_at)
                .limit(limit)
            )
            return list(session.exec(stmt).all())

    def list_requests(
        self,
        *,
        status: str | None = None,
        bot_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ToolApprovalRequest]:
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            stmt = select(ToolApprovalRequest)
            if status:
                stmt = stmt.where(ToolApprovalRequest.status == status)
            if bot_id:
                stmt = stmt.where(ToolApprovalRequest.bot_id == bot_id)
            stmt = stmt.order_by(ToolApprovalRequest.created_at.desc(), ToolApprovalRequest.id).offset(max(0, offset)).limit(min(max(1, limit), 200))
            return list(session.exec(stmt).all())


    def count_requests(self, *, status=None, bot_id=None):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            stmt = select(func.count()).select_from(ToolApprovalRequest)
            if status:
                stmt = stmt.where(ToolApprovalRequest.status == status)
            if bot_id:
                stmt = stmt.where(ToolApprovalRequest.bot_id == bot_id)
            return session.exec(stmt).one()

    def find_recoverable_mcp_requests(self, *, limit=20, lease_seconds=300):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            return list(session.exec(select(ToolApprovalRequest).where(
                ToolApprovalRequest.request_kind == KIND_MCP,
                ToolApprovalRequest.status != REQ_PENDING,
                ToolApprovalRequest.result_json.is_(None),
                or_(ToolApprovalRequest.execution_state == EXEC_PENDING,
                    and_(ToolApprovalRequest.execution_state == EXEC_RUNNING,
                         or_(ToolApprovalRequest.execution_started_at.is_(None),
                             ToolApprovalRequest.execution_started_at <= _utcnow() - timedelta(seconds=lease_seconds))))
            ).order_by(ToolApprovalRequest.created_at).limit(limit)).all())

    def prepare_harness_continuation(self, request_id):
        """Durably enqueue server-owned continuation without a fire-and-forget gap."""
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            session.exec(update(ToolApprovalRequest).where(
                ToolApprovalRequest.id == request_id,
                ToolApprovalRequest.request_kind == KIND_HARNESS,
                ToolApprovalRequest.status.in_([REQ_APPROVED, REQ_DENIED, REQ_RESPONDED]),
                ToolApprovalRequest.continuation_owner == "server",
                ToolApprovalRequest.continuation_state == CONT_NOT_NEEDED,
            ).values(continuation_state=CONT_PENDING, continuation_next_attempt_at=_utcnow(),
                     continuation_id=_continuation_id(request_id)))
            session.commit()
            return session.get(ToolApprovalRequest, request_id)

    def set_grant_state(self, request_id, state, error=None):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            allowed_prior = ["pending", "failed"] if state == "failed" else ["dispatching"]
            session.exec(update(ToolApprovalRequest).where(
                ToolApprovalRequest.id == request_id,
                ToolApprovalRequest.grant_state.in_(allowed_prior),
            ).values(grant_state=state, grant_error=error))
            session.commit()

    def claim_client_grant(self, request_id):
        """One publisher; an interrupted send is uncertain, never silently re-granted."""
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            result = session.exec(update(ToolApprovalRequest).where(
                ToolApprovalRequest.id == request_id,
                ToolApprovalRequest.grant_state.in_(["pending", "failed"]),
            ).values(grant_state="dispatching", grant_error=None))
            session.commit()
            return int(result.rowcount or 0) == 1

    def find_stranded_harness_continuations(self, *, limit=20):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            return list(session.exec(select(ToolApprovalRequest).where(
                ToolApprovalRequest.request_kind == KIND_HARNESS,
                ToolApprovalRequest.continuation_owner == "server",
                ToolApprovalRequest.continuation_state == CONT_NOT_NEEDED,
                ToolApprovalRequest.status.in_([REQ_APPROVED, REQ_DENIED, REQ_RESPONDED]),
            ).limit(limit)).all())

    def renew_mcp_execution_claim(self, request_id, claim_token):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            result = session.exec(update(ToolApprovalRequest).where(
                ToolApprovalRequest.id == request_id,
                ToolApprovalRequest.execution_state == EXEC_RUNNING,
                ToolApprovalRequest.execution_claim_token == claim_token,
            ).values(execution_started_at=_utcnow()))
            session.commit()
            return int(result.rowcount or 0) == 1

    def find_stranded_mcp_results(self, *, limit=20):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            return list(session.exec(select(ToolApprovalRequest).where(
                ToolApprovalRequest.request_kind == KIND_MCP,
                ToolApprovalRequest.result_json.is_not(None),
                ToolApprovalRequest.continuation_capable == True,  # noqa: E712
                ToolApprovalRequest.continuation_state == CONT_NOT_NEEDED,
                ToolApprovalRequest.status != REQ_CANCELLED,
            ).limit(limit)).all())
