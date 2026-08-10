"""Ops orchestration service (TASK-639).

Ties :class:`OpsStore` + :func:`validate_args` + :class:`Executor` together
so the MCP ``ops_run`` tool and the ``/v1/ops/*`` admin routes have one
call surface instead of duplicating validation / dispatch / reconcile logic.

Public methods intentionally return plain dicts (to_api shape) so the MCP
layer can pass them through unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .executor import (
    DispatchResult,
    Executor,
    ExecutorError,
    NohupSshExecutor,
    ReconcileResult,
)
from .models import (
    EXECUTOR_NOHUP_SSH,
    JOB_DISPATCHING,
    JOB_FAILED,
    JOB_LOST,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    JOB_TERMINAL_STATES,
    JOB_TIMED_OUT,
    OpsJob,
    OpsOperation,
)
from .store import OpsStore
from .validation import ArgValidationError, validate_args

logger = logging.getLogger(__name__)


# Runner state string → OpsJob state constant.
_RUNNER_STATE_MAP = {
    "running": JOB_RUNNING,
    "succeeded": JOB_SUCCEEDED,
    "failed": JOB_FAILED,
    "timed_out": JOB_TIMED_OUT,
}


class OpsDispatchError(RuntimeError):
    """Raised when a job cannot be dispatched. Carries a short code so the
    MCP tool response can differentiate a user-facing "operation disabled"
    vs a system-facing "executor unavailable"."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


# ---------------------------------------------------------------------------
# Argument redaction for display / logs
# ---------------------------------------------------------------------------

def _display_args(args: dict[str, Any], schema_json: str) -> dict[str, Any]:
    """Redact any arg whose schema entry carries ``x-sensitive: true``.

    Preserves keys + types for the UI (so the operator sees "field=***")
    without leaking values to logs or the approval card.
    """
    try:
        schema = json.loads(schema_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return dict(args)
    properties = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
    out: dict[str, Any] = {}
    for k, v in args.items():
        subschema = properties.get(k) if isinstance(properties, dict) else None
        if isinstance(subschema, dict) and subschema.get("x-sensitive"):
            out[k] = "***"
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# OpsService
# ---------------------------------------------------------------------------

class OpsService:
    """Facade over store + validation + executor."""

    def __init__(
        self,
        store: OpsStore,
        *,
        executor: Executor | None = None,
    ) -> None:
        self.store = store
        self._executors: dict[str, Executor] = {}
        default = executor or NohupSshExecutor()
        self._executors[default.kind()] = default

    def register_executor(self, executor: Executor) -> None:
        self._executors[executor.kind()] = executor

    def _resolve_executor(self, op: OpsOperation) -> Executor:
        kind = (op.executor_kind or EXECUTOR_NOHUP_SSH).strip()
        ex = self._executors.get(kind)
        if ex is None:
            raise OpsDispatchError(
                "executor_kind_unknown",
                f"no executor registered for kind {kind!r} (operation {op.slug})",
            )
        return ex

    # ---- catalog surface -------------------------------------------------

    def list_operations_for_agent(
        self,
        *,
        include_disabled: bool = False,
    ) -> list[dict[str, Any]]:
        rows = self.store.list_operations(include_disabled=include_disabled)
        return [row.to_agent_summary() for row in rows]

    def get_operation(self, slug: str) -> OpsOperation | None:
        return self.store.get_operation_by_slug(slug)

    # ---- dispatch --------------------------------------------------------

    def dispatch_job(
        self,
        *,
        operation_slug: str,
        args: dict[str, Any] | None,
        idempotency_key: str,
        caller_bot_id: str | None = None,
        caller_user_id: str | None = None,
        caller_turn_id: str | None = None,
        caller_session_key: str | None = None,
        caller_backend: str | None = None,
        approval_request_id: str | None = None,
    ) -> dict[str, Any]:
        """Validate + create + dispatch a job. Returns the job's to_api dict.

        Idempotent on ``idempotency_key`` — a second call for the same key
        returns the pre-existing job without re-dispatching. This makes
        approval-flow at-least-once continuations side-effect safe.
        """
        op = self.store.get_operation_by_slug(operation_slug)
        if op is None:
            raise OpsDispatchError(
                "operation_not_found",
                f"unknown operation slug: {operation_slug!r}",
            )
        if not op.enabled or op.soft_deleted_at is not None:
            raise OpsDispatchError(
                "operation_disabled",
                f"operation {op.slug!r} is disabled",
            )

        # Validate args against the operation's schema. Rejects unknown
        # properties AND merges declared defaults.
        try:
            merged = validate_args(
                args or {},
                op.args_schema_json,
                op.args_defaults_json,
            )
        except ArgValidationError as exc:
            raise OpsDispatchError(
                "args_invalid",
                "; ".join(exc.violations) or str(exc),
            ) from exc

        args_json = json.dumps(merged, ensure_ascii=False)
        display_args_json = json.dumps(
            _display_args(merged, op.args_schema_json), ensure_ascii=False,
        )

        job = self.store.create_job(
            operation=op,
            args_json=args_json,
            display_args_json=display_args_json,
            idempotency_key=idempotency_key,
            caller_bot_id=caller_bot_id,
            caller_user_id=caller_user_id,
            caller_turn_id=caller_turn_id,
            caller_session_key=caller_session_key,
            caller_backend=caller_backend,
            approval_request_id=approval_request_id,
        )
        # If this was an idempotent short-circuit and the job is already past
        # QUEUED, we don't re-dispatch — return whatever the store has.
        if job.state != JOB_QUEUED:
            return job.to_api(include_output=False)

        # Fresh QUEUED job — dispatch now.
        executor = self._resolve_executor(op)
        if not executor.available():
            self.store.mark_terminal(
                job.id,
                state=JOB_FAILED,
                error_text=f"executor {executor.kind()!r} is not available on this host",
            )
            raise OpsDispatchError(
                "executor_unavailable",
                f"executor {executor.kind()!r} is not available on this host "
                "(TASK-639 Slice C: openssh-client must be installed in the app image)",
            )

        try:
            result: DispatchResult = executor.dispatch(
                job_id=job.id,
                operation_slug=op.slug,
                target_host=op.target_host,
                run_as_user=op.run_as_user,
                working_directory=op.working_directory,
                command_script=op.command_script,
                env_args=merged,
                timeout_seconds=op.timeout_seconds,
                start_delay_seconds=op.start_delay_seconds,
                max_output_bytes=op.max_output_bytes,
            )
        except ExecutorError as exc:
            self.store.mark_terminal(
                job.id,
                state=JOB_FAILED,
                error_text=f"dispatch failed: {exc}",
            )
            raise OpsDispatchError("dispatch_failed", str(exc)) from exc

        self.store.mark_dispatching(
            job.id,
            host_unit_name=result.host_unit_name,
            status_file_path=result.status_file_path,
            log_file_path=result.log_file_path,
        )
        # Return the fresh row so the caller sees state=DISPATCHING.
        row = self.store.get_job(job.id)
        return (row or job).to_api(include_output=False)

    # ---- status ----------------------------------------------------------

    def get_job_status(
        self,
        job_id: str,
        *,
        output_tail_bytes: int | None = None,
        reconcile_if_active: bool = True,
    ) -> dict[str, Any] | None:
        job = self.store.get_job(job_id)
        if job is None:
            return None
        include_output = output_tail_bytes is not None and output_tail_bytes > 0
        # Terminal — return the stored result unchanged.
        if job.state in JOB_TERMINAL_STATES or not reconcile_if_active:
            return job.to_api(include_output=include_output)
        if not job.host_unit_name or not job.status_file_path:
            # Dispatch failed to record a unit name — no way to poll.
            return job.to_api(include_output=include_output)

        # Poll the host status file.
        op = self.store.get_operation_by_slug(job.operation_slug)
        if op is None:
            return job.to_api(include_output=include_output)
        executor = self._resolve_executor(op)
        if not executor.available():
            return job.to_api(include_output=include_output)

        try:
            rec: ReconcileResult = executor.reconcile(
                job_id=job.id,
                target_host=op.target_host,
                status_file_path=job.status_file_path,
                log_file_path=job.log_file_path or "",
                output_tail_bytes=output_tail_bytes or 4096,
            )
        except ExecutorError as exc:
            logger.warning("reconcile failed for job %s: %s", job.id, exc)
            self.store.touch_reconcile(job.id)
            return job.to_api(include_output=include_output)

        self.store.touch_reconcile(job.id)

        if rec.state is None:
            # Status file not yet present — still queued/dispatching.
            return job.to_api(include_output=include_output)

        new_state = _RUNNER_STATE_MAP.get(rec.state)
        if new_state == JOB_RUNNING:
            self.store.mark_running(job.id)
        elif new_state in JOB_TERMINAL_STATES:
            self.store.mark_terminal(
                job.id,
                state=new_state,
                exit_code=rec.exit_code,
                output_tail=rec.output_tail,
                error_text=rec.error,
            )

        fresh = self.store.get_job(job.id) or job
        return fresh.to_api(include_output=include_output)


__all__ = ["OpsService", "OpsDispatchError"]
