"""Operation catalog + job ledger SQLModel tables (TASK-639).

Two tables:

* ``ops_operations`` — operator-configured operations. ``slug`` is the
  stable identifier the agent references via ``ops_run(operation=slug, ...)``.
  ``command_script`` is operator-authored bash; the agent never supplies it.
  Arguments are declared via ``args_schema_json`` (JSON Schema; unknown
  properties are rejected at validation) and exposed to the script as
  ``OPS_ARG_<NAME>`` env vars — no string interpolation of agent input into
  shell text.

* ``ops_jobs`` — one row per invocation. Snapshots the operation version +
  script hash so subsequent catalog edits never mutate an in-flight job.
  Carries caller context (bot / user / turn / approval request) for audit
  and for the continuation dispatcher.

Both tables live in the same ``askllm`` DB served by the shared
:func:`~llm_bawt.utils.db.get_shared_engine` pool.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlmodel import Field, SQLModel


# ---------------------------------------------------------------------------
# Operation risk / category enums (open strings; UI provides suggestions)
# ---------------------------------------------------------------------------

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
RISK_CRITICAL = "critical"

# ---------------------------------------------------------------------------
# Executor kinds (only systemd_ssh is implemented in Slice C)
# ---------------------------------------------------------------------------

EXECUTOR_SYSTEMD_SSH = "systemd_ssh"

# ---------------------------------------------------------------------------
# Job state machine
# ---------------------------------------------------------------------------

# The catalog validation + row commit landed but the executor hasn't been
# invoked yet. Only transient — held for a few ms.
JOB_QUEUED = "queued"
# Executor invoked; host unit has not yet been confirmed accepted.
JOB_DISPATCHING = "dispatching"
# Host unit is active per systemd; runner has claimed the status file.
JOB_RUNNING = "running"
# Terminal: runner wrote exit_code=0 status.
JOB_SUCCEEDED = "succeeded"
# Terminal: runner wrote non-zero exit status.
JOB_FAILED = "failed"
# Terminal: runner exceeded RuntimeMaxSec / timeout.
JOB_TIMED_OUT = "timed_out"
# Terminal: operator/caller cancelled prior to run OR mid-run.
JOB_CANCELLED = "cancelled"
# Terminal: reconciler could not find the host unit AND no status file
# arrived within the stale window. Distinct from failed because we don't
# know the exit code.
JOB_LOST = "lost"

JOB_TERMINAL_STATES = {
    JOB_SUCCEEDED, JOB_FAILED, JOB_TIMED_OUT, JOB_CANCELLED, JOB_LOST,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# OpsOperation
# ---------------------------------------------------------------------------

class OpsOperation(SQLModel, table=True):
    """One operator-configured operation. Editable in BawtHub.

    Creating / enabling a row here makes it appear in ``ops_list_operations``
    and become invocable via ``ops_run(operation=slug, ...)`` without any
    code change or service reload.
    """

    __tablename__ = "ops_operations"

    id: str = Field(sa_column=Column(String(64), primary_key=True))
    slug: str = Field(
        sa_column=Column(String(128), nullable=False, unique=True, index=True)
    )
    title: str = Field(default="", sa_column=Column(String(256), nullable=False))
    description: str = Field(default="", sa_column=Column(Text, nullable=False))
    enabled: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, server_default="false", index=True),
    )
    # Only systemd_ssh in Slice C. Extra executor kinds land alongside their
    # implementation modules.
    executor_kind: str = Field(
        default=EXECUTOR_SYSTEMD_SSH,
        sa_column=Column(String(32), nullable=False),
    )
    # Where the systemd unit runs. For LAN work this is "nick@172.18.0.1"
    # (echo) or another SSH-accessible host. The executor SSHes there; the
    # app container never runs the command directly.
    target_host: str = Field(default="", sa_column=Column(String(256), nullable=False))
    run_as_user: str | None = Field(
        default=None, sa_column=Column(String(64), nullable=True)
    )
    working_directory: str | None = Field(
        default=None, sa_column=Column(String(1024), nullable=True)
    )
    # Operator-authored bash. Reads args as ``OPS_ARG_<NAME>`` env vars.
    # NEVER interpolate agent input into shell text — quote the env var.
    command_script: str = Field(default="", sa_column=Column(Text, nullable=False))
    # JSON Schema for the args dict. Empty {} = no args.
    # ``additionalProperties: false`` is enforced at validation time regardless
    # of the schema (unknown args are rejected).
    args_schema_json: str = Field(
        default="{}", sa_column=Column(Text, nullable=False)
    )
    args_defaults_json: str = Field(
        default="{}", sa_column=Column(Text, nullable=False)
    )
    # Executor guardrails.
    timeout_seconds: int = Field(
        default=300,
        sa_column=Column(Integer, nullable=False, server_default="300"),
    )
    # Delay before the host unit starts running. Non-zero for self-affecting
    # operations (bridge/redis/app restart) so the caller's response has time
    # to stream back before the restart lands.
    start_delay_seconds: int = Field(
        default=0, sa_column=Column(Integer, nullable=False, server_default="0"),
    )
    max_output_bytes: int = Field(
        default=65536,
        sa_column=Column(Integer, nullable=False, server_default="65536"),
    )
    max_concurrent: int | None = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )
    risk_level: str = Field(
        default=RISK_MEDIUM,
        sa_column=Column(String(16), nullable=False, server_default=RISK_MEDIUM),
    )
    category: str | None = Field(
        default=None, sa_column=Column(String(64), nullable=True, index=True)
    )
    # Informational — a prefix the approval-policy UI can suggest / preview.
    # Actual gating enforcement stays in ``tool_approval_policies``.
    approval_prompt_prefix: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    # Monotonic per-slug. Bumped on every ``update()``. Jobs snapshot this.
    version: int = Field(
        default=1, sa_column=Column(Integer, nullable=False, server_default="1"),
    )
    # sha256(command_script). Snapshotted on jobs so a later edit can't
    # rewrite an in-flight job's script.
    script_hash: str = Field(
        default="", sa_column=Column(String(64), nullable=False)
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    created_by: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True)
    )
    updated_by: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True)
    )
    # Soft-delete marker. Disabled + soft-deleted rows are hidden from
    # ``ops_list_operations`` but preserved for audit.
    soft_deleted_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )

    def to_api(self, *, include_script: bool = True) -> dict[str, Any]:
        return {
            "id": self.id,
            "slug": self.slug,
            "title": self.title,
            "description": self.description,
            "enabled": bool(self.enabled),
            "executor_kind": self.executor_kind,
            "target_host": self.target_host,
            "run_as_user": self.run_as_user,
            "working_directory": self.working_directory,
            "command_script": self.command_script if include_script else None,
            "args_schema_json": self.args_schema_json,
            "args_defaults_json": self.args_defaults_json,
            "timeout_seconds": int(self.timeout_seconds or 0),
            "start_delay_seconds": int(self.start_delay_seconds or 0),
            "max_output_bytes": int(self.max_output_bytes or 0),
            "max_concurrent": self.max_concurrent,
            "risk_level": self.risk_level,
            "category": self.category,
            "approval_prompt_prefix": self.approval_prompt_prefix,
            "version": int(self.version or 1),
            "script_hash": self.script_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "soft_deleted_at": (
                self.soft_deleted_at.isoformat() if self.soft_deleted_at else None
            ),
        }

    def to_agent_summary(self) -> dict[str, Any]:
        """The subset returned by ``ops_list_operations`` — enough for the
        agent to form a valid ``ops_run`` call, without leaking the script."""
        import json
        try:
            args_schema = json.loads(self.args_schema_json or "{}")
        except (json.JSONDecodeError, TypeError):
            args_schema = {}
        try:
            args_defaults = json.loads(self.args_defaults_json or "{}")
        except (json.JSONDecodeError, TypeError):
            args_defaults = {}
        return {
            "slug": self.slug,
            "title": self.title,
            "description": self.description,
            "risk_level": self.risk_level,
            "category": self.category,
            "target_host": self.target_host,
            "timeout_seconds": int(self.timeout_seconds or 0),
            "args_schema": args_schema,
            "args_defaults": args_defaults,
        }


# ---------------------------------------------------------------------------
# OpsJob
# ---------------------------------------------------------------------------

class OpsJob(SQLModel, table=True):
    """One invocation. Rows survive app / bridge / redis restarts so the
    reconciler + approval outbox can recover their state.

    The DB row is authoritative for state transitions; the host-side status
    file (``.logs/ops-jobs/{job_id}/status.json``) is the source of truth for
    the exit code, and the reconciler imports it into ``exit_code`` +
    ``output_tail`` on the next poll.
    """

    __tablename__ = "ops_jobs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_ops_jobs_idempotency_key"),
    )

    id: str = Field(sa_column=Column(String(64), primary_key=True))

    # Operation snapshot — do NOT resolve the operation by slug at read time
    # for a running/terminal job; the row already carries the exact version.
    operation_slug: str = Field(
        sa_column=Column(String(128), nullable=False, index=True)
    )
    operation_version: int = Field(
        sa_column=Column(Integer, nullable=False, server_default="1")
    )
    operation_script_hash: str = Field(
        default="", sa_column=Column(String(64), nullable=False)
    )

    # Validated caller args (post-JSON-Schema check) + a redacted variant for
    # UI / logs (fields marked ``x-sensitive`` in the schema are masked here).
    args_json: str = Field(default="{}", sa_column=Column(Text, nullable=False))
    display_args_json: str = Field(
        default="{}", sa_column=Column(Text, nullable=False)
    )

    # Caller context (nullable — direct MCP callers may not have all of these).
    caller_bot_id: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True, index=True)
    )
    caller_user_id: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True)
    )
    caller_turn_id: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True, index=True)
    )
    caller_session_key: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True)
    )
    caller_backend: str | None = Field(
        default=None, sa_column=Column(String(64), nullable=True)
    )
    # If this job was dispatched via an approved MCP-kind approval request,
    # record its id for audit + result linking.
    approval_request_id: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True, index=True)
    )

    # Lifecycle.
    state: str = Field(
        default=JOB_QUEUED,
        sa_column=Column(String(16), nullable=False, server_default=JOB_QUEUED, index=True),
    )
    host_unit_name: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True)
    )
    submitted_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
    dispatched_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    started_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    finished_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    exit_code: int | None = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )
    output_tail: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    error_text: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    status_file_path: str | None = Field(
        default=None, sa_column=Column(String(1024), nullable=True)
    )
    log_file_path: str | None = Field(
        default=None, sa_column=Column(String(1024), nullable=True)
    )
    # Deduplication key — the MCP approval request id for approved calls, or
    # an explicit invocation id passed by a direct caller. UNIQUE index.
    idempotency_key: str = Field(
        sa_column=Column(String(128), nullable=False)
    )
    retry_count: int = Field(
        default=0, sa_column=Column(Integer, nullable=False, server_default="0"),
    )
    last_reconcile_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )

    def to_api(self, *, include_output: bool = False) -> dict[str, Any]:
        import json
        try:
            args = json.loads(self.display_args_json or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        row: dict[str, Any] = {
            "id": self.id,
            "operation": self.operation_slug,
            "operation_version": int(self.operation_version or 1),
            "operation_script_hash": self.operation_script_hash,
            "args": args,
            "caller": {
                "bot_id": self.caller_bot_id,
                "user_id": self.caller_user_id,
                "turn_id": self.caller_turn_id,
                "session_key": self.caller_session_key,
                "backend": self.caller_backend,
            },
            "approval_request_id": self.approval_request_id,
            "state": self.state,
            "host_unit_name": self.host_unit_name,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "dispatched_at": self.dispatched_at.isoformat() if self.dispatched_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "exit_code": self.exit_code,
            "error_text": self.error_text,
            "idempotency_key": self.idempotency_key,
            "retry_count": int(self.retry_count or 0),
            "terminal": self.state in JOB_TERMINAL_STATES,
            "last_reconcile_at": (
                self.last_reconcile_at.isoformat() if self.last_reconcile_at else None
            ),
        }
        if include_output:
            row["output_tail"] = self.output_tail
            row["status_file_path"] = self.status_file_path
            row["log_file_path"] = self.log_file_path
        return row


__all__ = [
    "OpsOperation",
    "OpsJob",
    "RISK_LOW",
    "RISK_MEDIUM",
    "RISK_HIGH",
    "RISK_CRITICAL",
    "EXECUTOR_SYSTEMD_SSH",
    "JOB_QUEUED",
    "JOB_DISPATCHING",
    "JOB_RUNNING",
    "JOB_SUCCEEDED",
    "JOB_FAILED",
    "JOB_TIMED_OUT",
    "JOB_CANCELLED",
    "JOB_LOST",
    "JOB_TERMINAL_STATES",
]
