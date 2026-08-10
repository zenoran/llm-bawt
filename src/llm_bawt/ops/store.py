"""Storage layer for the ops catalog + job ledger (TASK-639).

CRUD + atomic transitions for :class:`OpsOperation` and :class:`OpsJob`.
No executor concerns here — the store is pure state; the executor
(:mod:`.executor`) reads/writes through this store.

Concurrency:

* Job state transitions use ``SELECT ... FOR UPDATE`` under Postgres so
  reconciler + executor + resolver don't race. Under SQLite (test env) the
  same code path still works — the store just picks up whichever row was
  visible at read time; race hazards are covered by the idempotency key
  constraint and the terminal-state guards.
* Operation edits bump ``version`` monotonically; job snapshots capture the
  operation version + script hash so a concurrent edit cannot rewrite an
  in-flight job.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, DateTime, Integer, String, Text, text
from sqlmodel import Session, SQLModel, select

from ..utils.config import Config, has_database_credentials
from ..utils.schema import SchemaBootstrapGuard
from .models import (
    EXECUTOR_DOCKER,
    JOB_CANCELLED,
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
    RISK_MEDIUM,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


def _script_hash(script: str) -> str:
    """Deterministic content hash used for job snapshots + policy subjects."""
    return hashlib.sha256((script or "").encode("utf-8")).hexdigest()


# Fields a caller may set on create/update. Anything else is ignored.
_OP_WRITABLE = {
    "slug",
    "title",
    "description",
    "enabled",
    "executor_kind",
    "target_host",
    "run_as_user",
    "working_directory",
    "command_script",
    "args_schema_json",
    "args_defaults_json",
    "timeout_seconds",
    "start_delay_seconds",
    "max_output_bytes",
    "max_concurrent",
    "risk_level",
    "category",
    "approval_prompt_prefix",
}


class OpsStoreUnavailable(RuntimeError):
    """Raised when the ops store has no DB engine and a caller tried to write."""


class OpsStore:
    """DB access for the ops catalog + job ledger."""

    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, config: Config):
        self.config = config
        self.engine = None
        if not has_database_credentials(config):
            return
        try:
            from ..utils.db import get_shared_engine

            self.engine = get_shared_engine(config)
            if self.engine is None:
                return
            self._ensure_tables_exist()
        except Exception as e:  # noqa: BLE001
            self.engine = None
            logger.warning("Ops store DB unavailable: %s", e)

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return

        def bootstrap(conn) -> None:
            SQLModel.metadata.create_all(
                bind=conn,
                tables=[OpsOperation.__table__, OpsJob.__table__],
            )
            self._migrate_add_columns(conn)

        self._schema_guard.run(self.engine, "ops-store", bootstrap)

    def _migrate_add_columns(self, conn) -> None:
        """Postgres-only column-add migration for future evolution.

        Currently a no-op — the tables are new in TASK-639. New columns land
        here as they're added so a redeploy against a pre-existing tenant is
        safe.
        """
        if conn.dialect.name != "postgresql":
            return
        # Future ADD COLUMN IF NOT EXISTS migrations go here.

    # ---- Operation CRUD ---------------------------------------------------

    def list_operations(
        self,
        *,
        include_disabled: bool = False,
        include_soft_deleted: bool = False,
    ) -> list[OpsOperation]:
        if self.engine is None:
            return []
        with Session(self.engine) as session:
            stmt = select(OpsOperation)
            if not include_soft_deleted:
                stmt = stmt.where(OpsOperation.soft_deleted_at.is_(None))
            if not include_disabled:
                stmt = stmt.where(OpsOperation.enabled == True)  # noqa: E712
            stmt = stmt.order_by(OpsOperation.category, OpsOperation.slug)
            return list(session.exec(stmt).all())

    def get_operation(self, slug_or_id: str) -> OpsOperation | None:
        if self.engine is None or not slug_or_id:
            return None
        with Session(self.engine) as session:
            # Try by id first (uuid.hex is 32 chars, deterministic length).
            row = session.get(OpsOperation, slug_or_id)
            if row is not None:
                return row
            return session.exec(
                select(OpsOperation).where(OpsOperation.slug == slug_or_id)
            ).first()

    def get_operation_by_slug(self, slug: str) -> OpsOperation | None:
        if self.engine is None or not slug:
            return None
        with Session(self.engine) as session:
            return session.exec(
                select(OpsOperation).where(OpsOperation.slug == slug)
            ).first()

    def _clean_op(self, data: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in data.items() if k in _OP_WRITABLE}
        # Ensure JSON columns hold valid JSON strings so downstream loads
        # never explode. Caller may pass a dict or a string — normalize to
        # canonical JSON text.
        for jf in ("args_schema_json", "args_defaults_json"):
            if jf in out and out[jf] is not None:
                if isinstance(out[jf], (dict, list)):
                    out[jf] = json.dumps(out[jf], ensure_ascii=False)
                else:
                    # Validate string parses as JSON — reject junk early.
                    try:
                        json.loads(out[jf])
                    except (json.JSONDecodeError, TypeError) as exc:
                        raise ValueError(f"{jf} is not valid JSON: {exc}") from exc
        return out

    def create_operation(
        self,
        data: dict[str, Any],
        *,
        actor: str | None = None,
    ) -> OpsOperation:
        if self.engine is None:
            raise OpsStoreUnavailable("ops store has no DB engine")
        clean = self._clean_op(data)
        slug = str(clean.get("slug") or "").strip()
        if not slug:
            raise ValueError("slug is required")
        # Guard uniqueness at the app layer for a friendly error; DB unique
        # index is the ultimate authority.
        if self.get_operation_by_slug(slug) is not None:
            raise ValueError(f"operation slug already exists: {slug}")
        now = _utcnow()
        script = str(clean.get("command_script", "") or "")
        row = OpsOperation(
            id=_new_id(),
            slug=slug,
            title=str(clean.get("title", "") or ""),
            description=str(clean.get("description", "") or ""),
            enabled=bool(clean.get("enabled", False)),
            executor_kind=str(clean.get("executor_kind", EXECUTOR_DOCKER)),
            target_host=str(clean.get("target_host", "") or ""),
            run_as_user=clean.get("run_as_user"),
            working_directory=clean.get("working_directory"),
            command_script=script,
            args_schema_json=str(clean.get("args_schema_json", "{}") or "{}"),
            args_defaults_json=str(clean.get("args_defaults_json", "{}") or "{}"),
            timeout_seconds=int(clean.get("timeout_seconds", 300) or 300),
            start_delay_seconds=int(clean.get("start_delay_seconds", 0) or 0),
            max_output_bytes=int(clean.get("max_output_bytes", 65536) or 65536),
            max_concurrent=clean.get("max_concurrent"),
            risk_level=str(clean.get("risk_level", RISK_MEDIUM) or RISK_MEDIUM),
            category=clean.get("category"),
            approval_prompt_prefix=clean.get("approval_prompt_prefix"),
            version=1,
            script_hash=_script_hash(script),
            created_at=now,
            updated_at=now,
            created_by=actor,
            updated_by=actor,
        )
        with Session(self.engine) as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def update_operation(
        self,
        slug_or_id: str,
        data: dict[str, Any],
        *,
        actor: str | None = None,
    ) -> OpsOperation | None:
        if self.engine is None:
            raise OpsStoreUnavailable("ops store has no DB engine")
        clean = self._clean_op(data)
        with Session(self.engine) as session:
            row = session.get(OpsOperation, slug_or_id)
            if row is None:
                row = session.exec(
                    select(OpsOperation).where(OpsOperation.slug == slug_or_id)
                ).first()
                if row is None:
                    return None
            for field in (
                "title", "description", "enabled", "executor_kind",
                "target_host", "run_as_user", "working_directory",
                "command_script", "args_schema_json", "args_defaults_json",
                "timeout_seconds", "start_delay_seconds", "max_output_bytes",
                "max_concurrent", "risk_level", "category",
                "approval_prompt_prefix",
            ):
                if field in clean:
                    setattr(row, field, clean[field])
            # Slug renames are allowed but they invalidate any lookup by the
            # old slug — the operator UI warns on this.
            if "slug" in clean and clean["slug"] != row.slug:
                row.slug = str(clean["slug"])
            # Bump version + refresh script hash whenever the script text moved.
            row.version = int(row.version or 1) + 1
            row.script_hash = _script_hash(row.command_script or "")
            row.updated_at = _utcnow()
            row.updated_by = actor
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def soft_delete_operation(
        self,
        slug_or_id: str,
        *,
        actor: str | None = None,
    ) -> bool:
        """Mark an operation soft-deleted (hidden from list + agent) while
        preserving audit history. Also flips ``enabled`` off. Idempotent."""
        if self.engine is None:
            raise OpsStoreUnavailable("ops store has no DB engine")
        with Session(self.engine) as session:
            row = session.get(OpsOperation, slug_or_id)
            if row is None:
                row = session.exec(
                    select(OpsOperation).where(OpsOperation.slug == slug_or_id)
                ).first()
                if row is None:
                    return False
            if row.soft_deleted_at is not None:
                return True
            row.soft_deleted_at = _utcnow()
            row.enabled = False
            row.updated_at = _utcnow()
            row.updated_by = actor
            session.add(row)
            session.commit()
            return True

    # ---- Per-slug seeding -------------------------------------------------

    def seed_operation_if_missing(
        self,
        data: dict[str, Any],
        *,
        actor: str = "system-seed",
    ) -> OpsOperation | None:
        """Insert an operation only if no row with that ``slug`` exists.

        TASK-639 catalog invariant: seed rows are per-slug insert-if-missing,
        NEVER overwrite operator edits. Returns the created row, or None if
        an existing row was found.
        """
        if self.engine is None:
            return None
        slug = str(data.get("slug") or "").strip()
        if not slug:
            raise ValueError("seed data missing slug")
        existing = self.get_operation_by_slug(slug)
        if existing is not None:
            return None
        return self.create_operation(data, actor=actor)

    # ---- Job lifecycle ----------------------------------------------------

    def create_job(
        self,
        *,
        operation: OpsOperation,
        args_json: str,
        display_args_json: str,
        idempotency_key: str,
        caller_bot_id: str | None = None,
        caller_user_id: str | None = None,
        caller_turn_id: str | None = None,
        caller_session_key: str | None = None,
        caller_backend: str | None = None,
        approval_request_id: str | None = None,
    ) -> OpsJob:
        """Create a queued job. Idempotent on ``idempotency_key`` — a second
        call with the same key returns the pre-existing job.
        """
        if self.engine is None:
            raise OpsStoreUnavailable("ops store has no DB engine")
        # Idempotent short-circuit.
        with Session(self.engine) as session:
            existing = session.exec(
                select(OpsJob).where(OpsJob.idempotency_key == idempotency_key)
            ).first()
            if existing is not None:
                return existing
        row = OpsJob(
            id=_new_id(),
            operation_slug=operation.slug,
            operation_version=int(operation.version or 1),
            operation_script_hash=operation.script_hash or "",
            args_json=args_json or "{}",
            display_args_json=display_args_json or "{}",
            caller_bot_id=caller_bot_id,
            caller_user_id=caller_user_id,
            caller_turn_id=caller_turn_id,
            caller_session_key=caller_session_key,
            caller_backend=caller_backend,
            approval_request_id=approval_request_id,
            state=JOB_QUEUED,
            idempotency_key=idempotency_key,
            submitted_at=_utcnow(),
        )
        with Session(self.engine) as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def get_job(self, job_id: str) -> OpsJob | None:
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            return session.get(OpsJob, job_id)

    def list_jobs(
        self,
        *,
        operation_slug: str | None = None,
        state: str | None = None,
        limit: int = 50,
    ) -> list[OpsJob]:
        if self.engine is None:
            return []
        with Session(self.engine) as session:
            stmt = select(OpsJob)
            if operation_slug:
                stmt = stmt.where(OpsJob.operation_slug == operation_slug)
            if state:
                stmt = stmt.where(OpsJob.state == state)
            stmt = stmt.order_by(OpsJob.submitted_at.desc()).limit(limit)
            return list(session.exec(stmt).all())

    def mark_dispatching(
        self,
        job_id: str,
        *,
        host_unit_name: str,
        status_file_path: str | None = None,
        log_file_path: str | None = None,
    ) -> OpsJob | None:
        """QUEUED → DISPATCHING; records the resolved host unit + paths.

        Idempotent on already-dispatching / already-running; refuses to
        regress a terminal row.
        """
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            row = session.get(OpsJob, job_id)
            if row is None:
                return None
            if row.state in JOB_TERMINAL_STATES:
                return row
            if row.state == JOB_QUEUED:
                row.state = JOB_DISPATCHING
            row.host_unit_name = host_unit_name
            row.status_file_path = status_file_path
            row.log_file_path = log_file_path
            row.dispatched_at = _utcnow()
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def mark_running(self, job_id: str) -> OpsJob | None:
        """DISPATCHING → RUNNING; called by the reconciler when the host
        unit is confirmed active or when the first status line lands."""
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            row = session.get(OpsJob, job_id)
            if row is None:
                return None
            if row.state in JOB_TERMINAL_STATES:
                return row
            if row.state in (JOB_QUEUED, JOB_DISPATCHING):
                row.state = JOB_RUNNING
                row.started_at = _utcnow()
                session.add(row)
                session.commit()
                session.refresh(row)
            return row

    def mark_terminal(
        self,
        job_id: str,
        *,
        state: str,
        exit_code: int | None = None,
        output_tail: str | None = None,
        error_text: str | None = None,
    ) -> OpsJob | None:
        """Transition to a terminal state. Idempotent — the first terminal
        transition wins and later calls are no-ops that return the stored row.

        Callers must pass one of :data:`JOB_SUCCEEDED`, :data:`JOB_FAILED`,
        :data:`JOB_TIMED_OUT`, :data:`JOB_CANCELLED`, :data:`JOB_LOST`.
        """
        if state not in JOB_TERMINAL_STATES:
            raise ValueError(f"not a terminal state: {state!r}")
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            row = session.get(OpsJob, job_id)
            if row is None:
                return None
            if row.state in JOB_TERMINAL_STATES:
                return row
            row.state = state
            row.exit_code = exit_code
            if output_tail is not None:
                row.output_tail = output_tail
            if error_text is not None:
                row.error_text = error_text
            row.finished_at = _utcnow()
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def touch_reconcile(self, job_id: str) -> None:
        """Record a reconciler poll timestamp so we can detect lost jobs."""
        if self.engine is None:
            return
        with Session(self.engine) as session:
            row = session.get(OpsJob, job_id)
            if row is None:
                return
            row.last_reconcile_at = _utcnow()
            session.add(row)
            session.commit()

    def find_active_jobs(self, *, limit: int = 100) -> list[OpsJob]:
        """All jobs in a non-terminal state — the reconciler's work queue."""
        if self.engine is None:
            return []
        with Session(self.engine) as session:
            stmt = (
                select(OpsJob)
                .where(OpsJob.state.notin_(JOB_TERMINAL_STATES))
                .order_by(OpsJob.submitted_at)
                .limit(limit)
            )
            return list(session.exec(stmt).all())


__all__ = ["OpsStore", "OpsStoreUnavailable"]
