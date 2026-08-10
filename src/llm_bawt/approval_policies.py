"""Storage + audit for approval-gated tool policies (TASK-289, TASK-290).

Source of truth for the feature lives here, in llm-bawt. Two tables:

* ``tool_approval_policies`` — the operator-configured rules. Compiled into the
  pure :class:`agent_bridge.approval.ApprovalPolicy` bundle that bridges fetch
  over HTTP and evaluate in their per-tool permission hook.
* ``tool_approval_requests`` — one row per gated tool call: full audit trail of
  what was asked, which policy matched, and how it resolved. Mirrors the
  ``chat_pending_questions`` design (TASK-269) so the deny→resolve→continuation
  lifecycle is durable and idempotent.

Evaluation semantics are NOT here — they're in the pure engine. This module is
storage, compilation, and the request state machine only.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, text
from sqlmodel import Field, Session, SQLModel, select

from agent_bridge.approval import (
    ApprovalPolicy,
    MatcherType,
    PolicyAction,
    PolicyBundle,
    Severity,
    compute_etag,
    humanize_subject,
)

from .utils.config import Config, has_database_credentials
from .utils.schema import SchemaBootstrapGuard

logger = logging.getLogger(__name__)


class ApprovalPersistError(RuntimeError):
    """A tool approval request could not be durably persisted.

    TASK-306 Section A: callers that gate a tool on approval require a
    *confirmed* commit. This is raised when the row cannot be written (no DB
    engine, or the insert/commit failed). It must be surfaced honestly to the
    agent/user — never swallowed — because a swallowed failure means the user
    never sees the approval and the agent proceeds on a false premise.
    """


def _new_id() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware_utc(dt: datetime | None) -> datetime | None:
    """Coerce a possibly-naive datetime to tz-aware UTC.

    Postgres round-trips ``TIMESTAMP WITH TIME ZONE`` as aware, but SQLite
    (test env) strips tzinfo on read. Comparisons of ``_utcnow()`` (aware)
    against a stored value would then raise ``TypeError``. Coerce here so
    the lease/backoff math works uniformly.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# Request lifecycle states.
REQ_PENDING = "pending"
REQ_APPROVED = "approved"
REQ_DENIED = "denied"
# Cancelled = user dismissed the request WITHOUT warning the agent. Unlike
# `denied` (which dispatches a "you were refused" continuation that costs the
# agent tokens to acknowledge), cancel is silent: no grant, no continuation.
REQ_CANCELLED = "cancelled"
# Responded = user declined to run the tool but sent the agent their own
# guidance (e.g. correcting a false-positive gate) instead of the canned deny.
# Like deny: no grant, tool not run — but the continuation is user-authored.
REQ_RESPONDED = "responded"
REQ_EXPIRED = "expired"
REQ_SUPERSEDED = "superseded"

# TASK-639: request kind. "harness" = classic bridge-hook approvals (Bash and
# other bridge-layer gated tools; the client dispatches a one-shot grant +
# continuation). "mcp" = server-side approval interception at the BawtHub
# MCP server; the approved tool is executed exactly-once server-side and the
# result is delivered back through a durable continuation outbox — the agent
# never re-issues the call. Default stays "harness" so existing rows are
# unambiguous.
KIND_HARNESS = "harness"
KIND_MCP = "mcp"

# TASK-639: execution state machine for MCP-kind approvals. Harness-kind rows
# leave this at EXEC_NOT_APPLICABLE — execution happens on the CLIENT after
# the one-shot grant lands.
EXEC_NOT_APPLICABLE = "not_applicable"
EXEC_PENDING = "pending"
EXEC_RUNNING = "running"
EXEC_SUCCEEDED = "succeeded"
EXEC_FAILED = "failed"
EXEC_SKIPPED = "skipped"

# TASK-639: continuation outbox state. Harness-kind = CONT_NOT_NEEDED (client
# dispatches its own continuation). MCP-kind = CONT_PENDING once the tool has
# executed; a lifespan worker moves it through DISPATCHING → DELIVERED (or
# FAILED with backoff).
CONT_NOT_NEEDED = "not_needed"
CONT_PENDING = "pending"
CONT_DISPATCHING = "dispatching"
CONT_DELIVERED = "delivered"
CONT_FAILED = "failed"


class ToolApprovalPolicy(SQLModel, table=True):
    """One operator-configured approval rule. Compiles to an ApprovalPolicy."""

    __tablename__ = "tool_approval_policies"

    id: str = Field(
        default_factory=_new_id,
        sa_column=Column(String(64), primary_key=True),
    )
    enabled: bool = Field(
        default=True, sa_column=Column(Boolean, nullable=False, index=True)
    )
    # "*" = any bridge, else a backend name ("claude-code", "codex", "openclaw").
    backend_scope: str = Field(
        default="*", sa_column=Column(String(64), nullable=False, index=True)
    )
    # "*" = any tool, else a tool name ("Bash", "Write", …). MCP-tail aware.
    tool_name: str = Field(
        default="*", sa_column=Column(String(128), nullable=False, index=True)
    )
    matcher_type: str = Field(default="always", sa_column=Column(String(16), nullable=False))
    pattern: str = Field(default="", sa_column=Column(Text, nullable=False))
    # Which tool-input field to derive the subject from ("" = per-tool default,
    # "*" = whole input JSON).
    field: str = Field(default="", sa_column=Column(String(128), nullable=False))
    action: str = Field(
        default="require_approval", sa_column=Column(String(24), nullable=False)
    )
    severity: str = Field(default="medium", sa_column=Column(String(16), nullable=False))
    category: str | None = Field(default=None, sa_column=Column(String(64), nullable=True))
    approval_prompt: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    # Lower = evaluated first (first match wins). Named order_index — ``order`` is
    # a SQL reserved word.
    order_index: int = Field(default=100, sa_column=Column(Integer, nullable=False, index=True))
    # Bumped on every update — gives the bundle a per-row revision for debugging
    # and lets the UI show "version N" (TASK-289 versioning semantics).
    version: int = Field(default=1, sa_column=Column(Integer, nullable=False))
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    created_by: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    updated_by: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))

    def to_policy(self) -> ApprovalPolicy:
        """Compile this row into the pure evaluation dataclass."""
        return ApprovalPolicy(
            id=self.id,
            backend_scope=self.backend_scope or "*",
            tool_name=self.tool_name or "*",
            matcher_type=MatcherType.coerce(self.matcher_type),
            pattern=self.pattern or "",
            field=self.field or "",
            action=PolicyAction.coerce(self.action),
            severity=Severity.coerce(self.severity),
            category=self.category,
            approval_prompt=self.approval_prompt,
            order=self.order_index,
            enabled=self.enabled,
            version=self.version,
        )

    def to_api(self) -> dict[str, Any]:
        """Full row as a JSON-able dict for the admin API."""
        return {
            "id": self.id,
            "enabled": self.enabled,
            "backend_scope": self.backend_scope,
            "tool_name": self.tool_name,
            "matcher_type": self.matcher_type,
            "pattern": self.pattern,
            "field": self.field,
            "action": self.action,
            "severity": self.severity,
            "category": self.category,
            "approval_prompt": self.approval_prompt,
            "order": self.order_index,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "updated_by": self.updated_by,
        }


class ToolApprovalRequest(SQLModel, table=True):
    """Durable audit + state for one gated tool call (mirrors PendingQuestion)."""

    __tablename__ = "tool_approval_requests"

    # SDK tool_use id doubles as PK so a duplicate APPROVAL_REQUIRED event
    # (Redis replay / multi-tab race) is idempotent.
    id: str = Field(sa_column=Column(String(128), primary_key=True))
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    bot_id: str = Field(sa_column=Column(String(128), nullable=False, index=True))
    user_id: str = Field(sa_column=Column(String(128), nullable=False, index=True))
    turn_id: str = Field(sa_column=Column(String(128), nullable=False, index=True))
    trigger_message_id: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True, index=True)
    )
    session_key: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    backend: str = Field(default="claude-code", sa_column=Column(String(64), nullable=False))
    tool_name: str = Field(sa_column=Column(String(128), nullable=False))
    tool_arguments_json: str = Field(sa_column=Column(Text, nullable=False))
    subject: str = Field(sa_column=Column(Text, nullable=False))
    grant_key: str = Field(sa_column=Column(String(64), nullable=False, index=True))
    policy_id: str | None = Field(default=None, sa_column=Column(String(64), nullable=True, index=True))
    severity: str = Field(default="medium", sa_column=Column(String(16), nullable=False))
    prompt: str = Field(default="", sa_column=Column(Text, nullable=False))
    # pending → approved | denied | expired | superseded
    status: str = Field(default=REQ_PENDING, sa_column=Column(String(24), nullable=False, index=True))
    resolved_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    resolved_by: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    resolved_turn_id: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))

    # TASK-639 --- MCP-kind extensions --------------------------------------
    # Two lifecycles share this table: the classic "harness" flow (Bash and
    # other bridge-hook gated tools; client re-issues on approve) and the new
    # "mcp" flow (BawtHub MCP server intercepts; server executes exactly-once
    # on approve; result delivered via the durable continuation outbox).
    request_kind: str = Field(
        default=KIND_HARNESS,
        sa_column=Column(String(16), nullable=False, index=True, server_default=KIND_HARNESS),
    )
    # For MCP-kind the row `id` is an app-generated approval-request id; the
    # SDK's tool_use_id (used by the client harness to correlate a resumed
    # ToolResultBlock) is captured separately when the caller-context header
    # provided it. For harness-kind this stays None (their `id` IS the SDK
    # tool_use_id, per pre-TASK-639 invariant).
    tool_use_id: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True, index=True)
    )
    # The BawtHub MCP server that received the intercepted call (e.g. "bawthub").
    # None for harness-kind.
    mcp_server: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    # sha256(tool_name + canonical(args_json)) — used to bind the internal
    # signed approval-bypass to the exact stored invocation so a model-forged
    # bypass header can't swap args at execute time.
    invocation_hash: str | None = Field(
        default=None, sa_column=Column(String(64), nullable=True)
    )
    # The signed per-turn caller context stamped by the Claude PreToolUse
    # hook (bot_id, user_id, turn_id, trigger_message_id, session_key,
    # backend, tool_use_id). Verified at MCP dispatch and again at resolve.
    caller_context_json: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    # Some callers (raw MCP clients that don't ride an active agent turn) can
    # be approved but cannot receive a continuation. When False the outbox
    # marks itself CONT_NOT_NEEDED and the result is only retrievable via the
    # approval API / job status endpoints.
    continuation_capable: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, server_default="false"),
    )

    # ---- MCP execution state machine (only meaningful for KIND_MCP rows) ---
    execution_state: str = Field(
        default=EXEC_NOT_APPLICABLE,
        sa_column=Column(String(16), nullable=False, server_default=EXEC_NOT_APPLICABLE, index=True),
    )
    execution_attempts: int = Field(
        default=0, sa_column=Column(Integer, nullable=False, server_default="0"),
    )
    execution_started_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    execution_finished_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    execution_error: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    # Normalized MCP result payload (JSON-serialized). For an ordinary success
    # this is the tool's return value; for a refusal/deny/respond it is a
    # structured refusal envelope. Never contains an inflight/pending marker.
    result_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    result_is_error: bool | None = Field(
        default=None, sa_column=Column(Boolean, nullable=True),
    )

    # ---- continuation outbox state (only meaningful for KIND_MCP rows) -----
    continuation_state: str = Field(
        default=CONT_NOT_NEEDED,
        sa_column=Column(String(16), nullable=False, server_default=CONT_NOT_NEEDED, index=True),
    )
    continuation_attempts: int = Field(
        default=0, sa_column=Column(Integer, nullable=False, server_default="0"),
    )
    continuation_last_error: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    continuation_next_attempt_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True, index=True)
    )
    continuation_delivered_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )

    def to_api(self) -> dict[str, Any]:
        try:
            args = json.loads(self.tool_arguments_json) if self.tool_arguments_json else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "bot_id": self.bot_id,
            "user_id": self.user_id,
            "turn_id": self.turn_id,
            "trigger_message_id": self.trigger_message_id,
            "session_key": self.session_key,
            "backend": self.backend,
            "tool_name": self.tool_name,
            "tool_arguments": args,
            "subject": self.subject,
            "label": humanize_subject(self.subject or ""),
            "grant_key": self.grant_key,
            "policy_id": self.policy_id,
            "severity": self.severity,
            "prompt": self.prompt,
            "status": self.status,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "resolved_turn_id": self.resolved_turn_id,
            # TASK-639 — MCP-kind fields. Present on every row (server_default
            # backfills), so ApprovalCard hydration stays field-compatible with
            # pre-TASK-639 harness rows.
            "request_kind": self.request_kind,
            "tool_use_id": self.tool_use_id,
            "mcp_server": self.mcp_server,
            "invocation_hash": self.invocation_hash,
            "continuation_capable": bool(self.continuation_capable),
            "execution_state": self.execution_state,
            "execution_attempts": int(self.execution_attempts or 0),
            "execution_started_at": (
                self.execution_started_at.isoformat() if self.execution_started_at else None
            ),
            "execution_finished_at": (
                self.execution_finished_at.isoformat() if self.execution_finished_at else None
            ),
            "execution_error": self.execution_error,
            "result_is_error": self.result_is_error,
            # result_json is intentionally excluded from the default API dict —
            # it can be large; callers that need it fetch via the dedicated
            # /v1/tool-approval-requests/{id}/result endpoint.
            "continuation_state": self.continuation_state,
            "continuation_attempts": int(self.continuation_attempts or 0),
            "continuation_delivered_at": (
                self.continuation_delivered_at.isoformat()
                if self.continuation_delivered_at else None
            ),
        }


# Fields a caller may set on create/update. Anything else is ignored.
_POLICY_WRITABLE = {
    "enabled", "backend_scope", "tool_name", "matcher_type", "pattern",
    "field", "action", "severity", "category", "approval_prompt", "order",
}


class ToolApprovalPolicyStore:
    """DB access for approval policies + request audit log."""

    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, config: Config):
        self.config = config
        self.engine = None
        if not has_database_credentials(config):
            return
        try:
            from .utils.db import get_shared_engine

            self.engine = get_shared_engine(config)  # TASK-202: shared pool
            if self.engine is None:
                return
            self._ensure_tables_exist()
        except Exception as e:  # noqa: BLE001
            self.engine = None
            logger.warning("Tool approval policies DB unavailable: %s", e)

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return
        def bootstrap(conn) -> None:
            SQLModel.metadata.create_all(
                bind=conn,
                tables=[
                    ToolApprovalPolicy.__table__,
                    ToolApprovalRequest.__table__,
                ],
            )
            self._migrate_add_columns(conn)

        self._schema_guard.run(self.engine, "tool-approval-policy-store", bootstrap)

    def _migrate_add_columns(self, conn) -> None:
        """Add columns introduced after initial schema creation.

        TASK-639 extends ``tool_approval_requests`` with the MCP-kind execution
        + continuation state so the same table serves both the classic bridge-
        hook (harness) flow and the new server-side MCP flow. Idempotent via
        ``ADD COLUMN IF NOT EXISTS``; safe to run every bootstrap.

        Postgres-only: SQLite (test env) got the full column list from the
        preceding ``create_all`` and doesn't accept ``ADD COLUMN IF NOT EXISTS``
        syntax. Skip cleanly there.
        """
        if conn.dialect.name != "postgresql":
            return
        migrations = [
            # request kind + SDK/MCP correlation
            f"ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            f"request_kind VARCHAR(16) NOT NULL DEFAULT '{KIND_HARNESS}'",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "tool_use_id VARCHAR(128)",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "mcp_server VARCHAR(128)",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "invocation_hash VARCHAR(64)",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "caller_context_json TEXT",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "continuation_capable BOOLEAN NOT NULL DEFAULT FALSE",
            # execution state machine
            f"ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            f"execution_state VARCHAR(16) NOT NULL DEFAULT '{EXEC_NOT_APPLICABLE}'",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "execution_attempts INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "execution_started_at TIMESTAMP WITH TIME ZONE",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "execution_finished_at TIMESTAMP WITH TIME ZONE",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "execution_error TEXT",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "result_json TEXT",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "result_is_error BOOLEAN",
            # continuation outbox
            f"ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            f"continuation_state VARCHAR(16) NOT NULL DEFAULT '{CONT_NOT_NEEDED}'",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "continuation_attempts INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "continuation_last_error TEXT",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "continuation_next_attempt_at TIMESTAMP WITH TIME ZONE",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "continuation_delivered_at TIMESTAMP WITH TIME ZONE",
            # indexes matching the SQLModel Field(index=True) declarations
            "CREATE INDEX IF NOT EXISTS ix_tool_approval_requests_request_kind "
            "ON tool_approval_requests (request_kind)",
            "CREATE INDEX IF NOT EXISTS ix_tool_approval_requests_tool_use_id "
            "ON tool_approval_requests (tool_use_id)",
            "CREATE INDEX IF NOT EXISTS ix_tool_approval_requests_execution_state "
            "ON tool_approval_requests (execution_state)",
            "CREATE INDEX IF NOT EXISTS ix_tool_approval_requests_continuation_state "
            "ON tool_approval_requests (continuation_state)",
            "CREATE INDEX IF NOT EXISTS "
            "ix_tool_approval_requests_continuation_next_attempt_at "
            "ON tool_approval_requests (continuation_next_attempt_at)",
        ]
        for stmt in migrations:
            conn.execute(text(stmt))

    # ---- policy CRUD -------------------------------------------------------

    def list_all(self) -> list[ToolApprovalPolicy]:
        if self.engine is None:
            return []
        with Session(self.engine) as session:
            return list(
                session.exec(
                    select(ToolApprovalPolicy).order_by(
                        ToolApprovalPolicy.order_index, ToolApprovalPolicy.id
                    )
                ).all()
            )

    def get(self, policy_id: str) -> ToolApprovalPolicy | None:
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            return session.get(ToolApprovalPolicy, policy_id)

    def _clean(self, data: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in data.items() if k in _POLICY_WRITABLE}
        # normalize enums to canonical lowercase strings via the coercers
        if "matcher_type" in out:
            out["matcher_type"] = MatcherType.coerce(out["matcher_type"]).value
        if "action" in out:
            out["action"] = PolicyAction.coerce(out["action"]).value
        if "severity" in out:
            out["severity"] = Severity.coerce(out["severity"]).value
        return out

    def create(self, data: dict[str, Any], actor: str | None = None) -> ToolApprovalPolicy:
        if self.engine is None:
            raise RuntimeError("Tool approval policies DB unavailable")
        clean = self._clean(data)
        now = _utcnow()
        row = ToolApprovalPolicy(
            id=_new_id(),
            enabled=bool(clean.get("enabled", True)),
            backend_scope=str(clean.get("backend_scope", "*") or "*"),
            tool_name=str(clean.get("tool_name", "*") or "*"),
            matcher_type=str(clean.get("matcher_type", "always")),
            pattern=str(clean.get("pattern", "") or ""),
            field=str(clean.get("field", "") or ""),
            action=str(clean.get("action", "require_approval")),
            severity=str(clean.get("severity", "medium")),
            category=clean.get("category"),
            approval_prompt=clean.get("approval_prompt"),
            order_index=int(clean.get("order", 100) or 0),
            version=1,
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

    def update(
        self, policy_id: str, data: dict[str, Any], actor: str | None = None
    ) -> ToolApprovalPolicy | None:
        if self.engine is None:
            raise RuntimeError("Tool approval policies DB unavailable")
        clean = self._clean(data)
        with Session(self.engine) as session:
            row = session.get(ToolApprovalPolicy, policy_id)
            if row is None:
                return None
            if "enabled" in clean:
                row.enabled = bool(clean["enabled"])
            if "backend_scope" in clean:
                row.backend_scope = str(clean["backend_scope"] or "*")
            if "tool_name" in clean:
                row.tool_name = str(clean["tool_name"] or "*")
            if "matcher_type" in clean:
                row.matcher_type = str(clean["matcher_type"])
            if "pattern" in clean:
                row.pattern = str(clean["pattern"] or "")
            if "field" in clean:
                row.field = str(clean["field"] or "")
            if "action" in clean:
                row.action = str(clean["action"])
            if "severity" in clean:
                row.severity = str(clean["severity"])
            if "category" in clean:
                row.category = clean["category"]
            if "approval_prompt" in clean:
                row.approval_prompt = clean["approval_prompt"]
            if "order" in clean:
                row.order_index = int(clean["order"] or 0)
            row.version += 1
            row.updated_at = _utcnow()
            row.updated_by = actor
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def delete(self, policy_id: str) -> bool:
        if self.engine is None:
            raise RuntimeError("Tool approval policies DB unavailable")
        with Session(self.engine) as session:
            row = session.get(ToolApprovalPolicy, policy_id)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    # ---- bundle compilation ------------------------------------------------

    def compile_bundle(self) -> PolicyBundle:
        """Compile all rows into the versioned bundle a bridge consumes."""
        policies = [row.to_policy() for row in self.list_all()]
        etag = compute_etag(1, policies)
        return PolicyBundle(version=1, etag=etag, policies=policies)

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
            return None
        with Session(self.engine) as session:
            return session.get(ToolApprovalRequest, request_id)

    def resolve_request(
        self,
        request_id: str,
        *,
        status: str,
        resolved_by: str | None = None,
        resolved_turn_id: str | None = None,
    ) -> ToolApprovalRequest | None:
        """Flip a pending request to approved/denied/expired. Idempotent."""
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            row = session.get(ToolApprovalRequest, request_id)
            if row is None:
                return None
            if row.status == REQ_PENDING:
                row.status = status
                row.resolved_at = _utcnow()
                row.resolved_by = resolved_by
                row.resolved_turn_id = resolved_turn_id
                session.add(row)
                session.commit()
                session.refresh(row)
            return row

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
        lease_seconds: int = 60,
    ) -> ToolApprovalRequest | None:
        """Atomically claim an approved MCP request for execution.

        Returns the claimed row (execution_state=EXEC_RUNNING) or None if:
          * request does not exist
          * status is not REQ_APPROVED
          * execution has already completed (EXEC_SUCCEEDED / EXEC_FAILED / EXEC_SKIPPED)
          * another worker holds a fresh RUNNING lease (started_at within lease_seconds)

        A stale RUNNING lease may be reclaimed only after ``lease_seconds`` has
        elapsed and no result_json is present. Attempts counter increments.
        """
        if self.engine is None:
            return None
        now = _utcnow()
        with Session(self.engine) as session:
            row = session.get(ToolApprovalRequest, request_id)
            if row is None:
                return None
            if row.status != REQ_APPROVED:
                return None
            # Already terminal — never re-execute.
            if row.execution_state in (EXEC_SUCCEEDED, EXEC_FAILED, EXEC_SKIPPED):
                return None
            # Live lease held by another worker?
            if row.execution_state == EXEC_RUNNING:
                started = _as_aware_utc(row.execution_started_at)
                if started is not None and row.result_json is None:
                    elapsed = (now - started).total_seconds()
                    if elapsed < lease_seconds:
                        return None
            row.execution_state = EXEC_RUNNING
            row.execution_started_at = now
            row.execution_attempts = int(row.execution_attempts or 0) + 1
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def complete_mcp_execution(
        self,
        request_id: str,
        *,
        result_json: str,
        is_error: bool,
        error: str | None = None,
        skipped: bool = False,
    ) -> ToolApprovalRequest | None:
        """Persist the terminal result of an MCP execution. Idempotent.

        Once a request has a stored result, subsequent complete calls are
        no-ops that return the stored row — this is the exactly-once guarantee
        against duplicate resolve calls and outbox retries.
        """
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            row = session.get(ToolApprovalRequest, request_id)
            if row is None:
                return None
            # Idempotent on already-terminal.
            if row.execution_state in (EXEC_SUCCEEDED, EXEC_FAILED, EXEC_SKIPPED):
                return row
            now = _utcnow()
            row.execution_state = (
                EXEC_SKIPPED if skipped
                else (EXEC_FAILED if is_error else EXEC_SUCCEEDED)
            )
            row.execution_finished_at = now
            row.result_json = result_json
            row.result_is_error = bool(is_error)
            if error:
                row.execution_error = error
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def enqueue_continuation(
        self,
        request_id: str,
    ) -> ToolApprovalRequest | None:
        """Mark a completed MCP execution as CONT_PENDING for the outbox worker.

        No-op if the row is not continuation_capable, has no terminal execution
        result, or is already in a continuation lifecycle. Idempotent.
        """
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            row = session.get(ToolApprovalRequest, request_id)
            if row is None:
                return None
            if not row.continuation_capable:
                return row
            if row.execution_state not in (EXEC_SUCCEEDED, EXEC_FAILED, EXEC_SKIPPED):
                return row
            # Already enqueued / in-flight / delivered — no-op.
            if row.continuation_state != CONT_NOT_NEEDED:
                return row
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
            return None
        now = _utcnow()
        with Session(self.engine) as session:
            row = session.get(ToolApprovalRequest, request_id)
            if row is None:
                return None
            if row.continuation_state == CONT_DELIVERED:
                return None
            if row.continuation_state == CONT_DISPATCHING:
                # Reclaim only after the lease expires and delivery didn't complete.
                if row.continuation_delivered_at is not None:
                    return None
                next_at = _as_aware_utc(row.continuation_next_attempt_at)
                if next_at is not None:
                    elapsed = (now - next_at).total_seconds()
                    if elapsed < lease_seconds:
                        return None
            if row.continuation_state not in (CONT_PENDING, CONT_DISPATCHING):
                return None
            row.continuation_state = CONT_DISPATCHING
            row.continuation_next_attempt_at = now
            row.continuation_attempts = int(row.continuation_attempts or 0) + 1
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def mark_continuation_delivered(
        self,
        request_id: str,
    ) -> ToolApprovalRequest | None:
        """CONT_DISPATCHING → CONT_DELIVERED. Idempotent."""
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            row = session.get(ToolApprovalRequest, request_id)
            if row is None:
                return None
            if row.continuation_state == CONT_DELIVERED:
                return row
            row.continuation_state = CONT_DELIVERED
            row.continuation_delivered_at = _utcnow()
            row.continuation_last_error = None
            row.continuation_next_attempt_at = None
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def mark_continuation_failed(
        self,
        request_id: str,
        *,
        error: str,
        max_attempts: int = 5,
        backoff_seconds: int = 30,
    ) -> ToolApprovalRequest | None:
        """Record a failed continuation dispatch. Reschedules with backoff or
        marks CONT_FAILED after ``max_attempts``. Never regresses a delivered
        row.
        """
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            row = session.get(ToolApprovalRequest, request_id)
            if row is None:
                return None
            if row.continuation_state == CONT_DELIVERED:
                return row
            attempts = int(row.continuation_attempts or 0)
            row.continuation_last_error = (error or "")[:2000]
            if attempts >= max_attempts:
                row.continuation_state = CONT_FAILED
                row.continuation_next_attempt_at = None
            else:
                # Exponential backoff capped at 10 minutes.
                delay = min(backoff_seconds * (2 ** max(0, attempts - 1)), 600)
                row.continuation_state = CONT_PENDING
                row.continuation_next_attempt_at = _utcnow() + timedelta(seconds=delay)
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

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
            return []
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
    ) -> list[ToolApprovalRequest]:
        if self.engine is None:
            return []
        with Session(self.engine) as session:
            stmt = select(ToolApprovalRequest)
            if status:
                stmt = stmt.where(ToolApprovalRequest.status == status)
            if bot_id:
                stmt = stmt.where(ToolApprovalRequest.bot_id == bot_id)
            stmt = stmt.order_by(ToolApprovalRequest.created_at.desc()).limit(limit)
            return list(session.exec(stmt).all())

    # ---- seeding -----------------------------------------------------------

    def seed_defaults(self) -> int:
        """Insert a conservative starter rule set if the table is empty.

        Only unambiguously destructive shell patterns, enabled by default —
        a safety feature that ships disabled protects nothing. Operators can
        disable or delete any rule from the admin UI. See docs/approval-policies.md.
        """
        if self.engine is None:
            raise RuntimeError("Tool approval policies DB unavailable")
        if self.list_all():
            return 0
        defaults = _DEFAULT_POLICIES
        for d in defaults:
            self.create(d, actor="seed")
        return len(defaults)


# Conservative default rule set (TASK-296). High/critical, shell-destructive only.
_DEFAULT_POLICIES: list[dict[str, Any]] = [
    {
        "backend_scope": "*", "tool_name": "Bash", "matcher_type": "regex",
        "pattern": r"\brm\b\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rRf][a-zA-Z]*",
        "action": "require_approval", "severity": "high", "category": "filesystem",
        "approval_prompt": "This will recursively/forcibly delete files. Approve?",
        "order": 10,
    },
    {
        "backend_scope": "*", "tool_name": "Bash", "matcher_type": "prefix",
        "pattern": "sudo ", "action": "require_approval", "severity": "high",
        "category": "privilege", "order": 20,
    },
    {
        "backend_scope": "*", "tool_name": "Bash", "matcher_type": "regex",
        "pattern": r"git\s+push\b.*(--force|-f)\b",
        "action": "require_approval", "severity": "high", "category": "git",
        "approval_prompt": "Force-push can overwrite remote history. Approve?",
        "order": 30,
    },
    {
        "backend_scope": "*", "tool_name": "Bash", "matcher_type": "regex",
        "pattern": r"(?i)\b(DROP\s+TABLE|DROP\s+DATABASE|TRUNCATE)\b",
        "action": "require_approval", "severity": "critical", "category": "database",
        "approval_prompt": "Destructive SQL. Approve?", "order": 40,
    },
    {
        "backend_scope": "*", "tool_name": "Bash", "matcher_type": "regex",
        "pattern": r"(mkfs|dd\s+.*of=/dev/|>\s*/dev/sd|chmod\s+-R\s+777|:\(\)\s*\{)",
        "action": "require_approval", "severity": "critical", "category": "system",
        "approval_prompt": "Potentially system-destroying command. Approve?", "order": 50,
    },
    {
        "backend_scope": "*", "tool_name": "Bash", "matcher_type": "regex",
        "pattern": r"(curl|wget)\b.*\|\s*(sudo\s+)?(ba)?sh\b",
        "action": "require_approval", "severity": "high", "category": "network",
        "approval_prompt": "Piping a remote script straight into a shell. Approve?",
        "order": 60,
    },
]
