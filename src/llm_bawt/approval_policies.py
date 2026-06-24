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
from datetime import datetime, timezone
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
)

from .utils.config import Config, has_database_credentials

logger = logging.getLogger(__name__)


def _new_id() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
            "grant_key": self.grant_key,
            "policy_id": self.policy_id,
            "severity": self.severity,
            "prompt": self.prompt,
            "status": self.status,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "resolved_turn_id": self.resolved_turn_id,
        }


# Fields a caller may set on create/update. Anything else is ignored.
_POLICY_WRITABLE = {
    "enabled", "backend_scope", "tool_name", "matcher_type", "pattern",
    "field", "action", "severity", "category", "approval_prompt", "order",
}


class ToolApprovalPolicyStore:
    """DB access for approval policies + request audit log."""

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
        SQLModel.metadata.create_all(
            self.engine,
            tables=[ToolApprovalPolicy.__table__, ToolApprovalRequest.__table__],
        )

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
    ) -> ToolApprovalRequest | None:
        """Persist a new pending approval. Idempotent on request_id."""
        if self.engine is None:
            return None
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
        except Exception:  # noqa: BLE001
            logger.exception("Failed to record approval request id=%s", request_id)
            return None

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
