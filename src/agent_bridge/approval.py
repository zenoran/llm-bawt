"""Approval-gated tool policy engine (pure, dependency-free).

This module is the single source of truth for *evaluating* approval policies.
It is deliberately I/O-free so the exact same matching semantics run in two
places:

* the **app** (``llm_bawt``) — compiles DB rows into ``ApprovalPolicy`` bundles
  and serves them to bridges, and recomputes grant keys when a user resolves an
  approval; and
* the **bridges** (``claude_code_bridge``, ``codex_bridge``) — evaluate the
  compiled bundle inside the per-tool permission hook before a tool runs.

Storage (SQLModel), HTTP, Redis, and SDK glue all live elsewhere. Keep this
file pure: stdlib only, no logging side effects on the hot path, deterministic.

Design (TASK-289):

* A policy targets a ``backend_scope`` ("*" = any bridge) and a ``tool_name``
  ("*" = any tool), matches a ``subject`` string derived from the tool input via
  ``field`` using one ``matcher_type``/``pattern``, and yields an ``action``.
* ``action`` is data-driven (TASK-289 favours data over hardcoded command
  checks): ``require_approval`` gates the call, ``allow`` whitelists it, ``deny``
  hard-blocks it. ``allow`` rules placed at a lower ``order`` let an operator
  carve safe exceptions out of a broad ``require_approval`` rule below them.
* Evaluation is **first match wins** over policies sorted by ``(order, id)``. No
  match → ``allow`` (default-allow, matching today's bypass behaviour).
* The **grant key** is a stable hash of (backend, tool, normalized subject). The
  bridge stores a grant under this key when the app tells it an approval was
  granted, and consumes it when the model re-issues the identical call on the
  continuation turn.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Bundle/protocol version. Bump when the compiled-bundle shape changes in a way
# a bridge must notice; bridges log a warning on a major mismatch (TASK-289
# "document failure and versioning semantics").
BUNDLE_VERSION = 1


class MatcherType(str, Enum):
    """How ``pattern`` is tested against the derived subject string."""

    ALWAYS = "always"        # match every invocation of the targeted tool
    EXACT = "exact"          # subject == pattern (after strip)
    PREFIX = "prefix"        # subject starts with pattern
    CONTAINS = "contains"    # pattern is a substring of subject
    GLOB = "glob"            # fnmatch glob (e.g. "rm -rf *")
    REGEX = "regex"          # re.search(pattern, subject)

    @classmethod
    def coerce(cls, value: Any) -> "MatcherType":
        """Tolerant parse — unknown/garbage degrades to EXACT, never raises."""
        if isinstance(value, MatcherType):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.EXACT


class PolicyAction(str, Enum):
    """The decision a matching policy produces."""

    REQUIRE_APPROVAL = "require_approval"
    ALLOW = "allow"
    DENY = "deny"

    @classmethod
    def coerce(cls, value: Any) -> "PolicyAction":
        if isinstance(value, PolicyAction):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.REQUIRE_APPROVAL


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def coerce(cls, value: Any) -> "Severity":
        if isinstance(value, Severity):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.MEDIUM


def _tool_tail(tool_name: str) -> str:
    """Strip MCP namespacing so ``mcp__srv__Bash`` targets a ``Bash`` policy.

    Mirrors ``_is_ask_user_question`` in the claude-code bridge.
    """
    if not tool_name:
        return ""
    return tool_name.rsplit("__", 1)[-1]


# Per-tool default field to derive the subject from, when a policy leaves
# ``field`` blank. Bash/shell calls match against the command string; anything
# else falls back to the whole compact-JSON input (``*``).
_DEFAULT_FIELD_BY_TOOL = {
    "Bash": "command",
    "BashOutput": "command",
    "Shell": "command",
}


def derive_subject(tool_name: str, tool_input: Any, field_name: str | None) -> str:
    """Compute the string a matcher tests against (TASK-292 normalization).

    * Explicit ``field_name`` (other than "*") selects that key from a dict
      input; a missing key yields "".
    * Blank ``field_name`` uses a per-tool default ("command" for shell tools),
      else the whole input.
    * "*" (or a non-dict input) serializes the entire input to compact, sorted
      JSON so a policy can match against any part of it.
    """
    fld = (field_name or "").strip()
    if not fld:
        fld = _DEFAULT_FIELD_BY_TOOL.get(_tool_tail(tool_name), "*")

    if fld != "*" and isinstance(tool_input, dict):
        val = tool_input.get(fld, "")
        if isinstance(val, str):
            return val
        if val is None:
            return ""
        return json.dumps(val, sort_keys=True, ensure_ascii=False, default=str)

    if isinstance(tool_input, str):
        return tool_input
    try:
        return json.dumps(tool_input, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(tool_input)


def _normalize_subject(subject: str) -> str:
    """Collapse whitespace so grant keys are stable across trivial reformatting.

    The model rarely reproduces a command byte-for-byte on the continuation
    turn (leading/trailing spaces, doubled spaces). Collapsing runs of
    whitespace to single spaces keeps the grant key matchable without being so
    loose that a different command sneaks through.
    """
    return re.sub(r"\s+", " ", subject).strip()


def grant_key(backend: str, tool_name: str, subject: str) -> str:
    """Stable hash identifying one approved (backend, tool, command) triple.

    Computed identically by the app (when recording a grant) and the bridge
    (when consuming it on the continuation turn). Uses the tool *tail* so MCP
    namespacing doesn't fork the key.
    """
    canonical = "\x1f".join(
        [
            (backend or "*").strip().lower(),
            _tool_tail(tool_name),
            _normalize_subject(subject),
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8", errors="surrogateescape")).hexdigest()


@dataclass(frozen=True)
class ApprovalPolicy:
    """One compiled, immutable approval rule. The wire/eval shape bridges see."""

    id: str
    backend_scope: str = "*"
    tool_name: str = "*"
    matcher_type: MatcherType = MatcherType.ALWAYS
    pattern: str = ""
    field: str = ""
    action: PolicyAction = PolicyAction.REQUIRE_APPROVAL
    severity: Severity = Severity.MEDIUM
    category: str | None = None
    approval_prompt: str | None = None
    order: int = 100
    enabled: bool = True
    version: int = 1

    def applies_to(self, backend: str, tool_name: str) -> bool:
        """Does this policy target this backend + tool (ignoring the matcher)?"""
        if not self.enabled:
            return False
        scope = (self.backend_scope or "*").strip()
        if scope not in ("*", "") and scope.lower() != (backend or "").strip().lower():
            return False
        want = (self.tool_name or "*").strip()
        if want in ("*", ""):
            return True
        return _tool_tail(tool_name) == _tool_tail(want)

    def matches_subject(self, subject: str) -> bool:
        """Test this policy's matcher against an already-derived subject."""
        mt = self.matcher_type
        if mt is MatcherType.ALWAYS:
            return True
        pat = self.pattern or ""
        if mt is MatcherType.EXACT:
            return subject.strip() == pat.strip()
        if mt is MatcherType.PREFIX:
            return subject.lstrip().startswith(pat)
        if mt is MatcherType.CONTAINS:
            return pat in subject
        if mt is MatcherType.GLOB:
            return fnmatch.fnmatch(subject, pat)
        if mt is MatcherType.REGEX:
            try:
                return re.search(pat, subject) is not None
            except re.error:
                # A malformed regex must never throw on the hot path and must
                # never silently match — treat as no-match. The admin UI should
                # validate patterns before save; this is the runtime backstop.
                return False
        return False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "backend_scope": self.backend_scope,
            "tool_name": self.tool_name,
            "matcher_type": self.matcher_type.value,
            "pattern": self.pattern,
            "field": self.field,
            "action": self.action.value,
            "severity": self.severity.value,
            "category": self.category,
            "approval_prompt": self.approval_prompt,
            "order": self.order,
            "enabled": self.enabled,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ApprovalPolicy":
        return cls(
            id=str(data.get("id", "")),
            backend_scope=str(data.get("backend_scope", "*") or "*"),
            tool_name=str(data.get("tool_name", "*") or "*"),
            matcher_type=MatcherType.coerce(data.get("matcher_type", "always")),
            pattern=str(data.get("pattern", "") or ""),
            field=str(data.get("field", "") or ""),
            action=PolicyAction.coerce(data.get("action", "require_approval")),
            severity=Severity.coerce(data.get("severity", "medium")),
            category=(data.get("category") or None),
            approval_prompt=(data.get("approval_prompt") or None),
            order=int(data.get("order", 100) or 0),
            enabled=bool(data.get("enabled", True)),
            version=int(data.get("version", 1) or 1),
        )


@dataclass(frozen=True)
class ApprovalDecision:
    """Outcome of evaluating a tool call against a policy bundle."""

    action: PolicyAction
    subject: str
    policy: ApprovalPolicy | None = None
    severity: Severity = Severity.MEDIUM
    prompt: str = ""
    grant_key: str = ""

    @property
    def requires_approval(self) -> bool:
        return self.action is PolicyAction.REQUIRE_APPROVAL

    @property
    def is_denied(self) -> bool:
        return self.action is PolicyAction.DENY

    @property
    def is_allowed(self) -> bool:
        return self.action is PolicyAction.ALLOW


def _default_prompt(tool_name: str, subject: str) -> str:
    tail = _tool_tail(tool_name) or "tool"
    shown = subject if len(subject) <= 300 else subject[:297] + "…"
    if tail in ("Bash", "Shell"):
        return f"Approve running this command?\n\n{shown}"
    return f"Approve {tail}?\n\n{shown}"


def evaluate(
    policies: list[ApprovalPolicy],
    backend: str,
    tool_name: str,
    tool_input: Any,
) -> ApprovalDecision:
    """Evaluate a tool call against the bundle. First applicable match wins.

    Returns an ``allow`` decision when nothing matches (default-allow). The
    returned decision always carries the derived ``subject`` and a precomputed
    ``grant_key`` so the caller doesn't recompute them.
    """
    subject = ""
    for policy in sorted(policies, key=lambda p: (p.order, p.id)):
        if not policy.applies_to(backend, tool_name):
            continue
        subject = derive_subject(tool_name, tool_input, policy.field)
        if not policy.matches_subject(subject):
            continue
        prompt = policy.approval_prompt or _default_prompt(tool_name, subject)
        return ApprovalDecision(
            action=policy.action,
            subject=subject,
            policy=policy,
            severity=policy.severity,
            prompt=prompt,
            grant_key=grant_key(backend, tool_name, subject),
        )

    # No policy matched → default allow. Derive the subject once for the key so
    # callers logging the (allowed) call still get a stable identifier.
    subject = derive_subject(tool_name, tool_input, None)
    return ApprovalDecision(
        action=PolicyAction.ALLOW,
        subject=subject,
        policy=None,
        grant_key=grant_key(backend, tool_name, subject),
    )


@dataclass(frozen=True)
class PolicyBundle:
    """A versioned, compiled set of policies a bridge fetches over HTTP.

    ``etag`` lets a bridge skip re-parsing when nothing changed (the app
    computes it from the policy contents + version).
    """

    version: int
    etag: str
    policies: list[ApprovalPolicy] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "etag": self.etag,
            "bundle_version": BUNDLE_VERSION,
            "policies": [p.to_dict() for p in self.policies],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyBundle":
        return cls(
            version=int(data.get("version", 1) or 1),
            etag=str(data.get("etag", "") or ""),
            policies=[ApprovalPolicy.from_dict(p) for p in (data.get("policies") or [])],
        )


def compute_etag(version: int, policies: list[ApprovalPolicy]) -> str:
    """Deterministic etag over the compiled bundle contents."""
    canonical = json.dumps(
        {
            "v": version,
            "bv": BUNDLE_VERSION,
            "p": [p.to_dict() for p in sorted(policies, key=lambda p: (p.order, p.id))],
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8", errors="surrogateescape")).hexdigest()[:16]
