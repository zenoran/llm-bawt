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
  match → ``allow`` (default-allow, matching today's bypass behaviour), EXCEPT
  for the fail-closed tools in :data:`_FAIL_CLOSED_TOOLS` (``ops_run``), where no
  match → ``require_approval``. Those tools execute operator-authored privileged
  scripts, so the gate must be structural: deleting the catch-all seed row must
  not silently ungate them. An explicit ``allow`` policy still carves exceptions.
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
# else falls back to the whole compact-JSON input (``*``). ``ops_run`` is
# special-cased below because its policy subject deliberately excludes the
# caller-provided idempotency key.
_DEFAULT_FIELD_BY_TOOL = {
    "Bash": "command",
    "BashOutput": "command",
    "Shell": "command",
}


# Tools whose no-match fallback is ``require_approval`` instead of the global
# default-allow (TASK-639). ``ops_run`` executes an operator-authored privileged
# script from the ops catalog; a missing or operator-deleted catch-all policy row
# must not silently turn that into an ungated tool. Keyed on the MCP tool tail so
# ``mcp__bawthub__ops_run`` matches. Read-only ``ops_list_operations`` /
# ``ops_job_status`` are deliberately NOT here — status reads stay ungated.
_FAIL_CLOSED_TOOLS = frozenset({"ops_run"})

# Tools whose subject is a shell command line, and therefore gets the
# inert-data trimming below before a pattern is tested against it.
_SHELL_TOOLS = frozenset(_DEFAULT_FIELD_BY_TOOL)

# Commands whose heredoc body is *data*, not code: they copy the body to a file
# or a pipe and never execute it. Everything else (``bash``, ``sh``, ``python``,
# ``ssh``, ``docker exec … sh``, …) runs what it is fed, so those bodies stay in
# the matched subject. Basename-compared, so ``/bin/cat`` counts.
_HEREDOC_DATA_SINKS = frozenset({"cat", "tee"})

# ``<<TAG`` / ``<<'TAG'`` / ``<<-"TAG"``. Deliberately does NOT match the ``<<<``
# here-string form (that body is a single inline word, not a data block).
_HEREDOC_START_RE = re.compile(r"<<-?[ \t]*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")

# Tokens that may precede the real command word in a pipeline segment.
_COMMAND_PREFIXES = frozenset({"sudo", "command", "nohup", "exec", "time", "env"})


def _heredoc_body_is_inert(line: str, op_index: int) -> bool:
    """Is the heredoc opened at ``op_index`` merely written somewhere, not run?

    Looks at the command word owning the redirect — the head of the last
    pipeline/chain segment before the ``<<``. Unknown or absent command word
    means "assume it executes", so the gate stays conservative.
    """
    segment = re.split(r"\|\||&&|[|;&]", line[:op_index])[-1]
    for token in segment.split():
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
            continue  # VAR=value prefix
        if token[0] in "-<>" or re.match(r"^\d+[<>]", token):
            continue  # flag or redirect
        if token in _COMMAND_PREFIXES:
            continue
        return token.rsplit("/", 1)[-1].strip("\"'") in _HEREDOC_DATA_SINKS
    return False


def strip_inert_heredoc_bodies(command: str) -> str:
    """Drop heredoc bodies that are written as data rather than executed.

    ``cat > file <<'EOF' … EOF`` is a *file write*. Matching a rule's regex
    against the body means the file's contents get judged as if they were the
    command — so writing a test fixture that merely mentions
    ``docker compose restart`` trips the docker rule (TASK-860). The opening
    line, the terminator, and every executable heredoc body are preserved.

    Pure and total: an unterminated or unrecognised heredoc keeps its body.
    """
    if "<<" not in command:
        return command
    lines = command.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        opened = [
            (m.group(2), _heredoc_body_is_inert(line, m.start()))
            for m in _HEREDOC_START_RE.finditer(line)
        ]
        for tag, inert in opened:
            body: list[str] = []
            while i < len(lines) and lines[i].strip() != tag:
                body.append(lines[i])
                i += 1
            terminated = i < len(lines)
            if not inert or not terminated:
                # Unterminated body may not be a heredoc at all — keep it.
                out.extend(body)
            if terminated:
                out.append(lines[i])
                i += 1
    return "\n".join(out)


def matchable_subject(tool_name: str, subject: str) -> str:
    """The text a policy pattern is tested against, given a derived subject.

    Identical to ``subject`` for every tool except the shell ones, where inert
    heredoc bodies are trimmed. The *raw* subject is still what gets recorded,
    displayed, and hashed into the grant key — only matching sees this.
    """
    if not subject or _tool_tail(tool_name) not in _SHELL_TOOLS:
        return subject
    return strip_inert_heredoc_bodies(subject)


def _derive_ops_run_subject(tool_input: Any) -> str:
    """Canonical policy/audit subject for a catalogued operation call.

    The operation slug leads so exact/prefix policies stay readable and stable.
    Args use compact sorted JSON; the idempotency key is transport metadata and
    must not change the approval subject or grant key.
    """
    if not isinstance(tool_input, dict):
        return "operation= args={}"
    operation = tool_input.get("operation", "")
    if operation is None:
        operation = ""
    elif not isinstance(operation, str):
        operation = json.dumps(operation, sort_keys=True, ensure_ascii=False, default=str)
    args = tool_input.get("args")
    if not isinstance(args, dict):
        args = {}
    args_json = json.dumps(
        args,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return f"operation={operation.strip()} args={args_json}"


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
    tool_tail = _tool_tail(tool_name)
    if not fld and tool_tail == "ops_run":
        return _derive_ops_run_subject(tool_input)
    if not fld:
        fld = _DEFAULT_FIELD_BY_TOOL.get(tool_tail, "*")

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


def humanize_subject(subject: str) -> str:
    """Derive a short, human-readable label from a raw approval subject.

    Strips SSH wrappers, output piping, cd prefixes, and multi-command chains
    to surface the primary operation — e.g. ``"bawthub › make rebuild-prod"``
    instead of ``ssh user@host "cd ~/dev/bawthub && make
    rebuild-prod" 2>&1 | tail -20``.

    Pure function — no I/O, no side effects.
    """
    if not subject or not subject.strip():
        return ""

    s = re.sub(r"\s+", " ", subject).strip()

    # ── project detection (from path or container name) ──────────────
    project = ""
    for frag, name in (
        ("dev/bawthub", "bawthub"),
        ("dev/llm-bawt", "llm-bawt"),
        ("dev/agent-skills", "agent-skills"),
    ):
        if frag in s:
            project = name
            break
    if not project:
        for frag, name in (
            ("bawthub-frontend", "bawthub"),
            ("bawthub-traefik", "bawthub"),
            ("llm-bawt-app", "llm-bawt"),
            ("llm-bawt-claude", "llm-bawt"),
        ):
            if frag in s:
                project = name
                break

    def _label(op: str) -> str:
        return f"{project} › {op}" if project else op

    # ── make target ──────────────────────────────────────────────────
    m = re.search(r"\bmake\s+([\w][\w-]*)", s)
    if m:
        return _label(f"make {m.group(1)}")

    # ── docker compose <action> [service] ────────────────────────────
    m = re.search(
        r"\bdocker(?:\s+|-)?compose\b[^|;&]*?"
        r"\b(up|down|restart|stop|start|build|logs|ps|config|rm|pull)\b"
        r"([^|;&]*)",
        s,
    )
    if m:
        action = m.group(1)
        rest_tokens = m.group(2).split()
        service = ""
        for tok in rest_tokens:
            bare = tok.strip("\"'")
            if bare.startswith("-"):
                continue
            if re.match(r"^2?>&?\d?$", bare):
                break
            if re.match(r"^[\w][\w.-]*$", bare):
                service = bare
                break
        op = f"docker compose {action}" + (f" {service}" if service else "")
        return _label(op)

    # ── docker <verb> <container> ────────────────────────────────────
    m = re.search(
        r"\bdocker\s+(restart|stop|start|kill|rm|run|exec|inspect|logs|images)"
        r"\s+([\w][\w.-]*)",
        s,
    )
    if m:
        return _label(f"docker {m.group(1)} {m.group(2)}")

    # ── git ──────────────────────────────────────────────────────────
    m = re.search(
        r"\bgit\s+(checkout|switch|push|pull|merge|rebase|reset|commit|add|branch)"
        r"\b(.*?)(?:\s*[;&|]|\s*$)",
        s,
        re.DOTALL,
    )
    if m:
        args = re.sub(r"\s+", " ", m.group(2)).strip()
        if len(args) > 40:
            args = args[:37] + "…"
        op = f"git {m.group(1)}" + (f" {args}" if args else "")
        return _label(op)

    # ── python scripts ───────────────────────────────────────────────
    if re.match(r"(?:python3?|\.venv)", s):
        return _label("python script")

    # ── fallback: truncate ───────────────────────────────────────────
    if len(s) > 72:
        return s[:69] + "…"
    return s


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
    label: str = ""

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
        if not policy.matches_subject(matchable_subject(tool_name, subject)):
            continue
        prompt = policy.approval_prompt or _default_prompt(tool_name, subject)
        return ApprovalDecision(
            action=policy.action,
            subject=subject,
            policy=policy,
            severity=policy.severity,
            prompt=prompt,
            grant_key=grant_key(backend, tool_name, subject),
            label=humanize_subject(subject),
        )

    # No policy matched. Derive the subject once for the key so callers logging
    # the call still get a stable identifier.
    subject = derive_subject(tool_name, tool_input, None)
    if _tool_tail(tool_name) in _FAIL_CLOSED_TOOLS:
        # Fail closed: a privileged catalogued operation is never ungated just
        # because nobody wrote (or somebody deleted) a policy row for it.
        return ApprovalDecision(
            action=PolicyAction.REQUIRE_APPROVAL,
            subject=subject,
            policy=None,
            severity=Severity.HIGH,
            prompt=(
                "No approval policy matched this operation, and catalogued "
                f"operations are gated by default.\n\n{subject}"
            ),
            grant_key=grant_key(backend, tool_name, subject),
            label=humanize_subject(subject),
        )
    return ApprovalDecision(
        action=PolicyAction.ALLOW,
        subject=subject,
        policy=None,
        grant_key=grant_key(backend, tool_name, subject),
        label=humanize_subject(subject),
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
