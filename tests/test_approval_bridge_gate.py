"""Tests for the claude-code bridge approval gate decision logic (TASK-292).

Exercises the bridge's _decide_approval / grant store / _evaluate_tool_gate
against injected policy bundles and grants — no Redis, no live SDK turn, no
HTTP. Skips cleanly if claude_agent_sdk isn't importable (i.e. when run outside
the bridge container).

Runnable under pytest, or standalone: ``python tests/test_approval_bridge_gate.py``.
"""

from __future__ import annotations

import asyncio
import time

try:
    from claude_code_bridge.bridge import ClaudeCodeBridge
    from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny

    _SDK_OK = True
except Exception:  # noqa: BLE001
    _SDK_OK = False

from agent_bridge.approval import ApprovalPolicy, MatcherType, PolicyAction, PolicyBundle


class _FakePub:
    class _R:
        connection_pool = None

    _redis = _R()

    def close(self):
        pass


def _bridge():
    b = ClaudeCodeBridge(_FakePub(), backend_name="claude-code", app_api_url="")
    pols = [
        ApprovalPolicy(id="allow1", order=10, tool_name="Bash",
                       matcher_type=MatcherType.PREFIX, pattern="rm -rf /tmp/safe",
                       action=PolicyAction.ALLOW),
        ApprovalPolicy(id="req1", order=100, tool_name="Bash",
                       matcher_type=MatcherType.PREFIX, pattern="rm -rf",
                       action=PolicyAction.REQUIRE_APPROVAL),
        ApprovalPolicy(id="deny1", order=5, tool_name="Bash",
                       matcher_type=MatcherType.CONTAINS, pattern="/dev/sda",
                       action=PolicyAction.DENY),
    ]
    b._policy_bundle = PolicyBundle(version=1, etag="x", policies=pols)
    b._policy_bundle_fetched_at = time.monotonic()  # fresh → no HTTP
    return b


def _skip_if_no_sdk():
    if not _SDK_OK:
        try:
            import pytest

            pytest.skip("claude_agent_sdk not importable (run inside bridge container)")
        except Exception:
            raise SystemExit(0)


def test_decide_allow_require_deny_default():
    _skip_if_no_sdk()
    b = _bridge()
    assert b._decide_approval("Bash", {"command": "rm -rf /tmp/safe/x"}).is_allowed
    assert b._decide_approval("Bash", {"command": "rm -rf /home/x"}).requires_approval
    assert b._decide_approval("Bash", {"command": "dd of=/dev/sda"}).is_denied
    assert b._decide_approval("Bash", {"command": "ls"}).is_allowed


def test_grant_one_shot_and_expiry():
    _skip_if_no_sdk()
    b = _bridge()
    gk = b._decide_approval("Bash", {"command": "rm -rf /home/x"}).grant_key
    assert not b._consume_grant(gk)
    b._grant_approval(gk, 600)
    assert b._consume_grant(gk)
    assert not b._consume_grant(gk)  # one-shot
    b._approval_grants["k2"] = time.monotonic() - 1  # already expired
    assert not b._consume_grant("k2")
    assert "k2" not in b._approval_grants  # pruned


def test_gate_allow_deny_require_emits_event():
    _skip_if_no_sdk()
    b = _bridge()

    class Ctx:
        tool_use_id = "tuid-1"

    captured: list[dict] = []
    b._publish_event = lambda *a, **k: captured.append(k)  # type: ignore[assignment]
    seq = [0]

    async def gate(cmd):
        return await b._evaluate_tool_gate("Bash", {"command": cmd}, Ctx(), "req", "sess", seq)

    assert isinstance(asyncio.run(gate("ls")), PermissionResultAllow)
    assert isinstance(asyncio.run(gate("dd of=/dev/sda")), PermissionResultDeny)
    assert isinstance(asyncio.run(gate("rm -rf /home/x")), PermissionResultDeny)
    assert captured[-1]["kind"].value == "approval_required"
    assert captured[-1]["tool_use_id"] == "tuid-1"
    assert captured[-1]["extra_raw"]["grant_key"]


def test_gate_granted_reissue_allows():
    _skip_if_no_sdk()
    b = _bridge()

    class Ctx:
        tool_use_id = "tuid-1"

    b._publish_event = lambda *a, **k: None  # type: ignore[assignment]
    gk = b._decide_approval("Bash", {"command": "rm -rf /home/x"}).grant_key
    b._grant_approval(gk, 600)

    async def gate():
        return await b._evaluate_tool_gate("Bash", {"command": "rm -rf /home/x"}, Ctx(), "r", "s", [0])

    assert isinstance(asyncio.run(gate()), PermissionResultAllow)


def test_gate_fail_closed_denies_when_unreachable():
    _skip_if_no_sdk()
    b = _bridge()
    b._approval_fail_closed = True
    b._policy_fetch_ok = False

    class Ctx:
        tool_use_id = "t"

    async def gate():
        return await b._evaluate_tool_gate("Bash", {"command": "ls"}, Ctx(), "r", "s", [0])

    assert isinstance(asyncio.run(gate()), PermissionResultDeny)


def _hook_decision(out: dict) -> str | None:
    """Extract permissionDecision from a PreToolUse hook output dict."""
    return (out or {}).get("hookSpecificOutput", {}).get("permissionDecision")


def test_hook_gate_allow_deny_require_emits_event():
    _skip_if_no_sdk()
    b = _bridge()
    captured: list[dict] = []
    b._publish_event = lambda *a, **k: captured.append(k)  # type: ignore[assignment]
    seq = [0]

    async def gate(cmd):
        return await b._evaluate_tool_gate_hook(
            "Bash", {"command": cmd}, "tuid-1", "req", "sess", seq
        )

    # ALLOW → empty dict (no decision; tool proceeds)
    assert asyncio.run(gate("ls")) == {}
    # DENY → permissionDecision="deny"
    assert _hook_decision(asyncio.run(gate("dd of=/dev/sda"))) == "deny"
    # REQUIRE_APPROVAL with no grant → "deny" + APPROVAL_REQUIRED event
    assert _hook_decision(asyncio.run(gate("rm -rf /home/x"))) == "deny"
    assert captured[-1]["kind"].value == "approval_required"
    assert captured[-1]["tool_use_id"] == "tuid-1"
    assert captured[-1]["extra_raw"]["grant_key"]


def test_hook_gate_granted_reissue_allows():
    _skip_if_no_sdk()
    b = _bridge()
    b._publish_event = lambda *a, **k: None  # type: ignore[assignment]
    gk = b._decide_approval("Bash", {"command": "rm -rf /home/x"}).grant_key
    b._grant_approval(gk, 600)

    async def gate():
        return await b._evaluate_tool_gate_hook(
            "Bash", {"command": "rm -rf /home/x"}, "tuid-1", "r", "s", [0]
        )

    # consumed grant → allow (empty dict, no decision)
    assert asyncio.run(gate()) == {}


def test_hook_gate_fail_closed_denies_when_unreachable():
    _skip_if_no_sdk()
    b = _bridge()
    b._approval_fail_closed = True
    b._policy_fetch_ok = False

    async def gate():
        return await b._evaluate_tool_gate_hook("Bash", {"command": "ls"}, "t", "r", "s", [0])

    assert _hook_decision(asyncio.run(gate())) == "deny"


def test_hook_gate_no_tool_use_id_still_denies_on_require():
    _skip_if_no_sdk()
    b = _bridge()
    b._publish_event = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not publish"))  # type: ignore[assignment]

    async def gate():
        # empty tool_use_id → can't persist a request row, but must still block
        return await b._evaluate_tool_gate_hook("Bash", {"command": "rm -rf /home/x"}, "", "r", "s", [0])

    assert _hook_decision(asyncio.run(gate())) == "deny"


def test_pre_tool_use_hook_passes_through_ask_user_question():
    _skip_if_no_sdk()
    b = _bridge()
    hook = b._make_pre_tool_use_hook(request_id="r", session_key="s", seq_holder=[0])

    async def run(name):
        return await hook({"tool_name": name, "tool_input": {}, "tool_use_id": "x"}, "x", {})

    # AskUserQuestion (bare + MCP-namespaced) must NOT be gated — handed back to SDK
    assert asyncio.run(run("AskUserQuestion")) == {}
    assert asyncio.run(run("mcp__foo__AskUserQuestion")) == {}


def test_pre_tool_use_hook_gates_dangerous_bash():
    _skip_if_no_sdk()
    b = _bridge()
    b._publish_event = lambda *a, **k: None  # type: ignore[assignment]
    hook = b._make_pre_tool_use_hook(request_id="r", session_key="s", seq_holder=[0])

    async def run(cmd):
        return await hook(
            {"tool_name": "Bash", "tool_input": {"command": cmd}, "tool_use_id": "x"},
            "x", {},
        )

    assert asyncio.run(run("ls")) == {}                                  # allow
    assert _hook_decision(asyncio.run(run("dd of=/dev/sda"))) == "deny"  # deny
    assert _hook_decision(asyncio.run(run("rm -rf /home/x"))) == "deny"  # require→deny


def test_pre_tool_use_hook_fails_open_on_unexpected_error():
    _skip_if_no_sdk()
    b = _bridge()

    def boom(*a, **k):
        raise RuntimeError("bundle blew up")

    b._get_policy_bundle = boom  # type: ignore[assignment]
    hook = b._make_pre_tool_use_hook(request_id="r", session_key="s", seq_holder=[0])

    async def run():
        return await hook(
            {"tool_name": "Bash", "tool_input": {"command": "ls"}, "tool_use_id": "x"},
            "x", {},
        )

    # default posture: gate error must NOT wedge tools
    assert asyncio.run(run()) == {}
    # fail-closed posture: gate error blocks instead
    b._approval_fail_closed = True
    assert _hook_decision(asyncio.run(run())) == "deny"


if __name__ == "__main__":
    import sys
    import traceback

    if not _SDK_OK:
        print("SKIP: claude_agent_sdk not importable here")
        sys.exit(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
        except SystemExit:
            print(f"SKIP {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
