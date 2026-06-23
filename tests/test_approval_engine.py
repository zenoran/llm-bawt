"""Unit tests for the pure approval-policy engine (agent_bridge.approval).

Pure, dependency-free assertions — runnable under pytest, or standalone via
``python tests/test_approval_engine.py`` when pytest isn't installed (the
bridge/app containers don't ship it).
"""

from __future__ import annotations

from agent_bridge.approval import (
    ApprovalPolicy,
    MatcherType,
    PolicyAction,
    PolicyBundle,
    Severity,
    compute_etag,
    derive_subject,
    evaluate,
    grant_key,
)


def _pol(**kw) -> ApprovalPolicy:
    base = dict(id="p1", tool_name="Bash", matcher_type=MatcherType.ALWAYS)
    base.update(kw)
    # allow string enums through the tolerant coercers
    if isinstance(base.get("matcher_type"), str):
        base["matcher_type"] = MatcherType.coerce(base["matcher_type"])
    if isinstance(base.get("action"), str):
        base["action"] = PolicyAction.coerce(base["action"])
    if isinstance(base.get("severity"), str):
        base["severity"] = Severity.coerce(base["severity"])
    return ApprovalPolicy(**base)


# ---- derive_subject --------------------------------------------------------

def test_derive_subject_bash_defaults_to_command():
    assert derive_subject("Bash", {"command": "rm -rf /x"}, None) == "rm -rf /x"


def test_derive_subject_mcp_namespaced_bash():
    assert derive_subject("mcp__srv__Bash", {"command": "ls"}, None) == "ls"


def test_derive_subject_explicit_field():
    assert derive_subject("Write", {"file_path": "/etc/passwd"}, "file_path") == "/etc/passwd"


def test_derive_subject_whole_input_json():
    s = derive_subject("Write", {"b": 2, "a": 1}, "*")
    assert s == '{"a": 1, "b": 2}'  # sorted keys


def test_derive_subject_missing_field_is_empty():
    assert derive_subject("Write", {"x": 1}, "file_path") == ""


# ---- matcher semantics -----------------------------------------------------

def test_always_matches():
    assert _pol(matcher_type=MatcherType.ALWAYS).matches_subject("anything")


def test_exact_match_trims():
    p = _pol(matcher_type=MatcherType.EXACT, pattern="rm -rf /")
    assert p.matches_subject("  rm -rf / ")
    assert not p.matches_subject("rm -rf /tmp")


def test_prefix_match():
    p = _pol(matcher_type=MatcherType.PREFIX, pattern="sudo ")
    assert p.matches_subject("sudo apt update")
    assert p.matches_subject("   sudo apt update")  # leading ws stripped
    assert not p.matches_subject("echo sudo")


def test_contains_match():
    p = _pol(matcher_type=MatcherType.CONTAINS, pattern="rm -rf")
    assert p.matches_subject("cd /tmp && rm -rf foo")


def test_glob_match():
    p = _pol(matcher_type=MatcherType.GLOB, pattern="git push*")
    assert p.matches_subject("git push origin main")
    assert not p.matches_subject("git status")


def test_regex_match():
    p = _pol(matcher_type=MatcherType.REGEX, pattern=r"\brm\b.*-[a-z]*f")
    assert p.matches_subject("rm -rf /x")
    assert not p.matches_subject("confirm something")


def test_bad_regex_never_throws_never_matches():
    p = _pol(matcher_type=MatcherType.REGEX, pattern="(unclosed")
    # must not raise, and must not match
    assert p.matches_subject("(unclosed") is False


# ---- applies_to scoping ----------------------------------------------------

def test_backend_scope_wildcard_and_specific():
    p = _pol(backend_scope="claude-code")
    assert p.applies_to("claude-code", "Bash")
    assert not p.applies_to("codex", "Bash")
    assert _pol(backend_scope="*").applies_to("codex", "Bash")


def test_tool_wildcard():
    p = _pol(tool_name="*")
    assert p.applies_to("claude-code", "Write")


def test_disabled_policy_never_applies():
    assert not _pol(enabled=False).applies_to("claude-code", "Bash")


# ---- evaluate: ordering + first-match-wins ---------------------------------

def test_default_allow_when_no_policy():
    d = evaluate([], "claude-code", "Bash", {"command": "ls"})
    assert d.action is PolicyAction.ALLOW
    assert d.policy is None


def test_require_approval_match():
    pols = [_pol(matcher_type=MatcherType.PREFIX, pattern="rm -rf", action="require_approval")]
    d = evaluate(pols, "claude-code", "Bash", {"command": "rm -rf /x"})
    assert d.requires_approval
    assert d.subject == "rm -rf /x"
    assert d.grant_key  # populated
    assert "rm -rf /x" in d.prompt


def test_allow_rule_overrides_lower_require_rule_by_order():
    # order 10 allow carves a safe hole out of the order 100 require-all rule
    allow_rule = _pol(id="a", order=10, matcher_type=MatcherType.PREFIX,
                      pattern="git status", action="allow")
    require_rule = _pol(id="b", order=100, matcher_type=MatcherType.PREFIX,
                        pattern="git", action="require_approval")
    pols = [require_rule, allow_rule]  # deliberately out of order in the list
    d = evaluate(pols, "claude-code", "Bash", {"command": "git status -s"})
    assert d.is_allowed
    # but a different git command still trips the require rule
    d2 = evaluate(pols, "claude-code", "Bash", {"command": "git push"})
    assert d2.requires_approval


def test_deny_action():
    pols = [_pol(matcher_type=MatcherType.CONTAINS, pattern="rm -rf /", action="deny",
                 severity="critical")]
    d = evaluate(pols, "claude-code", "Bash", {"command": "rm -rf / --no-preserve-root"})
    assert d.is_denied
    assert d.severity is Severity.CRITICAL


def test_scope_filters_out_other_backend():
    pols = [_pol(backend_scope="codex", matcher_type=MatcherType.ALWAYS,
                 action="require_approval")]
    d = evaluate(pols, "claude-code", "Bash", {"command": "anything"})
    assert d.is_allowed  # codex-scoped rule doesn't apply to claude-code


# ---- grant key stability ---------------------------------------------------

def test_grant_key_stable_across_whitespace():
    k1 = grant_key("claude-code", "Bash", "rm  -rf   /x")
    k2 = grant_key("claude-code", "Bash", "rm -rf /x")
    assert k1 == k2


def test_grant_key_differs_by_command():
    assert grant_key("claude-code", "Bash", "rm -rf /x") != grant_key(
        "claude-code", "Bash", "rm -rf /y"
    )


def test_grant_key_ignores_mcp_namespacing():
    assert grant_key("claude-code", "mcp__s__Bash", "ls") == grant_key(
        "claude-code", "Bash", "ls"
    )


def test_evaluate_grant_key_matches_standalone():
    d = evaluate([_pol(action="require_approval")], "claude-code", "Bash", {"command": "x y"})
    assert d.grant_key == grant_key("claude-code", "Bash", "x y")


# ---- serialization round-trips ---------------------------------------------

def test_policy_dict_roundtrip():
    p = _pol(matcher_type=MatcherType.REGEX, pattern="rm.*", action="deny",
             severity="high", category="filesystem", approval_prompt="careful!",
             order=5, version=3)
    p2 = ApprovalPolicy.from_dict(p.to_dict())
    assert p2 == p


def test_bundle_roundtrip_and_etag_deterministic():
    pols = [_pol(id="a"), _pol(id="b", order=5)]
    etag = compute_etag(2, pols)
    b = PolicyBundle(version=2, etag=etag, policies=pols)
    b2 = PolicyBundle.from_dict(b.to_dict())
    assert b2.policies == pols
    assert b2.version == 2
    # etag stable regardless of input ordering
    assert compute_etag(2, list(reversed(pols))) == etag


def test_etag_changes_with_content():
    a = compute_etag(1, [_pol(id="a", pattern="x")])
    b = compute_etag(1, [_pol(id="a", pattern="y")])
    assert a != b


def test_tolerant_coercion_of_garbage_enum_values():
    p = ApprovalPolicy.from_dict({"id": "x", "matcher_type": "bogus",
                                  "action": "nope", "severity": "weird"})
    assert p.matcher_type is MatcherType.EXACT
    assert p.action is PolicyAction.REQUIRE_APPROVAL
    assert p.severity is Severity.MEDIUM


if __name__ == "__main__":
    # Standalone runner for environments without pytest.
    import sys
    import traceback

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
