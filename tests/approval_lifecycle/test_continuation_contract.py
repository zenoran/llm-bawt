"""Exact authorization requires complete continuation inputs, not UI previews."""
import json

from llm_bawt.service.approval_continuations import build_continuation_prompt


def test_approved_subject_is_not_truncated():
    command = "printf " + "x" * 1000
    assert command in build_continuation_prompt(True, command, "Bash")


def test_approved_continuation_carries_all_original_arguments():
    arguments = {"command": "printf " + "x" * 1000, "timeout": 100, "cwd": "/repo"}
    prompt = build_continuation_prompt(
        True, "short policy field", "Bash", tool_arguments_json=json.dumps(arguments),
    )
    assert json.dumps(arguments, ensure_ascii=False, sort_keys=True) in prompt


def test_preview_explains_structural_ops_fallback():
    from llm_bawt.service.routes.approval_policies import PolicyPreview, preview_policy
    result = preview_policy(PolicyPreview(backend="mcp", tool_name="ops_run", tool_input={}, policies=[]))
    assert result["action"] == "require_approval"
    assert result["reason"] == "No matching policy; default require_approval"
