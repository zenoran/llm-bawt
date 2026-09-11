from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_bawt.mcp_server import step_tools


def run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.mark.parametrize("tool_name", ["add_steps", "set_steps"])
def test_step_list_tools_reject_invalid_type_before_http(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
) -> None:
    async def unexpected_request(*args: Any, **kwargs: Any) -> dict:
        raise AssertionError("invalid step types must not reach BawtHub")

    monkeypatch.setattr(step_tools, "_api_post", unexpected_request)
    monkeypatch.setattr(step_tools, "_api_put", unexpected_request)

    result = run(getattr(step_tools, tool_name)(
        "TASK-872",
        [{"title": "Deploy", "type": "DEPLOY"}],  # type: ignore[list-item]
    ))

    assert result == {
        "error": (
            "steps[0].type must be one of: PLAN, READ_FILE, EDIT_FILE, "
            "CREATE_FILE, DELETE_FILE, RUN_COMMAND, SEARCH, ASK_USER, REVIEW"
        )
    }


@pytest.mark.parametrize(
    ("tool_name", "transport_name", "expected_path"),
    [
        ("add_steps", "_api_post", "/tasks/TASK-872/steps"),
        ("set_steps", "_api_put", "/tasks/TASK-872/steps"),
    ],
)
def test_step_list_tools_forward_valid_types(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    transport_name: str,
    expected_path: str,
) -> None:
    calls: list[tuple[str, list[dict]]] = []

    async def fake_request(
        path: str,
        json: list[dict],
        headers: dict | None = None,
    ) -> list[dict]:
        calls.append((path, json))
        return json

    monkeypatch.setattr(step_tools, transport_name, fake_request)
    steps = [{"title": "Run tests", "type": "RUN_COMMAND"}]

    assert run(getattr(step_tools, tool_name)("TASK-872", steps)) == steps
    assert calls == [(expected_path, steps)]
