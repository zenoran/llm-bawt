import asyncio
import ast
import logging
from pathlib import Path

import httpx
import pytest

from codex_bridge.bridge import CodexBridge


class _Response:
    def __init__(self, *, error=None):
        self._error = error
        self.raise_calls = 0

    def raise_for_status(self):
        self.raise_calls += 1
        if self._error:
            raise self._error


class _Client:
    put_response: _Response
    put_calls: list[tuple[str, dict, dict]] = []

    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def put(self, url, *, params, json):
        self.put_calls.append((url, params, json))
        return self.put_response


def test_set_thread_session_checks_put_status(monkeypatch, caplog):
    error = httpx.HTTPStatusError(
        "thread update failed",
        request=httpx.Request("PUT", "http://app/v1/sessions/t-1/agent-session-key"),
        response=httpx.Response(500),
    )
    _Client.put_response = _Response(error=error)
    _Client.put_calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    bridge = CodexBridge.__new__(CodexBridge)
    bridge._app_api_url = "http://app"

    with caplog.at_level(logging.WARNING):
        asyncio.run(
            bridge._set_thread_session(
                "t-1", "al", "new-session", "gpt-5.6-sol"
            )
        )

    assert _Client.put_response.raise_calls == 1
    assert _Client.put_calls == [
        (
            "http://app/v1/sessions/t-1/agent-session-key",
            {"bot_id": "al"},
            {
                "backend": "codex",
                "session_key": "new-session",
                "model": "gpt-5.6-sol",
            },
        )
    ]
    assert "Failed to persist thread session for al/t-1" in caplog.text


@pytest.mark.parametrize(
    "path",
    [
        "src/codex_bridge/session_ops.py",
        "src/claude_code_bridge/session_ops.py",
    ],
)
def test_thread_session_put_is_status_checked(path):
    tree = ast.parse(Path(path).read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_set_thread_session"
    )
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "raise_for_status"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "resp"
    ]
    assert calls, f"{path}:_set_thread_session must check the PUT response status"


@pytest.mark.parametrize(
    "path",
    [
        "src/codex_bridge/session_ops.py",
        "src/claude_code_bridge/session_ops.py",
    ],
)
@pytest.mark.parametrize("retired", ["_get_session", "_set_session", "_clear_session"])
def test_scalar_session_methods_are_retired(path, retired):
    tree = ast.parse(Path(path).read_text())
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert retired not in names
