import asyncio
import ast
import logging
from pathlib import Path

import httpx
import pytest

from codex_bridge.bridge import CodexBridge


class _Response:
    def __init__(self, *, payload=None, error=None):
        self._payload = payload or {}
        self._error = error
        self.raise_calls = 0

    def json(self):
        return self._payload

    def raise_for_status(self):
        self.raise_calls += 1
        if self._error:
            raise self._error


class _Client:
    patch_response: _Response

    def __init__(self, **_kwargs):
        self.get_response = _Response(
            payload={
                "data": [
                    {
                        "slug": "al",
                        "agent_backend_config": {"session_key": "old"},
                    }
                ]
            }
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, _url):
        return self.get_response

    async def patch(self, _url, *, json):
        return self.patch_response


def test_set_session_checks_patch_status(monkeypatch, caplog):
    error = httpx.HTTPStatusError(
        "profile update failed",
        request=httpx.Request("PATCH", "http://app/v1/bots/al/profile"),
        response=httpx.Response(500),
    )
    _Client.patch_response = _Response(error=error)
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    bridge = CodexBridge.__new__(CodexBridge)
    bridge._app_api_url = "http://app"

    with caplog.at_level(logging.WARNING):
        asyncio.run(bridge._set_session("al", "new-session", "gpt-5.6-sol"))

    assert _Client.patch_response.raise_calls == 1
    assert "Failed to persist session for al" in caplog.text


def test_clear_session_checks_patch_status(monkeypatch, caplog):
    error = httpx.HTTPStatusError(
        "profile update failed",
        request=httpx.Request("PATCH", "http://app/v1/bots/al/profile"),
        response=httpx.Response(500),
    )
    _Client.patch_response = _Response(error=error)
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    bridge = CodexBridge.__new__(CodexBridge)
    bridge._app_api_url = "http://app"

    with caplog.at_level(logging.WARNING):
        cleared = asyncio.run(bridge._clear_session("al"))

    assert not cleared
    assert _Client.patch_response.raise_calls == 1
    assert "Failed to clear session for al" in caplog.text


@pytest.mark.parametrize(
    "path",
    [
        "src/codex_bridge/bridge.py",
        "src/claude_code_bridge/bridge.py",
    ],
)
@pytest.mark.parametrize("method_name", ["_set_session", "_clear_session"])
def test_bridge_session_patch_is_status_checked(path, method_name):
    tree = ast.parse(Path(path).read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "raise_for_status"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "patch_response"
    ]
    assert calls, f"{path}:{method_name} must check the PATCH response status"
