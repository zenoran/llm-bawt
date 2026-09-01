"""Codex usage probe self-heal: a 401 must force-rotate the token and retry.

The stored access token can be rejected upstream (revoked / rotated elsewhere)
while its JWT ``exp`` claim still looks valid — the probe must not fail
forever on it (the "Codex usage snapshot is stale; probe refresh failed" loop).
"""

from __future__ import annotations

import asyncio

from llm_bawt.service.usage import codex_oauth
from llm_bawt.service.usage.adapters import openai_chatgpt as mod


class _Tok:
    def __init__(self, token: str | None) -> None:
        self.token = token
        self.account_id = None


def _patched_adapter(monkeypatch, probe_results: list[tuple[dict | None, int | None]]):
    """Adapter with _probe_once returning canned results in order."""
    adapter = mod.OpenAIChatGPTUsageAdapter()
    seq = list(probe_results)

    async def fake_probe_once(result):  # noqa: ANN001
        fake_probe_once.tokens.append(result.token)
        return seq.pop(0)

    fake_probe_once.tokens = []
    monkeypatch.setattr(adapter, "_probe_once", fake_probe_once)
    return adapter, fake_probe_once


def test_probe_401_force_refreshes_and_retries(monkeypatch) -> None:
    calls: list[bool] = []

    def fake_get(*, force_refresh: bool = False) -> _Tok:
        calls.append(force_refresh)
        return _Tok("fresh" if force_refresh else "dead")

    monkeypatch.setattr(codex_oauth, "get_access_token", fake_get)
    snap = {"captured_at": 123}
    adapter, probe = _patched_adapter(monkeypatch, [(None, 401), (snap, 200)])

    out = asyncio.run(adapter._probe_snapshot())

    assert out is snap
    assert calls == [False, True]
    assert probe.tokens == ["dead", "fresh"]


def test_probe_non_401_failure_does_not_retry(monkeypatch) -> None:
    calls: list[bool] = []

    def fake_get(*, force_refresh: bool = False) -> _Tok:
        calls.append(force_refresh)
        return _Tok("tok")

    monkeypatch.setattr(codex_oauth, "get_access_token", fake_get)
    adapter, probe = _patched_adapter(monkeypatch, [(None, 500)])

    out = asyncio.run(adapter._probe_snapshot())

    assert out is None
    assert calls == [False]
    assert probe.tokens == ["tok"]


def test_probe_401_gives_up_when_refresh_yields_no_token(monkeypatch) -> None:
    def fake_get(*, force_refresh: bool = False) -> _Tok:
        return _Tok(None if force_refresh else "dead")

    monkeypatch.setattr(codex_oauth, "get_access_token", fake_get)
    adapter, probe = _patched_adapter(monkeypatch, [(None, 401)])

    out = asyncio.run(adapter._probe_snapshot())

    assert out is None
    assert probe.tokens == ["dead"]


def test_save_with_connected_at_does_not_crash(monkeypatch) -> None:
    """Regression: duplicate connected_at kwarg crashed _save for connected
    records, dropping freshly-rotated token pairs on the floor."""
    saved: dict = {}

    class FakeStore:
        available = True

        def save(self, record) -> None:  # noqa: ANN001
            saved["record"] = record

    monkeypatch.setattr(codex_oauth, "_credential_store", lambda: FakeStore())

    codex_oauth._save(
        {"connected_at": "2026-01-01T00:00:00Z", "auth_method": "cli_oauth"},
        {"tokens": {"access_token": "tok", "refresh_token": "ref"}},
    )

    assert saved["record"].connected_at == "2026-01-01T00:00:00Z"
    assert saved["record"].secret["codex_auth"]["tokens"]["refresh_token"] == "ref"
