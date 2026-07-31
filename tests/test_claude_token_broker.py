from __future__ import annotations

from claude_code_bridge import _bridge_helpers as helpers


def test_direct_turn_reads_broker_each_time_and_observes_rotation(monkeypatch):
    tokens = iter([("token-a", 1000), ("token-b", 2000)])
    calls: list[bool] = []

    def fake_fetch(*, force: bool = False):
        calls.append(force)
        return next(tokens)

    monkeypatch.setattr(helpers, "_fetch_broker_token", fake_fetch)

    assert helpers._get_fresh_oauth_token() == "token-a"
    assert helpers._get_fresh_oauth_token() == "token-b"
    assert calls == [False, False]


def test_confirmed_401_force_fetches_broker(monkeypatch):
    calls: list[bool] = []

    def fake_fetch(*, force: bool = False):
        calls.append(force)
        return "token-b", 2000

    monkeypatch.setattr(helpers, "_fetch_broker_token", fake_fetch)

    assert helpers._get_fresh_oauth_token(force_refresh=True) == "token-b"
    assert calls == [True]
