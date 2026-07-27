"""Upstream model discovery contracts used by the Add Model dialog."""

from __future__ import annotations

import pytest

from llm_bawt.service.model_discovery import (
    KimiCodingDiscoveryProvider,
    ModelDiscoveryError,
    discover_models,
)
from llm_bawt.service.routes import models as model_routes


def test_kimi_discovery_preserves_display_name_and_context(monkeypatch):
    payload = {
        "data": [
            {
                "id": "k3",
                "display_name": "K3",
                "context_length": 262_144,
                "supports_reasoning": True,
            },
            {
                "id": "kimi-for-coding",
                "display_name": "K2.7 Coding",
                "context_length": 262_144,
            },
            {
                "id": "kimi-for-coding-highspeed",
                "display_name": "K2.7 Coding Highspeed",
                "context_length": 262_144,
            },
        ]
    }

    class _Response:
        status_code = 200

        def __init__(self, body):
            self.body = body

        @staticmethod
        def raise_for_status():
            return None

        def json(self):
            return self.body

    def fake_get(url, *, headers, timeout):
        assert headers == {
            "Authorization": "Bearer coding-key",
            "Accept": "application/json",
        }
        assert timeout == 20.0
        if url == "https://api.kimi.com/coding/v1/models":
            return _Response(payload)
        assert url == "https://api.kimi.com/coding/v1/usages"
        return _Response({"user": {"membership": {"level": "LEVEL_BASIC"}}})

    monkeypatch.setenv("KIMI_CODING_API_KEY", "coding-key")
    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", fake_get)

    assert KimiCodingDiscoveryProvider().fetch() == [
        {"id": "k3", "description": "K3", "context_length": 262_144},
        {
            "id": "kimi-for-coding",
            "description": "K2.7 Coding",
            "context_length": 262_144,
        },
    ]


def test_kimi_discovery_keeps_highspeed_for_upgraded_members(monkeypatch):
    class _Response:
        @staticmethod
        def raise_for_status():
            return None

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(url, **_kwargs):
        if url.endswith("/models"):
            return _Response(
                {
                    "data": [
                        {
                            "id": "kimi-for-coding-highspeed",
                            "display_name": "K2.7 Coding Highspeed",
                            "context_length": 262_144,
                        }
                    ]
                }
            )
        return _Response({"user": {"membership": {"level": "LEVEL_ALLEGRETTO"}}})

    monkeypatch.setenv("KIMI_CODING_API_KEY", "coding-key")
    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", fake_get)

    assert KimiCodingDiscoveryProvider().fetch() == [
        {
            "id": "kimi-for-coding-highspeed",
            "description": "K2.7 Coding Highspeed",
            "context_length": 262_144,
        }
    ]


def test_kimi_discovery_requires_its_subscription_key(monkeypatch):
    monkeypatch.delenv("KIMI_CODING_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    with pytest.raises(ModelDiscoveryError) as exc_info:
        KimiCodingDiscoveryProvider().fetch()

    assert exc_info.value.status_code == 503
    assert "KIMI_CODING_API_KEY" in str(exc_info.value)


def test_kimi_provider_aliases_resolve(monkeypatch):
    expected = [{"id": "k3", "description": "K3", "context_length": 262_144}]
    monkeypatch.setattr(KimiCodingDiscoveryProvider, "fetch", lambda _self: expected)

    assert discover_models("kimi") == expected
    assert discover_models("kimi_coding") == expected
    assert discover_models("kimi-code") == expected


def test_upstream_lookup_caches_normalized_kimi_catalog(monkeypatch):
    model_routes._upstream_cache.clear()
    calls = 0

    def fake_discover(provider):
        nonlocal calls
        calls += 1
        assert provider == "kimi"
        return [{"id": "k3", "description": "K3", "context_length": 262_144}]

    monkeypatch.setattr("llm_bawt.service.model_discovery.discover_models", fake_discover)

    first = model_routes._upstream_lookup("kimi")
    second = model_routes._upstream_lookup("kimi")

    assert first == second
    assert calls == 1
