from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.integrations import x_api
from llm_bawt.service.providers import crypto
from llm_bawt.service.providers.base import CredentialStore
from llm_bawt.service.providers.registry import get_adapter
from llm_bawt.service.providers.x import XAdapter
from llm_bawt.service.routes import providers as routes


@pytest.fixture
def store(monkeypatch):
    class MemorySettings:
        engine = object()
        values = {}

        def get_scope_settings(self, *_args):
            return dict(self.values)

        def set_value(self, _scope, _id, key, value):
            self.values[key] = value

        def delete_value(self, _scope, _id, key):
            return self.values.pop(key, None) is not None

    settings = MemorySettings()
    monkeypatch.setattr("llm_bawt.service.providers.base.RuntimeSettingsStore", lambda _: settings)
    from cryptography.fernet import Fernet
    monkeypatch.setattr(crypto, "_get_fernet", lambda: fernet)
    fernet = Fernet(Fernet.generate_key())
    return settings


def response(status=200, data=None):
    return httpx.Response(status, json=data or {"data": {"project_id": "123"}})


def test_connect_route_encrypts_and_disconnect_removes_token(monkeypatch, store):
    calls = []
    monkeypatch.setattr(x_api.httpx, "get", lambda url, **kw: calls.append((url, kw)) or response())
    config = SimpleNamespace()
    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config=config))
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    secret = "secret-bearer-token"
    result = client.post("/v1/providers/x/connect/api-key", json={"api_key": secret})
    assert result.status_code == 200
    assert secret not in result.text
    raw = store.values["provider_connection:x"]
    assert crypto.is_encrypted(raw["secret_enc"])
    assert secret not in str(raw)
    assert CredentialStore(config).load("x").secret == {"api_key": secret}
    assert calls[0][0] == "https://api.x.com/2/usage/tweets"
    assert calls[0][1]["headers"]["Authorization"] == f"Bearer {secret}"
    descriptor = client.get("/v1/providers/x").json()
    assert descriptor["category"] == "search"
    assert descriptor["credential_label"] == "Bearer token"
    assert secret not in str(descriptor)
    assert descriptor["health"]["state"] == "ok"
    assert len(calls) == 1  # descriptor/health polling never calls X
    assert client.delete("/v1/providers/x").json()["disconnected"]
    assert get_adapter(config, "x").api_key() is None
    assert get_adapter(config, "x").health()["state"] == "unconfigured"


@pytest.mark.parametrize("status,code", [(401, "unauthorized"), (402, "credits_required"), (403, "forbidden"), (429, "rate_limited"), (500, "upstream_error"), (400, "invalid_query"), (302, "upstream_error")])
def test_failures_are_actionable_and_never_echo_upstream_body(monkeypatch, status, code):
    monkeypatch.setattr(x_api.httpx, "get", lambda *a, **kw: response(status, {"detail": "secret-bearer-token"}))
    with pytest.raises(x_api.XApiError) as err:
        x_api.request_x("secret-bearer-token", "tweets/search/recent")
    assert err.value.code == code
    assert "secret-bearer-token" not in str(err.value)


def test_bad_reconnect_preserves_existing_token(monkeypatch, store):
    monkeypatch.setattr(x_api.httpx, "get", lambda *a, **kw: response())
    adapter = XAdapter(SimpleNamespace())
    assert adapter.set_api_key("working-token").ok
    monkeypatch.setattr(x_api.httpx, "get", lambda *a, **kw: response(401))
    assert not adapter.set_api_key("bad-token").ok
    assert adapter.api_key() == "working-token"


@pytest.mark.parametrize("payload", [{}, {"data": []}, {"data": {}}, [1, 2]])
def test_probe_rejects_malformed_success(monkeypatch, store, payload):
    monkeypatch.setattr(x_api.httpx, "get", lambda *a, **kw: httpx.Response(200, json=payload))
    assert not XAdapter(SimpleNamespace()).set_api_key("token").ok
    assert store.values == {}


def test_timeout_is_sanitized(monkeypatch):
    def fail(*args, **kwargs):
        raise httpx.ReadTimeout("secret-bearer-token")
    monkeypatch.setattr(x_api.httpx, "get", fail)
    with pytest.raises(x_api.XApiError, match="unreachable") as err:
        x_api.request_x("secret-bearer-token", "usage/tweets")
    assert "secret-bearer-token" not in str(err.value)


def test_search_single_page_recency_fields_and_time_filters(monkeypatch):
    monkeypatch.setattr("llm_bawt.service.providers.api_key.resolve_api_key", lambda config, provider: "stored-token" if provider == "x" else None)
    calls = []
    data = {"data": [{"id": "1234", "text": "Server down", "author_id": "42", "created_at": "2026-09-21T23:00:00Z"}], "meta": {"result_count": 1, "next_token": "next-page"}}
    monkeypatch.setattr(x_api.httpx, "get", lambda url, **kw: calls.append((url, kw)) or response(data=data))
    start = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    end = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    result = x_api.recent_search(SimpleNamespace(), "from:BlizzardCS -is:retweet", max_results=10, start_time=start, end_time=end, next_token="prior-page")
    assert len(calls) == 1
    url, kw = calls[0]
    assert url.endswith("/tweets/search/recent")
    assert kw["params"]["sort_order"] == "recency"
    assert kw["params"]["next_token"] == "prior-page"
    assert kw["params"]["max_results"] == 10
    assert "expansions" not in kw["params"]  # no extra paid user resources
    assert kw["params"]["start_time"].endswith("Z")
    assert kw["follow_redirects"] is False
    assert result["next_token"] == "next-page"
    assert result["results"][0]["url"] == "https://x.com/i/status/1234"
    assert result["results"][0]["created_at"] == "2026-09-21T23:00:00Z"
    assert "stored-token" not in str(result)


@pytest.mark.parametrize("kwargs", [{"query": ""}, {"query": "a" * 513}, {"max_results": 5}, {"max_results": 101}, {"max_results": True}, {"start_time": "nonsense"}, {"start_time": "2020-01-01T00:00:00Z"}, {"start_time": "2026-01-01T00:00:00"}, {"next_token": " "}])
def test_invalid_inputs_never_make_paid_request(monkeypatch, kwargs):
    def fail(*a, **kw):
        pytest.fail("must reject before API call")
    monkeypatch.setattr(x_api.httpx, "get", fail)
    args = {"query": "outage", **kwargs}
    with pytest.raises(x_api.XApiError) as err:
        x_api.recent_search(SimpleNamespace(), **args)
    assert err.value.code == "invalid_request"


def test_missing_credentials_never_fall_back(monkeypatch):
    monkeypatch.setattr("llm_bawt.service.providers.api_key.resolve_api_key", lambda *a: None)
    with pytest.raises(x_api.XApiError) as err:
        x_api.recent_search(SimpleNamespace(), "outage")
    assert err.value.code == "not_connected"


def test_empty_search_is_distinct_from_failure(monkeypatch):
    monkeypatch.setattr("llm_bawt.service.providers.api_key.resolve_api_key", lambda *a: "token")
    monkeypatch.setattr(x_api.httpx, "get", lambda *a, **kw: response(data={"meta": {"result_count": 0}}))
    result = x_api.recent_search(SimpleNamespace(), "outage")
    assert result["count"] == 0
    assert "error" not in result


def test_mcp_error_is_structured(monkeypatch):
    from llm_bawt.mcp_server.search_tools import x_search
    def fail(*a, **kw):
        raise x_api.XApiError("credits_required", "Add X credits")
    monkeypatch.setattr(x_api, "recent_search", fail)
    result = asyncio.run(x_search("outage"))
    assert result["error_code"] == "credits_required"
    assert result["results"] == []


def test_x_not_a_first_bot_provider_or_automatic_search_target(store):
    from llm_bawt.seeding import FIRST_BOT_PROVIDERS
    from llm_bawt.search.multi import available_providers
    assert "x" not in [p.provider_id for p in FIRST_BOT_PROVIDERS]
    assert "x" not in [p.value for p in available_providers(SimpleNamespace())]
