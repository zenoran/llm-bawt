from __future__ import annotations

import base64
import json

import httpx

from llm_bawt.service.providers import codex
from llm_bawt.service.providers.base import AUTH_DEVICE_OAUTH

_HTTPX_CLIENT = httpx.Client


def _jwt(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


class FakeStore:
    def __init__(self) -> None:
        self.records: dict[str, object] = {}

    def load(self, provider: str):
        return self.records.get(provider)

    def save(self, record) -> None:
        self.records[record.provider] = record

    def delete(self, provider: str) -> bool:
        return self.records.pop(provider, None) is not None


def _adapter() -> tuple[codex.CodexAdapter, FakeStore]:
    adapter = object.__new__(codex.CodexAdapter)
    store = FakeStore()
    adapter.store = store
    return adapter, store


def test_start_device_flow_returns_openai_codes(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL(codex._DEVICE_CODE_URL)
        assert json.loads(request.content) == {"client_id": codex._CLIENT_ID}
        return httpx.Response(
            200,
            json={"device_auth_id": "device-1", "user_code": "ABCD-EFGH", "interval": 1},
        )

    monkeypatch.setattr(
        codex.httpx,
        "Client",
        lambda **kwargs: _HTTPX_CLIENT(transport=httpx.MockTransport(handler), **kwargs),
    )
    adapter, _ = _adapter()

    result = adapter.start_device_flow()

    assert result.device_code == "device-1"
    assert result.user_code == "ABCD-EFGH"
    assert result.verification_uri == "https://auth.openai.com/codex/device"
    assert result.interval == 3


def test_poll_device_flow_returns_pending_without_token_exchange(monkeypatch) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(403, json={"error": "authorization_pending"})

    monkeypatch.setattr(
        codex.httpx,
        "Client",
        lambda **kwargs: _HTTPX_CLIENT(transport=httpx.MockTransport(handler), **kwargs),
    )
    adapter, _ = _adapter()

    result = adapter.poll_device_flow("device-1", "ABCD-EFGH")

    assert result.status == "pending"
    assert len(requests) == 1
    assert json.loads(requests[0].content) == {
        "device_auth_id": "device-1",
        "user_code": "ABCD-EFGH",
    }


def test_poll_device_flow_persists_broker_bundle_and_descriptor(monkeypatch) -> None:
    access_token = _jwt({"exp": 2_000_000_000})
    id_token = _jwt(
        {
            "email": "nick@example.com",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "account-1",
                "chatgpt_plan_type": "plus",
            },
        }
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url == httpx.URL(codex._DEVICE_POLL_URL):
            return httpx.Response(
                200,
                json={"authorization_code": "auth-code", "code_verifier": "verifier"},
            )
        if request.url == httpx.URL(codex._TOKEN_EXCHANGE_URL):
            return httpx.Response(
                200,
                json={
                    "access_token": access_token,
                    "refresh_token": "refresh-token",
                    "id_token": id_token,
                },
            )
        raise AssertionError(f"unexpected request: {request.url}")

    monkeypatch.setattr(
        codex.httpx,
        "Client",
        lambda **kwargs: _HTTPX_CLIENT(transport=httpx.MockTransport(handler), **kwargs),
    )
    adapter, store = _adapter()

    result = adapter.poll_device_flow("device-1", "ABCD-EFGH")

    assert result.status == "authorized"
    record = store.records["openai_chatgpt"]
    assert record.auth_method == AUTH_DEVICE_OAUTH
    assert record.meta["account_id"] == "account-1"
    assert record.secret["codex_auth"]["tokens"] == {
        "id_token": id_token,
        "access_token": access_token,
        "refresh_token": "refresh-token",
        "account_id": "account-1",
    }
    descriptor = adapter.descriptor()
    assert descriptor["connection"]["connected"] is True
    assert descriptor["connection"]["status"] == "connected"
    assert descriptor["connection"]["meta"]["account_id"] == "account-1"
    assert descriptor["connection"]["meta"]["plan"] == "plus"

    exchange = requests[1]
    assert exchange.headers["content-type"].startswith("application/x-www-form-urlencoded")
    body = exchange.content.decode()
    assert "code=auth-code" in body
    assert "code_verifier=verifier" in body


def test_disconnect_reports_shared_bundle_removal() -> None:
    adapter, store = _adapter()
    store.records["openai_chatgpt"] = object()

    assert adapter.disconnect() is True
    assert store.records == {}
