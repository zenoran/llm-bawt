"""Synchronous inter-bot visibility trust-boundary regressions."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

import llm_bawt.mcp_server.server as mcp_server
from llm_bawt.service.routes.chat import _internal_inter_bot_sender


def _request(host: str, sender: str | None) -> Request:
    headers = []
    if sender is not None:
        headers.append((b"x-llm-bawt-inter-bot-sender", sender.encode()))
    return Request({
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": headers,
        "client": (host, 1234),
    })


def test_internal_sender_header_requires_loopback() -> None:
    assert _internal_inter_bot_sender(_request("127.0.0.1", "Vex")) == "Vex"
    assert _internal_inter_bot_sender(_request("::1", "Vex")) == "Vex"
    assert _internal_inter_bot_sender(_request("10.0.0.50", None)) is None

    with pytest.raises(HTTPException) as caught:
        _internal_inter_bot_sender(_request("10.0.0.50", "Vex"))
    assert caught.value.status_code == 403


def test_sync_dispatch_identifies_named_sender(monkeypatch) -> None:
    captured: dict = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "model": "test-model",
                "choices": [{"message": {"content": "reply"}}],
            }

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", Client)

    result = asyncio.run(mcp_server._dispatch_bot_message(
        {"messages": [{"role": "user", "content": "hello"}]},
        "snark",
        "vex",
        30,
    ))

    assert result["success"] is True
    assert captured["headers"] == {"X-LLM-Bawt-Inter-Bot-Sender": "vex"}


def test_sync_dispatch_omits_unknown_sender_header(monkeypatch) -> None:
    captured: dict = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "model": "test-model",
                "choices": [{"message": {"content": "reply"}}],
            }

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, _url, **kwargs):
            captured.update(kwargs)
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", Client)

    asyncio.run(mcp_server._dispatch_bot_message(
        {"messages": [{"role": "user", "content": "hello"}]},
        "snark",
        "unknown",
        30,
    ))

    assert captured["headers"] is None
