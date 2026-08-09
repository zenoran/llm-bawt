from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet

from llm_bawt import task_turn_context as codec
from llm_bawt.mcp_server import task_association


@pytest.fixture
def fernet(monkeypatch: pytest.MonkeyPatch) -> Fernet:
    instance = Fernet(Fernet.generate_key())
    monkeypatch.setattr(codec, "_get_fernet", lambda: instance)
    return instance


def _values() -> dict[str, str]:
    return {
        "session_id": str(uuid4()),
        "turn_id": "turn-" + "a" * 32,
        "trigger_message_id": str(uuid4()),
        "bot_id": "loopy",
        "user_id": "nick",
    }


def test_round_trip_preserves_all_trusted_identifiers(fernet: Fernet) -> None:
    values = _values()
    token = codec.mint_task_turn_context(**values, issued_at=123)

    opened = codec.open_task_turn_context(token)

    assert opened.session_id == values["session_id"]
    assert opened.turn_id == values["turn_id"]
    assert opened.trigger_message_id == values["trigger_message_id"]
    assert opened.bot_id == "loopy"
    assert opened.user_id == "nick"
    assert opened.issued_at == 123


def test_invalid_or_missing_capability_fails_closed(fernet: Fernet) -> None:
    with pytest.raises(codec.TaskTurnContextError, match="No trusted"):
        codec.open_task_turn_context(None)
    with pytest.raises(codec.TaskTurnContextError, match="invalid or expired"):
        codec.open_task_turn_context("not-a-token")


def test_mint_rejects_noncanonical_identifiers(fernet: Fernet) -> None:
    values = _values()
    values["session_id"] = values["session_id"].upper()
    with pytest.raises(codec.TaskTurnContextError, match="canonical lowercase UUID"):
        codec.mint_task_turn_context(**values)


def test_mint_accepts_delivery_shape_turn_id(fernet: Fernet) -> None:
    # Regression for TASK-783: inter-bot delivery turns use ``turn-delivery-<hex>``
    # (see ``llm_bawt.inter_bot_delivery.stable_ids``). Rejecting that shape
    # silently killed the whole ``associate_current_turn`` chain on every
    # delivery-dispatched claude-code turn — mint raised, exception was
    # swallowed to a warning at ``chat_streaming.py:365-373``, no capability
    # header was sent, MCP tool failed closed. The signed capability MUST
    # round-trip the delivery shape verbatim so downstream can distinguish it.
    values = _values()
    values["turn_id"] = "turn-delivery-" + "b" * 32

    token = codec.mint_task_turn_context(**values)
    opened = codec.open_task_turn_context(token)

    assert opened.turn_id == values["turn_id"]
    assert codec.is_delivery_turn_id(opened.turn_id) is True
    assert codec.is_delivery_turn_id("turn-" + "a" * 32) is False


def test_mint_rejects_arbitrary_turn_id_shapes(fernet: Fernet) -> None:
    # The regex loosening for delivery ids must not open the door to arbitrary
    # strings — every non-canonical shape (wrong length, wrong prefix, uppercase
    # hex, extra suffix, malformed delivery variant) still fails closed.
    values = _values()
    for bad in [
        "turn-" + "A" * 32,                    # uppercase hex
        "turn-" + "a" * 31,                    # short by one
        "turn-" + "a" * 33,                    # long by one
        "turn-delivery-" + "A" * 32,           # uppercase in delivery shape
        "turn-delivery-" + "a" * 31,           # delivery short by one
        "turndelivery-" + "a" * 32,            # missing hyphen
        "turn-delivery-" + "a" * 32 + "-extra",  # trailing junk
        "delivery-" + "a" * 32,                # missing turn- prefix
        "",
    ]:
        values["turn_id"] = bad
        with pytest.raises(codec.TaskTurnContextError, match="turn_id must match"):
            codec.mint_task_turn_context(**values)


def test_asgi_middleware_binds_and_resets_capability() -> None:
    observed: list[str | None] = []

    async def app(scope, receive, send):
        observed.append(task_association._current_capability.get())
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = task_association.TaskTurnCapabilityMiddleware(app)

    async def run() -> None:
        messages = [{"type": "http.request", "body": b"", "more_body": False}]

        async def receive():
            return messages.pop(0)

        async def send(_message):
            return None

        await middleware(
            {
                "type": "http",
                "headers": [
                    (b"x-llm-bawt-task-turn-context", b"opaque-value"),
                ],
            },
            receive,
            send,
        )

    asyncio.run(run())

    assert observed == ["opaque-value"]
    assert task_association._current_capability.get() is None


def test_association_posts_only_verified_server_context(
    fernet: Fernet,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _values()
    token = codec.mint_task_turn_context(**values)
    binding = task_association.set_current_task_turn_capability(token)
    monkeypatch.setenv("TASK_ASSOCIATION_INTERNAL_TOKEN", "internal-secret")
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"association": {"task": {"shortId": "TASK-701"}}}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def put(self, path: str, json: dict):
            captured["path"] = path
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(task_association.httpx, "AsyncClient", FakeClient)
    try:
        result = asyncio.run(task_association.associate_current_task("TASK-701"))
    finally:
        task_association.reset_current_task_turn_capability(binding)

    assert result["ok"] is True
    assert captured["path"] == "/internal/tasks/TASK-701/associations"
    assert captured["client"]["headers"] == {
        "Authorization": "Bearer internal-secret",
    }
    assert captured["json"] == {
        "sessionId": values["session_id"],
        "botId": "loopy",
        "userId": "nick",
        "source": "AGENT",
        "turn": {
            "triggerMessageId": values["trigger_message_id"],
            "turnId": values["turn_id"],
            "source": "AGENT",
        },
    }


def test_association_nulls_turn_id_for_delivery_shape(
    fernet: Fernet,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression for TASK-783: an inter-bot-delivery turn (``turn-delivery-<hex>``)
    # is legal in the signed capability but must not be persisted as an
    # ``AgentTaskTurn.turnId`` on the BawtHub side — its strict validator
    # accepts only ``turn-<32 hex>`` and would 400 the PUT. Per the TASK-706
    # design, delivery-turn attribution rides on ``triggerMessageId`` alone.
    values = _values()
    values["turn_id"] = "turn-delivery-" + "c" * 32
    token = codec.mint_task_turn_context(**values)
    binding = task_association.set_current_task_turn_capability(token)
    monkeypatch.setenv("TASK_ASSOCIATION_INTERNAL_TOKEN", "internal-secret")
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"association": {"task": {"shortId": "TASK-783"}}}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def put(self, path: str, json: dict):
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(task_association.httpx, "AsyncClient", FakeClient)
    try:
        result = asyncio.run(task_association.associate_current_task("TASK-783"))
    finally:
        task_association.reset_current_task_turn_capability(binding)

    assert result["ok"] is True
    assert captured["json"]["turn"]["triggerMessageId"] == values["trigger_message_id"]
    assert captured["json"]["turn"]["turnId"] is None
    # Canonical shape still round-trips verbatim.
    values["turn_id"] = "turn-" + "d" * 32
    token = codec.mint_task_turn_context(**values)
    binding = task_association.set_current_task_turn_capability(token)
    try:
        asyncio.run(task_association.associate_current_task("TASK-783"))
    finally:
        task_association.reset_current_task_turn_capability(binding)
    assert captured["json"]["turn"]["turnId"] == values["turn_id"]


def test_association_requires_internal_service_credential(
    fernet: Fernet,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = codec.mint_task_turn_context(**_values())
    binding = task_association.set_current_task_turn_capability(token)
    monkeypatch.delenv("TASK_ASSOCIATION_INTERNAL_TOKEN", raising=False)
    try:
        with pytest.raises(codec.TaskTurnContextError, match="service credential"):
            asyncio.run(task_association.associate_current_task("TASK-701"))
    finally:
        task_association.reset_current_task_turn_capability(binding)
