"""Unit tests for the ``self_fwd`` MCP tool.

Follows the monkeypatch pattern of test_mcp_task_tools_compact.py: call the tool
coroutine directly with asyncio.run and stub the two boundaries — tail retrieval
(``self_tools._tail_bubbles``) and delivery (``server.send_message_to_bot``,
which self_fwd imports locally at call time).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_bawt.mcp_server import self_tools
import llm_bawt.mcp_server.server as server


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _stub_tail(monkeypatch: pytest.MonkeyPatch, bubbles: list[dict], total: int | None = None) -> None:
    async def fake_tail(bot_id: str, count: int) -> tuple[list[dict], int]:
        return bubbles[-count:], total if total is not None else len(bubbles)

    monkeypatch.setattr(self_tools, "_tail_bubbles", fake_tail)


def _capture_send(monkeypatch: pytest.MonkeyPatch, result: dict | None = None) -> dict:
    captured: dict[str, Any] = {}

    async def fake_send(**kwargs: Any) -> dict:
        captured.update(kwargs)
        return result if result is not None else {"success": True, "dispatched": True}

    monkeypatch.setattr(server, "send_message_to_bot", fake_send)
    return captured


def test_self_fwd_forwards_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_tail(
        monkeypatch,
        [
            {"role": "user", "content": "hi there", "timestamp": 1.0},
            {"role": "assistant", "content": "hello back", "timestamp": 2.0},
        ],
    )
    captured = _capture_send(monkeypatch)

    res = _run(self_tools.self_fwd(sender_bot_id="vex", target_bot_id="snark", count=2))

    assert res["success"] is True
    assert res["forwarded"] == 2
    assert res["sender"] == "vex"
    assert res["target"] == "snark"
    # Delivery went through the reused send tool with the right routing.
    assert captured["target_bot_id"] == "snark"
    assert captured["sender_bot_id"] == "vex"
    msg = captured["message"]
    assert "FORWARDED CONTEXT from 'vex'" in msg
    assert "hi there" in msg and "hello back" in msg


def test_self_fwd_prepends_note(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_tail(monkeypatch, [{"role": "user", "content": "ctx", "timestamp": 1.0}])
    captured = _capture_send(monkeypatch)

    res = _run(
        self_tools.self_fwd(
            sender_bot_id="vex", target_bot_id="snark", count=1, note="heads up:"
        )
    )

    assert res["success"] is True
    assert captured["message"].startswith("heads up:")
    assert "FORWARDED CONTEXT" in captured["message"]


def test_self_fwd_rejects_default_sender(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_send(monkeypatch)

    res = _run(self_tools.self_fwd(sender_bot_id="default", target_bot_id="snark"))

    assert res["success"] is False
    assert "sender_bot_id is required" in res["error"]
    assert captured == {}  # no delivery attempted


def test_self_fwd_requires_target(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_send(monkeypatch)

    res = _run(self_tools.self_fwd(sender_bot_id="vex", target_bot_id=""))

    assert res["success"] is False
    assert "target_bot_id is required" in res["error"]
    assert captured == {}


def test_self_fwd_rejects_self_target(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_send(monkeypatch)

    res = _run(self_tools.self_fwd(sender_bot_id="vex", target_bot_id="Vex"))

    assert res["success"] is False
    assert "Cannot forward to yourself" in res["error"]
    assert captured == {}


def test_self_fwd_empty_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_tail(monkeypatch, [], total=0)
    captured = _capture_send(monkeypatch)

    res = _run(self_tools.self_fwd(sender_bot_id="vex", target_bot_id="snark"))

    assert res["success"] is False
    assert res["forwarded"] == 0
    assert "No messages found" in res["error"]
    assert captured == {}


def test_self_fwd_surfaces_in_turn_block(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_tail(monkeypatch, [{"role": "user", "content": "ctx", "timestamp": 1.0}])
    _capture_send(
        monkeypatch,
        result={"success": False, "sent": False, "in_turn": True},
    )

    res = _run(self_tools.self_fwd(sender_bot_id="vex", target_bot_id="snark"))

    # self_fwd mirrors the send's success flag and passes the raw result through.
    assert res["success"] is False
    assert res["forwarded"] == 1
    assert res["send_result"]["in_turn"] is True
