from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_bawt.mcp_server import task_tools


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_associate_current_tool_accepts_only_task_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_associate(task_ref: str) -> dict:
        calls.append(task_ref)
        return {"ok": True, "taskRef": task_ref}

    monkeypatch.setattr(task_tools, "associate_current_task", fake_associate)

    assert run(task_tools.associate_task_to_current_turn("TASK-701")) == {
        "ok": True,
        "taskRef": "TASK-701",
    }
    assert calls == ["TASK-701"]


def test_create_can_immediately_associate_new_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    async def fake_post(path: str, json: dict, headers: dict | None = None) -> dict:
        calls.append((path, json))
        return {"id": "task-id", "shortId": "TASK-900", "title": json["title"]}

    async def fake_associate(task_ref: str) -> dict:
        calls.append(("associate", task_ref))
        return {"ok": True, "taskRef": task_ref}

    monkeypatch.setattr(task_tools, "_api_post", fake_post)
    monkeypatch.setattr(task_tools, "associate_current_task", fake_associate)

    result = run(task_tools.create_task(
        "Follow-up",
        bot_id="loopy",
        associate_current_turn=True,
    ))

    assert result["task"]["shortId"] == "TASK-900"
    assert result["currentTurnAssociation"]["ok"] is True
    assert calls[-1] == ("associate", "TASK-900")


def test_create_without_flag_preserves_existing_response_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = {"id": "task-id", "shortId": "TASK-900"}

    async def fake_post(path: str, json: dict, headers: dict | None = None) -> dict:
        return created

    monkeypatch.setattr(task_tools, "_api_post", fake_post)

    assert run(task_tools.create_task("Follow-up")) is created


def test_update_can_associate_without_rewriting_task_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_associate(task_ref: str) -> dict:
        return {"ok": True, "taskRef": task_ref}

    monkeypatch.setattr(task_tools, "associate_current_task", fake_associate)

    assert run(task_tools.update_task(
        "TASK-701",
        associate_current_turn=True,
    )) == {"ok": True, "taskRef": "TASK-701"}


def test_update_can_claim_and_associate_current_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_patch(path: str, json: dict, headers: dict | None = None) -> dict:
        return {
            "id": "task-id",
            "shortId": "TASK-701",
            "title": "Traceability",
            "status": json["status"],
            "steps": [],
        }

    async def fake_associate(task_ref: str) -> dict:
        return {"ok": True, "taskRef": task_ref}

    monkeypatch.setattr(task_tools, "_api_patch", fake_patch)
    monkeypatch.setattr(task_tools, "associate_current_task", fake_associate)

    result = run(task_tools.update_task(
        "TASK-701",
        status="IN_PROGRESS",
        bot_id="loopy",
        associate_current_turn=True,
    ))

    assert result["task"]["status"] == "IN_PROGRESS"
    assert result["currentTurnAssociation"] == {
        "ok": True,
        "taskRef": "TASK-701",
    }
