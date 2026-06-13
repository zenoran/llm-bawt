import asyncio
import json
from typing import Any

import pytest

from llm_bawt.mcp_server import task_tools


GIANT_TEXT = "x" * 20_000


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _full_task() -> dict[str, Any]:
    return {
        "id": "task-id",
        "shortId": "TASK-266",
        "title": "Fix garage task context blowup",
        "description": GIANT_TEXT,
        "status": "REVIEW",
        "priority": "HIGH",
        "planned": True,
        "project": {
            "id": "project-id",
            "name": "Garage",
            "description": GIANT_TEXT,
            "agentBotId": "snark",
            "_count": {"tasks": 7},
        },
        "agentBotId": "snark",
        "createdAt": "2026-06-13T20:00:00Z",
        "updatedAt": "2026-06-13T21:00:00Z",
        "url": "https://example.test/tasks/TASK-266",
        "response": "done" * 1_000,
        "steps": [
            {"id": "step-1", "title": "Read logs", "status": "COMPLETED"},
            {"id": "step-2", "title": "Patch MCP", "status": "RUNNING"},
            {"id": "step-3", "title": "Verify", "status": "COMPLETED"},
        ],
        "dependencies": [{"id": "dep-1"}],
        "dependents": [{"id": "child-1"}, {"id": "child-2"}],
    }


def test_tasks_list_returns_compact_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get(path: str, params: dict | None = None) -> dict[str, Any]:
        assert path == "/tasks"
        assert params == {"limit": "20", "sort": "updated", "q": "garage"}
        return {"tasks": [_full_task()], "total": 1}

    monkeypatch.setattr(task_tools, "_api_get", fake_get)

    result = _run(task_tools.list_tasks(q="garage"))

    assert result["total"] == 1
    task = result["tasks"][0]
    assert task == {
        "id": "task-id",
        "shortId": "TASK-266",
        "title": "Fix garage task context blowup",
        "status": "REVIEW",
        "priority": "HIGH",
        "planned": True,
        "project": {
            "id": "project-id",
            "name": "Garage",
            "agentBotId": "snark",
            "taskCount": 7,
        },
        "agentBotId": "snark",
        "createdAt": "2026-06-13T20:00:00Z",
        "updatedAt": "2026-06-13T21:00:00Z",
        "url": "https://example.test/tasks/TASK-266",
        "descriptionChars": 20_000,
        "responseChars": 4_000,
        "stepCount": 3,
        "stepStatusCounts": {"COMPLETED": 2, "RUNNING": 1},
        "dependencyCount": 1,
        "dependentCount": 2,
    }
    assert "description" not in task
    assert "response" not in task
    assert "steps" not in task
    assert len(json.dumps(result)) < 1_000


def test_tasks_update_returns_compact_task(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_patch(
        path: str,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        calls.append({"path": path, "json": json, "headers": headers})
        return _full_task()

    monkeypatch.setattr(task_tools, "_api_patch", fake_patch)

    result = _run(
        task_tools.update_task(
            "TASK-266",
            status="REVIEW",
            response=GIANT_TEXT,
            bot_id="snark",
        )
    )

    assert calls == [
        {
            "path": "/tasks/TASK-266",
            "json": {"status": "REVIEW", "response": GIANT_TEXT},
            "headers": {"X-Agent-Bot-Id": "snark"},
        }
    ]
    assert result["ok"] is True
    assert result["updated"] == ["response", "status"]
    assert result["task"]["shortId"] == "TASK-266"
    assert result["task"]["descriptionChars"] == 20_000
    assert "description" not in result["task"]
    assert "response" not in result["task"]
    assert "steps" not in result["task"]
    assert len(json.dumps(result)) < 1_100


def test_projects_list_returns_compact_projects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get(path: str, params: dict | None = None) -> list[dict[str, Any]]:
        assert path == "/projects"
        assert params is None
        return [
            {
                "id": "project-id",
                "name": "Garage",
                "description": GIANT_TEXT,
                "color": "#94a3b8",
                "icon": "wrench",
                "agentBotId": "snark",
                "_count": {"tasks": 12},
                "tasks": [_full_task()],
                "createdAt": "2026-06-13T20:00:00Z",
                "updatedAt": "2026-06-13T21:00:00Z",
            }
        ]

    monkeypatch.setattr(task_tools, "_api_get", fake_get)

    result = _run(task_tools.list_projects())

    assert result == [
        {
            "id": "project-id",
            "name": "Garage",
            "descriptionChars": 20_000,
            "color": "#94a3b8",
            "icon": "wrench",
            "agentBotId": "snark",
            "taskCount": 12,
            "createdAt": "2026-06-13T20:00:00Z",
            "updatedAt": "2026-06-13T21:00:00Z",
        }
    ]
    assert len(json.dumps(result)) < 300
