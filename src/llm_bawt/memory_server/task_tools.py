"""MCP tools for the agent task system.

Registers task/project/step/activity tools on the shared FastMCP server.
Tools call the unmute REST API internally via httpx — agents see clean
MCP tool interfaces without needing to think about HTTP.

Imported by server.py to trigger tool registration on startup.
"""

from __future__ import annotations

import os
import logging
from typing import Any

import httpx

from .server import mcp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BASE_URL = os.getenv(
    "LLM_BAWT_TASK_API_URL",
    "http://echo.lan.zenoran.com",
)
_API_PREFIX = "/api/agents"
_TIMEOUT = 30.0

# ---------------------------------------------------------------------------
# HTTP client accessor (lazy singleton, mirrors _get_storage pattern)
# ---------------------------------------------------------------------------

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Get or create the singleton httpx client for the task API."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
    return _client


def _headers(bot_id: str | None = None) -> dict[str, str]:
    """Build request headers, optionally including bot actor identification."""
    h: dict[str, str] = {}
    if bot_id:
        h["X-Agent-Bot-Id"] = bot_id
    return h


async def _api_get(path: str, params: dict | None = None) -> Any:
    """GET helper with standard error handling."""
    client = _get_client()
    url = f"{_API_PREFIX}{path}"
    logger.debug("Task API GET %s params=%s", url, params)
    resp = await client.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


async def _api_post(
    path: str, json: dict | list, headers: dict | None = None,
) -> Any:
    """POST helper with standard error handling."""
    client = _get_client()
    url = f"{_API_PREFIX}{path}"
    logger.debug("Task API POST %s", url)
    resp = await client.post(url, json=json, headers=headers or {})
    resp.raise_for_status()
    return resp.json()


async def _api_patch(
    path: str, json: dict, headers: dict | None = None,
) -> Any:
    """PATCH helper with standard error handling."""
    client = _get_client()
    url = f"{_API_PREFIX}{path}"
    logger.debug("Task API PATCH %s", url)
    resp = await client.patch(url, json=json, headers=headers or {})
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Task Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_tasks(
    status: str | None = None,
    project_id: str | None = None,
    q: str | None = None,
    limit: int = 20,
) -> dict:
    """List agent tasks with optional filters.

    Use this to find tasks by status, project, or keyword search.
    Returns tasks sorted by most recently updated.

    Args:
        status: Filter by status. One of: QUEUED, PLANNING, REFINED,
                IN_PROGRESS, REVIEW, COMPLETED, FAILED, CANCELLED.
        project_id: Filter to tasks in a specific project (UUID).
                    Use "none" for unassigned tasks.
        q: Search query - matches title, description, and shortId.
        limit: Maximum tasks to return (default 20, max 50).

    Returns:
        Dict with "tasks" list and "total" count.
        Each task includes shortId, title, status, priority, project,
        steps, and dependencies.
    """
    logger.debug("MCP tool invoked: tools/list_tasks status=%s project=%s q=%s", status, project_id, q)
    params: dict[str, str] = {"limit": str(min(limit, 50)), "sort": "updated"}
    if status:
        params["status"] = status
    if project_id:
        params["projectId"] = project_id
    if q:
        params["q"] = q
    try:
        return await _api_get("/tasks", params=params)
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "status": e.response.status_code}


@mcp.tool()
async def get_task(
    task_id: str,
) -> dict:
    """Get full details of a single task by ID or shortId.

    Args:
        task_id: Task UUID or shortId (e.g. "TASK-42").

    Returns:
        Full task object with title, description, status, priority,
        response, steps (ordered), project info, and dependencies.
        Returns error dict if not found.
    """
    logger.debug("MCP tool invoked: tools/get_task id=%s", task_id)
    try:
        return await _api_get(f"/tasks/{task_id}")
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "status": e.response.status_code}


@mcp.tool()
async def update_task(
    task_id: str,
    status: str | None = None,
    response: str | None = None,
    model_id: str | None = None,
    title: str | None = None,
    description: str | None = None,
    priority: str | None = None,
    planned: bool | None = None,
    project_id: str | None = None,
    agent_bot_id: str | None = None,
    bot_id: str | None = None,
) -> dict:
    """Update a task's fields. Only provided fields are changed.

    Common patterns:
    - Start work: status="IN_PROGRESS", model_id="claude-opus-4-6"
    - Finish work: status="REVIEW", response="Summary of what was done"
    - Report failure: status="FAILED", response="What went wrong"

    IMPORTANT: Set status to REVIEW when done - only humans mark COMPLETED.

    Args:
        task_id: Task UUID or shortId (e.g. "TASK-42").
        status: New status (QUEUED, PLANNING, REFINED, IN_PROGRESS,
                REVIEW, COMPLETED, FAILED, CANCELLED).
        response: Summary text - your final answer / work output.
        model_id: Model identifier (e.g. "claude-opus-4-6").
        title: Updated task title.
        description: Updated description / spec.
        priority: URGENT, HIGH, MEDIUM, LOW, or NONE.
        planned: True after writing spec + steps.
        project_id: Move task to a different project (UUID).
        agent_bot_id: Assign task to a specific bot.
        bot_id: Your bot ID for activity attribution.

    Returns:
        Updated task object, or error dict.
    """
    logger.debug("MCP tool invoked: tools/update_task id=%s status=%s", task_id, status)
    body: dict[str, Any] = {}
    if status is not None:
        body["status"] = status
    if response is not None:
        body["response"] = response
    if model_id is not None:
        body["modelId"] = model_id
    if title is not None:
        body["title"] = title
    if description is not None:
        body["description"] = description
    if priority is not None:
        body["priority"] = priority
    if planned is not None:
        body["planned"] = planned
    if project_id is not None:
        body["projectId"] = project_id
    if agent_bot_id is not None:
        body["agentBotId"] = agent_bot_id

    if not body:
        return {"error": "No fields to update"}

    try:
        return await _api_patch(
            f"/tasks/{task_id}",
            json=body,
            headers=_headers(bot_id),
        )
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "status": e.response.status_code}


@mcp.tool()
async def create_task(
    title: str,
    description: str | None = None,
    project_id: str | None = None,
    priority: str = "MEDIUM",
    status: str = "QUEUED",
    steps: list[dict] | None = None,
    bot_id: str | None = None,
) -> dict:
    """Create a new agent task.

    Use this to break work into sub-tasks or queue follow-up work.

    Args:
        title: Task title (required).
        description: Detailed description or spec.
        project_id: Assign to a project (UUID). Omit for unassigned.
        priority: URGENT, HIGH, MEDIUM, LOW, or NONE (default MEDIUM).
        status: Initial status (default QUEUED).
        steps: Optional initial steps. Each dict needs "title" (str)
               and optional "type" (PLAN, READ_FILE, EDIT_FILE, etc.).
        bot_id: Your bot ID for activity attribution.

    Returns:
        Created task with generated shortId.
    """
    logger.debug("MCP tool invoked: tools/create_task title=%s", title)
    body: dict[str, Any] = {
        "title": title,
        "priority": priority,
        "status": status,
    }
    if description is not None:
        body["description"] = description
    if project_id is not None:
        body["projectId"] = project_id
    if steps is not None:
        body["steps"] = steps

    try:
        return await _api_post("/tasks", json=body, headers=_headers(bot_id))
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "status": e.response.status_code}


# ---------------------------------------------------------------------------
# Step Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def update_step(
    task_id: str,
    step_id: str,
    status: str | None = None,
    output: str | None = None,
    bot_id: str | None = None,
) -> dict:
    """Update a task step's status and/or output.

    Call this as you work through each step:
    1. Set status="RUNNING" when you start the step.
    2. Set status="COMPLETED", output="what you did" when done.
    3. Set status="FAILED", output="error details" if it fails.
    4. Set status="SKIPPED", output="reason" to skip.

    Args:
        task_id: Parent task UUID or shortId (e.g. "TASK-42").
        step_id: Step UUID (from the task's steps array).
        status: PENDING, RUNNING, COMPLETED, FAILED, or SKIPPED.
        output: Summary of what was done or error details.
        bot_id: Your bot ID for activity attribution.

    Returns:
        Updated step object, or error dict.
    """
    logger.debug("MCP tool invoked: tools/update_step task=%s step=%s status=%s", task_id, step_id, status)
    body: dict[str, Any] = {}
    if status is not None:
        body["status"] = status
    if output is not None:
        body["output"] = output

    if not body:
        return {"error": "No fields to update"}

    try:
        return await _api_patch(
            f"/tasks/{task_id}/steps/{step_id}",
            json=body,
            headers=_headers(bot_id),
        )
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "status": e.response.status_code}


@mcp.tool()
async def add_steps(
    task_id: str,
    steps: list[dict],
    bot_id: str | None = None,
) -> list[dict]:
    """Add new steps to a task.

    Steps are appended after existing steps. Use when planning work
    or when you discover additional steps mid-execution.

    Args:
        task_id: Task UUID or shortId (e.g. "TASK-42").
        steps: List of step dicts. Each needs "title" (str).
               Optional: "type" (default "PLAN") and "status"
               (default "PENDING").
               Types: PLAN, READ_FILE, EDIT_FILE, CREATE_FILE,
               DELETE_FILE, RUN_COMMAND, SEARCH, ASK_USER, REVIEW.
        bot_id: Your bot ID for activity attribution.

    Returns:
        List of created step objects.
    """
    logger.debug("MCP tool invoked: tools/add_steps task=%s count=%d", task_id, len(steps))
    try:
        return await _api_post(
            f"/tasks/{task_id}/steps",
            json=steps,
            headers=_headers(bot_id),
        )
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "status": e.response.status_code}


# ---------------------------------------------------------------------------
# Project Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_projects() -> list[dict]:
    """List all agent projects with task counts.

    Returns projects sorted by most recently updated.
    Use get_project() for full details including tasks and context.

    Returns:
        List of project objects with id, name, description,
        color, icon, agentBotId, and _count.tasks.
    """
    logger.debug("MCP tool invoked: tools/list_projects")
    try:
        return await _api_get("/projects")
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "status": e.response.status_code}


@mcp.tool()
async def get_project(
    project_id: str,
) -> dict:
    """Get a project's details including context prompt and all tasks.

    The contextPrompt contains project-specific instructions and
    conventions. Read this before working on tasks in the project.

    Args:
        project_id: Project UUID.

    Returns:
        Project with name, description, contextPrompt, tasks
        (with steps and dependencies), and configuration.
    """
    logger.debug("MCP tool invoked: tools/get_project id=%s", project_id)
    try:
        return await _api_get(f"/projects/{project_id}")
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "status": e.response.status_code}


# ---------------------------------------------------------------------------
# Activity Tool
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_activity(
    task_id: str | None = None,
    project_id: str | None = None,
    limit: int = 20,
) -> dict:
    """Get recent activity log entries.

    Shows what has happened on tasks/projects — status changes,
    dispatches, assignments, step updates, etc.

    Args:
        task_id: Filter to a specific task (UUID).
        project_id: Filter to a specific project (UUID).
        limit: Maximum entries to return (default 20, max 100).

    Returns:
        Dict with "activities" list and "total" count.
        Each entry has type, actorType, actorId, meta, createdAt,
        and related task/project summaries.
    """
    logger.debug("MCP tool invoked: tools/get_activity task=%s project=%s", task_id, project_id)
    params: dict[str, str] = {"limit": str(min(limit, 100))}
    if task_id:
        params["taskId"] = task_id
    if project_id:
        params["projectId"] = project_id
    try:
        return await _api_get("/activity", params=params)
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "status": e.response.status_code}


# ---------------------------------------------------------------------------
# Context Tool
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_task_context(
    task_id: str,
) -> str:
    """Get a formatted briefing document for a task.

    Combines task details, step checklist, dependencies, and the
    parent project's context prompt into a single readable document.
    Load this before starting work on a task.

    Args:
        task_id: Task UUID or shortId (e.g. "TASK-42").

    Returns:
        Formatted markdown text with everything you need to work
        on the task. Returns error string if not found.
    """
    logger.debug("MCP tool invoked: tools/get_task_context id=%s", task_id)
    try:
        task = await _api_get(f"/tasks/{task_id}")
    except httpx.HTTPStatusError as e:
        return f"Error fetching task: {e}"

    if "error" in task:
        return f"Task not found: {task_id}"

    # Build formatted context document
    lines: list[str] = []

    # Task header
    lines.append(f"# {task.get('shortId', '')} — {task.get('title', 'Untitled')}")
    lines.append("")
    lines.append(f"**Status:** {task.get('status', '?')}  ")
    lines.append(f"**Priority:** {task.get('priority', '?')}  ")
    if task.get("agentBotId"):
        lines.append(f"**Assigned to:** {task['agentBotId']}  ")
    if task.get("modelId"):
        lines.append(f"**Model:** {task['modelId']}  ")
    lines.append(f"**Created:** {task.get('createdAt', '?')}  ")
    lines.append(f"**Updated:** {task.get('updatedAt', '?')}  ")
    lines.append("")

    # Description
    if task.get("description"):
        lines.append("## Description")
        lines.append("")
        lines.append(task["description"])
        lines.append("")

    # Dependencies
    deps = task.get("dependsOn", [])
    if deps:
        lines.append("## Dependencies")
        lines.append("")
        for dep in deps:
            status_icon = "✅" if dep.get("status") == "COMPLETED" else "⏳"
            lines.append(f"- {status_icon} {dep.get('shortId', '?')} — {dep.get('title', '?')} ({dep.get('status', '?')})")
        lines.append("")

    # Steps
    steps = task.get("steps", [])
    if steps:
        lines.append("## Steps")
        lines.append("")
        for step in steps:
            status_map = {
                "PENDING": "[ ]",
                "RUNNING": "[~]",
                "COMPLETED": "[x]",
                "FAILED": "[!]",
                "SKIPPED": "[-]",
            }
            checkbox = status_map.get(step.get("status", ""), "[ ]")
            step_type = step.get("type", "")
            type_label = f" ({step_type})" if step_type else ""
            lines.append(f"- {checkbox} {step.get('title', '?')}{type_label}")
            if step.get("output"):
                # Indent output under step
                for out_line in step["output"].split("\n"):
                    lines.append(f"      {out_line}")
        lines.append("")

    # Existing response
    if task.get("response"):
        lines.append("## Previous Response")
        lines.append("")
        lines.append(task["response"])
        lines.append("")

    # Project context
    project = task.get("project")
    if project:
        lines.append(f"## Project: {project.get('name', '?')}")
        lines.append("")
        # Fetch full project for context prompt
        try:
            full_project = await _api_get(f"/projects/{project['id']}")
            if full_project.get("contextPrompt"):
                lines.append("### Project Context")
                lines.append("")
                lines.append(full_project["contextPrompt"])
                lines.append("")
            if full_project.get("description"):
                lines.append("### Project Description")
                lines.append("")
                lines.append(full_project["description"])
                lines.append("")
        except httpx.HTTPStatusError:
            lines.append("_(Could not load project context)_")
            lines.append("")

    return "\n".join(lines)
