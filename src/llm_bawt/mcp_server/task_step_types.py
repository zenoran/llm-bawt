"""Canonical task-step payload types shared by BawtHub MCP tools."""

from __future__ import annotations

from typing import Literal, NotRequired, Required, TypedDict

StepType = Literal[
    "PLAN",
    "READ_FILE",
    "EDIT_FILE",
    "CREATE_FILE",
    "DELETE_FILE",
    "RUN_COMMAND",
    "SEARCH",
    "ASK_USER",
    "REVIEW",
]

STEP_TYPES: tuple[StepType, ...] = (
    "PLAN",
    "READ_FILE",
    "EDIT_FILE",
    "CREATE_FILE",
    "DELETE_FILE",
    "RUN_COMMAND",
    "SEARCH",
    "ASK_USER",
    "REVIEW",
)
_STEP_TYPE_SET = frozenset(STEP_TYPES)


class TaskStepInput(TypedDict, total=False):
    """JSON shape accepted by task creation and step-list mutation tools."""

    title: Required[str]
    type: NotRequired[StepType]
    status: NotRequired[str]
    output: NotRequired[str]
    file_path: NotRequired[str]
    filePath: NotRequired[str]


def validate_step_inputs(steps: list[TaskStepInput]) -> str | None:
    """Return a precise validation error without sending bad data downstream."""
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            return f"steps[{index}] must be an object"
        title = step.get("title")
        if not isinstance(title, str) or not title.strip():
            return f"steps[{index}].title must be a non-empty string"
        step_type = step.get("type")
        if step_type is not None and step_type not in _STEP_TYPE_SET:
            return (
                f"steps[{index}].type must be one of: "
                + ", ".join(STEP_TYPES)
            )
    return None
