"""Data-driven Claude Agent SDK tool policy.

The application resolves the configured base list and sends it with every
Claude Code command.  The bridge applies the same normalization defensively and
adds transport-mandated exclusions for proxy-routed turns.
"""

from __future__ import annotations

import json
from typing import Any

CLAUDE_CODE_DISALLOWED_TOOLS_KEY = "claude_code_disallowed_tools"
DEFAULT_DISALLOWED_TOOLS: tuple[str, ...] = (
    "EnterPlanMode",
    "ExitPlanMode",
    "EnterWorktree",
    "ExitWorktree",
)
PROXY_DISALLOWED_TOOLS: tuple[str, ...] = ("WebSearch", "WebFetch")

_MAX_TOOL_COUNT = 256
_MAX_TOOL_NAME_LENGTH = 256


class InvalidToolPolicy(ValueError):
    """Raised when a configured disallowed-tools value is not a string list."""


def validate_disallowed_tools(value: Any) -> list[str]:
    """Validate and normalize an explicitly configured tool-name list.

    Unknown names are intentionally accepted so operators can disable tools
    introduced by newer SDK versions without waiting for an application release.
    Order is preserved and duplicates are removed.  An explicit empty list is a
    valid policy.
    """
    if not isinstance(value, list):
        raise InvalidToolPolicy("value must be a JSON array of tool names")
    if len(value) > _MAX_TOOL_COUNT:
        raise InvalidToolPolicy(f"value may contain at most {_MAX_TOOL_COUNT} tool names")

    normalized: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise InvalidToolPolicy(f"item {index} must be a string")
        name = item.strip()
        if not name:
            raise InvalidToolPolicy(f"item {index} must not be empty")
        if len(name) > _MAX_TOOL_NAME_LENGTH:
            raise InvalidToolPolicy(
                f"item {index} exceeds {_MAX_TOOL_NAME_LENGTH} characters"
            )
        if name not in seen:
            normalized.append(name)
            seen.add(name)
    return normalized


def configured_disallowed_tools(value: Any) -> list[str]:
    """Return a safe base policy, falling back to defaults on malformed input."""
    if value is None:
        return list(DEFAULT_DISALLOWED_TOOLS)
    try:
        return validate_disallowed_tools(value)
    except InvalidToolPolicy:
        return list(DEFAULT_DISALLOWED_TOOLS)


def decode_disallowed_tools_field(raw: Any) -> list[str]:
    """Decode a Redis command field and return a safe normalized base policy."""
    if raw is None:
        return list(DEFAULT_DISALLOWED_TOOLS)
    value = raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            return list(DEFAULT_DISALLOWED_TOOLS)
    return configured_disallowed_tools(value)


def effective_disallowed_tools(raw: Any, *, use_proxy: bool) -> list[str]:
    """Build the SDK policy for a turn, adding proxy-only exclusions."""
    tools = decode_disallowed_tools_field(raw)
    if use_proxy:
        seen = set(tools)
        for name in PROXY_DISALLOWED_TOOLS:
            if name not in seen:
                tools.append(name)
                seen.add(name)
    return tools
