"""Clarify deferred-only discovery at the model-facing tool boundary.

An empty ToolSearch result is not a connection check or an inventory of already
loaded tools. Preserve the SDK schema and description; add stable guidance only
for its client-side ToolSearch tool. Never change execution or approval policy.
"""
from __future__ import annotations

from typing import Any

TOOL_SEARCH_GUIDANCE = (
    "Tool availability: first inspect the callable tools already supplied in this "
    "request. Call an already-loaded tool directly; do not search to load it again. "
    "ToolSearch searches deferred tools only. 'No matching deferred tools found' "
    "does NOT mean an already-loaded tool is unavailable or its MCP server is down. "
    "For a deferred MCP tool, use its fully qualified advertised name, for example "
    "select:mcp__bawthub__ops_list_operations, not just ops_list_operations. "
    "Before claiming unavailability, check loaded definitions and exact-name "
    "discovery; distinguish absent schema, failed connection, and denied approval. "
    "A search miss never authorizes an HTTP/SSH bypass."
)


def model_tool_description(tool: dict[str, Any]) -> Any:
    """Keep ordinary descriptions identical and ToolSearch guidance idempotent."""
    description = tool.get("description")
    if tool.get("name") != "ToolSearch":
        return description
    if not isinstance(description, str):
        description = ""
    if TOOL_SEARCH_GUIDANCE not in description:
        description = f"{description}\n\n{TOOL_SEARCH_GUIDANCE}".lstrip()
    return description
