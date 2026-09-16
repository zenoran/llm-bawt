"""Compact discovery metadata without changing dispatch or validation.

FastMCP still owns callable signatures and validation. The public list_tools seam
only replaces model-facing descriptions and removes generated schema titles.
Titles are annotations, not field names; never recursively strip arbitrary keys.
"""

from copy import deepcopy
from typing import Any

from mcp.types import Tool

from .approval_interceptor import ApprovalAwareFastMCP
from .catalog_contracts import TOOL_SUMMARIES, tool_description

# JSON Schema locations that hold schemas (not arbitrary user data/defaults).
_SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
_SCHEMA_LISTS = ("allOf", "anyOf", "oneOf", "prefixItems")
_SCHEMA_SINGLES = (
    "additionalProperties", "unevaluatedProperties", "propertyNames", "items",
    "contains", "not", "if", "then", "else", "additionalItems", "unevaluatedItems",
)


def compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Copy a schema, dropping only presentation titles at schema nodes."""
    result = deepcopy(schema)

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        node.pop("title", None)
        for key in _SCHEMA_MAPS:
            children = node.get(key)
            if isinstance(children, dict):
                for child in children.values():
                    visit(child)
        for key in _SCHEMA_LISTS:
            children = node.get(key)
            if isinstance(children, list):
                for child in children:
                    visit(child)
        for key in _SCHEMA_SINGLES:
            child = node.get(key)
            if isinstance(child, list):  # draft-07 tuple items
                for entry in child:
                    visit(entry)
            else:
                visit(child)

    visit(result)
    return result


class CatalogFastMCP(ApprovalAwareFastMCP):
    """Approval-aware server with a compact, skill-linked discovery catalog."""

    async def list_tools(self) -> list[Tool]:
        tools = await super().list_tools()
        return [
            tool.model_copy(update={
                # Unmigrated/new tools remain callable and discoverable. Tests
                # require explicit coverage before they can land in this repo.
                "description": tool_description(tool.name)
                if tool.name in TOOL_SUMMARIES else tool.description,
                "inputSchema": compact_schema(tool.inputSchema),
            })
            for tool in tools
        ]
