"""Regression: a deferred-search miss must not imply loaded MCP tools are absent."""
from copy import deepcopy

import pytest

from claude_code_bridge.proxy.tool_discovery import TOOL_SEARCH_GUIDANCE, model_tool_description
from claude_code_bridge.proxy.translate import _tools_to_responses
from claude_code_bridge.proxy.translate_cc import _tools_to_cc


@pytest.mark.parametrize("convert,nested", [(_tools_to_responses, False), (_tools_to_cc, True)])
def test_search_guidance_and_loaded_ops_schema_are_both_preserved(convert, nested):
    tools = [
        {"name": "ToolSearch", "description": "Fetch deferred tool definitions.",
         "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        {"name": "mcp__bawthub__ops_list_operations", "description": "Read operation catalog.",
         "input_schema": {"type": "object", "properties": {"include_disabled": {"type": "boolean"}}}},
    ]
    original = deepcopy(tools)
    result = convert(tools)
    functions = [item["function"] if nested else item for item in result]
    by_name = {item["name"]: item for item in functions}
    search = by_name["ToolSearch"]
    assert search["description"].startswith(tools[0]["description"])
    assert TOOL_SEARCH_GUIDANCE in search["description"]
    assert "does NOT mean" in search["description"]
    assert "select:mcp__bawthub__ops_list_operations" in search["description"]
    for tool in tools:
        assert by_name[tool["name"]]["parameters"] == tool["input_schema"]
    assert by_name[tools[1]["name"]]["description"] == tools[1]["description"]
    assert tools == original
    assert convert(tools) == result  # stable cache prefix


def test_description_guidance_is_idempotent_and_handles_missing_description():
    rendered = model_tool_description({"name": "ToolSearch"})
    assert rendered == TOOL_SEARCH_GUIDANCE
    assert model_tool_description({"name": "ToolSearch", "description": rendered}) == rendered
    assert model_tool_description({"name": "other"}) is None
    assert model_tool_description({"name": "other", "description": "unchanged"}) == "unchanged"


@pytest.mark.parametrize("convert", [_tools_to_responses, _tools_to_cc])
def test_no_search_tool_is_invented_and_server_tool_filter_is_unchanged(convert):
    assert convert([]) is None
    assert convert([{"name": "web_search", "type": "web_search_20260209"}]) is None
    output = convert([{"name": "Read", "input_schema": {"type": "object"}}])
    assert "ToolSearch" not in str(output)
    assert "description" not in (output[0].get("function") or output[0])
