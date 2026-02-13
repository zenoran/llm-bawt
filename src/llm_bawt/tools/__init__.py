"""Tool calling support for LLMs.

This module provides prompt-based tool calling that works with any model,
allowing bots to autonomously search memories, store facts, manage profiles,
search the web, and switch models.

Consolidated tools (8 total):
- memory: Search/store/delete facts
- history: Search/retrieve/forget messages (with date filtering)
- profile: Get/set/delete user attributes
- self: Bot personality reflection and development
- search: Web/news search
- home: Home Assistant control/status
- model: List/current/switch models
- time: Get current time
"""

from .definitions import (
    # Tool instances
    MEMORY_TOOL,
    HISTORY_TOOL,
    PROFILE_TOOL,
    SELF_TOOL,
    SEARCH_TOOL,
    HOME_TOOL,
    MODEL_TOOL,
    TIME_TOOL,
    # Tool categories
    CORE_TOOLS,
    SEARCH_TOOLS,
    HOME_TOOLS,
    MODEL_TOOLS,
    ALL_TOOLS,
    # Legacy mapping
    LEGACY_TOOL_MAP,
    normalize_legacy_tool_call,
    # Functions
    get_tools_prompt,
    get_tools_list,
    get_tool_by_name,
    # Types
    Tool,
    ToolParameter,
)
from .parser import parse_tool_calls, ToolCall, has_tool_call, format_tool_result, KNOWN_TOOLS
from .executor import ToolExecutor
from .loop import ToolLoop, query_with_tools
from .streaming import stream_with_tools

__all__ = [
    # Tool instances
    "MEMORY_TOOL",
    "HISTORY_TOOL",
    "PROFILE_TOOL",
    "SELF_TOOL",
    "SEARCH_TOOL",
    "HOME_TOOL",
    "MODEL_TOOL",
    "TIME_TOOL",
    # Tool categories
    "CORE_TOOLS",
    "SEARCH_TOOLS",
    "HOME_TOOLS",
    "MODEL_TOOLS",
    "ALL_TOOLS",
    # Legacy mapping
    "LEGACY_TOOL_MAP",
    "normalize_legacy_tool_call",
    # Functions
    "get_tools_prompt",
    "get_tools_list",
    "get_tool_by_name",
    # Types
    "Tool",
    "ToolParameter",
    # Parser
    "parse_tool_calls",
    "ToolCall",
    "has_tool_call",
    "format_tool_result",
    "KNOWN_TOOLS",
    # Executor
    "ToolExecutor",
    # Loop
    "ToolLoop",
    "query_with_tools",
    # Streaming
    "stream_with_tools",
]
