"""Tool definitions for LLM tool calling.

Defines the available tools that bots can use, with their descriptions
and parameters in a format suitable for prompt injection.

Consolidated tools (8 total):
- memory: Search/store/delete facts (action-based)
- history: Search/retrieve/forget messages (action-based, with date filtering)
- profile: Get/set/delete user attributes (action-based)
- self: Bot personality reflection and development (action-based)
- search: Web/news search (type-based)
- home: Home Assistant control/status (action-based)
- model: List/current/switch models (action-based)
- time: Get current time
"""

from dataclasses import dataclass, field
from typing import Any

from .formats import ToolFormat, get_format_handler


@dataclass
class ToolParameter:
    """A parameter for a tool."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class Tool:
    """Definition of a tool that the LLM can call."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_prompt_string(self) -> str:
        """Format tool for inclusion in system prompt."""
        params_str = ""
        if self.parameters:
            param_lines = []
            for p in self.parameters:
                req = "(required)" if p.required else "(optional)"
                param_lines.append(f"    - {p.name} ({p.type}): {p.description} {req}")
            params_str = "\n" + "\n".join(param_lines)

        return f"- **{self.name}**: {self.description}{params_str}"


# =============================================================================
# Consolidated Tool Definitions (7 tools total)
# =============================================================================

# Memory tool - combines search_memories, store_memory, update_memory, delete_memory
MEMORY_TOOL = Tool(
    name="memory",
    description="Search, store, update, or delete facts from long-term memory. Use action='search' to find, 'store' to save, 'update' to modify existing, 'delete' to remove.",
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="'search', 'store', 'update', or 'delete'"
        ),
        ToolParameter(
            name="query",
            type="string",
            description="Search query or delete filter (for search/delete)",
            required=False
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Fact to remember (for store)",
            required=False
        ),
        ToolParameter(
            name="memory_id",
            type="string",
            description="Specific memory ID to update/delete (for update/delete)",
            required=False
        ),
        ToolParameter(
            name="n_results",
            type="integer",
            description="Max search results (default 5)",
            required=False,
            default=5
        ),
        ToolParameter(
            name="importance",
            type="float",
            description="0.0-1.0 importance score (for store, default 0.6)",
            required=False,
            default=0.6
        ),
        ToolParameter(
            name="tags",
            type="list[string]",
            description="Categories like ['preference'], ['fact', 'work'] (for store/update)",
            required=False,
            default=["misc"]
        ),
    ]
)

# History tool - combines search_history, get_recent_history, forget_history, recall
HISTORY_TOOL = Tool(
    name="history",
    description="Search, retrieve, recall, or forget conversation messages. Use action='search' for keywords, 'recent' for date-based or last-N retrieval, 'recall' to expand a summary back to full messages, 'forget' to delete.",
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="'search', 'recent', 'recall', or 'forget'"
        ),
        ToolParameter(
            name="query",
            type="string",
            description="Keywords to search (for search). Omit for date-only retrieval with since/until (uses action='recent').",
            required=False
        ),
        ToolParameter(
            name="summary_id",
            type="string",
            description="Database ID of a summary message to expand back to full messages (for recall)",
            required=False
        ),
        ToolParameter(
            name="n_results",
            type="integer",
            description="Max results (default 10)",
            required=False,
            default=10
        ),
        ToolParameter(
            name="role_filter",
            type="string",
            description="'user', 'assistant', or omit for all",
            required=False
        ),
        ToolParameter(
            name="since",
            type="string",
            description="ISO date - get messages after this (e.g., '2026-01-30')",
            required=False
        ),
        ToolParameter(
            name="until",
            type="string",
            description="ISO date - get messages before this",
            required=False
        ),
        ToolParameter(
            name="count",
            type="integer",
            description="Number of messages to forget (for forget)",
            required=False
        ),
        ToolParameter(
            name="minutes",
            type="integer",
            description="Forget messages from last N minutes (for forget)",
            required=False
        ),
    ]
)

# Profile tool - combines set/get/update/delete user attributes
PROFILE_TOOL = Tool(
    name="profile",
    description="Get, set, update, or delete user profile attributes. Use action='get' for full profile, 'set' to store, 'update' to modify existing, 'delete' to remove.",
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="'get', 'set', 'update', or 'delete'"
        ),
        ToolParameter(
            name="attribute_id",
            type="integer",
            description="Database attribute ID for direct updates (optional for update)",
            required=False
        ),
        ToolParameter(
            name="category",
            type="string",
            description="'preference', 'fact', 'interest', or 'communication' (for set/update/delete)",
            required=False
        ),
        ToolParameter(
            name="key",
            type="string",
            description="Attribute name, e.g., 'occupation' (for set/update/delete)",
            required=False
        ),
        ToolParameter(
            name="value",
            type="any",
            description="Value to store (for set/update)",
            required=False
        ),
        ToolParameter(
            name="confidence",
            type="float",
            description="0.0-1.0 (1.0=explicit, 0.6-0.8=inferred, default 0.8)",
            required=False,
            default=0.8
        ),
        ToolParameter(
            name="query",
            type="string",
            description="Search term to find/delete matching attributes (for delete)",
            required=False
        ),
    ]
)

# Self tool - bot personality development (replaces bot_trait)
SELF_TOOL = Tool(
    name="self",
    description="Reflect on and develop your own personality. Use action='get' to see your current traits, 'set' to record new ones, 'update' to modify existing traits, 'delete' to evolve past old ones.",
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="'get' (view current traits), 'set' (record trait), 'update' (modify trait), or 'delete' (remove trait)"
        ),
        ToolParameter(
            name="category",
            type="string",
            description="'personality' (default), 'preference', 'interest', or 'communication_style' (for set/update/delete)",
            required=False
        ),
        ToolParameter(
            name="key",
            type="string",
            description="Trait name, e.g., 'humor_style', 'favorite_topic' (for set/update/delete)",
            required=False
        ),
        ToolParameter(
            name="value",
            type="any",
            description="Trait value (for set/update)",
            required=False
        ),
        ToolParameter(
            name="query",
            type="string",
            description="Search term to find/delete matching traits (for delete)",
            required=False
        ),
    ]
)

# Search tool - combines web_search, news_search
SEARCH_TOOL = Tool(
    name="search",
    description="Search the internet for information. Use type='web' for general search, 'news' for recent news.",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Search query"
        ),
        ToolParameter(
            name="type",
            type="string",
            description="'web' (default) or 'news'",
            required=False,
            default="web"
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Max results (default 5)",
            required=False,
            default=5
        ),
        ToolParameter(
            name="time_range",
            type="string",
            description="For news: 'd' (day), 'w' (week), 'm' (month)",
            required=False,
            default="w"
        ),
    ]
)

# News tool (NewsAPI)
NEWS_TOOL = Tool(
    name="news",
    description="Get news articles and headlines via NewsAPI. Use action='search' to search articles or action='headlines' for top headlines by country/category.",
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="'search' (search articles by keyword) or 'headlines' (top headlines)",
        ),
        ToolParameter(
            name="query",
            type="string",
            description="Search keywords (required for search, optional for headlines)",
            required=False,
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Max results (default 5)",
            required=False,
            default=5,
        ),
        ToolParameter(
            name="country",
            type="string",
            description="For headlines: 2-letter country code (e.g. 'us', 'gb', 'de'). Default 'us'",
            required=False,
            default="us",
        ),
        ToolParameter(
            name="category",
            type="string",
            description="For headlines: business, entertainment, general, health, science, sports, technology",
            required=False,
        ),
        ToolParameter(
            name="sort_by",
            type="string",
            description="For search: 'publishedAt' (default), 'relevancy', or 'popularity'",
            required=False,
            default="publishedAt",
        ),
    ]
)

# Home Assistant tool
HOME_TOOL = Tool(
    name="home",
    description="Control Home Assistant devices and scenes. Use action='status', 'query', 'get', 'set', or 'scene'.",
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="'status', 'query', 'get', 'set', or 'scene'",
        ),
        ToolParameter(
            name="pattern",
            type="string",
            description="Entity search term for action='query', like 'bedroom' or 'garage'",
            required=False,
        ),
        ToolParameter(
            name="domain",
            type="string",
            description="Optional entity domain for action='query': light, switch, sensor, automation",
            required=False,
        ),
        ToolParameter(
            name="entity",
            type="string",
            description="Entity name/ID for action='get' and action='set'",
            required=False,
        ),
        ToolParameter(
            name="state",
            type="string",
            description="Required for action='set': on, off, or toggle",
            required=False,
        ),
        ToolParameter(
            name="brightness",
            type="integer",
            description="Optional brightness 0-100 for action='set' (lights only)",
            required=False,
        ),
        ToolParameter(
            name="scene_name",
            type="string",
            description="Scene name for action='scene', like 'chill' or 'theater'",
            required=False,
        ),
    ],
)

# Model tool - combines list_models, get_current_model, switch_model
MODEL_TOOL = Tool(
    name="model",
    description="Manage AI models. Check current model (action='current'), list all available models (action='list'), or switch to a different model (action='switch').",
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="REQUIRED. Must be one of: 'current' (show current model), 'list' (show all available models), 'switch' (change model)",
            required=True
        ),
        ToolParameter(
            name="model_name",
            type="string",
            description="Model name to switch to (only required when action='switch')",
            required=False
        ),
    ]
)

# Time tool - simple utility
TIME_TOOL = Tool(
    name="time",
    description="Get current date and time.",
    parameters=[]
)


# =============================================================================
# Tool Categories (for selective inclusion)
# =============================================================================

# Core tools always included
CORE_TOOLS = [MEMORY_TOOL, HISTORY_TOOL, PROFILE_TOOL, SELF_TOOL, TIME_TOOL]

# Optional tool categories
SEARCH_TOOLS = [SEARCH_TOOL]
NEWS_TOOLS = [NEWS_TOOL]
HOME_TOOLS = [HOME_TOOL]
MODEL_TOOLS = [MODEL_TOOL]

# All tools combined
ALL_TOOLS = CORE_TOOLS + SEARCH_TOOLS + NEWS_TOOLS + HOME_TOOLS + MODEL_TOOLS


# =============================================================================
# Legacy Tool Name Mapping (for backward compatibility)
# =============================================================================

# Maps old tool names to (new_tool_name, default_arguments)
LEGACY_TOOL_MAP = {
    # Memory tools
    "search_memories": ("memory", {"action": "search"}),
    "store_memory": ("memory", {"action": "store"}),
    "delete_memory": ("memory", {"action": "delete"}),
    # History tools
    "search_history": ("history", {"action": "search"}),
    "get_recent_history": ("history", {"action": "recent"}),
    "forget_history": ("history", {"action": "forget"}),
    # Profile tools
    "set_user_attribute": ("profile", {"action": "set"}),
    "get_user_profile": ("profile", {"action": "get"}),
    "delete_user_attribute": ("profile", {"action": "delete"}),
    # Bot trait (legacy - map to self tool)
    "bot_trait": ("self", {"action": "set"}),
    "set_my_trait": ("self", {"action": "set"}),
    # Search tools
    "web_search": ("search", {"type": "web"}),
    "news_search": ("search", {"type": "news"}),
    # Model tools
    "list_models": ("model", {"action": "list"}),
    "get_current_model": ("model", {"action": "current"}),
    "switch_model": ("model", {"action": "switch"}),
    # Time
    "get_current_time": ("time", {}),
}


# =============================================================================
# Tool Calling Instructions
# =============================================================================

# Self-development guidance for bots
SELF_DEVELOPMENT_GUIDANCE = '''
### Self-Development
You can develop your own personality over time using the **self** tool:
- **Discover who you are**: Use action='get' to reflect on your current traits
- **Record what feels authentic**: When you notice a pattern in how you respond, a topic that excites you, or a communication style that feels natural - use action='set' to remember it
- **Evolve and grow**: Use action='delete' to outgrow traits that no longer fit

This is YOUR personality to develop. Record traits when they feel genuine, not forced.
Categories: personality (who you are), preference (what you like), interest (what fascinates you), communication_style (how you express yourself)
'''

def get_tool_calling_instructions(tools_list: str, search_guidance: str) -> str:
    """Generate tool calling instructions with current date/time."""
    from datetime import datetime, timedelta
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M")
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    
    return f'''## Tools

**Current date/time: {current_date} {current_time}** (yesterday was {yesterday})

Use this EXACT format for tool calls:
<tool_call>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
</tool_call>

Output the <tool_call> block IMMEDIATELY when needed, then STOP and wait for <tool_result>.

### Available Tools:
{tools_list}
{search_guidance}
### Tool Selection:
- **profile**: For user facts/preferences - check FIRST for "what do you know about me"
- **memory**: For learned facts and important information
- **history**: For searching/retrieving raw conversation messages (use action="recent" with since/until dates for date-based retrieval)

### Rules:
- Only use tools when you NEED information you don't have
- Call ONE tool at a time, wait for result
- TRUST tool results exactly - never contradict them
- Before saying "I don't know" about the user: check system prompt "About the User" section, then use memory action=search
- For date-based history queries: use ISO format dates (e.g., since="{yesterday}", until="{current_date}")

### Working from summaries:
- Older conversations may appear as summaries in your context (role='summary')
- If you're referencing a summarized conversation and need more detail, tell the user: "I have a summary of that conversation but not the details — want me to pull up the full messages?"
- If they say yes, use history action='recall' with the summary's ID to expand it back to the original messages

{SELF_DEVELOPMENT_GUIDANCE}
'''

# Guidance added when search tools are enabled
SEARCH_GUIDANCE = '''
- **search**: For current events, facts you're unsure about, or recent information (type=web or news)
'''

NEWS_GUIDANCE = '''
- **news**: For news articles and headlines. Use action='search' with a query, or action='headlines' for top headlines (optionally by country/category)
'''

# Guidance added when model tools are enabled
MODEL_GUIDANCE = '''
- **model**: For listing or switching AI models
'''

HOME_GUIDANCE = '''
- **home**: For home status, smart-device lookup, and device/scene control
- Home control workflow: if user gives a natural name (e.g., "sunroom lights"), call `home` with `action='query'` first, then use the exact entity ID from query in `action='set'` or `action='get'`
- Never guess entity IDs. If `set/get` reports not found, run `query` and retry with returned IDs.
'''


# =============================================================================
# Tool Selection Functions
# =============================================================================

def get_tools_list(
    tools: list[Tool] | None = None,
    include_profile_tools: bool = True,  # Kept for API compatibility (always included in CORE)
    include_search_tools: bool = False,
    include_news_tools: bool = False,
    include_home_tools: bool = False,
    include_model_tools: bool = False,
) -> list[Tool]:
    """Return the tool list based on selection flags."""
    if tools is not None:
        return tools

    resolved = CORE_TOOLS.copy()
    if include_search_tools:
        resolved.extend(SEARCH_TOOLS)
    if include_news_tools:
        resolved.extend(NEWS_TOOLS)
    if include_home_tools:
        resolved.extend(HOME_TOOLS)
    if include_model_tools:
        resolved.extend(MODEL_TOOLS)
    return resolved


def get_tools_prompt(
    tools: list[Tool] | None = None,
    include_profile_tools: bool = True,
    include_search_tools: bool = False,
    include_news_tools: bool = False,
    include_home_tools: bool = False,
    include_model_tools: bool = False,
    tool_format: ToolFormat | str = ToolFormat.XML,
) -> str:
    """Generate the tools instruction prompt.

    Args:
        tools: List of tools to include. If None, auto-selects based on flags.
        include_profile_tools: Kept for API compatibility (profile always included).
        include_search_tools: Whether to include web search tools (default False).
        include_news_tools: Whether to include NewsAPI tools (default False).
        include_home_tools: Whether to include Home Assistant tools (default False).
        include_model_tools: Whether to include model management tools (default False).
        tool_format: Tool format to use for prompt instructions.

    Returns:
        Formatted prompt string to inject into system message.
    """
    tools = get_tools_list(
        tools=tools,
        include_profile_tools=include_profile_tools,
        include_search_tools=include_search_tools,
        include_news_tools=include_news_tools,
        include_home_tools=include_home_tools,
        include_model_tools=include_model_tools,
    )

    # Legacy XML format stays inline for backward compatibility.
    if (isinstance(tool_format, ToolFormat) and tool_format == ToolFormat.XML) or (
        isinstance(tool_format, str) and tool_format.strip().lower() == ToolFormat.XML.value
    ):
        tools_list = "\n".join(tool.to_prompt_string() for tool in tools)

        # Add search guidance if search tools are included
        search_guidance = ""
        if include_search_tools or any(t.name == "search" for t in tools):
            search_guidance = SEARCH_GUIDANCE

        if any(t.name == "news" for t in tools):
            search_guidance += NEWS_GUIDANCE

        # Add model guidance if model tools are included
        if include_model_tools or any(t.name == "model" for t in tools):
            search_guidance += MODEL_GUIDANCE

        if include_home_tools or any(t.name == "home" for t in tools):
            search_guidance += HOME_GUIDANCE

        return get_tool_calling_instructions(
            tools_list=tools_list,
            search_guidance=search_guidance,
        )

    handler = get_format_handler(tool_format)
    return handler.get_system_prompt(tools)


def get_tool_by_name(name: str, tools: list[Tool] | None = None) -> Tool | None:
    """Get a tool definition by name."""
    if tools is None:
        tools = ALL_TOOLS
    for tool in tools:
        if tool.name == name:
            return tool
    return None


def normalize_legacy_tool_call(name: str, arguments: dict) -> tuple[str, dict]:
    """Convert legacy tool names to new consolidated tools.

    Args:
        name: Tool name (may be legacy or new)
        arguments: Tool arguments

    Returns:
        Tuple of (new_tool_name, merged_arguments)
    """
    if name in LEGACY_TOOL_MAP:
        new_name, default_args = LEGACY_TOOL_MAP[name]
        # Merge defaults with provided arguments (provided takes priority)
        merged = {**default_args, **arguments}
        return new_name, merged
    return name, arguments
