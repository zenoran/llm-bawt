# HA Native MCP Integration — Implementation Plan

> **Goal**: Replace the custom 5-tool HA MCP server on `vex@ubuntu` with a direct connection to Home Assistant's **native MCP server** at `/api/mcp`. This exposes ~22 HA intent tools (HassTurnOn, HassLightSet, HassSetPosition, etc.) directly to the LLM — no more single `home` tool with action dispatch. The device CSV currently injected into the system prompt via HA's template is no longer needed.

> **Transport**: HA native MCP uses **Streamable HTTP** (not SSE). Requires `mcp` SDK ≥1.26.0 with `mcp.client.streamable_http.streamablehttp_client`.

> **Auth**: HA long-lived access token passed as `Authorization: Bearer <token>` HTTP header.

---

## Overview of Changes

| # | File | What Changes |
|---|------|-------------|
| 1 | `src/llm_bawt/utils/config.py` | Add 3 new config fields |
| 2 | `.env.docker` | Add new env vars |
| 3 | `src/llm_bawt/integrations/ha_mcp/client.py` | **Rewrite** — new `HomeAssistantNativeClient` |
| 4 | `src/llm_bawt/integrations/ha_mcp/__init__.py` | Update exports |
| 5 | `src/llm_bawt/tools/definitions.py` | Add dynamic HA tool support, update tool categories |
| 6 | `src/llm_bawt/tools/executor.py` | Add HA native tool passthrough execution |
| 7 | `src/llm_bawt/tools/formats/native_openai.py` | Remove hardcoded `home` tool guidance, handle HA tools |
| 8 | `src/llm_bawt/core/base.py` | Update init to use native client + tool discovery |
| 9 | `src/llm_bawt/core/pipeline.py` | Pass HA tools through pipeline |
| 10 | `src/llm_bawt/tools/loop.py` | Handle HA tool execution in loop |
| 11 | `src/llm_bawt/service/routes/botchat.py` | Update sanitizer for new HA context |
| 12 | `src/llm_bawt/core/status.py` | Update status to show native MCP info |
| 13 | `tests/test_ha_native_mcp.py` | New test file |

---

## Step 1: Add Config Fields

**File**: `src/llm_bawt/utils/config.py`

Find the existing HA_MCP config fields (around line 203-220). Add these **after** `HA_MCP_AUTH_TOKEN`:

```python
    HA_NATIVE_MCP_URL: str = Field(
        default="",
        description="Home Assistant native MCP endpoint URL, e.g. http://hass.home:8123/api/mcp (Set via LLM_BAWT_HA_NATIVE_MCP_URL)",
    )
    HA_NATIVE_MCP_TOKEN: str = Field(
        default="",
        description="Home Assistant long-lived access token for native MCP auth (Set via LLM_BAWT_HA_NATIVE_MCP_TOKEN)",
    )
    HA_MCP_TOOL_EXCLUDE: str = Field(
        default="GetDateTime,HassCancelAllTimers,HassBroadcast",
        description="Comma-separated list of HA MCP tool names to exclude (Set via LLM_BAWT_HA_MCP_TOOL_EXCLUDE)",
    )
```

**That's it for config.py.** The existing `HA_MCP_ENABLED`, `HA_MCP_URL`, etc. stay for backward compat with the custom server. The new native client activates when `HA_NATIVE_MCP_URL` is non-empty.

---

## Step 2: Add Env Vars to .env.docker

**File**: `.env.docker`

Find the existing HA_MCP block (lines 26-30). Add these lines **after** `LLM_BAWT_HA_MCP_AUTH_TOKEN=`:

```dotenv
# --- HA Native MCP (direct connection to HA's /api/mcp endpoint) ---
# When set, this takes priority over the custom HA MCP server above
LLM_BAWT_HA_NATIVE_MCP_URL=
LLM_BAWT_HA_NATIVE_MCP_TOKEN=
# Comma-separated tool names to exclude from HA's native MCP tools
LLM_BAWT_HA_MCP_TOOL_EXCLUDE=GetDateTime,HassCancelAllTimers,HassBroadcast
```

---

## Step 3: Rewrite the HA MCP Client

**File**: `src/llm_bawt/integrations/ha_mcp/client.py`

Replace the **entire file** with this:

```python
"""Home Assistant MCP client.

Two modes:
1. Native MCP — connects directly to HA's /api/mcp via Streamable HTTP.
   Discovers tools dynamically. This is the preferred mode.
2. Legacy MCP — connects to the custom FastMCP server via SSE (backward compat).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from ...utils.config import Config

logger = logging.getLogger(__name__)


@dataclass
class HAToolDefinition:
    """A tool discovered from HA's native MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


class HomeAssistantNativeClient:
    """Client that connects to HA's native MCP server at /api/mcp.

    On init, connects via Streamable HTTP, calls list_tools() to discover
    available tools, caches them, and provides call_tool() for execution.
    """

    def __init__(self, config: Config):
        self._url = (getattr(config, "HA_NATIVE_MCP_URL", "") or "").rstrip("/")
        self._token = (getattr(config, "HA_NATIVE_MCP_TOKEN", "") or "").strip()
        self._timeout = float(max(1, int(getattr(config, "HA_MCP_TIMEOUT", 10) or 10)))

        # Parse exclude list from config
        exclude_str = (getattr(config, "HA_MCP_TOOL_EXCLUDE", "") or "").strip()
        self._exclude_names: set[str] = set()
        if exclude_str:
            self._exclude_names = {n.strip() for n in exclude_str.split(",") if n.strip()}

        # Cached tool definitions (populated by discover_tools())
        self._tools: list[HAToolDefinition] = []
        self._tool_names: set[str] = set()
        self._initialized = False

    @property
    def available(self) -> bool:
        """True if URL and token are configured."""
        return bool(self._url) and bool(self._token)

    @property
    def initialized(self) -> bool:
        """True if tools have been discovered."""
        return self._initialized

    @property
    def tools(self) -> list[HAToolDefinition]:
        """Return cached tool definitions."""
        return self._tools

    @property
    def tool_names(self) -> set[str]:
        """Return set of available tool names."""
        return self._tool_names

    def is_ha_tool(self, name: str) -> bool:
        """Check if a tool name belongs to HA native tools."""
        return name in self._tool_names

    def discover_tools(self) -> list[HAToolDefinition]:
        """Connect to HA MCP, list tools, cache and return them.

        Filters out tools in the exclude list. Safe to call multiple times
        (will re-discover each time).
        """
        if not self.available:
            logger.warning("HA native MCP not configured (missing URL or token)")
            return []

        try:
            raw_tools = self._run_async(self._discover_tools_async())
            self._tools = []
            self._tool_names = set()

            for t in raw_tools:
                if t.name in self._exclude_names:
                    logger.debug(f"Excluding HA tool: {t.name}")
                    continue
                tool_def = HAToolDefinition(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema or {"type": "object", "properties": {}},
                )
                self._tools.append(tool_def)
                self._tool_names.add(t.name)

            self._initialized = True
            logger.info(
                f"Discovered {len(self._tools)} HA native MCP tools "
                f"(excluded {len(self._exclude_names & {t.name for t in raw_tools})})"
            )
            return self._tools

        except Exception as e:
            logger.error(f"Failed to discover HA native MCP tools: {e}")
            self._initialized = False
            return []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call an HA tool by name and return the text result."""
        if not self.available:
            return "Error: HA native MCP not configured"
        if not self._initialized:
            return "Error: HA native MCP tools not discovered yet"
        if name not in self._tool_names:
            return f"Error: Unknown HA tool '{name}'"

        try:
            result = self._run_async(self._call_tool_async(name, arguments))
            return self._extract_text(result)
        except Exception as e:
            logger.error(f"HA tool call failed: {name}({arguments}): {e}")
            return f"Error calling HA tool {name}: {e}"

    # --- Async internals ---

    async def _discover_tools_async(self):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        headers = {"Authorization": f"Bearer {self._token}"}
        async with streamablehttp_client(self._url, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return result.tools

    async def _call_tool_async(self, name: str, arguments: dict[str, Any]):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        headers = {"Authorization": f"Bearer {self._token}"}
        async with streamablehttp_client(self._url, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(name, arguments)

    # --- Sync/async bridge ---

    @staticmethod
    def _run_async(coro):
        """Run an async coroutine from sync context, handling existing event loops."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: Any = None
        error: BaseException | None = None

        def _runner():
            nonlocal result, error
            try:
                result = asyncio.run(coro)
            except BaseException as exc:
                error = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join(timeout=30)
        if error is not None:
            raise error
        return result

    @staticmethod
    def _extract_text(result: Any) -> str:
        """Extract text content from MCP tool result."""
        if hasattr(result, "content"):
            parts: list[str] = []
            for item in getattr(result, "content", []) or []:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            joined = "\n".join(p for p in parts if p).strip()
            if joined:
                return joined

        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    return joined

        if isinstance(result, str):
            return result.strip()

        return json.dumps(result, default=str, ensure_ascii=False)


# --- Legacy client kept for backward compatibility ---


class HomeAssistantMCPClient:
    """Legacy client for the custom FastMCP server (SSE transport).

    Kept for backward compatibility. Use HomeAssistantNativeClient instead.
    """

    def __init__(self, config: Config):
        self._base_url = (getattr(config, "HA_MCP_URL", "") or "").rstrip("/")
        self._timeout = float(max(1, int(getattr(config, "HA_MCP_TIMEOUT", 10) or 10)))
        self._auth_token = (getattr(config, "HA_MCP_AUTH_TOKEN", "") or "").strip()

    @property
    def available(self) -> bool:
        return bool(self._base_url)

    def status(self) -> str:
        return self._call_text_tool("status", {})

    def status_raw(self) -> str:
        result = self._call_tool(name="status", arguments={})
        return json.dumps(self._to_serializable(result), indent=2, default=str, ensure_ascii=False)

    def query(self, pattern: str | None = None, domain: str | None = None) -> str:
        args: dict[str, Any] = {}
        if pattern:
            args["pattern"] = pattern
        if domain:
            args["domain"] = domain
        return self._call_text_tool("query", args)

    def get(self, entity: str) -> str:
        return self._call_text_tool("get", {"entity": entity})

    def set(self, entity: str, state: str, brightness: int | None = None) -> str:
        args: dict[str, Any] = {"entity": entity, "state": state}
        if brightness is not None:
            args["brightness"] = brightness
        return self._call_text_tool("set", args)

    def scene(self, name: str) -> str:
        return self._call_text_tool("scene", {"name": name})

    def _call_text_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = self._call_tool(name=name, arguments=arguments)
        text = self._extract_text(result)
        if text:
            return text
        return json.dumps(result, default=str)

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self._base_url:
            raise RuntimeError("HA MCP URL is not configured")
        sse_url = f"{self._base_url}/sse"
        try:
            return self._run_async(self._call_tool_async(sse_url, name, arguments))
        except Exception as e:
            raise ConnectionError(f"HA MCP request failed: {e}") from e

    async def _call_tool_async(self, sse_url: str, name: str, arguments: dict[str, Any]) -> Any:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        async with sse_client(sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(name, arguments)

    @staticmethod
    def _run_async(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: Any = None
        error: BaseException | None = None

        def _runner():
            nonlocal result, error
            try:
                result = asyncio.run(coro)
            except BaseException as exc:
                error = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if error is not None:
            raise error
        return result

    @staticmethod
    def _extract_text(result: Any) -> str:
        if hasattr(result, "content"):
            parts: list[str] = []
            for item in getattr(result, "content", []) or []:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            return "\n".join(p for p in parts if p).strip()

        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list):
                parts_list: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text" and isinstance(item.get("text"), str):
                            parts_list.append(item["text"])
                        elif item.get("type") == "json":
                            parts_list.append(json.dumps(item.get("json"), default=str))
                return "\n".join(p for p in parts_list if p).strip()

            structured = result.get("structuredContent")
            if isinstance(structured, dict) and "result" in structured:
                embedded = structured["result"]
                if isinstance(embedded, str):
                    return embedded.strip()

        if isinstance(result, str):
            return result.strip()
        return ""

    @classmethod
    def _to_serializable(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): cls._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._to_serializable(item) for item in value]
        if hasattr(value, "model_dump"):
            try:
                return cls._to_serializable(value.model_dump())
            except Exception:
                pass
        if hasattr(value, "dict"):
            try:
                return cls._to_serializable(value.dict())
            except Exception:
                pass
        return str(value)
```

---

## Step 4: Update `__init__.py`

**File**: `src/llm_bawt/integrations/ha_mcp/__init__.py`

Replace entirely with:

```python
"""Home Assistant MCP integration."""

from .client import HomeAssistantMCPClient, HomeAssistantNativeClient, HAToolDefinition

__all__ = ["HomeAssistantMCPClient", "HomeAssistantNativeClient", "HAToolDefinition"]
```

---

## Step 5: Update Tool Definitions

**File**: `src/llm_bawt/tools/definitions.py`

### 5a: Add function to convert HA tools to Tool dataclass

Add this function **right before** `get_tools_list()` (around line 580):

```python
def ha_tools_to_tool_definitions(ha_tools: list) -> list[Tool]:
    """Convert HAToolDefinition objects to Tool dataclass instances.

    Takes the tool definitions discovered from HA's native MCP server and
    converts them into the same Tool format used by all other llm-bawt tools.
    This allows them to be included in tool lists and schema generation.

    Args:
        ha_tools: List of HAToolDefinition from HomeAssistantNativeClient.

    Returns:
        List of Tool dataclass instances.
    """
    tools = []
    for ha_tool in ha_tools:
        params = []
        schema = ha_tool.input_schema or {}
        properties = schema.get("properties", {})
        required_names = set(schema.get("required", []))

        for prop_name, prop_def in properties.items():
            # Map JSON schema types to our ToolParameter types
            prop_type = prop_def.get("type", "string")
            if prop_type == "array":
                items = prop_def.get("items", {})
                item_type = items.get("type", "string")
                prop_type = f"array[{item_type}]"
            elif prop_type == "number":
                prop_type = "number"

            description = prop_def.get("description", "")
            # Add enum values to description if present
            if "enum" in prop_def:
                enum_str = ", ".join(str(v) for v in prop_def["enum"])
                description = f"{description} (one of: {enum_str})".strip()
            elif "items" in prop_def and "enum" in prop_def.get("items", {}):
                enum_str = ", ".join(str(v) for v in prop_def["items"]["enum"])
                description = f"{description} (values: {enum_str})".strip()

            # Add min/max to description if present
            if "minimum" in prop_def or "maximum" in prop_def:
                range_parts = []
                if "minimum" in prop_def:
                    range_parts.append(f"min={prop_def['minimum']}")
                if "maximum" in prop_def:
                    range_parts.append(f"max={prop_def['maximum']}")
                description = f"{description} ({', '.join(range_parts)})".strip()

            params.append(ToolParameter(
                name=prop_name,
                type=prop_type,
                description=description or f"Parameter '{prop_name}'",
                required=prop_name in required_names,
            ))

        tools.append(Tool(
            name=ha_tool.name,
            description=ha_tool.description or f"Home Assistant tool: {ha_tool.name}",
            parameters=params,
        ))

    return tools
```

### 5b: Update `get_tools_list()` to accept HA tools

Find `get_tools_list()` and change its signature and body:

**OLD** (around line 580-600):
```python
def get_tools_list(
    tools: list[Tool] | None = None,
    include_profile_tools: bool = True,
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
```

**NEW**:
```python
def get_tools_list(
    tools: list[Tool] | None = None,
    include_profile_tools: bool = True,
    include_search_tools: bool = False,
    include_news_tools: bool = False,
    include_home_tools: bool = False,
    include_model_tools: bool = False,
    ha_native_tools: list[Tool] | None = None,
) -> list[Tool]:
    """Return the tool list based on selection flags.

    Args:
        ha_native_tools: Pre-converted HA native MCP tool definitions.
            When provided, these replace HOME_TOOLS entirely.
    """
    if tools is not None:
        return tools

    resolved = CORE_TOOLS.copy()
    if include_search_tools:
        resolved.extend(SEARCH_TOOLS)
    if include_news_tools:
        resolved.extend(NEWS_TOOLS)
    if ha_native_tools:
        # Native HA tools replace the legacy home tool
        resolved.extend(ha_native_tools)
    elif include_home_tools:
        # Fallback to legacy home tool
        resolved.extend(HOME_TOOLS)
    if include_model_tools:
        resolved.extend(MODEL_TOOLS)
    return resolved
```

### 5c: Update `get_tools_prompt()` for HA native tools

Find `get_tools_prompt()`. Add `ha_native_tools: list[Tool] | None = None` parameter. Pass it through to `get_tools_list()`. Also update the guidance section.

In the function signature, add `ha_native_tools: list[Tool] | None = None` parameter.

In the body where it calls `get_tools_list()`, pass `ha_native_tools=ha_native_tools`.

In the XML format branch where `HOME_GUIDANCE` is appended, change the condition:

**OLD**:
```python
        if include_home_tools or any(t.name == "home" for t in tools):
            search_guidance += HOME_GUIDANCE
```

**NEW**:
```python
        if ha_native_tools:
            search_guidance += HA_NATIVE_GUIDANCE
        elif include_home_tools or any(t.name == "home" for t in tools):
            search_guidance += HOME_GUIDANCE
```

### 5d: Add `HA_NATIVE_GUIDANCE` constant

Add this constant near `HOME_GUIDANCE` (around line 566):

```python
HA_NATIVE_GUIDANCE = '''
- **Home Assistant tools** (HassTurnOn, HassTurnOff, HassLightSet, etc.): Control smart home devices directly
- These tools use **friendly names** (e.g., name="kitchen lights"), NOT entity IDs
- Use name, area, and/or floor parameters to identify devices
- For device state queries, use **GetLiveContext** tool
- HassTurnOn: Turn on lights, switches, scripts. For covers/blinds, this OPENS them.
- HassTurnOff: Turn off lights, switches. For covers/blinds, this CLOSES them.
- HassLightSet: Set brightness (0-100), color, or color temperature for lights
- HassSetPosition: Set cover/blind position (0-100)
- Execute tool calls immediately when user gives a command — do not describe what you plan to do.
'''
```

---

## Step 6: Update Tool Executor

**File**: `src/llm_bawt/tools/executor.py`

### 6a: Add `ha_native_client` to constructor

Find `__init__` (line 129). Add a new parameter after `home_client`:

```python
    def __init__(
        self,
        memory_client: "MemoryClient | None" = None,
        profile_manager: "ProfileManager | None" = None,
        search_client: "SearchClient | None" = None,
        home_client: "HomeAssistantMCPClient | None" = None,
        ha_native_client: "HomeAssistantNativeClient | None" = None,   # <-- ADD THIS
        news_client: "NewsAPIClient | None" = None,
        ...
    ):
```

And store it:
```python
        self.ha_native_client = ha_native_client
```

Also add the TYPE_CHECKING import at the top of the file (near line 28 where `HomeAssistantMCPClient` is imported):

```python
if TYPE_CHECKING:
    ...
    from ..integrations.ha_mcp.client import HomeAssistantMCPClient, HomeAssistantNativeClient
```

### 6b: Update `execute()` method to handle HA native tools

Find the `execute()` method. It currently looks up `self._handlers.get(tool_name)`. Before that lookup, add a check for HA native tools:

Find the line where it does the handler lookup (should be something like `handler = self._handlers.get(tool_name)`). **Before** that line, add:

```python
        # Check if this is an HA native tool call
        if self.ha_native_client and self.ha_native_client.is_ha_tool(tool_name):
            return self._execute_ha_native_tool(tool_call)
```

### 6c: Add `_execute_ha_native_tool()` method

Add this new method in the executor class (right after `_execute_home`, around line 1570):

```python
    def _execute_ha_native_tool(self, tool_call: ToolCall) -> str:
        """Execute an HA native MCP tool call (passthrough to HA)."""
        if not self.ha_native_client:
            return format_tool_result(tool_call.name, "HA native MCP client not available")

        tool_name = tool_call.name
        arguments = tool_call.arguments or {}

        # Strip None values — HA doesn't want them
        clean_args = {k: v for k, v in arguments.items() if v is not None}

        logger.info(f"HA native tool call: {tool_name}({clean_args})")
        try:
            result = self.ha_native_client.call_tool(tool_name, clean_args)
            logger.debug(f"HA native tool result: {result[:200] if result else '(empty)'}")
            return format_tool_result(tool_name, result or "Done")
        except Exception as e:
            logger.error(f"HA native tool call failed: {tool_name}: {e}")
            return format_tool_result(tool_name, f"Error: {e}")
```

---

## Step 7: Update Native OpenAI Format Handler

**File**: `src/llm_bawt/tools/formats/native_openai.py`

### 7a: Update `get_system_prompt()` to handle HA native tools

The current code has hardcoded `home` tool guidance. Replace the `home_guidance` block.

**FIND** (around line 52-66):
```python
        home_guidance = ""
        if "home" in tool_names:
            home_guidance = (
                "Home tool guidance:\n"
                "- For natural names like 'sunroom lights', call home(action='query', pattern='sunroom', domain='light') first.\n"
                ...
            )
```

**REPLACE WITH**:
```python
        home_guidance = ""
        # Check if HA native tools are present (they start with "Hass" or "Get")
        ha_native_names = [n for n in tool_names if n.startswith("Hass") or n in ("GetLiveContext", "GetDateTime")]
        if ha_native_names:
            home_guidance = (
                "Home Assistant tool guidance:\n"
                "- These tools use friendly device NAMES (e.g., name='kitchen lights'), NOT entity IDs.\n"
                "- Use name, area, and/or floor parameters to target devices.\n"
                "- HassTurnOn opens covers/blinds. HassTurnOff closes them.\n"
                "- HassLightSet sets brightness (0-100), color, or color temperature.\n"
                "- HassSetPosition sets cover/blind position (0-100).\n"
                "- For device state queries, use GetLiveContext.\n"
                "- Execute tool calls immediately — do not describe what you're about to do.\n"
            )
        elif "home" in tool_names:
            home_guidance = (
                "Home tool guidance:\n"
                "- For natural names like 'sunroom lights', call home(action='query', pattern='sunroom', domain='light') first.\n"
                "- Use exact entity IDs returned by query in subsequent home(action='get'/'set') calls.\n"
                "- Covers/blinds use state='close' or state='open' (NOT 'off'/'on') with action='set'.\n"
                "- Lights and switches use state='on', 'off', or 'toggle'.\n"
                "- Locks use state='lock' or state='unlock'.\n"
                "- If a set/get call reports not found, run query and retry with the suggested exact ID.\n"
                "- If asked for current home status, call home(action='status') before answering.\n"
                "- If asked for 'raw output', return the exact tool output verbatim; do not invent or normalize JSON fields.\n"
                "- Execute tool calls immediately when the user confirms — do not describe what you're about to do without calling the tool.\n"
            )
```

### 7b: Update `get_tools_schema()` to handle HA native tools with pre-built schemas

The current `get_tools_schema()` builds schemas from `ToolParameter` objects. HA tools come with their own JSON Schema already in `input_schema`. We need to detect this.

Find `get_tools_schema()`. The HA native tool `Tool` objects will have parameters built from the schema by `ha_tools_to_tool_definitions()`, so they'll work with the existing code. **No change needed here** — the conversion in Step 5a handles it.

---

## Step 8: Update Core Base

**File**: `src/llm_bawt/core/base.py`

### 8a: Add import for new client

Find the existing import (line 26):
```python
from ..integrations.ha_mcp.client import HomeAssistantMCPClient
```

Change to:
```python
from ..integrations.ha_mcp.client import HomeAssistantMCPClient, HomeAssistantNativeClient
```

### 8b: Add instance variable

Find where `self.home_client` is declared (around line 94-100). Add:

```python
        self.home_client: HomeAssistantMCPClient | None = None
        self.ha_native_client: HomeAssistantNativeClient | None = None
```

### 8c: Update `_init_home_assistant()`

Find `_init_home_assistant()` (around line 567). Replace it entirely:

```python
    def _init_home_assistant(self, config: Config):
        """Initialize Home Assistant integration.

        Priority: HA native MCP > legacy custom MCP server.
        """
        if not self.bot.uses_tools:
            return

        # Try native MCP first (direct connection to HA's /api/mcp)
        native_url = getattr(config, "HA_NATIVE_MCP_URL", "") or ""
        native_token = getattr(config, "HA_NATIVE_MCP_TOKEN", "") or ""
        if native_url and native_token:
            try:
                client = HomeAssistantNativeClient(config)
                if client.available:
                    tools = client.discover_tools()
                    if tools:
                        self.ha_native_client = client
                        logger.info(f"HA native MCP initialized with {len(tools)} tools")
                        return
                    else:
                        logger.warning("HA native MCP connected but no tools discovered")
            except Exception as e:
                logger.warning(f"Failed to initialize HA native MCP: {e}")

        # Fallback to legacy custom MCP server
        if not getattr(config, "HA_MCP_ENABLED", False):
            return
        try:
            client = HomeAssistantMCPClient(config)
            if client.available:
                self.home_client = client
                logger.debug("Legacy Home Assistant MCP client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize legacy HA MCP client: {e}")
```

### 8d: Update `_get_tool_definitions()`

Find `_get_tool_definitions()` (around line 555). Update it to pass HA native tools:

```python
    def _get_tool_definitions(self) -> list:
        """Get the list of tool definitions for the current configuration."""
        from ..tools.definitions import get_tools_list, ha_tools_to_tool_definitions

        include_home = self.home_client is not None
        include_search = self.search_client is not None

        # Convert HA native tools if available
        ha_native_tools = None
        if self.ha_native_client and self.ha_native_client.initialized:
            ha_native_tools = ha_tools_to_tool_definitions(self.ha_native_client.tools)

        tools = get_tools_list(
            include_search_tools=include_search,
            include_news_tools=self.news_client is not None,
            include_home_tools=include_home,
            include_model_tools=self.model_lifecycle is not None,
            ha_native_tools=ha_native_tools,
        )
        if self.memory is None:
            disallowed = {"memory", "history", "profile", "self"}
            tools = [t for t in tools if t.name not in disallowed]
        return tools
```

### 8e: Update `query()` to pass `ha_native_client`

Find the `query()` method (around line 304). Where it calls `query_with_tools()`, add `ha_native_client`:

Find this pattern:
```python
        home_client=self.home_client if self.bot.uses_tools else None,
```

Change to:
```python
        home_client=self.home_client if self.bot.uses_tools else None,
        ha_native_client=self.ha_native_client if self.bot.uses_tools else None,
```

### 8f: Update `_build_context_messages()` — determine if tools are available

Find where `use_tools` is determined. It currently checks `home_client is not None`. Update to also check `ha_native_client`:

Wherever you see:
```python
memory_client is not None or home_client is not None
```

Change to:
```python
memory_client is not None or home_client is not None or ha_native_client is not None
```

Or equivalent. Search for all such patterns in base.py.

---

## Step 9: Update Pipeline

**File**: `src/llm_bawt/core/pipeline.py`

### 9a: Add `ha_native_client` parameter to constructor

Find `__init__` and add `ha_native_client` parameter wherever `home_client` is accepted. Store it as `self.ha_native_client`.

### 9b: Update `_stage_pre_process()` tool detection

Find where it checks if tools should be used (around line 237-277). Add `ha_native_client`:

The check currently is something like:
```python
ctx.use_tools = bot.uses_tools and (memory_client is not None or home_client is not None)
```

Change to:
```python
ctx.use_tools = bot.uses_tools and (memory_client is not None or home_client is not None or self.ha_native_client is not None)
```

### 9c: Update `_stage_context_build()` tool definitions

Find where `get_tools_list()` is called. Add the `ha_native_tools` parameter:

```python
        # Convert HA native tools if available
        ha_native_tool_defs = None
        if self.ha_native_client and self.ha_native_client.initialized:
            from ..tools.definitions import ha_tools_to_tool_definitions
            ha_native_tool_defs = ha_tools_to_tool_definitions(self.ha_native_client.tools)

        tool_definitions = get_tools_list(
            include_home_tools=self.home_client is not None,
            ...,
            ha_native_tools=ha_native_tool_defs,
        )
```

Do the same for `get_tools_prompt()` — pass `ha_native_tools=ha_native_tool_defs`.

### 9d: Update `_stage_execute()` — pass `ha_native_client` to `query_with_tools()`

Find where `query_with_tools()` is called (around line 515-552). Add:
```python
            ha_native_client=self.ha_native_client,
```

---

## Step 10: Update Tool Loop

**File**: `src/llm_bawt/tools/loop.py`

### 10a: Add `ha_native_client` parameter to `ToolLoop.__init__()`

Find `__init__` (line 41). Add parameter `ha_native_client=None`. Pass it to `ToolExecutor`:

```python
        self.executor = ToolExecutor(
            ...
            home_client=home_client,
            ha_native_client=ha_native_client,      # <-- ADD
            ...
        )
```

### 10b: Add `ha_native_client` parameter to `query_with_tools()`

Find `query_with_tools()` (line 458). Add `ha_native_client=None` parameter. Pass to `ToolLoop`:

```python
def query_with_tools(
    ...,
    home_client=None,
    ha_native_client=None,       # <-- ADD
    ...
):
    loop = ToolLoop(
        ...,
        home_client=home_client,
        ha_native_client=ha_native_client,   # <-- ADD
        ...
    )
```

### 10c: Add TYPE_CHECKING import

At the top of loop.py, update the TYPE_CHECKING block to import `HomeAssistantNativeClient`:

```python
if TYPE_CHECKING:
    ...
    from ..integrations.ha_mcp.client import HomeAssistantMCPClient, HomeAssistantNativeClient
```

---

## Step 11: Update Botchat Context Sanitizer

**File**: `src/llm_bawt/service/routes/botchat.py`

The sanitizer currently strips a few HA boilerplate lines. With native MCP, the HA system context sent to botchat will still contain the device CSV, but since we're now using native HA tools, we should strip the entire device list and the "act as smart home manager" instructions.

Update `_sanitize_client_context()`:

```python
def _sanitize_client_context(text: str) -> str:
    """Strip HA boilerplate and device CSV from client context.

    With HA native MCP tools, the device list is unnecessary — the LLM
    uses GetLiveContext and tools like HassTurnOn with friendly names.
    The HA prompt instructions also conflict with our tool-calling system.
    """
    lines = text.splitlines()
    cleaned = []
    in_csv_block = False

    for line in lines:
        stripped = line.strip().lower()

        # Skip known conflicting HA instructions
        if stripped in _HA_CONFLICTING_LINES:
            continue

        # Skip the entire CSV device list block
        if "entity_id,name,state" in stripped or "```csv" in stripped:
            in_csv_block = True
            continue
        if in_csv_block:
            if stripped.startswith("```") or stripped == "":
                in_csv_block = False
            continue

        # Skip HA prompt boilerplate that conflicts with our tool system
        if any(phrase in stripped for phrase in (
            "i want you to act as smart home manager",
            "i will provide information of smart home",
            "available devices:",
            "the current state of devices",
            "use execute_services function",
        )):
            continue

        cleaned.append(line)

    result = "\n".join(cleaned).strip()
    # Remove excessive blank lines
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    return result
```

---

## Step 12: Update Status

**File**: `src/llm_bawt/core/status.py`

Find the status dataclass fields for HA (around line 116). Add:

```python
    ha_native_mcp_url: str | None = None
    ha_native_mcp_tools: int = 0
```

Find where these are populated from config (around line 583). Add:

```python
    ha_native_mcp_url=config.HA_NATIVE_MCP_URL or None,
```

Find the status display in `src/llm_bawt/cli/app.py` (around line 352). After the `ha_mcp_enabled` display, add:

```python
    if s_cfg.ha_native_mcp_url:
        console.print(f"  HA Native MCP: {s_cfg.ha_native_mcp_url} ({s_cfg.ha_native_mcp_tools} tools)")
```

---

## Step 13: Write Tests

**File**: `tests/test_ha_native_mcp.py`

Create this new test file:

```python
"""Tests for HA native MCP integration."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from llm_bawt.integrations.ha_mcp.client import (
    HomeAssistantNativeClient,
    HAToolDefinition,
)
from llm_bawt.tools.definitions import (
    Tool,
    ToolParameter,
    ha_tools_to_tool_definitions,
    get_tools_list,
)


@dataclass
class MockConfig:
    HA_NATIVE_MCP_URL: str = "http://hass.home:8123/api/mcp"
    HA_NATIVE_MCP_TOKEN: str = "test-token"
    HA_MCP_TIMEOUT: int = 10
    HA_MCP_TOOL_EXCLUDE: str = "GetDateTime,HassCancelAllTimers"


class TestHomeAssistantNativeClient:
    """Test the native HA MCP client."""

    def test_available_with_url_and_token(self):
        config = MockConfig()
        client = HomeAssistantNativeClient(config)
        assert client.available is True

    def test_not_available_without_url(self):
        config = MockConfig(HA_NATIVE_MCP_URL="")
        client = HomeAssistantNativeClient(config)
        assert client.available is False

    def test_not_available_without_token(self):
        config = MockConfig(HA_NATIVE_MCP_TOKEN="")
        client = HomeAssistantNativeClient(config)
        assert client.available is False

    def test_exclude_list_parsing(self):
        config = MockConfig(HA_MCP_TOOL_EXCLUDE="GetDateTime, HassBroadcast , ")
        client = HomeAssistantNativeClient(config)
        assert client._exclude_names == {"GetDateTime", "HassBroadcast"}

    def test_exclude_list_empty(self):
        config = MockConfig(HA_MCP_TOOL_EXCLUDE="")
        client = HomeAssistantNativeClient(config)
        assert client._exclude_names == set()

    def test_is_ha_tool_before_discovery(self):
        config = MockConfig()
        client = HomeAssistantNativeClient(config)
        assert client.is_ha_tool("HassTurnOn") is False
        assert client.initialized is False

    def test_is_ha_tool_after_manual_setup(self):
        config = MockConfig()
        client = HomeAssistantNativeClient(config)
        # Manually set tools as if discovered
        client._tools = [HAToolDefinition(name="HassTurnOn", description="Turn on")]
        client._tool_names = {"HassTurnOn"}
        client._initialized = True
        assert client.is_ha_tool("HassTurnOn") is True
        assert client.is_ha_tool("FakeTool") is False


class TestHAToolConversion:
    """Test converting HA tool definitions to Tool dataclass."""

    def test_basic_conversion(self):
        ha_tools = [
            HAToolDefinition(
                name="HassTurnOn",
                description="Turns on a device",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "area": {"type": "string"},
                    },
                },
            ),
        ]
        tools = ha_tools_to_tool_definitions(ha_tools)
        assert len(tools) == 1
        assert tools[0].name == "HassTurnOn"
        assert tools[0].description == "Turns on a device"
        assert len(tools[0].parameters) == 2
        assert tools[0].parameters[0].name == "name"
        assert tools[0].parameters[0].type == "string"

    def test_enum_in_description(self):
        ha_tools = [
            HAToolDefinition(
                name="HassSetVolume",
                description="Sets volume",
                input_schema={
                    "type": "object",
                    "properties": {
                        "volume_level": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Volume percentage",
                        },
                    },
                },
            ),
        ]
        tools = ha_tools_to_tool_definitions(ha_tools)
        param = tools[0].parameters[0]
        assert "min=0" in param.description
        assert "max=100" in param.description

    def test_array_type_conversion(self):
        ha_tools = [
            HAToolDefinition(
                name="HassTurnOn",
                description="Turn on",
                input_schema={
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            ),
        ]
        tools = ha_tools_to_tool_definitions(ha_tools)
        assert tools[0].parameters[0].type == "array[string]"

    def test_empty_schema(self):
        ha_tools = [
            HAToolDefinition(name="GetDateTime", description="Get time", input_schema={}),
        ]
        tools = ha_tools_to_tool_definitions(ha_tools)
        assert len(tools) == 1
        assert len(tools[0].parameters) == 0


class TestToolListIntegration:
    """Test that HA native tools integrate into the tool list correctly."""

    def test_ha_native_tools_replace_home_tool(self):
        ha_tools = [
            Tool(name="HassTurnOn", description="Turn on", parameters=[]),
            Tool(name="HassTurnOff", description="Turn off", parameters=[]),
        ]
        tools = get_tools_list(
            include_home_tools=True,  # would normally include HOME_TOOL
            ha_native_tools=ha_tools,  # but native takes priority
        )
        tool_names = [t.name for t in tools]
        assert "HassTurnOn" in tool_names
        assert "HassTurnOff" in tool_names
        assert "home" not in tool_names  # legacy should NOT be present

    def test_legacy_home_when_no_native(self):
        tools = get_tools_list(include_home_tools=True, ha_native_tools=None)
        tool_names = [t.name for t in tools]
        assert "home" in tool_names

    def test_no_home_tools_at_all(self):
        tools = get_tools_list(include_home_tools=False, ha_native_tools=None)
        tool_names = [t.name for t in tools]
        assert "home" not in tool_names
```

---

## Step 14: Docker / Production Config

Once all code changes are done, update the production `.env` files:

**On the server** (`~/.config/llm-bawt/.env` or Docker env):
```dotenv
# Disable legacy custom MCP server (no longer needed)
LLM_BAWT_HA_MCP_ENABLED=false

# Enable native HA MCP
LLM_BAWT_HA_NATIVE_MCP_URL=http://hass.home:8123/api/mcp
LLM_BAWT_HA_NATIVE_MCP_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4MWM2YzY0ZDBiZWU0NDI4ODVlZGU1MjNlYjQ1YzgzYSIsImlhdCI6MTc3MDkzMzk0MCwiZXhwIjoyMDg2MjkzOTQwfQ.p-k0fuM-oAF_OpSK86cF-ycWfAsA4ZB5PQBfMywlRuk
LLM_BAWT_HA_MCP_TOOL_EXCLUDE=GetDateTime,HassCancelAllTimers,HassBroadcast
```

> **NOTE**: `hass.home` resolves from the Docker container only if DNS is configured. You may need to use the IP `10.0.0.99` instead, depending on Docker network setup. Test with `curl http://10.0.0.99:8123/api/mcp` from inside the container.

---

## Step 15: End-to-End Testing

After deploying:

1. **Check status**: `llm --status` should show `HA Native MCP: http://...  (22 tools)`
2. **Test voice**: Say "turn on the kitchen lights" — should call `HassTurnOn(name="kitchen lights")` directly (one tool call)
3. **Test state query**: "what lights are on?" — should call `GetLiveContext` then answer
4. **Test brightness**: "set bedroom lights to 30%" — should call `HassLightSet(name="bedroom lights", brightness=30)`
5. **Test blinds**: "close the sunroom blinds" — should call `HassTurnOff(name="sunroom blinds")` (HA's intent system maps this correctly)
6. **Test script**: "close all blinds" — should call `close_all_blinds` (HA script exposed as tool)
7. **Test media**: "pause the living room TV" — should call `HassMediaPause(name="living room")`
8. **Check debug log**: `.logs/debug_turn.txt` — verify tool schemas are present and tool calls work

---

## Summary of How It All Fits Together

```
User says "close the sunroom blinds"
    ↓
LLM sees 22 HA tools in its tool schema (HassTurnOn, HassTurnOff, HassLightSet, etc.)
    ↓
LLM calls: HassTurnOff(name="sunroom blinds")
    ↓
ToolExecutor.execute() → sees "HassTurnOff" is in ha_native_client.tool_names
    ↓
_execute_ha_native_tool() → ha_native_client.call_tool("HassTurnOff", {"name": "sunroom blinds"})
    ↓
HomeAssistantNativeClient → Streamable HTTP POST to http://hass.home:8123/api/mcp
    ↓
HA resolves "sunroom blinds" → cover.sunroom_blind_back_right (and others)
    ↓
HA executes cover.close_cover service
    ↓
Response text: "Done" or JSON confirmation
    ↓
LLM sees result, responds to user: "Done, I've closed the sunroom blinds."
```

**Key wins**:
- 1 tool call instead of query-then-set (2 calls)
- No entity ID resolution needed — HA handles it
- Covers, media, climate, vacuum all work — no more missing domains
- New HA scripts/automations auto-appear as tools (no code change needed)
- Legacy `home` tool still works if native isn't configured (backward compat)
