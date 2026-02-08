"""Parser for tool calls in LLM output.

Extracts tool calls from model responses that use the prompt-based
tool calling format with <tool_call> markers.
"""

import json
import re
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def _try_fix_json(json_str: str) -> str | None:
    """Try to fix common JSON errors from LLM output.
    
    Models sometimes output malformed JSON like:
    - Extra closing braces: {"name": "foo", "arguments": {}}}
    - Missing quotes around keys
    - Trailing commas
    
    Returns:
        Fixed JSON string, or None if unfixable.
    """
    original = json_str
    
    # Fix extra closing braces (common: }}} instead of }})
    # Count opening and closing braces
    open_count = json_str.count('{')
    close_count = json_str.count('}')
    if close_count > open_count:
        # Remove extra closing braces from the end
        excess = close_count - open_count
        while excess > 0 and json_str.rstrip().endswith('}'):
            json_str = json_str.rstrip()[:-1]
            excess -= 1
    
    # Fix missing closing braces
    if open_count > close_count:
        json_str = json_str + ('}' * (open_count - close_count))
    
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    if json_str != original:
        return json_str
    return None


@dataclass
class ToolCall:
    """A parsed tool call from model output."""
    name: str
    arguments: dict[str, Any]
    raw_text: str  # Original text that was parsed
    
    def __str__(self) -> str:
        return f"ToolCall({self.name}, {self.arguments})"


# Known tool names for detecting raw JSON tool calls
# Includes both new consolidated tools and legacy names for backward compatibility
KNOWN_TOOLS = {
    # New consolidated tools (7 total)
    "memory", "history", "profile", "bot_trait", "search", "model", "time",
    # Legacy tool names (for backward compatibility)
    "search_memories", "store_memory", "delete_memory",
    "search_history", "get_recent_history", "forget_history",
    "set_user_attribute", "get_user_profile", "delete_user_attribute",
    "set_my_trait",
    "web_search", "news_search",
    "list_models", "get_current_model", "switch_model",
    "get_current_time",
}

# Pattern to match tool calls: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL | re.IGNORECASE
)

# Plain tool call: tool_name { ... } (common when models see tool list but no format)
_KNOWN_TOOLS_PATTERN = "|".join(sorted(KNOWN_TOOLS, key=len, reverse=True))
PLAIN_TOOL_CALL_PATTERN = re.compile(
    rf'^\s*(?P<name>{_KNOWN_TOOLS_PATTERN})\s*(?:\n|\s)+(?P<args>\{{.*?\}})',
    re.DOTALL | re.IGNORECASE
)

# Alternative patterns for models that might vary the format slightly
ALT_PATTERNS = [
    # Without closing tag (model might stop after JSON)
    re.compile(r'<tool_call>\s*(\{[^<]*\})', re.DOTALL | re.IGNORECASE),
    # With different casing or spacing
    re.compile(r'<TOOL_CALL>\s*(\{.*?\})\s*</TOOL_CALL>', re.DOTALL),
    # Function call style
    re.compile(r'<function_call>\s*(\{.*?\})\s*</function_call>', re.DOTALL | re.IGNORECASE),
    # ChatML-style: <|im_start|>tool_call {"name": ...} (no closing tag usually)
    re.compile(r'<\|im_start\|>tool_call\s*(\{[^\n]*\})', re.DOTALL | re.IGNORECASE),
    # ChatML with newline before JSON
    re.compile(r'<\|im_start\|>tool_call\s*\n\s*(\{.*?\})', re.DOTALL | re.IGNORECASE),
    # Markdown code block with JSON tool call (common with smaller models)
    # Matches: ```json, ```python, or just ``` followed by tool JSON
    re.compile(r'```(?:json|python)?\s*\n?\s*(\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\})\s*\n?```', re.DOTALL | re.IGNORECASE),
    # Inline markdown code (single backticks): `{"name": "...", "arguments": {...}}`
    re.compile(r'`(\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\})`', re.DOTALL | re.IGNORECASE),
    # Raw JSON tool call without any wrapper (last resort - only if it looks like a tool call)
    re.compile(r'^(\{"name"\s*:\s*"(?:get_user_profile|search_memories|store_memory|delete_memory|search_history|forget_history|set_user_attribute|delete_user_attribute|web_search|news_search|list_models|get_current_model|switch_model|get_current_time)"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\})$', re.DOTALL | re.MULTILINE),
]


def _normalize_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Normalize common parameter aliases from loose model output."""
    if not arguments:
        return arguments
    if name == "get_recent_history":
        if "n_messages" not in arguments:
            if "n" in arguments:
                arguments["n_messages"] = arguments.pop("n")
            elif "limit" in arguments:
                arguments["n_messages"] = arguments.pop("limit")
            elif "count" in arguments:
                arguments["n_messages"] = arguments.pop("count")
    if name == "search_history":
        if "n_results" not in arguments:
            if "n" in arguments:
                arguments["n_results"] = arguments.pop("n")
            elif "limit" in arguments:
                arguments["n_results"] = arguments.pop("limit")
    return arguments


def parse_tool_calls(text: str) -> tuple[list[ToolCall], str]:
    """Parse tool calls from model output.
    
    Args:
        text: The model's response text.
        
    Returns:
        Tuple of (list of ToolCall objects, remaining text after tool calls removed)
    """
    tool_calls = []
    remaining_text = text
    
    # Try plain tool call pattern first (tool_name {json})
    plain_match = PLAIN_TOOL_CALL_PATTERN.search(text)
    if plain_match:
        name = plain_match.group("name").strip()
        json_str = plain_match.group("args")
        try:
            arguments = json.loads(json_str)
        except json.JSONDecodeError:
            fixed_json = _try_fix_json(json_str)
            if fixed_json:
                arguments = json.loads(fixed_json)
            else:
                arguments = {}
        arguments = _normalize_tool_arguments(name, arguments if isinstance(arguments, dict) else {})
        tool_calls.append(
            ToolCall(
                name=name,
                arguments=arguments,
                raw_text=plain_match.group(0),
            )
        )
        remaining_text = remaining_text.replace(plain_match.group(0), "", 1).strip()
        return tool_calls, remaining_text

    # Try main pattern next
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    
    # If no matches, try alternative patterns
    if not matches:
        for pattern in ALT_PATTERNS:
            matches = list(pattern.finditer(text))
            if matches:
                break
    
    for match in matches:
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
            
            # Support both formats:
            # {"name": "tool", "arguments": {...}}
            # {"name": "tool", "args": {...}}
            name = data.get("name")
            arguments = data.get("arguments") or data.get("args") or {}
            
            if name:
                arguments = _normalize_tool_arguments(name, arguments)
                tool_calls.append(ToolCall(
                    name=name,
                    arguments=arguments,
                    raw_text=match.group(0)
                ))
                # Remove the tool call from remaining text
                remaining_text = remaining_text.replace(match.group(0), "", 1)
            else:
                logger.warning(f"Tool call missing 'name' field: {json_str}")
                
        except json.JSONDecodeError as e:
            # Try to fix common JSON errors from models
            fixed_json = _try_fix_json(json_str)
            if fixed_json:
                try:
                    data = json.loads(fixed_json)
                    name = data.get("name")
                    arguments = data.get("arguments") or data.get("args") or {}
                    if name:
                        arguments = _normalize_tool_arguments(name, arguments)
                        tool_calls.append(ToolCall(
                            name=name,
                            arguments=arguments,
                            raw_text=match.group(0)
                        ))
                        remaining_text = remaining_text.replace(match.group(0), "", 1)
                        logger.debug(f"Fixed malformed JSON: {json_str} -> {fixed_json}")
                        continue
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Failed to parse tool call JSON: {e}\nText: {json_str}")
    
    # Clean up remaining text
    remaining_text = remaining_text.strip()
    
    return tool_calls, remaining_text


# Pattern to detect markdown code blocks that might contain tool calls
# Matches ```json or ```python followed by {"name": ...
# Patterns to detect tool calls in various formats
MARKDOWN_TOOL_PATTERN = re.compile(
    r'```(?:json|python)?\s*\n?\s*\{"name"\s*:',
    re.IGNORECASE
)

# Pattern to detect inline code tool calls: `{"name": ...}`
INLINE_CODE_TOOL_PATTERN = re.compile(
    r'`\{"name"\s*:',
    re.IGNORECASE
)



def has_tool_call(text: str) -> bool:
    """Quick check if text contains a tool call marker."""
    lower = text.lower()
    if (
        "<tool_call>" in lower 
        or "<function_call>" in lower
        or "<|im_start|>tool_call" in lower
    ):
        return True
    
    # Check for markdown code block tool calls (common with smaller models)
    if MARKDOWN_TOOL_PATTERN.search(text):
        return True
    
    # Check for inline code tool calls: `{"name": ...}`
    if INLINE_CODE_TOOL_PATTERN.search(text):
        return True
    
    # Check for raw JSON that looks like a tool call (when response is just the JSON)
    stripped = text.strip()
    if stripped.startswith('{"name"') and '"arguments"' in stripped:
        # Verify it's a known tool
        for tool in KNOWN_TOOLS:
            if f'"{tool}"' in stripped:
                return True

    # Check for plain tool call format (tool_name {json})
    if PLAIN_TOOL_CALL_PATTERN.search(text):
        return True
    
    return False


def format_tool_result(tool_name: str, result: Any, error: str | None = None) -> str:
    """Format a tool result for injection back into the conversation.
    
    Args:
        tool_name: Name of the tool that was called.
        result: The result from the tool execution.
        error: Optional error message if tool failed.
        
    Returns:
        Formatted string to inject as a message.
    """
    if error:
        return (
            f"<tool_result name=\"{tool_name}\" status=\"error\">\n"
            f"{error}\n"
            f"</tool_result>\n"
            f"[IMPORTANT: The tool failed. Report this error to the user accurately.]"
        )
    
    # Format result nicely
    if isinstance(result, (dict, list)):
        result_str = json.dumps(result, indent=2, default=str)
    else:
        result_str = str(result)
    
    return (
        f"<tool_result name=\"{tool_name}\" status=\"success\">\n"
        f"{result_str}\n"
        f"</tool_result>"
    )


def format_memories_for_result(memories: list[dict]) -> str:
    """Format memory search results for the model.
    
    Args:
        memories: List of memory dicts from search.
        
    Returns:
        Human-readable formatted string.
    """
    if not memories:
        return "No memories found matching your query."
    
    lines = [f"Found {len(memories)} relevant memories:"]
    for i, mem in enumerate(memories, 1):
        content = mem.get("content", "")
        relevance = mem.get("relevance", 0)
        memory_id = mem.get("id", "")[:8] if mem.get("id") else ""
        lines.append(f"{i}. [id:{memory_id}] (relevance: {relevance:.2f}) {content}")
    
    return "\n".join(lines)
