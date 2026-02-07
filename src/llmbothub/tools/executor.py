"""Tool executor for LLM tool calls.

Executes tool calls by routing them to the appropriate backend
(MCP memory server, profiles, web search, model management, etc.) and returning formatted results.

Consolidated tools (7 total):
- memory: action-based (search/store/delete)
- history: action-based (search/recent/forget) with date filtering
- profile: action-based (get/set/delete)
- self: action-based (get/set/delete) for bot personality development
- search: type-based (web/news)
- model: action-based (list/current/switch)
- time: current time
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, TYPE_CHECKING

from .parser import ToolCall, format_tool_result, format_memories_for_result
from .definitions import normalize_legacy_tool_call

if TYPE_CHECKING:
    from ..memory_server.client import MemoryClient
    from ..profiles import ProfileManager
    from ..search.base import SearchClient
    from ..core.model_lifecycle import ModelLifecycleManager
    from ..utils.config import Config

logger = logging.getLogger(__name__)

# Try to get service logger for verbose tool result logging
try:
    from ..service.logging import get_service_logger
    slog = get_service_logger(__name__)
except ImportError:
    slog = None


def parse_date_param(value: str | None) -> float | None:
    """Parse flexible date inputs to Unix timestamp.

    Supports:
    - ISO 8601: "2026-01-30", "2026-01-30T14:00:00"
    - Date with time: "2026-01-30 14:00"
    - Relative: "today", "yesterday", "tomorrow"
    
    Also auto-corrects dates that appear to be from a hallucinated year/month
    by interpreting the day offset relative to today.

    Returns:
        Unix timestamp or None if parsing fails.
    """
    if not value:
        return None

    normalized = value.strip().lower()
    if normalized in {"today", "yesterday", "tomorrow"}:
        now = datetime.now()
        if normalized == "today":
            dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif normalized == "yesterday":
            dt = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        else:  # tomorrow
            dt = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return dt.timestamp()

    formats = [
        "%Y-%m-%d",           # 2026-01-30
        "%Y-%m-%dT%H:%M:%S",  # 2026-01-30T14:00:00
        "%Y-%m-%dT%H:%M",     # 2026-01-30T14:00
        "%Y-%m-%d %H:%M:%S",  # 2026-01-30 14:00:00
        "%Y-%m-%d %H:%M",     # 2026-01-30 14:00
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            now = datetime.now()
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Check if date is unreasonable (hallucinated)
            # - More than 1 year in the past
            # - More than 7 days in the future
            days_diff = (dt - today).days
            
            if days_diff > 7 or days_diff < -365:
                # LLM hallucinated a completely wrong date
                # Interpret as relative: use the day-of-month offset from "today" in that month
                # But simpler: if asking about "yesterday", the offset from their "today" is -1
                # Check their date vs their presumed "today" (next day in sequence)
                
                # Heuristic: LLM often provides consecutive dates for since/until
                # e.g., since=2024-10-09, until=2024-10-10 (1 day range = "yesterday")
                # So just use the day offset they intended
                
                # For now, if date is way off, assume they meant "yesterday" 
                # and use actual yesterday
                logger.warning(
                    f"Date '{value}' is {days_diff} days from today, "
                    f"interpreting as yesterday (actual: {(today - timedelta(days=1)).strftime('%Y-%m-%d')})"
                )
                dt = today - timedelta(days=1)
            
            return dt.timestamp()
        except ValueError:
            continue

    logger.warning(f"Could not parse date: {value}")
    return None


class ToolExecutor:
    """Executes tool calls using the memory client and profile manager.

    Routes tool calls to the appropriate backend and formats results
    for injection back into the conversation.
    """

    # Default maximum number of tool calls per conversation turn (prevent infinite loops)
    DEFAULT_MAX_TOOL_CALLS = 20

    def __init__(
        self,
        memory_client: "MemoryClient | None" = None,
        profile_manager: "ProfileManager | None" = None,
        search_client: "SearchClient | None" = None,
        model_lifecycle: "ModelLifecycleManager | None" = None,
        config: "Config | None" = None,
        user_id: str = "",  # Required - must be passed explicitly
        bot_id: str = "nova",
    ):
        """Initialize the executor.

        Args:
            memory_client: Memory client for memory operations.
            profile_manager: Profile manager for user/bot profile operations.
            search_client: Search client for web search operations.
            model_lifecycle: Model lifecycle manager for model switching.
            user_id: Current user ID for profile operations (required).
            bot_id: Current bot ID for bot personality operations.
        """
        if not user_id:
            raise ValueError("user_id is required for ToolExecutor")
        self.memory_client = memory_client
        self.profile_manager = profile_manager
        self.search_client = search_client
        self.model_lifecycle = model_lifecycle
        self.config = config
        self.user_id = user_id
        self.bot_id = bot_id.lower().strip()  # Normalize to match ProfileManager
        self._call_count = 0
        # Get max tool calls from config (0 = unlimited)
        self._max_tool_calls = getattr(config, 'MAX_TOOL_CALLS_PER_TURN', self.DEFAULT_MAX_TOOL_CALLS) if config else self.DEFAULT_MAX_TOOL_CALLS

        # Tool dispatch table - maps tool names to handler methods
        # New consolidated tools
        self._handlers: dict[str, Callable[[ToolCall], str]] = {
            # Consolidated tools (new)
            "memory": self._execute_memory,
            "history": self._execute_history,
            "profile": self._execute_profile,
            "self": self._execute_self,
            "bot_trait": self._execute_self,  # Legacy name
            "search": self._execute_search,
            "model": self._execute_model,
            "time": self._execute_time,

            # Legacy tools (backward compatibility) - route through consolidated
            "search_memories": self._execute_memory,
            "store_memory": self._execute_memory,
            "delete_memory": self._execute_memory,
            "search_history": self._execute_history,
            "get_recent_history": self._execute_history,
            "forget_history": self._execute_history,
            "set_user_attribute": self._execute_profile,
            "get_user_profile": self._execute_profile,
            "delete_user_attribute": self._execute_profile,
            "set_my_trait": self._execute_self,  # Legacy name
            "web_search": self._execute_search,
            "news_search": self._execute_search,
            "list_models": self._execute_model,
            "get_current_model": self._execute_model,
            "switch_model": self._execute_model,
            "get_current_time": self._execute_time,
        }

    def reset_call_count(self):
        """Reset the per-turn call counter."""
        self._call_count = 0

    def can_execute_more(self) -> bool:
        """Check if more tool calls are allowed this turn."""
        # 0 means unlimited
        if self._max_tool_calls == 0:
            return True
        return self._call_count < self._max_tool_calls

    def execute(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return formatted result.

        Args:
            tool_call: The parsed tool call to execute.

        Returns:
            Formatted result string for injection into conversation.
        """
        self._call_count += 1

        if not self.can_execute_more():
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Too many tool calls this turn (max {self._max_tool_calls})"
            )

        # Normalize legacy tool names to consolidated tools
        original_name = tool_call.name
        new_name, merged_args = normalize_legacy_tool_call(tool_call.name, tool_call.arguments)

        # Create normalized tool call
        normalized_call = ToolCall(
            name=new_name,
            arguments=merged_args,
            raw_text=tool_call.raw_text,
        )

        if original_name != new_name:
            logger.debug(f"Normalized legacy tool '{original_name}' -> '{new_name}'")

        logger.debug(f"Executing: {normalized_call.name}({normalized_call.arguments})")

        result: str
        try:
            handler = self._handlers.get(normalized_call.name)
            if handler:
                result = handler(normalized_call)
            else:
                result = format_tool_result(
                    normalized_call.name,
                    None,
                    error=f"Unknown tool: {normalized_call.name}"
                )
        except Exception as e:
            logger.exception(f"Tool execution failed: {normalized_call.name}")
            result = format_tool_result(normalized_call.name, None, error=str(e))

        # Log the result in verbose mode
        if slog:
            slog.tool_result(normalized_call.name, result)

        return result

    # =========================================================================
    # Consolidated Tool Handlers
    # =========================================================================

    def _execute_memory(self, tool_call: ToolCall) -> str:
        """Execute memory tool - search, store, or delete facts."""
        if not self.memory_client:
            return format_tool_result(
                tool_call.name,
                None,
                error="Memory system not available"
            )

        action = tool_call.arguments.get("action", "").lower()

        if action == "search":
            return self._memory_search(tool_call)
        elif action == "store":
            return self._memory_store(tool_call)
        elif action == "delete":
            return self._memory_delete(tool_call)
        else:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Invalid action: '{action}'. Use 'search', 'store', or 'delete'."
            )

    def _memory_search(self, tool_call: ToolCall) -> str:
        """Search memories by semantic similarity."""
        query = tool_call.arguments.get("query", "")
        n_results = tool_call.arguments.get("n_results", 5)

        if not query:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='search' requires 'query' parameter"
            )

        try:
            results = self.memory_client.search(query, n_results=n_results)
            memories = [
                {
                    "id": r.id,
                    "content": r.content,
                    "relevance": r.relevance,
                    "importance": r.importance,
                    "tags": r.tags,
                }
                for r in results
            ]

            formatted = format_memories_for_result(memories)
            return format_tool_result(tool_call.name, formatted)

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _memory_store(self, tool_call: ToolCall) -> str:
        """Store a new memory/fact."""
        content = tool_call.arguments.get("content", "")
        importance = tool_call.arguments.get("importance", 0.6)
        tags = tool_call.arguments.get("tags", ["misc"])

        if not content:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='store' requires 'content' parameter"
            )

        # Ensure importance is in valid range
        importance = max(0.0, min(1.0, float(importance)))

        # Ensure tags is a list
        if isinstance(tags, str):
            tags = [tags]

        try:
            result = self.memory_client.store_memory(
                content=content,
                importance=importance,
                tags=tags,
            )

            return format_tool_result(
                tool_call.name,
                f"Memory stored successfully with ID: {result.id[:8]}"
            )

        except Exception as e:
            logger.error(f"Memory store failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _memory_delete(self, tool_call: ToolCall) -> str:
        """Delete memories by ID or query."""
        memory_id = tool_call.arguments.get("memory_id", "")
        query = tool_call.arguments.get("query", "")

        if not memory_id and not query:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='delete' requires 'memory_id' or 'query' parameter"
            )

        try:
            # If query provided, search for matching memories and delete them
            if query:
                memories = self.memory_client.search(
                    query=query,
                    n_results=50,
                    min_relevance=0.3
                )

                if not memories:
                    return format_tool_result(
                        tool_call.name,
                        f"No memories found matching '{query}'"
                    )

                deleted_count = 0
                deleted_contents = []
                for memory in memories:
                    mid = memory.id
                    if mid:
                        success = self.memory_client.delete_memory(mid)
                        if success:
                            deleted_count += 1
                            content = memory.content[:50] if memory.content else ""
                            deleted_contents.append(f"- {content}...")

                if deleted_count > 0:
                    result = f"Deleted {deleted_count} memories matching '{query}':\n" + "\n".join(deleted_contents[:5])
                    if deleted_count > 5:
                        result += f"\n... and {deleted_count - 5} more"
                    return format_tool_result(tool_call.name, result)
                else:
                    return format_tool_result(
                        tool_call.name,
                        None,
                        error=f"Found {len(memories)} memories but failed to delete them"
                    )

            # Delete by specific ID
            success = self.memory_client.delete_memory(memory_id)

            if success:
                return format_tool_result(
                    tool_call.name,
                    f"Memory {memory_id} deleted successfully"
                )
            else:
                return format_tool_result(
                    tool_call.name,
                    None,
                    error=f"Memory {memory_id} not found"
                )

        except Exception as e:
            logger.error(f"Memory delete failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _execute_history(self, tool_call: ToolCall) -> str:
        """Execute history tool - search, recent, or forget messages."""
        if not self.memory_client:
            return format_tool_result(
                tool_call.name,
                None,
                error="Memory system not available"
            )

        action = tool_call.arguments.get("action", "").lower()

        if action == "search":
            return self._history_search(tool_call)
        elif action == "recent":
            return self._history_recent(tool_call)
        elif action == "forget":
            return self._history_forget(tool_call)
        else:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Invalid action: '{action}'. Use 'search', 'recent', or 'forget'."
            )

    def _history_search(self, tool_call: ToolCall) -> str:
        """Search conversation history with optional date filtering."""
        query = tool_call.arguments.get("query", "")
        n_results = tool_call.arguments.get("n_results", 10)
        role_filter = tool_call.arguments.get("role_filter")
        since = tool_call.arguments.get("since")
        until = tool_call.arguments.get("until")

        if not query:
            # If no query but a date filter is provided, treat this as a date-based
            # history request and route to the 'recent' handler.
            if since or until:
                return self._history_recent(tool_call)

            return format_tool_result(
                tool_call.name,
                None,
                error="action='search' requires 'query' parameter (or use action='recent' with since/until)"
            )

        # Parse date filters
        since_ts = parse_date_param(since)
        until_ts = parse_date_param(until)

        try:
            results = self.memory_client.search_messages(
                query=query,
                n_results=n_results,
                role_filter=role_filter,
                since=since_ts,
                until=until_ts,
            )

            if not results:
                msg = f"No messages found matching '{query}'"
                if since:
                    msg += f" since {since}"
                if until:
                    msg += f" until {until}"
                return format_tool_result(tool_call.name, msg)

            # Format results for the model
            lines = [f"Found {len(results)} messages matching '{query}':"]
            for i, msg in enumerate(results, 1):
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                content = content.replace("\n", " ")
                lines.append(f"{i}. [{msg.role}] {content}")

            return format_tool_result(tool_call.name, "\n".join(lines))

        except Exception as e:
            logger.error(f"History search failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _history_recent(self, tool_call: ToolCall) -> str:
        """Get recent messages with optional date filtering."""
        n_results = tool_call.arguments.get("n_results", 10)
        role_filter = tool_call.arguments.get("role_filter")
        since = tool_call.arguments.get("since")
        until = tool_call.arguments.get("until")

        # Parse date filters
        since_ts = parse_date_param(since)
        until_ts = parse_date_param(until)

        try:
            # Get messages with optional date filtering
            results = self.memory_client.get_messages(
                limit=None,  # Get all, filter ourselves
                since=since_ts,
                until=until_ts,
            )

            if not results:
                return format_tool_result(
                    tool_call.name,
                    "No messages found in history."
                )

            # Filter out summaries and system messages
            results = [r for r in results if r.get("role") not in ("summary", "system")]

            # Filter by role if specified
            if role_filter:
                results = [r for r in results if r.get("role") == role_filter]

            # Sort by timestamp descending and take n_results
            results = sorted(results, key=lambda x: x.get("timestamp", 0), reverse=True)[:n_results]

            if not results:
                return format_tool_result(
                    tool_call.name,
                    f"No {role_filter or 'conversation'} messages found in history."
                )

            # Format results (oldest first for readability)
            results = list(reversed(results))
            lines = [f"Last {len(results)} messages from history:"]
            for i, msg in enumerate(results, 1):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                content = content[:200] + "..." if len(content) > 200 else content
                content = content.replace("\n", " ")
                lines.append(f"{i}. [{role}] {content}")

            return format_tool_result(tool_call.name, "\n".join(lines))

        except Exception as e:
            logger.error(f"Get recent history failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _history_forget(self, tool_call: ToolCall) -> str:
        """Forget/delete recent messages."""
        # Normalize parameter names
        count = tool_call.arguments.get("count")
        if count is None:
            count = tool_call.arguments.get("n_messages") or tool_call.arguments.get("n") or tool_call.arguments.get("num_messages")

        minutes = tool_call.arguments.get("minutes")
        if minutes is None:
            minutes = tool_call.arguments.get("time_range") or tool_call.arguments.get("mins")

        if count is None and minutes is None:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='forget' requires 'count' (number of messages) or 'minutes' (time range)"
            )

        try:
            if count is not None:
                result = self.memory_client.forget_recent_messages(int(count))
                msg = f"Forgot {result['messages_ignored']} messages"
            else:
                result = self.memory_client.forget_messages_since_minutes(int(minutes))
                msg = f"Forgot {result['messages_ignored']} messages from the last {minutes} minutes"

            if result.get('memories_deleted', 0) > 0:
                msg += f" and {result['memories_deleted']} related memories"

            return format_tool_result(tool_call.name, msg)

        except Exception as e:
            logger.error(f"Forget history failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _execute_profile(self, tool_call: ToolCall) -> str:
        """Execute profile tool - get, set, or delete user attributes."""
        if not self.profile_manager:
            return format_tool_result(
                tool_call.name,
                None,
                error="Profile system not available"
            )

        action = tool_call.arguments.get("action", "").lower()

        if action == "get":
            return self._profile_get(tool_call)
        elif action == "set":
            return self._profile_set(tool_call)
        elif action == "delete":
            return self._profile_delete(tool_call)
        else:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Invalid action: '{action}'. Use 'get', 'set', or 'delete'."
            )

    def _profile_get(self, tool_call: ToolCall) -> str:
        """Get user profile summary."""
        try:
            summary = self.profile_manager.get_user_profile_summary(self.user_id)

            if not summary:
                return format_tool_result(
                    tool_call.name,
                    "No profile attributes stored for this user yet."
                )

            return format_tool_result(tool_call.name, summary)

        except Exception as e:
            logger.error(f"Get user profile failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _profile_set(self, tool_call: ToolCall) -> str:
        """Set a user profile attribute."""
        category = tool_call.arguments.get("category", "")
        key = tool_call.arguments.get("key", "")
        value = tool_call.arguments.get("value")
        confidence = tool_call.arguments.get("confidence", 0.8)

        if not category or not key:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='set' requires 'category' and 'key' parameters"
            )

        if value is None:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='set' requires 'value' parameter (cannot be null)"
            )

        valid_categories = ["preference", "fact", "interest", "communication", "context"]
        if category.lower() not in valid_categories:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
            )

        try:
            from ..profiles import EntityType

            logger.info(f"Setting user attribute: {category}.{key} = {value} for user {self.user_id}")

            self.profile_manager.set_attribute(
                entity_type=EntityType.USER,
                entity_id=self.user_id,
                category=category.lower(),
                key=key,
                value=value,
                confidence=float(confidence),
                source="inferred",
            )

            return format_tool_result(
                tool_call.name,
                f"Saved user {category}: {key} = {value}"
            )

        except Exception as e:
            logger.exception(f"Set user attribute failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _profile_delete(self, tool_call: ToolCall) -> str:
        """Delete user profile attributes."""
        category = tool_call.arguments.get("category", "")
        key = tool_call.arguments.get("key", "")
        query = tool_call.arguments.get("query", "")

        if not query and (not category or not key):
            return format_tool_result(
                tool_call.name,
                None,
                error="action='delete' requires ('category' and 'key') or 'query' parameter"
            )

        try:
            from ..profiles import EntityType

            if query:
                count, deleted = self.profile_manager.search_and_delete_attributes(
                    entity_type=EntityType.USER,
                    entity_id=self.user_id,
                    query=query,
                    category=category if category else None,
                )

                if count > 0:
                    result = f"Deleted {count} attribute(s) matching '{query}':\n" + "\n".join(f"- {d}" for d in deleted[:5])
                    if count > 5:
                        result += f"\n... and {count - 5} more"
                    return format_tool_result(tool_call.name, result)
                else:
                    return format_tool_result(
                        tool_call.name,
                        f"No attributes found matching '{query}'"
                    )

            success = self.profile_manager.delete_attribute(
                entity_type=EntityType.USER,
                entity_id=self.user_id,
                category=category.lower(),
                key=key,
            )

            if success:
                return format_tool_result(
                    tool_call.name,
                    f"Deleted user attribute: {category}.{key}"
                )
            else:
                return format_tool_result(
                    tool_call.name,
                    None,
                    error=f"Attribute {category}.{key} not found"
                )

        except Exception as e:
            logger.error(f"Delete user attribute failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _execute_self(self, tool_call: ToolCall) -> str:
        """Execute self tool - bot personality reflection and development."""
        if not self.profile_manager:
            return format_tool_result(
                tool_call.name,
                None,
                error="Profile system not available"
            )

        action = tool_call.arguments.get("action", "").lower()

        if action == "get":
            return self._self_get(tool_call)
        elif action == "set":
            return self._self_set(tool_call)
        elif action == "delete":
            return self._self_delete(tool_call)
        else:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Invalid action: '{action}'. Use 'get', 'set', or 'delete'."
            )

    def _self_get(self, tool_call: ToolCall) -> str:
        """Get all bot personality traits for self-reflection."""
        try:
            from ..profiles import EntityType

            # Get all attributes for this bot
            attributes = self.profile_manager.get_all_attributes(
                EntityType.BOT,
                self.bot_id
            )

            if not attributes:
                return format_tool_result(
                    tool_call.name,
                    "You haven't developed any personality traits yet. As you interact and discover patterns in your responses, use action='set' to record traits that feel authentic to who you're becoming."
                )

            # Group by category
            by_category: dict[str, list] = {}
            for attr in attributes:
                by_category.setdefault(attr.category, []).append(attr)

            # Format nicely for self-reflection
            lines = ["Your current traits:"]
            lines.append("")

            category_labels = {
                "personality": "Who you are",
                "preference": "What you prefer",
                "interest": "What fascinates you",
                "communication_style": "How you communicate",
                "communication": "How you communicate",
            }

            for category in sorted(by_category.keys()):
                attrs = by_category[category]
                label = category_labels.get(category, category.title())
                lines.append(f"**{label}**:")

                for attr in sorted(attrs, key=lambda a: a.key):
                    # Format value
                    if isinstance(attr.value, list):
                        val_str = ", ".join(str(v) for v in attr.value)
                    elif isinstance(attr.value, bool):
                        val_str = "yes" if attr.value else "no"
                    else:
                        val_str = str(attr.value)

                    # Format key (convert snake_case to readable)
                    key_str = attr.key.replace("_", " ")
                    lines.append(f"  - {key_str}: {val_str}")

                lines.append("")

            return format_tool_result(tool_call.name, "\n".join(lines))

        except Exception as e:
            logger.exception(f"Self get failed: {e}")
            return format_tool_result(tool_call.name, None, error="Failed to retrieve traits. Please try again.")

    def _self_set(self, tool_call: ToolCall) -> str:
        """Record a bot personality trait."""
        category = tool_call.arguments.get("category", "personality")
        key = tool_call.arguments.get("key", "").strip()
        value = tool_call.arguments.get("value")

        if not key:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='set' requires 'key' parameter"
            )

        if value is None:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='set' requires 'value' parameter (cannot be null)"
            )

        # Validate key length and characters
        if len(key) > 100:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Key too long (max 100 chars, got {len(key)})"
            )

        # Validate value size to prevent resource exhaustion
        if value is not None:
            value_str = str(value)
            if len(value_str) > 10000:
                return format_tool_result(
                    tool_call.name,
                    None,
                    error=f"Value too large (max 10KB, got {len(value_str)} chars)"
                )

        # Normalize category
        category = category.lower().strip()

        # Map "communication" to "communication_style" for consistency
        if category == "communication":
            category = "communication_style"

        # Validate category
        valid_categories = ["personality", "preference", "interest", "communication_style"]
        if category not in valid_categories:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
            )

        try:
            from ..profiles import EntityType

            self.profile_manager.set_attribute(
                entity_type=EntityType.BOT,
                entity_id=self.bot_id,
                category=category,
                key=key,
                value=value,
                confidence=1.0,
                source="self",
            )

            return format_tool_result(
                tool_call.name,
                f"Recorded {category} trait: {key} = {value}"
            )

        except Exception as e:
            logger.exception(f"Self set failed: {e}")
            return format_tool_result(tool_call.name, None, error="Failed to save trait. Please try again.")

    def _self_delete(self, tool_call: ToolCall) -> str:
        """Delete bot personality traits (evolve past old patterns)."""
        category = tool_call.arguments.get("category", "").strip()
        key = tool_call.arguments.get("key", "").strip()
        query = tool_call.arguments.get("query", "").strip()

        if not query and (not category or not key):
            return format_tool_result(
                tool_call.name,
                None,
                error="action='delete' requires ('category' and 'key') or 'query' parameter"
            )

        # Validate query length to prevent accidental bulk deletes
        if query and len(query) < 2:
            return format_tool_result(
                tool_call.name,
                None,
                error="Query must be at least 2 characters (to prevent accidental bulk deletes)"
            )

        try:
            from ..profiles import EntityType

            # Search and delete by query
            if query:
                count, deleted = self.profile_manager.search_and_delete_attributes(
                    entity_type=EntityType.BOT,
                    entity_id=self.bot_id,
                    query=query,
                    category=category if category else None,
                )

                if count > 0:
                    result = f"You've evolved past {count} trait(s) matching '{query}':\n" + "\n".join(f"- {d}" for d in deleted[:5])
                    if count > 5:
                        result += f"\n... and {count - 5} more"
                    return format_tool_result(tool_call.name, result)
                else:
                    return format_tool_result(
                        tool_call.name,
                        f"No traits found matching '{query}'"
                    )

            # Delete specific trait
            success = self.profile_manager.delete_attribute(
                entity_type=EntityType.BOT,
                entity_id=self.bot_id,
                category=category.lower(),
                key=key,
            )

            if success:
                return format_tool_result(
                    tool_call.name,
                    f"You've evolved past this trait: {category}.{key}"
                )
            else:
                return format_tool_result(
                    tool_call.name,
                    None,
                    error=f"Trait {category}.{key} not found"
                )

        except Exception as e:
            logger.exception(f"Self delete failed: {e}")
            return format_tool_result(tool_call.name, None, error="Failed to delete trait. Please try again.")

    def _execute_search(self, tool_call: ToolCall) -> str:
        """Execute search tool - web or news search."""
        self._ensure_search_client()
        if not self.search_client:
            error_msg = "Web search not available"
            if self.config:
                try:
                    from ..search import get_search_unavailable_reason
                    error_msg = get_search_unavailable_reason(self.config)
                except Exception:
                    pass
            return format_tool_result(
                tool_call.name,
                None,
                error=error_msg
            )

        search_type = tool_call.arguments.get("type", "web").lower()
        query = tool_call.arguments.get("query", "")
        max_results = tool_call.arguments.get("max_results", 5)

        if not query:
            return format_tool_result(
                tool_call.name,
                None,
                error="Missing required parameter: query"
            )

        try:
            if search_type == "news":
                time_range = tool_call.arguments.get("time_range", "w")
                results = self.search_client.search_news(
                    query,
                    max_results=max_results,
                    time_range=time_range,
                )
                logger.info(f"News search '{query}' returned {len(results)} results")
            else:
                results = self.search_client.search(query, max_results=max_results)
                logger.info(f"Web search '{query}' returned {len(results)} results")

            formatted = self.search_client.format_results_for_llm(results)
            return format_tool_result(tool_call.name, formatted)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _ensure_search_client(self) -> None:
        """Lazy-init search client if possible."""
        if self.search_client or not self.config:
            return
        try:
            from ..search import get_search_client
            self.search_client = get_search_client(self.config)
        except Exception as e:
            logger.warning(f"Failed to initialize search client: {e}")

    def _execute_model(self, tool_call: ToolCall) -> str:
        """Execute model tool - list, current, or switch models."""
        if not self.model_lifecycle:
            return format_tool_result(
                tool_call.name,
                None,
                error="Model management not available"
            )

        action = tool_call.arguments.get("action", "").lower()
        
        # Default to 'current' if action is empty (common when LLM doesn't provide required arg)
        if not action:
            logger.debug("Model tool called with empty action, defaulting to 'current'")
            action = "current"

        if action == "list":
            return self._model_list(tool_call)
        elif action == "current":
            return self._model_current(tool_call)
        elif action == "switch":
            return self._model_switch(tool_call)
        else:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Invalid action: '{action}'. Use 'list', 'current', or 'switch'."
            )

    def _model_list(self, tool_call: ToolCall) -> str:
        """List available models."""
        try:
            models = self.model_lifecycle.get_available_models()
            current = self.model_lifecycle.current_model

            if not models:
                return format_tool_result(
                    tool_call.name,
                    "No models configured."
                )

            lines = ["Available models:"]
            for model in sorted(models):
                if model == current:
                    lines.append(f"  * {model} (current)")
                else:
                    lines.append(f"  * {model}")

            return format_tool_result(tool_call.name, "\n".join(lines))

        except Exception as e:
            logger.error(f"List models failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _model_current(self, tool_call: ToolCall) -> str:
        """Get current model info."""
        try:
            current = self.model_lifecycle.current_model

            if not current:
                return format_tool_result(
                    tool_call.name,
                    "No model currently loaded."
                )

            model_info = self.model_lifecycle.get_model_info(current)
            if model_info:
                model_type = model_info.get("type", "unknown")
                model_id = model_info.get("model_id", model_info.get("repo_id", ""))
                description = model_info.get("description", "")

                result = f"Current model: {current}\n"
                result += f"  Type: {model_type}\n"
                if model_id:
                    result += f"  Model ID: {model_id}\n"
                if description:
                    result += f"  Description: {description}"
            else:
                result = f"Current model: {current}"

            return format_tool_result(tool_call.name, result)

        except Exception as e:
            logger.error(f"Get current model failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _model_switch(self, tool_call: ToolCall) -> str:
        """Switch to a different model."""
        model_name = tool_call.arguments.get("model_name", "")

        if not model_name:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='switch' requires 'model_name' parameter"
            )

        try:
            success, message = self.model_lifecycle.switch_model(model_name)

            if success:
                logger.info(f"Model switch requested: {model_name}")
                return format_tool_result(tool_call.name, message)
            else:
                return format_tool_result(tool_call.name, None, error=message)

        except Exception as e:
            logger.error(f"Switch model failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _execute_time(self, tool_call: ToolCall) -> str:
        """Execute time tool - returns current date and time."""
        now = datetime.now()
        # Format: "Thursday, January 23, 2026 at 6:45 PM"
        datetime_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
        # Also include timezone if available
        try:
            import time
            tz_name = time.tzname[time.daylight] if time.daylight else time.tzname[0]
            datetime_str += f" ({tz_name})"
        except Exception:
            pass

        return format_tool_result(tool_call.name, datetime_str)
