import os
import json
import time
import logging
from typing import TYPE_CHECKING

from ..clients.base import LLMClient
from ..models.message import Message
from rich.rule import Rule
from .config import Config

if TYPE_CHECKING:
    from ..memory.postgresql import PostgreSQLShortTermManager as ShortTermMemoryManager

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Rough token estimate for a text string.

    Uses ~4 chars per token as a fast heuristic, within 10-15%
    of tiktoken for English text. No external dependencies.
    """
    return len(text) // 4


def estimate_messages_tokens(messages: list) -> int:
    """Estimate total tokens for a list of Message objects.

    Adds a small per-message overhead for role/framing.
    """
    return sum(len(m.content) // 4 + 4 for m in messages)


class HistoryManager:
    messages: list[Message] = []
    client: LLMClient
    config: Config
    _db_backend: "ShortTermMemoryManager | None"
    bot_id: str

    def __init__(
        self,
        client: LLMClient,
        config: Config,
        db_backend: "ShortTermMemoryManager | None" = None,
        bot_id: str = "nova",
    ):
        self.client = client
        self.config = config
        self.bot_id = bot_id
        self.messages = []
        self._db_backend = db_backend
        
        # Use per-bot history files for local mode
        if config.HISTORY_FILE:
            # Get the base path and insert bot_id
            base_path = config.HISTORY_FILE
            dir_name = os.path.dirname(base_path)
            file_name = os.path.basename(base_path)
            name, ext = os.path.splitext(file_name)
            # Create bot-specific filename: history.json -> history_spark.json
            self.history_file = os.path.join(dir_name, f"{name}_{bot_id}{ext}")
        else:
            self.history_file = None
            
        if db_backend:
            logger.debug(f"HistoryManager using PostgreSQL short-term backend for bot: {bot_id}")
        else:
            logger.debug(f"HistoryManager using text file backend for bot: {bot_id} ({self.history_file})")

    def load_history(self, since_minutes: int | None = None):
        self.messages = []
        
        # Use PostgreSQL backend if available
        if self._db_backend:
            try:
                self.messages = self._db_backend.get_messages(since_minutes=since_minutes)
                logger.debug(f"Loaded {len(self.messages)} messages from PostgreSQL short-term memory")
                return
            except Exception as e:
                logger.warning(f"Failed to load from PostgreSQL, falling back to file: {e}")
                # Fall through to file-based loading
        
        # File-based fallback
        if not self.history_file or not os.path.exists(self.history_file):
            print("No history file found. Skipping history load.")
            logger.debug("No history file found. Skipping history load.")
            return
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                message_dicts = json.load(f)
                self.messages = [Message.from_dict(msg) for msg in message_dicts]
                logger.debug(f"Loaded {len(self.messages)} messages from history file.")
        except UnicodeDecodeError as e:
            self.client.console.print(f"[bold red]Error loading history:[/bold red] Unable to decode file. Ensure it is saved in UTF-8 format. ({e})")
        except json.JSONDecodeError as e:
            self.client.console.print(f"[bold red]Error loading history:[/bold red] Invalid JSON format in {self.history_file}. ({e})")
        except Exception as e:
            self.client.console.print(f"[bold red]Error loading history:[/bold red] {e}")
        if since_minutes is not None and self.messages:
            # Note: since_minutes is actually in seconds for backward compatibility
            # (HISTORY_DURATION is in seconds, not minutes)
            logger.debug(f"Loading history from {since_minutes} seconds ago ({len(self.messages)} messages)")
            cutoff = time.time() - since_minutes
            self.messages = [msg for msg in self.messages if msg.timestamp >= cutoff]

    def save_history(self):
        """Persist message history to the history file."""
        # PostgreSQL backend saves on add_message, so nothing to do here
        if self._db_backend:
            return
            
        if not self.history_file:
            return
        try:
            message_dicts = [msg.to_dict() for msg in self.messages]
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(message_dicts, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.client.console.print(f"[bold red]Error saving history:[/bold red] {e}")

    def get_context_messages(self, max_tokens: int = 0):
        """Get messages to be used as context for the LLM.

        Includes (in order):
        - System messages (always)
        - Summaries of older sessions (role='summary') with time context
        - Recent messages (within HISTORY_DURATION)

        If max_tokens > 0, applies a token budget:
        1. System messages are always included
        2. Protected recent turns are always included (newest N pairs)
        3. Remaining budget fills from newest-first, dropping oldest messages

        Args:
            max_tokens: Maximum token budget for the returned messages.
                        0 = no limit (backward-compatible default).
        """
        import time
        from datetime import datetime
        
        cutoff = time.time() - self.config.HISTORY_DURATION
        system_messages = []
        summary_messages = []
        recent_messages = []
        
        for msg in self.messages:
            if msg.role == "system":
                system_messages.append(msg)
            elif msg.role == "summary":
                # Add time context to summaries
                time_ago = self._format_time_ago(msg.timestamp)
                enhanced_content = f"[Previous conversation {time_ago}]\n{msg.content}"
                summary_messages.append(Message(
                    role="summary",
                    content=enhanced_content,
                    timestamp=msg.timestamp
                ))
            elif msg.timestamp >= cutoff:
                # Recent messages
                recent_messages.append(msg)
        
        if not system_messages:
            system_messages.append(Message(role="system", content=self.config.SYSTEM_MESSAGE))

        # Without a token budget, return everything
        if max_tokens <= 0:
            return system_messages + summary_messages + recent_messages

        # ── Token-budget enforcement ──────────────────────────────
        protected_turns = getattr(self.config, 'MEMORY_PROTECTED_RECENT_TURNS', 3)
        # Protect the last N user+assistant pairs (= 2*N messages from the end)
        n_protected = min(protected_turns * 2, len(recent_messages))
        protected = recent_messages[-n_protected:] if n_protected > 0 else []
        droppable = recent_messages[:-n_protected] if n_protected > 0 else list(recent_messages)

        # Calculate baseline usage (system + protected)
        system_cost = estimate_messages_tokens(system_messages)
        protected_cost = estimate_messages_tokens(protected)
        used = system_cost + protected_cost
        budget = max_tokens

        # Fill from summaries (oldest context → newest)
        included_summaries: list[Message] = []
        max_summaries = getattr(self.config, 'SUMMARIZATION_MAX_IN_CONTEXT', 5)
        for s in summary_messages[-max_summaries:]:
            cost = estimate_messages_tokens([s])
            if used + cost <= budget:
                included_summaries.append(s)
                used += cost

        # Fill droppable messages (newest-first to keep recent context)
        included_droppable: list[Message] = []
        for msg in reversed(droppable):
            cost = estimate_messages_tokens([msg])
            if used + cost <= budget:
                included_droppable.insert(0, msg)  # Maintain chronological order
                used += cost
            else:
                # Once we can't fit one, stop (all remaining are older)
                break

        dropped = len(droppable) - len(included_droppable)
        summary_cost = estimate_messages_tokens(included_summaries)
        droppable_cost = estimate_messages_tokens(included_droppable)

        logger.debug(
            f"Token budget: {budget} total | "
            f"system={system_cost}, protected={protected_cost} ({len(protected)} msgs), "
            f"summaries={summary_cost} ({len(included_summaries)} of {len(summary_messages)}), "
            f"history={droppable_cost} ({len(included_droppable)} msgs, {dropped} dropped) | "
            f"used={used}"
        )

        return system_messages + included_summaries + included_droppable + protected
    
    def _format_time_ago(self, timestamp: float) -> str:
        """Format a timestamp as a human-readable relative time."""
        import time
        from datetime import datetime
        
        now = time.time()
        diff = now - timestamp
        
        if diff < 60:
            return "just now"
        elif diff < 3600:
            minutes = int(diff / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif diff < 7200:
            return "about 1 hour ago"
        elif diff < 86400:
            hours = int(diff / 3600)
            return f"{hours} hours ago"
        elif diff < 172800:
            return "yesterday"
        elif diff < 604800:
            days = int(diff / 86400)
            return f"{days} days ago"
        elif diff < 1209600:
            return "last week"
        elif diff < 2592000:
            weeks = int(diff / 604800)
            return f"{weeks} weeks ago"
        elif diff < 5184000:
            return "last month"
        else:
            months = int(diff / 2592000)
            return f"{months} months ago"
    
    def get_context_messages_excluding_last(self):
        """Get messages to be used as context, excluding the most recent one."""
        context_messages = self.get_context_messages()
        if len(context_messages) > 1:
            return context_messages[:-1]
        elif context_messages and context_messages[0].role == "system":
            return context_messages
        else:
            if any(msg.role == "system" for msg in context_messages):
                return [msg for msg in context_messages if msg.role == "system"]
            else:
                return [Message(role="system", content=self.config.SYSTEM_MESSAGE)]

    def add_message(self, role, content):
        """Append a message to history and save."""
        message = Message(role, content)
        self.messages.append(message)
        
        # Save to PostgreSQL if available, otherwise file
        if self._db_backend:
            try:
                self._db_backend.add_message(role, content, message.timestamp)
            except Exception as e:
                logger.warning(f"Failed to save to PostgreSQL: {e}")
                self.save_history()  # Fall back to file
        else:
            self.save_history()


    def print_history(self, pairs_limit=None):
        """Print the conversation history in a formatted way.

        Args:
            pairs_limit: Number of recent conversation pairs to show (-1 for all).
        """
        if not self.messages:
            self.client.console.print("[italic]No conversation history found.[/italic]")
            return

        non_system_messages = [msg for msg in self.messages if msg.role != "system"]

        if not non_system_messages:
            self.client.console.print("[italic]No conversation messages found.[/italic]")
            return

        if pairs_limit is not None and pairs_limit != -1:
            messages_to_show = min(pairs_limit * 2, len(non_system_messages))
            non_system_messages = non_system_messages[-messages_to_show:]

        self.client.console.print()
        self.client.console.print("[bold]Conversation History:[/bold]")
        self.client.console.print(Rule(style="#555555"))

        for msg in non_system_messages:
            if msg.role == "user":
                self.client._print_user_message(msg.content)
            elif msg.role == "assistant":
                panel_title, panel_border_style = self.client.get_styling()
                parts = msg.content.split("\n\n", 1)
                first_part = parts[0]
                second_part = parts[1] if len(parts) > 1 else None
                self.client._print_assistant_message(
                    first_part,
                    second_part=second_part,
                    panel_title=panel_title,
                    panel_border_style=panel_border_style
                )
            self.client.console.print(Rule(style="#333333"))
    
    def clear_history(self):
        """Clear the conversation history from memory and disk."""
        self.messages = []
        
        # Clear PostgreSQL if available
        if self._db_backend:
            try:
                self._db_backend.clear()
                self.client.console.print("[bold red]History cleared (PostgreSQL).[/bold red]")
                return
            except Exception as e:
                logger.warning(f"Failed to clear PostgreSQL history: {e}")
                # Fall through to file-based clearing
        
        if self.history_file and os.path.exists(self.history_file):
            try:
                os.remove(self.history_file)
                self.client.console.print("[bold red]History cleared.[/bold red]")
            except Exception as e:
                self.client.console.print(f"[bold red]Error clearing history file:[/bold red] {e}")
        else:
            self.client.console.print("[dim]No history file found to clear or history file path not set.[/dim]")
            
    def get_last_assistant_message(self) -> str | None:
        """Get the content of the last assistant message, or None."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def remove_last_message_if_partial(self, role: str):
        """Remove the last message if it matches the specified role (used for cleanup on error/interrupt)."""
        if self.messages and self.messages[-1].role == role:
            self.messages.pop()
            # Remove from PostgreSQL as well if available
            if self._db_backend:
                try:
                    self._db_backend.remove_last_message_if_partial(role)
                except Exception as e:
                    logger.warning(f"Failed to remove from PostgreSQL: {e}")
            else:
                self.save_history()