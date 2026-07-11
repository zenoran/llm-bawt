import os
import json
import time
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Any

from ..clients.base import LLMClient
from ..models.message import Message
from rich.rule import Rule
from .config import Config

if TYPE_CHECKING:
    from ..memory.postgresql import PostgreSQLShortTermManager as ShortTermMemoryManager

logger = logging.getLogger(__name__)

# A role='system' row that carries tool-execution evidence rather than the
# persona/system prompt. Detected by content prefix so the two are separable in
# the context payload (chat keeps these; the plain system row is discarded by
# both delivery modes). Keep in sync with core/base.py's inline check.
_TOOL_RESULT_PREFIXES = ("[Tool Results]", "[Tools used:")


def _is_tool_result_system(msg) -> bool:
    if getattr(msg, "role", "") != "system":
        return False
    content = getattr(msg, "content", "") or ""
    return content.startswith(_TOOL_RESULT_PREFIXES)


@dataclass
class ContextPayload:
    """The ONE context-assembly result (TASK-493).

    Single source of truth for *what* prior context a conversation gets — for
    BOTH chat turns and agent session-seeds. The two orthogonal user controls
    resolve here identically for every bot_type:

    - ``continuity`` (bool): carry prior context at all.
    - ``history_scope`` ("inline+summaries" | "inline"): whether summaries are
      part of it.

    Delivery (chat inline vs agent seed) is NOT decided here — it is expressed by
    which composed view the caller reads (``inline_history`` vs ``seed_messages``).
    The buckets are kept separate so each delivery composes its own shape without
    re-deriving the summary gate. ``summary_messages`` is already gated: it is
    empty when summaries are excluded.
    """

    system_messages: list = field(default_factory=list)       # plain persona/system rows
    summary_messages: list = field(default_factory=list)       # already gated (empty if excluded)
    tool_result_messages: list = field(default_factory=list)   # role=system tool-evidence rows
    regular_messages: list = field(default_factory=list)       # user / assistant

    @property
    def inline_history(self) -> list:
        """Rows a CHAT turn appends after its own system-prompt builder.

        Order preserves what the pre-refactor loop produced: tool-result system
        rows (hoisted to the front by ``get_context_messages``), then summaries,
        then the regular conversation. The plain persona system row is omitted —
        the chat path builds its own.
        """
        return self.tool_result_messages + self.summary_messages + self.regular_messages

    @property
    def seed_messages(self) -> list:
        """Rows an AGENT cold-start seed carries: summaries + conversation, no
        system rows at all (the SDK injects its own byte-stable system prompt —
        TASK-288). Matches the pre-refactor ``role != 'system'`` strip.
        """
        return self.summary_messages + self.regular_messages


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
        settings_getter: Callable[[str, Any], Any] | None = None,
    ):
        self.client = client
        self.config = config
        self.bot_id = bot_id
        self.messages = []
        self._db_backend = db_backend
        self._settings_getter = settings_getter
        
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

    def _setting(self, key: str, fallback: Any) -> Any:
        """Resolve setting through optional runtime resolver."""
        if self._settings_getter is None:
            return fallback
        try:
            return self._settings_getter(key, fallback)
        except Exception:
            return fallback

    def load_history(self, since_minutes: int | None = None):
        self.messages = []

        # Conversation offset marker (moved forward by the `/new` command).
        # Raw messages older than the marker are dropped from the live context;
        # summaries + long-term memory are unaffected. None = no offset set.
        after_timestamp = self._setting("conversation_offset", None)
        try:
            after_timestamp = float(after_timestamp) if after_timestamp is not None else None
        except (TypeError, ValueError):
            after_timestamp = None

        # Use PostgreSQL backend if available
        if self._db_backend:
            try:
                self.messages = self._db_backend.get_messages(
                    since_minutes=since_minutes,
                    after_timestamp=after_timestamp,
                )
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
            # Note: since_minutes param is actually in seconds (legacy naming)
            logger.debug(f"Loading history from {since_minutes} seconds ago ({len(self.messages)} messages)")
            cutoff = time.time() - since_minutes
            self.messages = [msg for msg in self.messages if msg.timestamp >= cutoff]
        if after_timestamp is not None and self.messages:
            # Honor the /new conversation offset; always keep summary rows.
            self.messages = [
                msg for msg in self.messages
                if msg.role == "summary" or msg.timestamp >= after_timestamp
            ]

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

    def _compact_summary_content(self, content: str) -> str:
        """Reduce summary verbosity for prompt context while preserving key continuity."""
        text = (content or "").strip()
        if not text:
            return text

        compact_enabled = bool(
            self._setting("summarization_compact_context", getattr(self.config, "SUMMARIZATION_COMPACT_CONTEXT", True))
        )
        if not compact_enabled:
            return text

        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if not lines:
            return text

        from ..memory.summarization import compress_structured_summary_text, extract_summary_sections

        prefix_lines: list[str] = []
        sections = extract_summary_sections(compress_structured_summary_text(text))
        section_summary = (sections.get("summary") or "").strip()
        section_key_details = (sections.get("key_details") or "").strip()
        section_intent = (sections.get("intent") or "").strip()
        section_open_loops = (sections.get("open_loops") or "").strip()

        for line in lines:
            lowered = line.lower()
            if lowered.startswith("[historical summary]"):
                continue
            if lowered.startswith("summary:") or lowered.startswith("key details:") or lowered.startswith("intent:") or lowered.startswith("tone:") or lowered.startswith("open loops:"):
                continue
            # Keep non-section metadata prefix lines (legacy lines like "On YYYY-MM-DD...")
            if ":" not in line and not line.startswith("["):
                prefix_lines.append(line)

        compact_lines: list[str] = []
        compact_lines.extend(prefix_lines[:2])
        if section_summary:
            compact_lines.append(section_summary)
        if section_key_details:
            compact_lines.append(f"Details: {section_key_details}")
        elif section_intent:
            compact_lines.append(f"Intent: {section_intent}")
        if section_open_loops:
            compact_lines.append(f"Open: {section_open_loops}")

        # Legacy fallback: keep as-is if we couldn't identify structured sections.
        if not compact_lines:
            return text

        return "\n".join(compact_lines).strip()

    def get_context_messages(self, max_tokens: int = 0):
        """Get messages to be used as context for the LLM.

        Includes (in order):
        - System messages (always)
        - Summaries of older sessions (role='summary') with time context
        - Recent messages (newest-first within token budget)

        If max_tokens > 0, applies a token budget:
        1. System messages are always included
        2. Protected recent turns are always included (newest N pairs)
        3. Remaining budget fills from newest-first, dropping oldest messages
        4. Optional max_context_messages caps raw history messages before token budgeting

        Args:
            max_tokens: Maximum token budget for the returned messages.
                        0 = no limit (backward-compatible default).
        """
        system_messages = []
        summary_messages = []
        regular_messages = []

        for msg in self.messages:
            if msg.role == "system":
                system_messages.append(msg)
            elif msg.role == "summary":
                # Add time context to summaries
                time_ago = self._format_time_ago(msg.timestamp)
                compact_summary = self._compact_summary_content(msg.content)
                enhanced_content = f"[Previous conversation {time_ago}]\n{compact_summary}"
                summary_messages.append(Message(
                    role="summary",
                    content=enhanced_content,
                    timestamp=msg.timestamp
                ))
            else:
                regular_messages.append(msg)

        if not system_messages:
            system_messages.append(Message(role="system", content=self.config.SYSTEM_MESSAGE))

        max_context_messages = int(
            self._setting(
                "max_context_messages",
                getattr(self.config, "MAX_CONTEXT_MESSAGES", 0),
            )
            or 0
        )
        if max_context_messages > 0 and len(regular_messages) > max_context_messages:
            regular_messages = regular_messages[-max_context_messages:]

        # Without a token budget, return everything
        if max_tokens <= 0:
            return system_messages + summary_messages + regular_messages

        # ── Token-budget enforcement ──────────────────────────────
        protected_turns = int(
            self._setting(
                "memory_protected_recent_turns",
                getattr(self.config, "MEMORY_PROTECTED_RECENT_TURNS", 3),
            )
        )
        # Protect the last N user+assistant pairs (= 2*N messages from the end)
        n_protected = min(protected_turns * 2, len(regular_messages))
        protected = regular_messages[-n_protected:] if n_protected > 0 else []
        droppable = regular_messages[:-n_protected] if n_protected > 0 else list(regular_messages)

        # Calculate baseline usage (system + protected)
        system_cost = estimate_messages_tokens(system_messages)
        protected_cost = estimate_messages_tokens(protected)
        used = system_cost + protected_cost
        budget = max_tokens

        # Fill from summaries (oldest context → newest)
        included_summaries: list[Message] = []
        max_summaries = int(
            self._setting(
                "summarization_max_in_context",
                getattr(self.config, "SUMMARIZATION_MAX_IN_CONTEXT", 5),
            )
        )
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

        remaining = budget - used

        def pct(v: int) -> str:
            return f"{v * 100 // budget}%" if budget > 0 else "n/a"

        logger.debug(
            f"Token budget: {budget} total | used={used} ({pct(used)}) | remaining={remaining} | "
            f"system={system_cost} ({pct(system_cost)}), "
            f"protected={protected_cost} ({pct(protected_cost)}, {len(protected)} msgs), "
            f"summaries={summary_cost} ({pct(summary_cost)}, {len(included_summaries)}/{len(summary_messages)}), "
            f"history={droppable_cost} ({pct(droppable_cost)}, {len(included_droppable)} msgs, {dropped} dropped), "
            f"message_cap={max_context_messages or 'none'}"
        )

        return system_messages + included_summaries + included_droppable + protected
    
    def build_context_payload(
        self,
        *,
        continuity: bool,
        history_scope: str = "inline+summaries",
        delivery: str = "inline",
        max_tokens: int = 0,
    ) -> ContextPayload:
        """The ONE context-assembly handler (TASK-493).

        Decides *what* prior context a conversation gets, identically for chat
        turns and agent seeds. Wraps the shared assembler ``get_context_messages``
        and categorises its output into buckets, applying the single summary gate:

            summaries included  iff  continuity AND history_scope == "inline+summaries"

        System-row handling is NOT decided here — the caller reads the view that
        matches its delivery (``inline_history`` for chat, ``seed_messages`` for a
        seed). ``delivery`` is accepted for call-site clarity and future use; it
        does not change the buckets (both plain-system and tool-result-system rows
        are always separated so either delivery can compose correctly).

        Args:
            continuity: master gate — carry prior context at all.
            history_scope: "inline+summaries" (default) or "inline" (no summaries).
            delivery: "inline" (chat) or "seed" (agent) — documentary only.
            max_tokens: token budget passed through to ``get_context_messages``.
        """
        include_summaries = bool(continuity) and history_scope == "inline+summaries"

        payload = ContextPayload()
        for msg in self.get_context_messages(max_tokens=max_tokens):
            role = getattr(msg, "role", "")
            if role == "summary":
                if include_summaries:
                    payload.summary_messages.append(msg)
                continue
            if role == "system":
                if _is_tool_result_system(msg):
                    payload.tool_result_messages.append(msg)
                else:
                    payload.system_messages.append(msg)
                continue
            payload.regular_messages.append(msg)
        return payload

    def _format_time_ago(self, timestamp: float) -> str:
        """Format a timestamp as a human-readable relative time."""
        import time
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

    def add_message(self, role, content, message_id=None, attachments=None, reasoning=None):
        """Append a message to history and save.

        If ``message_id`` is provided, it is used as the persistent ID so
        the frontend's user-message UUID matches the server-side history
        row — enabling tool-call events keyed by trigger_message_id to
        join cleanly with chat history without a separate id-mapping step.

        ``attachments`` (TASK-225): optional tiny JSONB payload written
        to the ``{bot}_messages.attachments`` column — a list of
        ``{"asset_id": "ma_...", "kind": "image"}`` refs.  Only
        meaningful on user-role rows in practice; the file-fallback path
        ignores it (the on-disk format has no concept of attachments).
        """
        provided_id = str(message_id).strip() if message_id else None
        message = Message(role, content, db_id=provided_id or str(uuid.uuid4()))
        self.messages.append(message)

        # Save to PostgreSQL if available, otherwise file
        if self._db_backend:
            # EPIC TASK-217 / TASK-357: the DB backend now RAISES on a failed
            # INSERT instead of swallowing it and handing back a valid-looking
            # id. When the DB is configured, a failed persist must NOT be
            # silently masked by a file fallback — that is exactly what left a
            # turn_logs row pointing at an uncommitted (orphan) {bot}_messages
            # row. Propagate the failure so the turn-orchestration layer marks
            # the turn as errored (background_service catches this and sets the
            # turn_log status="error", then re-raises) rather than recording a
            # ghost reference. Every caller wraps add_message in its own
            # try/except, so the raise aborts the turn loudly and safely.
            persisted_id = self._db_backend.add_message(
                role, content, message.timestamp,
                message_id=provided_id,
                attachments=attachments,
                reasoning=reasoning,
            )
            if persisted_id:
                message.db_id = str(persisted_id)
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
