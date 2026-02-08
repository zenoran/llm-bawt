"""Base class for LLM orchestration.

BaseLLMBawt provides the shared logic for both CLI (LLMBawt) and service
(ServiceLLMBawt) modes, including:
- Bot initialization
- Memory client initialization
- Profile management
- Pipeline execution

Subclasses override _initialize_client() for their specific client types.
"""

import json
import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

from ..bots import Bot, BotManager
from ..clients import LLMClient
from ..profiles import ProfileManager, EntityType
from ..memory_server.client import MemoryClient, get_memory_client
from ..search import get_search_client, SearchClient
from ..tools import get_tools_prompt, get_tools_list, query_with_tools
from ..utils.config import Config, has_database_credentials
from ..utils.paths import resolve_log_dir
from ..utils.history import HistoryManager, Message
from ..adapters import get_adapter, ModelAdapter
from .pipeline import RequestPipeline, PipelineContext
from .prompt_builder import PromptBuilder, SectionPosition
from .model_lifecycle import ModelLifecycleManager, get_model_lifecycle

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
console = Console()


def _text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, text1, text2).ratio()


class BaseLLMBawt(ABC):
    """Base class for LLM orchestration with memory and tools.
    
    Provides common initialization and query logic. Subclasses implement
    _initialize_client() for their specific model backends.
    """
    
    def __init__(
        self,
        resolved_model_alias: str,
        config: Config,
        local_mode: bool = False,
        bot_id: str = "nova",
        user_id: str = "",  # Required - must be passed explicitly
        verbose: bool = False,
        debug: bool = False,
        existing_client: LLMClient | None = None,
    ):
        """Initialize the LLM orchestrator.
        
        Args:
            resolved_model_alias: The model alias from models.yaml
            config: Application configuration
            local_mode: If True, skip database features
            bot_id: Bot personality to use
            user_id: User profile ID (required)
            verbose: Enable verbose output (--verbose)
            debug: Enable debug output (--debug)
            existing_client: Reuse this client instead of creating a new one
        """
        if not user_id:
            raise ValueError("user_id is required - set LLM_BAWT_DEFAULT_USER or pass --user")
        self.resolved_model_alias = resolved_model_alias
        self.config = config
        self.model_definition = self.config.defined_models.get("models", {}).get(resolved_model_alias)
        self.tool_format = self.config.get_tool_format(model_alias=resolved_model_alias, model_def=self.model_definition)
        self.local_mode = local_mode
        self.bot_id = bot_id
        self.user_id = user_id
        self.verbose = verbose
        self.debug = debug
        
        # Will be initialized below
        self.memory: MemoryClient | None = None
        self.profile_manager: ProfileManager | None = None
        self.search_client: SearchClient | None = None
        self.model_lifecycle: ModelLifecycleManager | None = None
        self.client: LLMClient
        self.bot: Bot
        self.history_manager: HistoryManager
        self._db_available: bool = False
        self.adapter: ModelAdapter
        
        if not self.model_definition:
            raise ValueError(f"Could not find model definition for: '{resolved_model_alias}'")
        
        # Initialize model lifecycle manager (singleton)
        self.model_lifecycle = get_model_lifecycle(config)
        
        # Initialize adapter based on model definition
        self.adapter = get_adapter(
            self.resolved_model_alias,
            self.model_definition
        )
        
        # Initialize LLM client - reuse existing if provided
        if existing_client is not None:
            self.client = existing_client
            logger.debug(f"Reusing existing client for model '{resolved_model_alias}'")
        else:
            self.client = self._initialize_client()
        
        # Log effective per-model configuration
        if self.verbose:
            logger.info(
                f"Model '{resolved_model_alias}' config: "
                f"context_window={self.client.effective_context_window}, "
                f"max_tokens={self.client.effective_max_tokens}, "
                f"type={self.model_definition.get('type', 'unknown')}"
            )
        
        # Register client with lifecycle manager
        self.model_lifecycle.register_client(resolved_model_alias, self.client)
        
        # Initialize bot
        self._init_bot(config)
        
        # Initialize memory and profiles
        self._init_memory(config)
        
        # Initialize search client
        self._init_search(config)
        
        # Build system prompt
        self._init_system_prompt()
        
        # Initialize history manager
        self._init_history()
    
    @abstractmethod
    def _initialize_client(self) -> LLMClient:
        """Initialize the LLM client. Implemented by subclasses."""
        pass
    
    def _init_bot(self, config: Config):
        """Initialize bot configuration."""
        bot_manager = BotManager(config)
        bot = bot_manager.get_bot(self.bot_id)
        
        if not bot:
            logger.warning(f"Bot '{self.bot_id}' not found, falling back to default")
            bot = bot_manager.get_default_bot()
            self.bot_id = bot.slug
        
        self.bot = bot
        self.client.bot_name = self.bot.name
        self.client.model_alias = self.resolved_model_alias

        if self.verbose:
            logger.info(f"Using bot: {self.bot.name} ({self.bot_id})")
    
    def _init_memory(self, config: Config):
        """Initialize memory client if database is available."""
        if self.local_mode:
            logger.debug("Local mode enabled - using filesystem for history")
            return
        
        if not has_database_credentials(config):
            logger.debug("Database credentials not configured")
            return
        
        if not self.bot.requires_memory:
            logger.debug(f"Bot '{self.bot.name}' has requires_memory=false")
            self._db_available = True  # DB available but not needed
            return
        
        try:
            self.memory = get_memory_client(
                config=config,
                bot_id=self.bot_id,
                user_id=self.user_id,
                server_url=getattr(config, "MEMORY_SERVER_URL", None),
            )
            self._db_available = True
            logger.debug(f"Memory client initialized for bot: {self.bot_id}")
            
            # Initialize profile manager
            self.profile_manager = ProfileManager(config)
            
        except Exception as e:
            # Log without full traceback for connection errors (avoid 10-page stack traces)
            error_str = str(e)
            if "could not translate host name" in error_str or "Connection refused" in error_str:
                logger.error(f"Failed to initialize memory: {e}")
            else:
                logger.exception(f"Failed to initialize memory: {e}")
            if self.verbose:
                console.print(f"[yellow]Warning:[/yellow] Memory client failed: {e}")
    
    def _init_search(self, config: Config):
        """Initialize search client if bot uses search."""
        if not getattr(self.bot, 'uses_search', False):
            return
        
        try:
            self.search_client = get_search_client(config)
            if self.search_client and self.search_client.is_available():
                logger.debug(f"Search client initialized: {self.search_client.PROVIDER.value}")
            else:
                logger.debug("Search requested but no provider available")
                self.search_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize search: {e}")
            self.search_client = None
    
    def _init_system_prompt(self):
        """Build the system prompt using PromptBuilder."""
        builder = PromptBuilder()
        
        # Section 1: User profile context
        if self._db_available and self.profile_manager:
            try:
                user_context = self.profile_manager.get_user_profile_summary(self.user_id)
                if user_context:
                    builder.add_section(
                        "user_context",
                        f"## About the User\n{user_context}",
                        position=SectionPosition.USER_CONTEXT,
                    )
                    if self.verbose:
                        console.print(f"[dim]─── User Profile ({self.user_id}) ───[/dim]")
                        console.print(f"[cyan]{user_context}[/cyan]")
                        console.print(f"[dim]{'─' * 30}[/dim]")
            except Exception as e:
                logger.warning(f"Failed to load user profile: {e}")
        
        # Section 2: Bot personality traits
        if self._db_available and self.profile_manager:
            try:
                bot_context = self.profile_manager.get_bot_profile_summary(self.bot_id)
                if bot_context:
                    builder.add_section(
                        "bot_traits",
                        f"## Your Developed Traits\n{bot_context}",
                        position=SectionPosition.BOT_TRAITS,
                    )
                    if self.verbose:
                        console.print(f"[dim]─── Bot Profile ({self.bot_id}) ───[/dim]")
                        console.print(f"[magenta]{bot_context}[/magenta]")
                        console.print(f"[dim]{'─' * 30}[/dim]")
            except Exception as e:
                logger.warning(f"Failed to load bot profile: {e}")
        
        # Section 3: Base bot prompt
        if self.bot.system_prompt:
            builder.add_section(
                "base_prompt",
                self.bot.system_prompt,
                position=SectionPosition.BASE_PROMPT,
            )
        
        # Section 4: Tool instructions (added at query time if needed)
        # We store the builder, not the final string, so we can add context later
        self._prompt_builder = builder
        
        # For backward compatibility, also set config.SYSTEM_MESSAGE
        # This will be the base prompt without per-query memory context
        self.config.SYSTEM_MESSAGE = builder.build()
        
        if self.verbose:
            console.print(builder.get_verbose_summary())
    
    def _init_history(self):
        """Initialize history manager."""
        self.history_manager = HistoryManager(
            client=self.client,
            config=self.config,
            db_backend=self.memory.get_short_term_manager() if self.memory else None,
            bot_id=self.bot_id,
        )
        self.load_history()
    
    def load_history(self, since_minutes: int | None = None) -> list[dict]:
        """Load conversation history."""
        if since_minutes is None:
            since_minutes = self.config.HISTORY_DURATION
        self.history_manager.load_history(since_minutes=since_minutes)
        return [msg.to_dict() for msg in self.history_manager.messages]
    
    def query(self, prompt: str, plaintext_output: bool = False, stream: bool = True) -> str:
        """Send a query to the LLM and return the response.
        
        This is the main entry point for queries. It uses the RequestPipeline
        for modular processing.
        
        Args:
            prompt: User's question/message
            plaintext_output: If True, skip rich formatting
            stream: If True, stream the response
            
        Returns:
            Assistant's response string
        """
        try:
            # Add user message to history first
            self.history_manager.add_message("user", prompt)
            
            # Build context and execute via pipeline or direct method
            context_messages = self._build_context_messages(prompt)
            
            # Use tool loop if bot has tools enabled
            if self.bot.uses_tools and self.memory:
                tool_definitions = self._get_tool_definitions()
                assistant_response, tool_context = query_with_tools(
                    messages=context_messages,
                    client=self.client,
                    memory_client=self.memory,
                    profile_manager=self.profile_manager,
                    search_client=self.search_client,
                    model_lifecycle=self.model_lifecycle,
                    config=self.config,
                    user_id=self.user_id,
                    bot_id=self.bot_id,
                    stream=stream,
                    tool_format=self.tool_format,
                    tools=tool_definitions,
                    adapter=self.adapter,
                )

                # Render the final response (tool loop returns raw text, never renders)
                if assistant_response:
                    if not plaintext_output:
                        self.client._print_assistant_message(assistant_response)
                    else:
                        print(assistant_response)

                # Save tool context to history
                if tool_context:
                    self.history_manager.add_message("system", f"[Tool Results]\n{tool_context}")
            else:
                assistant_response = self.client.query(
                    context_messages,
                    plaintext_output=plaintext_output,
                    stream=stream,
                )
            
            if assistant_response:
                self.history_manager.add_message("assistant", assistant_response)
                # Memory extraction for all memory-enabled bots
                if self.memory:
                    self._trigger_memory_extraction(prompt, assistant_response)

            # Write debug turn log if enabled
            if self.debug:
                self._write_debug_turn_log(context_messages, prompt, assistant_response)

            return assistant_response
        
        except KeyboardInterrupt:
            console.print("[bold red]Query interrupted.[/bold red]")
            self.history_manager.remove_last_message_if_partial("assistant")
            return ""
        except Exception as e:
            console.print(f"[bold red]Error during query:[/bold red] {e}")
            logger.exception(f"Error during query: {e}")
            self.history_manager.remove_last_message_if_partial("assistant")
            return ""
    
    def _build_context_messages(self, prompt: str) -> list[Message]:
        """Build messages list with system prompt, memory context, and history."""
        messages = []
        
        # Start with a copy of the prompt builder
        builder = self._prompt_builder.copy()
        
        # Add tool instructions OR memory context
        if self.bot.uses_tools and self.memory:
            tool_definitions = self._get_tool_definitions()
            tools_prompt = get_tools_prompt(
                tools=tool_definitions,
                tool_format=self.tool_format,
            )
            builder.add_section(
                "tools",
                tools_prompt,
                position=SectionPosition.TOOLS,
            )
            # Cold-start memory priming: inject top memories when history is thin
            # This gives the model immediate context about the user on first interaction
            history_count = len(self.history_manager.messages)
            if history_count <= 3:
                cold_start_context = self._retrieve_cold_start_memories(prompt)
                if cold_start_context:
                    builder.add_section(
                        "cold_start_memory",
                        cold_start_context,
                        position=SectionPosition.MEMORY_CONTEXT,
                    )
        elif self.memory:
            # Retrieve and add memory context
            memory_context = self._retrieve_memory_context(prompt)
            if memory_context:
                builder.add_section(
                    "memory_context",
                    memory_context,
                    position=SectionPosition.MEMORY_CONTEXT,
                )
        
        # Build final system message
        system_content = builder.build()
        if system_content:
            messages.append(Message(role="system", content=system_content))
        
        if self.debug:
            logger.debug(f"System message: {len(system_content)} chars")
            logger.debug(f"Sections: {[s.name for s in builder.enabled_sections]}")
        
        # Determine if we should include history
        include_history = True
        if self.bot.uses_tools and self.memory:
            if getattr(self.config, "TOOLS_SKIP_HISTORY", True):
                include_history = not self._should_skip_history(prompt)
        
        if include_history:
            history = self.history_manager.get_context_messages()
            for msg in history:
                if msg.role in ("user", "assistant"):
                    messages.append(msg)
        else:
            # Always include current prompt when history is skipped
            if prompt:
                messages.append(Message(role="user", content=prompt))
        
        return messages

    def _get_tool_definitions(self) -> list:
        include_search = self.search_client is not None
        include_models = self.model_lifecycle is not None
        return get_tools_list(
            include_search_tools=include_search,
            include_model_tools=include_models,
        )
    
    def _should_skip_history(self, prompt: str) -> bool:
        """Return True if history should be skipped for this prompt."""
        prompt_lower = (prompt or "").lower()
        search_triggers = (
            "search", "web_search", "news", "current", "today",
            "latest", "now", "date", "time", "headline",
        )
        return any(token in prompt_lower for token in search_triggers)
    
    def _retrieve_memory_context(self, prompt: str) -> str:
        """Retrieve relevant memories and format as context string."""
        if not self.memory:
            return ""
        
        try:
            n_results = self.config.MEMORY_N_RESULTS
            min_relevance = self.config.MEMORY_MIN_RELEVANCE
            
            memory_results = self.memory.search(
                prompt, n_results=n_results, min_relevance=min_relevance
            )
            
            if not memory_results:
                return ""
            
            # Convert to dicts
            memories = [
                {
                    "content": m.content,
                    "relevance": m.relevance,
                    "tags": m.tags,
                    "importance": m.importance,
                }
                for m in memory_results
            ]
            
            # Deduplicate against recent history
            history_contents = [msg.content for msg in self.history_manager.messages[-10:]]
            unique_memories = []
            for mem in memories:
                is_dup = any(
                    _text_similarity(mem["content"], h) >= self.config.MEMORY_DEDUP_SIMILARITY
                    for h in history_contents
                )
                if not is_dup:
                    unique_memories.append(mem)
            
            if not unique_memories:
                return ""
            
            # Format as context string
            from ..memory.context_builder import build_memory_context_string
            
            # Get user's display name
            user_name = None
            if self.profile_manager:
                try:
                    profile, _ = self.profile_manager.get_or_create_profile(
                        EntityType.USER, self.user_id
                    )
                    user_name = profile.display_name
                except Exception:
                    pass
            
            return build_memory_context_string(unique_memories, user_name=user_name)
            
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return ""
    
    def _retrieve_cold_start_memories(self, prompt: str) -> str:
        """Retrieve a small set of high-importance memories for cold-start context.

        Called when a tool-enabled bot has sparse history (≤3 messages).
        Injects a few key facts about the user into the system prompt so the
        model has immediate context without needing to call the memory tool first.

        Args:
            prompt: Current user prompt (used for relevance search)

        Returns:
            Formatted memory context string, or empty string if nothing found
        """
        if not self.memory:
            return ""

        try:
            # Fetch a small number of high-relevance memories
            results = self.memory.search(
                prompt,
                n_results=3,
                min_relevance=self.config.MEMORY_MIN_RELEVANCE,
            )

            if not results:
                return ""

            memories = [
                {
                    "content": m.content,
                    "relevance": m.relevance,
                    "tags": m.tags,
                    "importance": m.importance,
                }
                for m in results
            ]

            from ..memory.context_builder import build_memory_context_string

            user_name = None
            if self.profile_manager:
                try:
                    profile, _ = self.profile_manager.get_or_create_profile(
                        EntityType.USER, self.user_id
                    )
                    user_name = profile.display_name
                except Exception:
                    pass

            context = build_memory_context_string(memories, user_name=user_name)
            if context:
                logger.debug(
                    f"Cold-start memory priming: injected {len(memories)} memories "
                    f"({len(context)} chars) into system prompt"
                )
            return context

        except Exception as e:
            logger.warning(f"Cold-start memory retrieval failed: {e}")
            return ""

    def _trigger_memory_extraction(self, user_prompt: str, assistant_response: str):
        """Trigger background memory extraction (non-blocking)."""
        def extract():
            try:
                from ..service import ServiceClient
                from ..service.tasks import create_extraction_task

                client = ServiceClient()
                if client.is_available():
                    task = create_extraction_task(
                        user_message=user_prompt,
                        assistant_message=assistant_response,
                        bot_id=self.bot_id,
                        user_id=self.user_id,
                        message_ids=[str(uuid.uuid4()), str(uuid.uuid4())],
                        model=self.resolved_model_alias,
                    )
                    client.submit_task(task)
            except Exception as e:
                logger.debug(f"Memory extraction failed: {e}")

        thread = threading.Thread(target=extract, daemon=True)
        thread.start()
        thread.join(timeout=0.5)

    def _write_debug_turn_log(
        self,
        context_messages: list[Message],
        user_prompt: str,
        assistant_response: str,
    ):
        """Write the current turn's request/response data to a debug log file.

        Only called when --debug is enabled. Overwrites the file on each turn
        to show the most recent request/response for review.

        Args:
            context_messages: Full list of messages sent to the LLM
            user_prompt: The user's input for this turn
            assistant_response: The assistant's response
        """
        try:
            logs_dir = resolve_log_dir()
            logs_dir.mkdir(parents=True, exist_ok=True)

            log_file = logs_dir / "debug_turn.txt"

            # Build the log content
            lines = []
            lines.append("=" * 80)
            lines.append(f"DEBUG TURN LOG - {datetime.now().isoformat()}")
            lines.append(f"Model: {self.resolved_model_alias}")
            lines.append(f"Bot: {self.bot_id} ({self.bot.name})")
            lines.append(f"User: {self.user_id}")
            lines.append("=" * 80)
            lines.append("")

            # Request data - all context messages
            lines.append("─" * 40)
            lines.append("REQUEST MESSAGES")
            lines.append("─" * 40)
            for i, msg in enumerate(context_messages):
                lines.append(f"\n[{i}] Role: {msg.role}")
                lines.append(f"    Timestamp: {msg.timestamp}")
                lines.append(f"    Content ({len(msg.content)} chars):")
                lines.append("    " + "─" * 36)
                # Indent content for readability
                for content_line in msg.content.split("\n"):
                    lines.append(f"    {content_line}")
                lines.append("")

            # Response data
            lines.append("─" * 40)
            lines.append("RESPONSE")
            lines.append("─" * 40)
            lines.append(f"Length: {len(assistant_response)} chars")
            lines.append("")
            lines.append(assistant_response)
            lines.append("")

            # Also dump as JSON for machine parsing
            lines.append("─" * 40)
            lines.append("JSON FORMAT (for parsing)")
            lines.append("─" * 40)
            json_data = {
                "timestamp": datetime.now().isoformat(),
                "model": self.resolved_model_alias,
                "bot_id": self.bot_id,
                "user_id": self.user_id,
                "request": [msg.to_dict() for msg in context_messages],
                "response": assistant_response,
            }
            lines.append(json.dumps(json_data, indent=2, ensure_ascii=False))

            # Write to file (overwrite)
            log_file.write_text("\n".join(lines), encoding="utf-8")
            logger.debug(f"Debug turn log written to: {log_file}")

        except Exception as e:
            logger.warning(f"Failed to write debug turn log: {e}")
    
    def refine_prompt(self, prompt: str, history: list | None = None) -> str:
        """Refine the user's prompt using context."""
        try:
            from ..bots import get_system_prompt
            
            messages = []
            refine_prompt = get_system_prompt("refine") or "You are a prompt refinement assistant."
            messages.append(Message(role="system", content=refine_prompt))
            
            if history:
                history_text = []
                for msg in history:
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        history_text.append(f"{msg.role}: {msg.content}")
                    elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        history_text.append(f"{msg['role']}: {msg['content']}")
                
                if history_text:
                    context = f"Previous conversation:\n{chr(10).join(history_text)}\n\nUser's prompt: {prompt}"
                    messages.append(Message(role="user", content=context))
                else:
                    messages.append(Message(role="user", content=f"User's prompt: {prompt}"))
            else:
                messages.append(Message(role="user", content=f"User's prompt: {prompt}"))
            
            return self.client.query(messages, plaintext_output=True, stream=False)
        except Exception as e:
            logger.warning(f"Prompt refinement failed: {e}")
            return prompt
