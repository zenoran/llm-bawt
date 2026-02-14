"""Modular request processing pipeline.

RequestPipeline encapsulates the full lifecycle of an LLM request with
clear decision points and extensible hooks at each stage.

Stages:
1. PRE_PROCESS: Validate input, apply transformations
2. CONTEXT_BUILD: Assemble system prompt via PromptBuilder
3. MEMORY_RETRIEVAL: Search memories if applicable
4. HISTORY_FILTER: Decide what history to include
5. MESSAGE_ASSEMBLY: Build final messages list
6. EXECUTE: Tool loop or direct LLM query
7. POST_PROCESS: Memory extraction, history save

Each stage can be observed (--verbose) or deeply inspected (--debug).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, TYPE_CHECKING
import logging
import time

if TYPE_CHECKING:
    from ..models.message import Message
    from ..bots import Bot
    from ..utils.config import Config
    from .prompt_builder import PromptBuilder
    from ..adapters import ModelAdapter

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Processing stages in the request pipeline."""
    PRE_PROCESS = auto()
    CONTEXT_BUILD = auto()
    MEMORY_RETRIEVAL = auto()
    HISTORY_FILTER = auto()
    MESSAGE_ASSEMBLY = auto()
    EXECUTE = auto()
    POST_PROCESS = auto()


@dataclass
class PipelineContext:
    """Shared context passed through all pipeline stages.
    
    This replaces the ad-hoc state management scattered across methods.
    Each stage reads what it needs and writes its outputs here.
    """
    # Input
    prompt: str
    user_id: str
    bot_id: str
    stream: bool = True
    plaintext_output: bool = False
    
    # Stage outputs
    prompt_builder: "PromptBuilder | None" = None
    memory_context: str = ""
    memory_results: list[dict] = field(default_factory=list)
    include_history: bool = True
    messages: list["Message"] = field(default_factory=list)
    response: str = ""
    tool_context: str = ""
    tool_call_details: list[dict] = field(default_factory=list)
    tool_format: str = "xml"
    tool_definitions: list[Any] = field(default_factory=list)
    
    # Decision flags (set by decision points)
    use_tools: bool = False
    use_memory: bool = False
    use_search: bool = False
    skip_history: bool = False
    
    # Timing/debug info
    stage_timings: dict[str, float] = field(default_factory=dict)
    stage_outputs: dict[str, Any] = field(default_factory=dict)
    
    def record_timing(self, stage: PipelineStage, elapsed_ms: float):
        """Record timing for a stage."""
        self.stage_timings[stage.name] = elapsed_ms
    
    def record_output(self, stage: PipelineStage, output: Any):
        """Record debug output for a stage."""
        self.stage_outputs[stage.name] = output


# Type for stage hooks
StageHook = Callable[[PipelineContext], None]


class RequestPipeline:
    """Orchestrates the LLM request lifecycle with modular stages.
    
    Each stage is a discrete processing step with clear inputs/outputs.
    Hooks can be registered to observe or modify behavior at each stage.
    
    Example:
        pipeline = RequestPipeline(config, bot, memory_client)
        pipeline.add_hook(PipelineStage.MEMORY_RETRIEVAL, my_custom_retriever)
        result = pipeline.execute(PipelineContext(prompt="hello", ...))
    """
    
    def __init__(
        self,
        config: "Config",
        bot: "Bot",
        memory_client: Any = None,
        profile_manager: Any = None,
        search_client: Any = None,
        home_client: Any = None,
        model_lifecycle: Any = None,
        history_manager: Any = None,
        llm_client: Any = None,
        adapter: "ModelAdapter | None" = None,
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initialize the pipeline.

        Args:
            config: Application configuration
            bot: Bot configuration for this request
            memory_client: Optional memory client for retrieval/storage
            profile_manager: Optional profile manager for user/bot attributes
            search_client: Optional search client for web search tools
            model_lifecycle: Optional model lifecycle manager for model switching
            history_manager: History manager for conversation history
            llm_client: LLM client for query execution
            adapter: Optional model adapter for model-specific stop sequences
            verbose: Enable verbose logging (--verbose)
            debug: Enable debug logging with full I/O (--debug)
        """
        self.config = config
        self.bot = bot
        self.memory_client = memory_client
        self.profile_manager = profile_manager
        self.search_client = search_client
        self.home_client = home_client
        self.model_lifecycle = model_lifecycle
        self.history_manager = history_manager
        self.llm_client = llm_client
        self.adapter = adapter
        self.verbose = verbose
        self.debug = debug
        
        # Stage hooks: stage -> list of callables
        self._hooks: dict[PipelineStage, list[StageHook]] = {
            stage: [] for stage in PipelineStage
        }
        
        # Decision point overrides
        self._decision_overrides: dict[str, bool] = {}
    
    def add_hook(self, stage: PipelineStage, hook: StageHook) -> "RequestPipeline":
        """Register a hook to be called during a stage.
        
        Hooks are called in order of registration after the default stage logic.
        
        Args:
            stage: Which stage to hook
            hook: Callable that receives PipelineContext
            
        Returns:
            self for chaining
        """
        self._hooks[stage].append(hook)
        return self
    
    def override_decision(self, decision: str, value: bool) -> "RequestPipeline":
        """Override a decision point.
        
        Args:
            decision: One of 'use_tools', 'use_memory', 'use_search', 'skip_history'
            value: Override value
            
        Returns:
            self for chaining
        """
        self._decision_overrides[decision] = value
        return self
    
    def execute(self, ctx: PipelineContext) -> str:
        """Execute the full pipeline.
        
        Args:
            ctx: Pipeline context with input parameters
            
        Returns:
            LLM response string
        """
        stages = [
            (PipelineStage.PRE_PROCESS, self._stage_pre_process),
            (PipelineStage.CONTEXT_BUILD, self._stage_context_build),
            (PipelineStage.MEMORY_RETRIEVAL, self._stage_memory_retrieval),
            (PipelineStage.HISTORY_FILTER, self._stage_history_filter),
            (PipelineStage.MESSAGE_ASSEMBLY, self._stage_message_assembly),
            (PipelineStage.EXECUTE, self._stage_execute),
            (PipelineStage.POST_PROCESS, self._stage_post_process),
        ]
        
        for stage, handler in stages:
            start = time.perf_counter()
            
            if self.verbose:
                logger.info(f"Pipeline: {stage.name}")
            
            try:
                handler(ctx)
                
                # Run registered hooks
                for hook in self._hooks[stage]:
                    hook(ctx)
                    
            except Exception as e:
                logger.exception(f"Pipeline error in {stage.name}: {e}")
                raise
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            ctx.record_timing(stage, elapsed_ms)
            
            if self.debug:
                logger.debug(f"  {stage.name} completed in {elapsed_ms:.1f}ms")
        
        if self.verbose:
            self._log_summary(ctx)
        
        return ctx.response
    
    # ─────────────────────────────────────────────────────────────────────────
    # Stage implementations
    # ─────────────────────────────────────────────────────────────────────────
    
    def _stage_pre_process(self, ctx: PipelineContext):
        """Stage 1: Validate input and set decision flags."""
        # Set decision flags based on bot config and available clients
        if "use_memory" in self._decision_overrides:
            ctx.use_memory = bool(self._decision_overrides["use_memory"])
        else:
            ctx.use_memory = (
                getattr(self.bot, "requires_memory", False)
                and self.memory_client is not None
            )
        
        if "use_tools" in self._decision_overrides:
            ctx.use_tools = bool(self._decision_overrides["use_tools"])
        else:
            ctx.use_tools = (
                getattr(self.bot, "uses_tools", False)
                and (self.memory_client is not None or self.home_client is not None)
            )
        
        if "use_search" in self._decision_overrides:
            ctx.use_search = bool(self._decision_overrides["use_search"])
        else:
            ctx.use_search = (
                getattr(self.bot, "uses_search", False)
                and self.search_client is not None
            )
        
        if self.debug:
            logger.debug(
                f"  Decisions: use_memory={ctx.use_memory}, "
                f"use_tools={ctx.use_tools}, use_search={ctx.use_search}"
            )
        
        ctx.record_output(PipelineStage.PRE_PROCESS, {
            "use_memory": ctx.use_memory,
            "use_tools": ctx.use_tools,
            "use_search": ctx.use_search,
        })
    
    def _stage_context_build(self, ctx: PipelineContext):
        """Stage 2: Build system prompt using PromptBuilder."""
        from .prompt_builder import PromptBuilder, SectionPosition
        from ..utils.temporal import build_temporal_context
        
        builder = PromptBuilder()

        builder.add_section(
            "temporal_context",
            build_temporal_context(self.history_manager.messages if self.history_manager else None),
            position=SectionPosition.DATETIME,
        )
        
        # Section 1: User profile context
        if self.profile_manager and ctx.use_memory:
            try:
                user_context = self.profile_manager.get_user_profile_summary(ctx.user_id)
                if user_context:
                    builder.add_section(
                        "user_context",
                        f"## About the User\n{user_context}",
                        position=SectionPosition.USER_CONTEXT,
                    )
            except Exception as e:
                logger.warning(f"Failed to get user profile: {e}")
        
        # Section 2: Bot traits (developed personality)
        if self.profile_manager and ctx.use_memory:
            try:
                bot_context = self.profile_manager.get_bot_profile_summary(ctx.bot_id)
                if bot_context:
                    builder.add_section(
                        "bot_traits",
                        f"## Your Developed Traits\n{bot_context}",
                        position=SectionPosition.BOT_TRAITS,
                    )
            except Exception as e:
                logger.warning(f"Failed to get bot profile: {e}")
        
        # Section 3: Base bot prompt
        if self.bot and self.bot.system_prompt:
            builder.add_section(
                "base_prompt",
                self.bot.system_prompt,
                position=SectionPosition.BASE_PROMPT,
            )
        
        # Section 4: Tools (full set for tool bots, read-only memory for non-tool memory bots)
        if ctx.use_tools:
            from ..tools import get_tools_prompt, get_tools_list
            include_models = self.model_lifecycle is not None
            tool_definitions = get_tools_list(
                include_search_tools=ctx.use_search,
                include_home_tools=self.home_client is not None,
                include_model_tools=include_models,
            )
            if self.memory_client is None:
                disallowed = {"memory", "history", "profile", "self"}
                tool_definitions = [t for t in tool_definitions if t.name not in disallowed]
            tool_format = self.config.get_tool_format(
                model_alias=getattr(self.llm_client, "model_alias", None)
            )
            ctx.tool_format = tool_format
            ctx.tool_definitions = tool_definitions
            tools_prompt = get_tools_prompt(
                tools=tool_definitions,
                tool_format=tool_format,
            )
            builder.add_section(
                "tools",
                tools_prompt,
                position=SectionPosition.TOOLS,
            )
        elif ctx.use_memory and self.memory_client:
            # Non-tool memory bots get a read-only memory search tool
            # so they can search memories on-demand instead of upfront injection
            from ..tools import get_tools_prompt
            from ..tools.definitions import MEMORY_TOOL
            tool_format = self.config.get_tool_format(
                model_alias=getattr(self.llm_client, "model_alias", None)
            )
            ctx.tool_format = tool_format
            ctx.tool_definitions = [MEMORY_TOOL]
            ctx.use_tools = True  # Enable tool loop for execution
            tools_prompt = get_tools_prompt(
                tools=[MEMORY_TOOL],
                tool_format=tool_format,
            )
            builder.add_section(
                "tools",
                tools_prompt,
                position=SectionPosition.TOOLS,
            )
        
        ctx.prompt_builder = builder
        
        if self.verbose:
            from rich.console import Console
            console = Console()
            console.print(builder.get_verbose_summary())
        
        ctx.record_output(PipelineStage.CONTEXT_BUILD, {
            "sections": [s.name for s in builder.sections],
        })
    
    def _stage_memory_retrieval(self, ctx: PipelineContext):
        """Stage 3: Cold-start memory priming for all memory-enabled bots.

        All memory bots use on-demand memory search via tools. This stage
        only injects a small number of memories when history is sparse
        (cold-start), giving the model immediate user context.
        """
        if not ctx.use_memory or not self.memory_client:
            return

        history_count = len(self.history_manager.messages) if self.history_manager else 0
        if history_count > 3:
            # Enough history — let the model use memory tool on-demand
            return

        # Cold-start: inject a small number of memories
        try:
            from .prompt_builder import SectionPosition
            results = self.memory_client.search(
                ctx.prompt, n_results=3,
                min_relevance=self.config.MEMORY_MIN_RELEVANCE,
            )
            if results:
                memories = [
                    {"content": m.content, "relevance": m.relevance,
                     "tags": m.tags, "importance": m.importance}
                    for m in results
                ]
                from ..memory.context_builder import build_memory_context_string
                user_name = None
                if self.profile_manager:
                    try:
                        from ..profiles import EntityType
                        profile, _ = self.profile_manager.get_or_create_profile(
                            EntityType.USER, ctx.user_id
                        )
                        user_name = profile.display_name
                    except Exception:
                        pass
                cold_ctx = build_memory_context_string(memories, user_name=user_name)
                if cold_ctx and ctx.prompt_builder:
                    ctx.prompt_builder.add_section(
                        "cold_start_memory",
                        cold_ctx,
                        position=SectionPosition.MEMORY_CONTEXT,
                    )
                    logger.debug(
                        f"Cold-start memory priming: {len(memories)} memories "
                        f"({len(cold_ctx)} chars)"
                    )
        except Exception as e:
            logger.warning(f"Cold-start memory retrieval failed: {e}")

        ctx.record_output(PipelineStage.MEMORY_RETRIEVAL, {
            "cold_start": True,
            "memory_count": len(ctx.memory_results),
        })
    
    def _stage_history_filter(self, ctx: PipelineContext):
        """Stage 4: Decide what conversation history to include.

        History is always included. The two-layer architecture (raw messages +
        summaries) with token budgeting handles context overflow properly.
        """
        ctx.include_history = True

        # Apply override if set (for testing)
        if "skip_history" in self._decision_overrides:
            ctx.skip_history = self._decision_overrides["skip_history"]
            ctx.include_history = not ctx.skip_history

        if self.debug:
            logger.debug(f"  include_history={ctx.include_history}")

        ctx.record_output(PipelineStage.HISTORY_FILTER, {
            "include_history": ctx.include_history,
        })
    
    def _stage_message_assembly(self, ctx: PipelineContext):
        """Stage 5: Assemble final messages list."""
        from ..models.message import Message
        
        messages = []
        
        # System message from prompt builder
        if ctx.prompt_builder:
            system_content = ctx.prompt_builder.build()
            if system_content:
                messages.append(Message(role="system", content=system_content))
                
                if self.debug:
                    logger.debug(f"  System message: {len(system_content)} chars")
        
        # Conversation history
        if ctx.include_history and self.history_manager:
            # Compute token budget from the client's effective context window
            max_context_tokens = getattr(self.config, 'MAX_CONTEXT_TOKENS', 0)
            if max_context_tokens <= 0 and self.llm_client:
                # Auto: reserve half context window for output, use rest for input
                ctx_window = getattr(self.llm_client, 'effective_context_window', 0)
                if ctx_window > 0:
                    max_output = getattr(self.llm_client, 'effective_max_tokens', 4096)
                    max_context_tokens = ctx_window - max_output

            history = self.history_manager.get_context_messages(
                max_tokens=max_context_tokens
            )
            history_count = 0
            for msg in history:
                # Include user, assistant, and summary messages (summary → system for API)
                if msg.role in ("user", "assistant", "summary"):
                    messages.append(msg)
                    history_count += 1
            
            if self.debug:
                logger.debug(f"  Added {history_count} history messages (budget: {max_context_tokens} tokens)")
        
        # Current user prompt (if not already in history)
        # History manager adds the message before this, so we check
        if not ctx.include_history and ctx.prompt:
            messages.append(Message(role="user", content=ctx.prompt))
        
        ctx.messages = messages
        
        ctx.record_output(PipelineStage.MESSAGE_ASSEMBLY, {
            "message_count": len(messages),
            "roles": [m.role for m in messages],
        })
    
    def _stage_execute(self, ctx: PipelineContext):
        """Stage 6: Execute LLM query (with or without tools)."""
        if not self.llm_client:
            logger.error("No LLM client configured")
            ctx.response = ""
            return
        
        if ctx.use_tools and (self.memory_client or self.home_client):
            # Use tool loop
            from ..tools import query_with_tools
            response, tool_context, tool_call_details = query_with_tools(
                messages=ctx.messages,
                client=self.llm_client,
                memory_client=self.memory_client,
                profile_manager=self.profile_manager,
                search_client=self.search_client,
                home_client=self.home_client,
                model_lifecycle=self.model_lifecycle,
                config=self.config,
                user_id=ctx.user_id,
                bot_id=ctx.bot_id,
                stream=ctx.stream,
                tool_format=ctx.tool_format,
                tools=ctx.tool_definitions,
                adapter=self.adapter,
                history_manager=self.history_manager,
            )
            
            ctx.response = response
            ctx.tool_context = tool_context
            ctx.tool_call_details = tool_call_details
        else:
            # Direct query
            ctx.response = self.llm_client.query(
                ctx.messages,
                plaintext_output=ctx.plaintext_output,
                stream=ctx.stream,
            )
        
        if self.debug:
            logger.debug(f"  Response: {len(ctx.response)} chars")
        
        ctx.record_output(PipelineStage.EXECUTE, {
            "response_length": len(ctx.response),
            "used_tools": bool(ctx.tool_context),
        })
    
    def _stage_post_process(self, ctx: PipelineContext):
        """Stage 7: Post-processing (history save)."""
        if not ctx.response:
            return

        # Save to history
        if self.history_manager:
            self.history_manager.add_message("assistant", ctx.response)

            # Save tool context if present (with timestamp for freshness)
            if ctx.tool_context:
                from datetime import datetime
                ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                self.history_manager.add_message(
                    "system", f"[Tool Results @ {ts}]\n{ctx.tool_context}"
                )

        ctx.record_output(PipelineStage.POST_PROCESS, {
            "saved_to_history": self.history_manager is not None,
            "extraction_triggered": False,
        })
    
    def _log_summary(self, ctx: PipelineContext):
        """Log a summary of the pipeline execution."""
        total_ms = sum(ctx.stage_timings.values())
        
        logger.info(f"Pipeline completed in {total_ms:.1f}ms")
        for stage, ms in ctx.stage_timings.items():
            logger.info(f"  {stage}: {ms:.1f}ms")
