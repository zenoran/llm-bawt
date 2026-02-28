"""Background task service and chat execution runtime."""

import asyncio
import json
import os
import threading
import time
import uuid
from datetime import datetime
from queue import PriorityQueue
from typing import Any, AsyncIterator
from urllib.parse import urlparse

from ..bots import BotManager, StreamingEmoteFilter, get_bot, strip_emotes
from ..utils.config import Config
from ..utils.paths import resolve_log_dir
from .logging import RequestContext, generate_request_id, get_service_logger
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ServiceStatusResponse,
    UsageInfo,
)
from .tasks import Task, TaskResult, TaskStatus, TaskType

log = get_service_logger(__name__)

SERVICE_VERSION = "0.1.0"


def _is_tcp_listening(host: str, port: int) -> bool:
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


def _message_to_dict(msg: Any) -> dict[str, Any]:
    """Normalize message objects for JSON logging."""
    if hasattr(msg, "to_dict"):
        value = msg.to_dict()
        if isinstance(value, dict):
            if not value.get("id") and value.get("db_id"):
                value = dict(value)
                value["id"] = value.get("db_id")
            return value
        return {"value": value}
    if hasattr(msg, "role"):
        return {
            "role": getattr(msg, "role", "unknown"),
            "content": getattr(msg, "content", ""),
            "timestamp": getattr(msg, "timestamp", 0),
            "id": getattr(msg, "db_id", None),
        }
    if isinstance(msg, dict):
        return msg
    return {"value": str(msg)}


def _normalize_tool_call_details(tool_calls: list[dict] | None) -> list[dict]:
    """Normalize tool-call details into request/response shape."""
    if not tool_calls:
        return []

    normalized: list[dict] = []
    for idx, item in enumerate(tool_calls, start=1):
        if not isinstance(item, dict):
            normalized.append({"index": idx, "name": "unknown", "arguments": {}, "result": str(item)})
            continue
        name = item.get("tool") or item.get("name") or "unknown"
        args = item.get("parameters") or item.get("arguments") or {}
        result = item.get("result") or item.get("response") or ""
        normalized.append(
            {
                "index": idx,
                "iteration": item.get("iteration", 1),
                "name": name,
                "arguments": args if isinstance(args, dict) else {"raw": args},
                "result": result,
            }
        )
    return normalized


def _write_debug_turn_log(
    prepared_messages: list,
    user_prompt: str,
    response: str,
    model: str,
    bot_id: str,
    user_id: str,
    tool_calls: list[dict] | None = None,
) -> None:
    """Write the current turn's request/response data to a debug log file.

    Called when debug logging is enabled. Overwrites the file on each turn
    to show the most recent request/response for review.
    """
    try:
        logs_dir = resolve_log_dir()
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / "debug_turn.txt"

        # Build the log content
        lines = []
        lines.append("=" * 80)
        lines.append(f"DEBUG TURN LOG - {datetime.now().isoformat()}")
        lines.append(f"Model: {model}")
        lines.append(f"Bot: {bot_id}")
        lines.append(f"User: {user_id}")
        lines.append("=" * 80)
        lines.append("")

        # Request data - all context messages
        lines.append("─" * 40)
        lines.append("REQUEST MESSAGES")
        lines.append("─" * 40)
        for i, msg in enumerate(prepared_messages):
            role = msg.role if hasattr(msg, 'role') else msg.get('role', 'unknown')
            content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
            timestamp = msg.timestamp if hasattr(msg, 'timestamp') else msg.get('timestamp', 0)
            lines.append(f"\n[{i}] Role: {role}")
            lines.append(f"    Timestamp: {timestamp}")
            lines.append(f"    Content ({len(content)} chars):")
            lines.append("    " + "─" * 36)
            # Indent content for readability
            for content_line in str(content).split("\n"):
                lines.append(f"    {content_line}")
            lines.append("")

        # Tool calls section (between request and response)
        if tool_calls:
            total_calls = len(tool_calls)
            iterations = max((tc.get("iteration", 1) for tc in tool_calls), default=1)
            lines.append("─" * 40)
            lines.append(f"TOOL CALLS ({total_calls} call{'s' if total_calls != 1 else ''} across {iterations} iteration{'s' if iterations != 1 else ''})")
            lines.append("─" * 40)
            for idx, tc in enumerate(tool_calls, 1):
                lines.append("")
                lines.append(f"[{idx}] Tool: {tc.get('tool', 'unknown')}")
                params = tc.get('parameters', {})
                if isinstance(params, dict):
                    for pk, pv in params.items():
                        lines.append(f"    {pk}: {pv}")
                else:
                    lines.append(f"    Parameters: {params}")
                result = tc.get('result', '')
                lines.append(f"    Result ({len(result)} chars):")
                lines.append("    " + "─" * 36)
                for result_line in str(result)[:2000].split("\n"):
                    lines.append(f"    {result_line}")
            lines.append("")

        # Response data
        lines.append("─" * 40)
        lines.append("RESPONSE")
        lines.append("─" * 40)
        lines.append(f"Length: {len(response)} chars")
        lines.append("")
        lines.append(response)
        lines.append("")

        # Also dump as JSON for machine parsing
        lines.append("─" * 40)
        lines.append("JSON FORMAT (for parsing)")
        lines.append("─" * 40)

        json_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "bot_id": bot_id,
            "user_id": user_id,
            "request": [_message_to_dict(msg) for msg in prepared_messages],
            "tool_calls": _normalize_tool_call_details(tool_calls),
            "response": response,
        }
        lines.append(json.dumps(json_data, indent=2, ensure_ascii=False, default=str))

        # Write to file (overwrite)
        log_file.write_text("\n".join(lines), encoding="utf-8")
        log.debug(f"Debug turn log written to: {log_file}")

    except Exception as e:
        log.warning(f"Failed to write debug turn log: {e}")


_memory_mcp_thread: threading.Thread | None = None


def _ensure_memory_mcp_server(config: Config) -> None:
    """Ensure an MCP memory server is running and configure the service to use it.

    This makes memory retrieval happen via MCP tool calls (e.g. tools/search_memories),
    which can be logged distinctly from embedded DB access.
    """
    global _memory_mcp_thread

    # Default to local MCP memory server for llm-service if not configured.
    if not getattr(config, "MEMORY_SERVER_URL", None):
        config.MEMORY_SERVER_URL = "http://127.0.0.1:8001"

    parsed = urlparse(config.MEMORY_SERVER_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8001

    # If something is already listening, assume it's the memory MCP server.
    if _is_tcp_listening(host, port):
        log.info("Memory MCP server already listening at %s", config.MEMORY_SERVER_URL)
        return

    # Start the MCP memory server in-process (HTTP transport) on a daemon thread.
    def _run():
        try:
            from ..memory_server.server import run_server
            run_server(transport="streamable-http", host=host, port=port)
        except Exception as e:
            log.error("Failed to start MCP memory server: %s", e)

    _memory_mcp_thread = threading.Thread(target=_run, daemon=True, name="memory-mcp")
    _memory_mcp_thread.start()
    log.info("Started MCP memory server at %s", config.MEMORY_SERVER_URL)

class BackgroundService:
    """
    The main background service that processes async tasks.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.start_time = time.time()
        self.tasks_processed = 0
        self._task_queue: PriorityQueue = PriorityQueue()
        self._results: dict[str, TaskResult] = {}
        self._submitted_tasks: dict[str, Task] = {}
        self._result_events: dict[str, asyncio.Event] = {}
        self._shutdown_event = asyncio.Event()
        self._worker_task: asyncio.Task | None = None
        
        # Initialize model lifecycle manager (singleton)
        from ..core.model_lifecycle import get_model_lifecycle
        self._model_lifecycle = get_model_lifecycle(config)
        
        # Cached LLMBawt instances keyed by (model_alias, bot_id, user_id)
        self._llm_bawt_cache: dict[tuple[str, str, str], Any] = {}
        self._cache_lock = asyncio.Lock()
        
        # Cached LLM clients keyed by model_alias only
        # This prevents loading the same model (especially GGUF) multiple times
        # when different bot contexts need the same underlying model
        self._client_cache: dict[str, Any] = {}
        
        # Lock to serialize LLM calls - prevents CUDA crashes from concurrent access
        # llama-cpp-python is NOT thread-safe for concurrent inference
        self._llm_lock = asyncio.Lock()
        
        # Single-threaded executor for LLM calls - ensures only one runs at a time
        from concurrent.futures import ThreadPoolExecutor
        self._llm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
        
        # Cancellation support: when a new request comes in, cancel the current one
        # This handles the case where UI sends partial transcriptions that build up
        self._current_generation_cancel: threading.Event | None = None
        self._generation_done: threading.Event | None = None  # Signals when generation finishes
        self._cancel_lock = threading.Lock()
        
        # Memory client cache keyed by bot_id
        self._memory_clients: dict[tuple[str, str], Any] = {}
        from .turn_logs import TurnLogStore
        self._turn_log_store = TurnLogStore(config, ttl_hours=168)

        # Extraction client (API model like Grok, separate from chat model)
        self._extraction_client: Any | None = None
        self._extraction_client_model: str | None = None
        
        # Session model overrides: when user switches model via tool, 
        # remember it for the rest of the session. Keyed by (bot_id, user_id).
        self._session_model_overrides: dict[tuple[str, str], str] = {}
        
        # Model configuration
        self._available_models: list[str] = []
        self._default_model: str | None = None
        self._default_bot: str = config.DEFAULT_BOT or "nova"
        self._load_available_models()
        
        # Register callback for model unload - clears caches when model changes
        self._model_lifecycle.on_model_unloaded(self._on_model_unloaded)
    
    def _on_model_unloaded(self, model_alias: str):
        """Called when a model is unloaded - clears related caches."""
        log.debug(f"Model '{model_alias}' unloaded - clearing caches")
        
        # Clear client cache for this model
        if model_alias in self._client_cache:
            del self._client_cache[model_alias]
        
        # Clear all LLMBawt instances that use this model
        keys_to_remove = [
            key for key in self._llm_bawt_cache
            if key[0] == model_alias
        ]
        for key in keys_to_remove:
            del self._llm_bawt_cache[key]
        
        log.debug(f"Cleared {len(keys_to_remove)} cached instances for model '{model_alias}'")

    def invalidate_bot_instances(self, bot_id: str) -> int:
        """Invalidate cached ServiceLLMBawt instances for a bot across models/users."""
        normalized_bot_id = (bot_id or "").strip().lower()
        if not normalized_bot_id:
            return 0
        keys_to_remove = [
            key for key in self._llm_bawt_cache
            if key[1] == normalized_bot_id
        ]
        for key in keys_to_remove:
            del self._llm_bawt_cache[key]
        if keys_to_remove:
            log.debug(
                "Cleared %s cached instances for bot '%s'",
                len(keys_to_remove),
                normalized_bot_id,
            )
        return len(keys_to_remove)

    def invalidate_all_instances(self) -> int:
        """Invalidate all cached ServiceLLMBawt instances."""
        cleared = len(self._llm_bawt_cache)
        self._llm_bawt_cache.clear()
        if cleared:
            log.debug("Cleared all cached instances (%s)", cleared)
        return cleared

    def clear_session_model_overrides(self, bot_id: str | None = None, user_id: str | None = None) -> int:
        """Clear session model overrides, optionally scoped by bot and/or user."""
        if not self._session_model_overrides:
            return 0

        normalized_bot = (bot_id or "").strip().lower() if bot_id is not None else None
        normalized_user = (user_id or "").strip() if user_id is not None else None

        keys_to_remove: list[tuple[str, str]] = []
        for key in self._session_model_overrides:
            key_bot, key_user = key
            if normalized_bot is not None and key_bot != normalized_bot:
                continue
            if normalized_user is not None and key_user != normalized_user:
                continue
            keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._session_model_overrides[key]

        if keys_to_remove:
            scope = []
            if normalized_bot is not None:
                scope.append(f"bot={normalized_bot}")
            if normalized_user is not None:
                scope.append(f"user={normalized_user}")
            detail = " ".join(scope) if scope else "all sessions"
            log.info("Cleared %s session model override(s) for %s", len(keys_to_remove), detail)

        return len(keys_to_remove)
    
    def _load_available_models(self):
        """Load list of available models from config."""
        models = self.config.defined_models.get("models", {})
        self._available_models = list(models.keys())
        log.debug(f"Loaded {len(self._available_models)} models from config")

        # Inject virtual model definitions for agent-backend bots.
        # Each backend name (e.g. "openclaw") becomes a model alias with
        # type: "agent_backend" so the entire pipeline flows normally.
        bot_manager = BotManager(self.config)
        for bot in bot_manager.list_bots():
            backend_name = getattr(bot, "agent_backend", None)
            if not backend_name:
                continue
            backend_config = getattr(bot, "agent_backend_config", {}) or {}
            if backend_name not in models:
                virtual_def = {
                    "type": "agent_backend",
                    "backend": backend_name,
                    "bot_config": backend_config,
                    "tool_support": "none",
                }
                models[backend_name] = virtual_def
                if backend_name not in self._available_models:
                    self._available_models.append(backend_name)
                log.info(
                    "Registered virtual model '%s' for agent backend (bot=%s)",
                    backend_name,
                    bot.slug,
                )
            # Ensure the bot's default_model points at its backend
            if not bot.default_model:
                bot.default_model = backend_name

        # Set default model from bot/config selection or use first available
        selection = bot_manager.select_model(None, bot_slug=self._default_bot)
        self._default_model = selection.alias
        if not self._default_model and self._available_models:
            self._default_model = self._available_models[0]

    def _resolve_request_model(
        self,
        requested_model: str | None,
        bot_id: str,
        local_mode: bool,
    ) -> tuple[str, list[str]]:
        """Resolve the model alias for a request using shared bot/config logic.

        Returns:
            Tuple of (resolved_model_alias, list_of_warnings)
        """
        warnings: list[str] = []
        bot_manager = BotManager(self.config)
        selection = bot_manager.select_model(requested_model, bot_slug=bot_id, local_mode=local_mode)
        model_alias = selection.alias

        if model_alias and model_alias in self._available_models:
            return model_alias, warnings

        # If explicit model is invalid, warn and fall back
        if model_alias and model_alias not in self._available_models:
            fallback = bot_manager.select_model(None, bot_slug=bot_id, local_mode=local_mode)
            if fallback.alias and fallback.alias in self._available_models:
                msg = f"Model '{model_alias}' not available on service, using '{fallback.alias}'"
                log.warning(msg)
                warnings.append(msg)
                return fallback.alias, warnings

        if self._available_models:
            fallback_model = self._available_models[0]
            msg = f"Model '{model_alias}' not available on service, using '{fallback_model}'"
            log.warning(msg)
            warnings.append(msg)
            return fallback_model, warnings

        raise ValueError("No models available on service.")
    
    async def _start_generation(self) -> tuple[threading.Event, threading.Event]:
        """Start a new generation, cancelling and waiting for any in-progress one.
        
        Returns:
            tuple of (cancel_event, done_event):
            - cancel_event: The generation should check this periodically and abort if set
            - done_event: The generation MUST set this when complete (in finally block)
        
        This ensures only one generation runs at a time by:
        1. Signalling the previous generation to cancel
        2. Waiting for it to actually finish (up to 5 seconds)
        3. Then allowing the new generation to start
        """
        loop = asyncio.get_event_loop()
        
        with self._cancel_lock:
            # Cancel any existing generation and wait for it to finish
            if self._current_generation_cancel is not None:
                log.debug("Cancelling previous generation for new request")
                self._current_generation_cancel.set()
                
                # Wait for the previous generation to signal it's done
                if self._generation_done is not None:
                    done_event = self._generation_done
                    # Release lock while waiting to avoid deadlock
                    self._cancel_lock.release()
                    try:
                        # Wait in executor to avoid blocking the event loop
                        await loop.run_in_executor(
                            None,  # Use default executor
                            lambda: done_event.wait(timeout=5.0)
                        )
                    finally:
                        self._cancel_lock.acquire()
            
            # Create new events for this generation
            cancel_event = threading.Event()
            done_event = threading.Event()
            self._current_generation_cancel = cancel_event
            self._generation_done = done_event
            return cancel_event, done_event
    
    def _end_generation(self, cancel_event: threading.Event, done_event: threading.Event):
        """Mark a generation as complete."""
        # Signal that we're done FIRST (before acquiring lock)
        done_event.set()
        
        with self._cancel_lock:
            # Only clear if this is still the current generation
            if self._current_generation_cancel is cancel_event:
                self._current_generation_cancel = None
                self._generation_done = None

    def _persist_turn_log(
        self,
        *,
        turn_id: str,
        request_id: str | None,
        path: str,
        stream: bool,
        model: str | None,
        bot_id: str,
        user_id: str,
        status: str,
        latency_ms: float | None,
        user_prompt: str,
        prepared_messages: list,
        response_text: str,
        tool_calls: list[dict] | None = None,
        error_text: str | None = None,
    ) -> None:
        """Persist one turn record to short-lived DB storage."""
        try:
            request_payload = {
                "messages": [_message_to_dict(msg) for msg in prepared_messages],
            }
            self._turn_log_store.save_turn(
                turn_id=turn_id,
                request_id=request_id,
                path=path,
                stream=stream,
                model=model,
                bot_id=bot_id,
                user_id=user_id,
                status=status,
                latency_ms=latency_ms,
                user_prompt=user_prompt,
                request_payload=request_payload,
                response_text=response_text,
                tool_calls=_normalize_tool_call_details(tool_calls),
                error_text=error_text,
            )
        except Exception as e:
            log.debug("Failed to persist turn log: %s", e)
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
    
    @property
    def model_lifecycle(self):
        """Get the model lifecycle manager for tool access."""
        return self._model_lifecycle
    
    def get_memory_client(self, bot_id: str, user_id: str | None = None):
        """Get or create memory client for a bot/user pair."""
        cache_key = (bot_id, user_id)

        if cache_key not in self._memory_clients:
            try:
                from ..memory_server.client import get_memory_client
                self._memory_clients[cache_key] = get_memory_client(
                    config=self.config,
                    bot_id=bot_id,
                    user_id=user_id,
                    server_url=getattr(self.config, "MEMORY_SERVER_URL", None),
                )
                log.memory_operation("client_init", bot_id, details="MemoryClient created")
            except Exception as e:
                log.warning(f"Memory client unavailable for {bot_id}: {e}")
                self._memory_clients[cache_key] = None
        return self._memory_clients.get(cache_key)
    
    def _get_llm_bawt(self, model_alias: str, bot_id: str, user_id: str, local_mode: bool = False):
        """Get or create an LLMBawt instance with caching.
        
        This method enforces single-model loading through the ModelLifecycleManager.
        If a different model is requested, the current model will be unloaded first.
        
        Model selection priority:
        1. Pending model switch (from switch_model tool) - becomes the new session model
        2. Current session model override (from previous switch)
        3. Model from API request
        """
        from .core import ServiceLLMBawt
        
        # Session key for model overrides (per bot+user)
        session_key = (bot_id, user_id)
        
        # Check for pending model switch (from switch_model tool)
        pending = self._model_lifecycle.clear_pending_switch()
        if pending:
            log.info(f"🔄 Switching to model: {pending}")
            # Store as session override so subsequent requests use this model
            self._session_model_overrides[session_key] = pending
            model_alias = pending
        elif session_key in self._session_model_overrides:
            # Use the session model override from a previous switch
            model_alias = self._session_model_overrides[session_key]
            log.debug(f"Using session model override: {model_alias}")
        
        cache_key = (model_alias, bot_id, user_id)
        
        # Check if we need to switch models (different model requested)
        current_model = self._model_lifecycle.current_model
        if current_model and current_model != model_alias:
            log.info(f"🔄 Model: {current_model} → {model_alias}")
            # Unloading will trigger _on_model_unloaded callback which clears caches
            self._model_lifecycle.unload_current_model()
        
        if cache_key in self._llm_bawt_cache:
            log.cache_hit("llm_bawt", f"{model_alias}/{bot_id}/{user_id}")
            log.debug(f"Reusing cached ServiceLLMBawt instance for {cache_key}")
            return self._llm_bawt_cache[cache_key]
        
        log.cache_miss("llm_bawt", f"{model_alias}/{bot_id}/{user_id}")
        
        # Check if we already have a client for this model in the client cache
        # If so, reuse it to avoid reloading GGUF models into VRAM
        existing_client = self._client_cache.get(model_alias)
        
        # Get model type for logging
        model_def = self.config.defined_models.get("models", {}).get(model_alias, {})
        model_type = model_def.get("type", "unknown")
        
        # Only log model loading if we don't have the client cached
        if not existing_client:
            log.model_loading(model_alias, model_type, cached=False)
        else:
            log.model_loading(model_alias, model_type, cached=True)
        try:
            # Create a copy of config for each ServiceLLMBawt instance
            # This is necessary because it modifies config.SYSTEM_MESSAGE
            # based on the bot's system prompt
            instance_config = self.config.model_copy(deep=True)
            llm_bawt = ServiceLLMBawt(
                resolved_model_alias=model_alias,
                config=instance_config,
                local_mode=local_mode,
                bot_id=bot_id,
                user_id=user_id,
                existing_client=existing_client,  # Reuse cached client if available
            )
            self._llm_bawt_cache[cache_key] = llm_bawt
            
            # Also cache the client for future reuse by extraction tasks
            if model_alias not in self._client_cache:
                self._client_cache[model_alias] = llm_bawt.client
            # Note: BaseLLMBawt.__init__ already registers with lifecycle manager
            
        except Exception as e:
            log.model_error(model_alias, str(e))
            raise
        
        return self._llm_bawt_cache[cache_key]
    
    def get_client(self, model_alias: str):
        """Get LLM client for a given model (for extraction tasks).
        
        Uses a dedicated client cache to avoid reloading models.
        GGUF models especially are expensive to load into VRAM,
        so we cache the client independently from LLMBawt instances.
        """
        if model_alias in self._client_cache:
            log.cache_hit("llm_client", model_alias)
            return self._client_cache[model_alias]
        
        log.cache_miss("llm_client", model_alias)
        
        # Check if we already have an LLMBawt instance with this model
        # and can reuse its client
        for (cached_model, _, _), llm_bawt in self._llm_bawt_cache.items():
            if cached_model == model_alias:
                log.debug(f"Reusing client from existing LLMBawt instance for '{model_alias}'")
                self._client_cache[model_alias] = llm_bawt.client
                return llm_bawt.client
        
        # Need to create a new client - use spark bot (no memory overhead)
        log.debug(f"Creating new client for model '{model_alias}' (extraction context)")
        llm_bawt = self._get_llm_bawt(model_alias, "spark", "system", local_mode=True)
        self._client_cache[model_alias] = llm_bawt.client
        return llm_bawt.client
    
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
        """
        Handle an OpenAI-compatible chat completion request.
        
        This is the main entry point for the API. It:
        1. Uses llm_bawt's internal history + memory for context
        2. Augments with bot system prompt and memory context (if enabled)
        3. Runs blocking LLM calls in a thread pool
        4. Stores messages and extracts memories for future use
        """
        # Create request context for logging
        if request.client_system_context is not None:
            req_path = f"/v1/botchat/{request.bot_id}/{request.user}/chat/completions"
        else:
            req_path = "/v1/chat/completions"
        ctx = RequestContext(
            request_id=generate_request_id(),
            method="POST",
            path=req_path,
            model=request.model,
            bot_id=request.bot_id,
            user_id=request.user,
            stream=False,
        )

        # Log incoming request (verbose mode will show the full payload)
        log.api_request(ctx, request.model_dump(exclude_none=True))
        
        bot_id = request.bot_id or self._default_bot
        user_id = request.user or self.config.DEFAULT_USER
        local_mode = not request.augment_memory

        # Resolve model using shared bot/config logic
        # Agent-backend bots resolve to their virtual model (e.g. "openclaw")
        try:
            model_alias, model_warnings = self._resolve_request_model(request.model, bot_id, local_mode)
        except Exception as e:
            log.api_error(ctx, str(e), 400)
            raise

        ctx.model = model_alias

        # Debug: Show memory settings
        log.debug(
            f"Memory settings: augment_memory={request.augment_memory}, "
            f"local_mode={local_mode}, bot={bot_id}, user={user_id}"
        )
        
        # Get cached LLMBawt instance
        llm_bawt = self._get_llm_bawt(model_alias, bot_id, user_id, local_mode)
        
        # Get the user's prompt (last user message)
        user_prompt = ""
        for m in reversed(request.messages):
            if m.role == "user":
                user_prompt = m.content or ""
                break
        
        if not user_prompt:
            raise ValueError("No user message found in request")
        
        # Start new generation (cancels and waits for any previous one)
        cancel_event, done_event = await self._start_generation()
        turn_log_id = f"turn-{uuid.uuid4().hex}"
        
        try:
            # Run the blocking query in single-thread executor
            loop = asyncio.get_event_loop()
            llm_start_time = time.time()
            cancelled = False
            
            def _do_query():
                nonlocal cancelled
                # Check if already cancelled before starting
                if cancel_event.is_set():
                    cancelled = True
                    return ""
                
                # Inject client-supplied system context (e.g. HA device list)
                llm_bawt._client_system_context = request.client_system_context
                llm_bawt._ha_mode = request.ha_mode

                # Prepare messages with history and memory context
                prepared_messages = llm_bawt.prepare_messages_for_query(user_prompt)

                # Log what we're sending to the LLM (verbose mode)
                log.llm_context(prepared_messages)

                # Execute the query with prepared messages
                response, tool_context, tool_call_details = llm_bawt.execute_llm_query(
                    prepared_messages,
                    plaintext_output=True,
                    stream=False,
                )

                # For agent backend clients, extract tool call details
                # from the structured result stashed by the backend.
                from ..clients.agent_backend_client import AgentBackendClient
                if isinstance(llm_bawt.client, AgentBackendClient):
                    for tc in llm_bawt.client.get_tool_calls():
                        result_payload = tc.get("result")
                        if result_payload is None:
                            result_payload = "Result not exposed by OpenClaw API (see assistant response)."
                        tool_call_details.append({
                            "iteration": 1,
                            "tool": tc.get("display_name") or tc.get("name", "unknown"),
                            "parameters": tc.get("arguments", {}),
                            "result": result_payload,
                        })

                # Write debug turn log if enabled (check config or env var)
                if self.config.DEBUG_TURN_LOG or os.environ.get("LLM_BAWT_DEBUG_TURN_LOG"):
                    _write_debug_turn_log(
                        prepared_messages=prepared_messages,
                        user_prompt=user_prompt,
                        response=response,
                        model=model_alias,
                        bot_id=bot_id,
                        user_id=user_id,
                        tool_calls=tool_call_details,
                    )

                # Check if cancelled during generation
                if cancel_event.is_set():
                    log.info("Generation cancelled - newer request received")
                    cancelled = True
                    return ""

                # Finalize (add to history, trigger memory extraction)
                llm_bawt.finalize_response(response, tool_context)

                self._persist_turn_log(
                    turn_id=turn_log_id,
                    request_id=ctx.request_id,
                    path=ctx.path,
                    stream=False,
                    model=model_alias,
                    bot_id=bot_id,
                    user_id=user_id,
                    status="ok",
                    latency_ms=(time.time() - llm_start_time) * 1000,
                    user_prompt=user_prompt,
                    prepared_messages=prepared_messages,
                    response_text=response,
                    tool_calls=tool_call_details,
                )

                return response
            
            response_text = await loop.run_in_executor(self._llm_executor, _do_query)
            llm_elapsed_ms = (time.time() - llm_start_time) * 1000
            
            # If cancelled, return empty response (the new request will handle it)
            if cancelled:
                return ChatCompletionResponse(
                    model=model_alias,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=""),
                            finish_reason="cancelled",
                        )
                    ],
                    usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                )
        finally:
            self._end_generation(cancel_event, done_event)
        
        # Post-process for voice_optimized bots (strip emotes for TTS)
        bot = get_bot(bot_id)
        if bot and bot.voice_optimized:
            original_len = len(response_text)
            response_text = strip_emotes(response_text)
            if len(response_text) != original_len:
                log.debug(f"Stripped emotes for TTS: {original_len} -> {len(response_text)} chars")
        
        # Estimate token counts (rough approximation: 1 token ≈ 4 characters)
        prompt_text = " ".join(m.content or "" for m in request.messages)
        prompt_tokens = len(prompt_text) // 4
        completion_tokens = len(response_text) // 4
        total_tokens = prompt_tokens + completion_tokens
        
        # Log response (verbose shows content summary with tokens/sec)
        log.llm_response(response_text, tokens=completion_tokens, elapsed_ms=llm_elapsed_ms)
        log.api_response(ctx, status=200, tokens=total_tokens)
        
        # Build response
        response = ChatCompletionResponse(
            model=model_alias,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )
        
        return response

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """
        Handle a streaming chat completion request.
        
        Uses llm_bawt's internal history + memory for context.
        Yields Server-Sent Events (SSE) formatted chunks.
        """
        import json
        # Create request context for logging
        if request.client_system_context is not None:
            req_path = f"/v1/botchat/{request.bot_id}/{request.user}/chat/completions"
        else:
            req_path = "/v1/chat/completions"
        ctx = RequestContext(
            request_id=generate_request_id(),
            method="POST",
            path=req_path,
            model=request.model,
            bot_id=request.bot_id,
            user_id=request.user,
            stream=True,
        )

        # Log incoming request (verbose mode will show the full payload)
        log.api_request(ctx, request.model_dump(exclude_none=True))
        
        bot_id = request.bot_id or self._default_bot
        user_id = request.user or self.config.DEFAULT_USER
        local_mode = not request.augment_memory

        # Resolve model using shared bot/config logic
        # Agent-backend bots resolve to their virtual model (e.g. "openclaw")
        try:
            model_alias, model_warnings = self._resolve_request_model(request.model, bot_id, local_mode)
        except Exception as e:
            log.api_error(ctx, str(e), 400)
            raise

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # Get cached LLMBawt instance
        try:
            llm_bawt = self._get_llm_bawt(model_alias, bot_id, user_id, local_mode)
        except Exception as e:
            log.api_error(ctx, str(e), 500)
            warning_data = {
                "object": "service.warning",
                "model": model_alias,
                "warnings": [f"request_failed: {e}"],
            }
            yield f"data: {json.dumps(warning_data)}\n\n"
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_alias,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        # Get the user's prompt (last user message)
        user_prompt = ""
        for m in reversed(request.messages):
            if m.role == "user":
                user_prompt = m.content or ""
                break
        
        if not user_prompt:
            warning_data = {
                "object": "service.warning",
                "model": model_alias,
                "warnings": ["request_failed: No user message found in request"],
            }
            yield f"data: {json.dumps(warning_data)}\n\n"
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_alias,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        chunk_queue: asyncio.Queue = asyncio.Queue()
        full_response_holder = [""]  # Use list to allow mutation in nested function
        tool_context_holder = [""]  # Store tool context from native tool calls
        tool_call_details_holder: list[dict] = []
        timing_holder = [0.0, 0.0]  # [start_time, end_time]
        cancelled_holder = [False]  # Track if we were cancelled
        
        # Capture the event loop before entering the thread
        loop = asyncio.get_running_loop()
        
        # Start new generation (cancels and waits for any previous one)
        cancel_event, done_event = await self._start_generation()
        turn_log_id = f"turn-{uuid.uuid4().hex}"
        
        def _stream_to_queue():
            """Run streaming in a thread and push chunks to the async queue."""
            try:
                # Check if already cancelled before starting
                if cancel_event.is_set():
                    cancelled_holder[0] = True
                    return
                
                # Inject client-supplied system context (e.g. HA device list)
                llm_bawt._client_system_context = request.client_system_context
                llm_bawt._ha_mode = request.ha_mode

                # Use llm_bawt.prepare_messages_for_query to get full context
                # (history from DB + memory + system prompt)
                messages = llm_bawt.prepare_messages_for_query(user_prompt)
                
                # Log what we're sending to the LLM (verbose mode)
                log.llm_context(messages)
                
                # Track when first token arrives
                timing_holder[0] = time.time()
                
                # Choose streaming method based on whether bot uses tools
                if llm_bawt.bot.uses_tools and (llm_bawt.memory or llm_bawt.home_client or llm_bawt.ha_native_client):
                    # Check if client supports native streaming with tools (OpenAI)
                    use_native_streaming = (
                        llm_bawt.client.supports_native_tools()
                        and llm_bawt.tool_format in ("native", "NATIVE_OPENAI")
                        and hasattr(llm_bawt.client, "stream_with_tools")
                    )

                    # Resolve per-bot generation parameters
                    gen_kwargs = llm_bawt._get_generation_kwargs()

                    if use_native_streaming:
                        # Native streaming with tools - streams content AND handles tool calls
                        from ..tools.executor import ToolExecutor
                        from ..tools.formats import get_format_handler
                        from ..models.message import Message as Msg

                        log.debug("Using native streaming with tools")

                        tool_definitions = llm_bawt._get_tool_definitions()
                        handler = get_format_handler(llm_bawt.tool_format)
                        tools_schema = handler.get_tools_schema(tool_definitions)

                        # Log tool names for debugging
                        tool_names = [t.get("function", {}).get("name", "?") for t in tools_schema]
                        ha_tools = [n for n in tool_names if n.startswith("Hass") or n in ("GetLiveContext",)]
                        log.info(f"📋 {len(tools_schema)} tools in schema ({len(ha_tools)} HA: {', '.join(ha_tools[:5])}{'...' if len(ha_tools) > 5 else ''})")

                        executor = ToolExecutor(
                            memory_client=llm_bawt.memory,
                            profile_manager=llm_bawt.profile_manager,
                            search_client=llm_bawt.search_client,
                            home_client=llm_bawt.home_client,
                            ha_native_client=llm_bawt.ha_native_client,
                            model_lifecycle=llm_bawt.model_lifecycle,
                            config=llm_bawt.config,
                            user_id=llm_bawt.user_id,
                            bot_id=llm_bawt.bot_id,
                        )

                        def native_stream_with_tool_loop():
                            """Stream with native tool support, handling tool calls inline."""
                            import json as _json
                            from ..tools.parser import ToolCall

                            current_msgs = list(messages)
                            max_iterations = 3 if llm_bawt._ha_mode else 5
                            has_executed_tools = False

                            for iteration in range(max_iterations):
                                # After first tool execution, don't pass tools_schema
                                # so the model generates a text response instead of
                                # looping on more tool calls.
                                current_tools = None if has_executed_tools else tools_schema

                                for item in llm_bawt.client.stream_with_tools(
                                    current_msgs,
                                    tools_schema=current_tools,
                                    tool_choice="auto",
                                    **gen_kwargs,
                                ):
                                    if isinstance(item, str):
                                        # Content chunk - yield to user immediately
                                        yield item
                                    elif isinstance(item, dict) and "tool_calls" in item:
                                        # Tool calls at end of stream
                                        tool_calls = item["tool_calls"]
                                        content = item.get("content", "")

                                        if not tool_calls:
                                            return  # No tools, done

                                        # Execute tools and log with their arguments
                                        tool_results = []
                                        for tc in tool_calls:
                                            func = tc.get("function", {})
                                            name = func.get("name", "")
                                            args_str = func.get("arguments", "{}")
                                            try:
                                                args = _json.loads(args_str) if args_str else {}
                                            except _json.JSONDecodeError:
                                                args = {}

                                            log.info(f"🔧 {name}({args})")
                                            loop.call_soon_threadsafe(
                                                chunk_queue.put_nowait,
                                                {
                                                    "event": "tool_call",
                                                    "name": name,
                                                    "arguments": args,
                                                },
                                            )

                                            tool_call_obj = ToolCall(name=name, arguments=args, raw_text="")
                                            result = executor.execute(tool_call_obj)
                                            tool_results.append({
                                                "tool_call_id": tc.get("id", ""),
                                                "content": result,
                                            })
                                            tool_call_details_holder.append(
                                                {
                                                    "iteration": iteration + 1,
                                                    "tool": name,
                                                    "parameters": args,
                                                    "result": result,
                                                }
                                            )

                                        has_executed_tools = True

                                        # Store tool context
                                        tool_context_holder[0] = "\n\n".join(
                                            f"[{tc['function']['name']}]\n{tr['content']}"
                                            for tc, tr in zip(tool_calls, tool_results)
                                        )

                                        # Build continuation messages
                                        assistant_msg = Msg(
                                            role="assistant",
                                            content=content,
                                            tool_calls=[
                                                {"id": tc.get("id"), "name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                                                for tc in tool_calls
                                            ],
                                        )
                                        current_msgs.append(assistant_msg)

                                        for tc, tr in zip(tool_calls, tool_results):
                                            current_msgs.append(Msg(
                                                role="tool",
                                                content=tr["content"],
                                                tool_call_id=tr["tool_call_id"],
                                            ))

                                        # Continue to next iteration (will stream the follow-up)
                                        break
                                else:
                                    # Stream finished without tool calls dict (pure content response)
                                    return

                            log.warning(f"Tool loop: max iterations ({max_iterations}) reached")

                        stream_iter = native_stream_with_tool_loop()
                    else:
                        # Fall back to text-based streaming for non-native models (GGUF, etc.)
                        from ..tools import stream_with_tools
                        log.debug(f"Using stream_with_tools for tool format: {llm_bawt.tool_format}")
                        
                        adapter = getattr(llm_bawt, 'adapter', None)
                        if adapter:
                            log.debug(f"Passing adapter '{adapter.name}' to stream_with_tools")
                        else:
                            log.warning("No adapter found on llm_bawt instance")

                        def stream_fn(msgs, stop_sequences=None):
                            return llm_bawt.client.stream_raw(msgs, stop=stop_sequences, **gen_kwargs)

                        stream_iter = stream_with_tools(
                            messages=messages,
                            stream_fn=stream_fn,
                            memory_client=llm_bawt.memory,
                            profile_manager=llm_bawt.profile_manager,
                            search_client=llm_bawt.search_client,
                            home_client=llm_bawt.home_client,
                            ha_native_client=llm_bawt.ha_native_client,
                            model_lifecycle=llm_bawt.model_lifecycle,
                            config=llm_bawt.config,
                            user_id=llm_bawt.user_id,
                            bot_id=llm_bawt.bot_id,
                            tool_format=llm_bawt.tool_format,
                            adapter=adapter,
                            history_manager=llm_bawt.history_manager,
                            tool_call_details=tool_call_details_holder,
                        )
                else:
                    # Resolve per-bot generation parameters
                    gen_kwargs = llm_bawt._get_generation_kwargs()
                    # Pass adapter stop sequences even without tools
                    adapter = getattr(llm_bawt, 'adapter', None)
                    adapter_stops = adapter.get_stop_sequences() if adapter else []
                    stream_iter = llm_bawt.client.stream_raw(
                        messages, stop=adapter_stops or None, **gen_kwargs
                    )
                
                # Stream chunks to queue
                for chunk in stream_iter:
                    # Check for cancellation - new request came in
                    if cancel_event.is_set():
                        log.info("Generation cancelled - newer request received")
                        cancelled_holder[0] = True
                        return
                    
                    full_response_holder[0] += chunk
                    # Put chunk in queue - use call_soon_threadsafe with captured loop
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
                
                timing_holder[1] = time.time()
                
                # Emit tool call SSE events for agent backend clients.
                # The backend stashes tool call metadata in last_result after
                # query() completes — push them to the chunk queue so the
                # SSE consumer can broadcast them to the frontend.
                from ..clients.agent_backend_client import AgentBackendClient
                if isinstance(llm_bawt.client, AgentBackendClient):
                    for tc in llm_bawt.client.get_tool_calls():
                        result_payload = tc.get("result")
                        if result_payload is None:
                            result_payload = "Result not exposed by OpenClaw API (see assistant response)."
                        loop.call_soon_threadsafe(
                            chunk_queue.put_nowait,
                            {
                                "event": "tool_call",
                                "name": tc.get("display_name") or tc.get("name", "unknown"),
                                "arguments": tc.get("arguments", {}),
                                "result": result_payload,
                            },
                        )
                        loop.call_soon_threadsafe(
                            chunk_queue.put_nowait,
                            {
                                "event": "tool_result",
                                "name": tc.get("display_name") or tc.get("name", "unknown"),
                                "result": result_payload,
                            },
                        )
                        tool_call_details_holder.append({
                            "iteration": 1,
                            "tool": tc.get("display_name") or tc.get("name", "unknown"),
                            "parameters": tc.get("arguments", {}),
                            "result": result_payload,
                        })
                
                # Finalize: add response to history and trigger memory extraction
                # Only if we weren't cancelled
                if full_response_holder[0] and not cancel_event.is_set():
                    # Calculate elapsed time and log with tokens/sec
                    elapsed_ms = (timing_holder[1] - timing_holder[0]) * 1000
                    # Apply adapter output cleaning as safety net
                    adapter = getattr(llm_bawt, 'adapter', None)
                    if adapter:
                        cleaned = adapter.clean_output(full_response_holder[0])
                        if cleaned != full_response_holder[0]:
                            log.info(f"Adapter '{adapter.name}' cleaned response: "
                                     f"{len(full_response_holder[0])} -> {len(cleaned)} chars")
                            full_response_holder[0] = cleaned
                    
                    log.llm_response(full_response_holder[0], elapsed_ms=elapsed_ms)
                    llm_bawt.finalize_response(full_response_holder[0], tool_context_holder[0])

                    self._persist_turn_log(
                        turn_id=turn_log_id,
                        request_id=ctx.request_id,
                        path=ctx.path,
                        stream=True,
                        model=model_alias,
                        bot_id=bot_id,
                        user_id=user_id,
                        status="ok",
                        latency_ms=elapsed_ms,
                        user_prompt=user_prompt,
                        prepared_messages=messages,
                        response_text=full_response_holder[0],
                        tool_calls=tool_call_details_holder,
                    )

                    # Write debug turn log if enabled (check config or env var)
                    if self.config.DEBUG_TURN_LOG or os.environ.get("LLM_BAWT_DEBUG_TURN_LOG"):
                        _write_debug_turn_log(
                            prepared_messages=messages,
                            user_prompt=user_prompt,
                            response=full_response_holder[0],
                            model=model_alias,
                            bot_id=bot_id,
                            user_id=user_id,
                            tool_calls=tool_call_details_holder,
                        )
                    
            except Exception as e:
                if not cancel_event.is_set():
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, None)  # Sentinel
        
        try:
            # Check if this bot needs emote filtering for TTS
            bot = get_bot(bot_id)
            emote_filter = StreamingEmoteFilter() if (bot and bot.voice_optimized) else None
            
            # Send service warnings (e.g. model fallback) before content
            if model_warnings:
                warning_data = {
                    "object": "service.warning",
                    "model": model_alias,
                    "warnings": model_warnings,
                }
                yield f"data: {json.dumps(warning_data)}\n\n"

            # Start streaming in single-thread executor
            loop.run_in_executor(self._llm_executor, _stream_to_queue)

            # Yield SSE chunks (with keepalive to prevent client timeout
            # during slow backends like vLLM first-inference)
            while True:
                try:
                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Send SSE comment as keepalive to prevent client/proxy timeout
                    yield ": keepalive\n\n"
                    continue
                
                if chunk is None:
                    # Stream complete - flush any buffered content from emote filter
                    if emote_filter:
                        final_chunk = emote_filter.flush()
                        if final_chunk:
                            data = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_alias,
                                "choices": [{"index": 0, "delta": {"content": final_chunk}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                    
                    # Send final chunk
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_alias,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                
                if isinstance(chunk, Exception):
                    # Upstream stream failed; close SSE cleanly so downstream clients
                    # don't see an incomplete chunked read protocol error.
                    log.error("Streaming backend failed: %s", chunk)
                    warning_data = {
                        "object": "service.warning",
                        "model": model_alias,
                        "warnings": [f"stream_interrupted: {chunk}"],
                    }
                    yield f"data: {json.dumps(warning_data)}\n\n"

                    # Flush any buffered content from emote filter.
                    if emote_filter:
                        final_chunk = emote_filter.flush()
                        if final_chunk:
                            data = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_alias,
                                "choices": [{"index": 0, "delta": {"content": final_chunk}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(data)}\n\n"

                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_alias,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    yield "data: [DONE]\n\n"
                    break

                if isinstance(chunk, dict) and chunk.get("event") == "tool_call":
                    has_result = chunk.get("result") is not None
                    event_data = {
                        "object": "service.tool_call",
                        "model": model_alias,
                        "tool": chunk.get("name", "unknown"),
                        "arguments": chunk.get("arguments", {}),
                        "result": chunk.get("result"),
                        "status": "completed" if has_result else "pending",
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                    continue

                if isinstance(chunk, dict) and chunk.get("event") == "tool_result":
                    event_data = {
                        "object": "service.tool_result",
                        "model": model_alias,
                        "tool": chunk.get("name", "unknown"),
                        "result": chunk.get("result"),
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                    continue
                
                # Apply emote filter for voice_optimized bots
                if emote_filter:
                    chunk = emote_filter.process(chunk)
                    if not chunk:
                        # Chunk was filtered out or buffered
                        continue
                
                # Normal chunk
                data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_alias,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(data)}\n\n"
        finally:
            # Mark generation as complete
            self._end_generation(cancel_event, done_event)

    async def process_task(self, task: Task) -> TaskResult:
        """Process a single background task."""
        start_time = time.time()
        task_type_str = task.task_type.value
        
        log.debug(f"Processing task {task.task_id[:8]} ({task_type_str})")
        
        try:
            if task.task_type == TaskType.CONTEXT_COMPACTION:
                result = await self._process_compaction(task)
            elif task.task_type == TaskType.EMBEDDING_GENERATION:
                result = await self._process_embeddings(task)
            elif task.task_type == TaskType.MEANING_UPDATE:
                result = await self._process_meaning_update(task)
            elif task.task_type == TaskType.MEMORY_MAINTENANCE:
                result = await self._process_maintenance(task)
            elif task.task_type == TaskType.PROFILE_MAINTENANCE:
                result = await self._process_profile_maintenance(task)
            elif task.task_type == TaskType.HISTORY_SUMMARIZATION:
                result = await self._process_history_summarization(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            elapsed_ms = (time.time() - start_time) * 1000
            log.task_completed(task.task_id, task_type_str, elapsed_ms, result)
            
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                processing_time_ms=elapsed_ms,
            )
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            log.task_failed(task.task_id, task_type_str, str(e), elapsed_ms)
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                processing_time_ms=elapsed_ms,
            )
    
    async def _process_compaction(self, task: Task) -> dict:
        """Process a context compaction task."""
        # TODO: Implement with summarization model
        return {"compacted": False, "reason": "Not yet implemented"}
    
    async def _process_embeddings(self, task: Task) -> dict:
        """Process an embedding generation task."""
        # TODO: Implement with embedding model
        return {"embeddings_generated": 0, "reason": "Not yet implemented"}

    async def _process_meaning_update(self, task: Task) -> dict:
        """Process a meaning update task (via MCP tools)."""
        bot_id = task.bot_id
        payload = task.payload
        memory_id = payload.get("memory_id")
        if not memory_id:
            return {"updated": False, "reason": "No memory_id provided"}

        memory_client = self.get_memory_client(bot_id)
        if not memory_client:
            return {"updated": False, "reason": "Memory client unavailable"}

        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            lambda: memory_client.update_memory_meaning(
                memory_id=memory_id,
                intent=payload.get("intent"),
                stakes=payload.get("stakes"),
                emotional_charge=payload.get("emotional_charge"),
                recurrence_keywords=payload.get("recurrence_keywords"),
                updated_tags=payload.get("updated_tags"),
            ),
        )
        return {"updated": bool(success), "memory_id": memory_id}

    async def _process_maintenance(self, task: Task) -> dict:
        """Process a unified memory maintenance task.
        
        Uses a cached LLM client if available, otherwise runs without LLM.
        Will NOT load a new model to avoid VRAM conflicts.
        """
        bot_id = task.bot_id
        payload = task.payload

        memory_client = self.get_memory_client(bot_id)
        if not memory_client:
            return {"error": "Memory client unavailable"}

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: memory_client.run_maintenance(
                run_consolidation=payload.get("run_consolidation", True),
                run_recurrence_detection=payload.get("run_recurrence_detection", True),
                run_decay_pruning=payload.get("run_decay_pruning", False),
                run_orphan_cleanup=payload.get("run_orphan_cleanup", False),
                dry_run=payload.get("dry_run", False),
            ),
        )
        return result

    async def _process_profile_maintenance(self, task: Task) -> dict:
        """Process profile maintenance - consolidate attributes into summary.
        
        Uses configured maintenance model resolution and loads that model
        deterministically when needed.
        """
        from ..memory.profile_maintenance import ProfileMaintenanceService
        from ..profiles import ProfileManager
        
        entity_id = task.payload.get("entity_id", task.user_id)
        entity_type = task.payload.get("entity_type", "user")
        dry_run = task.payload.get("dry_run", False)
        
        requested_model = (
            task.payload.get("model")
            or getattr(self.config, "MAINTENANCE_MODEL", None)
            or (self.config.PROFILE_MAINTENANCE_MODEL or None)
            or getattr(self.config, "SUMMARIZATION_MODEL", None)
        )
        try:
            model_to_use, _ = self._resolve_request_model(
                requested_model,
                task.bot_id or self._default_bot,
                local_mode=False,
            )
        except Exception as e:
            log.error(f"Failed to resolve model for profile maintenance: {e}")
            return {"error": f"Failed to resolve model: {e}"}

        if not model_to_use:
            err = "No model available for profile maintenance"
            log.error(err)
            return {"error": err}

        model_def = self.config.defined_models.get("models", {}).get(model_to_use, {})
        if model_def.get("type") == "openai" and not (self.config.OPENAI_API_KEY or self.config.XAI_API_KEY):
            err = f"Profile maintenance model '{model_to_use}' requires API key configuration"
            log.error(err)
            return {"error": err}

        if model_to_use not in self._available_models:
            err = f"Model '{model_to_use}' unavailable for profile maintenance"
            log.error(err)
            return {"error": err}
        
        log.info(f"🔧 Profile maintenance: {entity_type}/{entity_id} (model={model_to_use})")
        
        # Get or create profile manager
        profile_manager = ProfileManager(self.config)
        
        # Get LLM client - try cached first
        llm_client = None
        if model_to_use and model_to_use in self._client_cache:
            llm_client = self._client_cache[model_to_use]
        else:
            # Check LLMBawt cache
            for (cached_model, _, _), llm_bawt in self._llm_bawt_cache.items():
                if not model_to_use or cached_model == model_to_use:
                    llm_client = llm_bawt.client
                    break
        
        # If no cached client, load the default model
        if not llm_client:
            log.info(f"⏳ Loading model for profile maintenance: {model_to_use}")
            try:
                # Create an LLMBawt instance which will load the model
                llm_bawt = self._get_llm_bawt(
                    model_alias=model_to_use,
                    bot_id=task.bot_id or self._default_bot,
                    user_id=entity_id,
                    local_mode=False,
                )
                llm_client = llm_bawt.client
            except Exception as e:
                log.error(f"Failed to load model for profile maintenance: {e}")
                return {"error": f"Failed to load model: {e}"}
        
        service = ProfileMaintenanceService(profile_manager, llm_client)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._llm_executor,  # Use same executor as extraction
            lambda: service.run(entity_id, entity_type, dry_run)
        )
        
        return {
            "entity_id": result.entity_id,
            "attributes_before": result.attributes_before,
            "attributes_after": result.attributes_after,
            "categories_updated": result.categories_updated,
            "error": result.error,
        }

    def _get_extraction_client(self) -> tuple[Any | None, str | None]:
        """Get or create the extraction client for memory extraction.

        Uses ``EXTRACTION_MODEL`` from config.  Only API-based models
        (``openai`` or ``grok`` type) are supported to avoid VRAM conflicts
        with the local chat model.

        Returns:
            ``(client, model_alias)`` or ``(None, None)`` if unavailable.
        """
        if self._extraction_client is not None:
            return self._extraction_client, self._extraction_client_model

        extraction_model = getattr(self.config, "EXTRACTION_MODEL", None)
        if not extraction_model:
            log.debug("No EXTRACTION_MODEL configured — memory extraction will use heuristics")
            return None, None

        try:
            model_alias, model_def = self._resolve_request_model(
                extraction_model,
                bot_id="nova",
                local_mode=False,
            )
            model_type = model_def.get("type")

            if model_type not in ("openai", "grok"):
                log.warning(
                    f"EXTRACTION_MODEL '{extraction_model}' is type '{model_type}'. "
                    f"Only 'openai' and 'grok' types are supported for extraction. "
                    f"Falling back to heuristics."
                )
                return None, None

            llm_bawt = self._get_llm_bawt(
                model_alias=model_alias,
                bot_id="nova",
                user_id="system",
            )
            client = llm_bawt.client

            self._extraction_client = client
            self._extraction_client_model = model_alias
            log.info(f"Initialized extraction client: {model_alias} ({model_type})")
            return client, model_alias

        except Exception as e:
            log.error(f"Failed to initialize extraction client for '{extraction_model}': {e}")
            return None, None

    async def _extract_from_summaries(
        self,
        summarization_results: list[dict],
        bot_id: str,
        user_id: str,
    ) -> dict:
        """Extract facts from newly created summaries using an API model.

        Called after ``summarize_eligible_sessions()`` completes.  Only
        processes results where ``created`` is ``True``.

        Args:
            summarization_results: Result dicts from ``HistorySummarizer``
            bot_id: Bot whose memories are being updated
            user_id: User identity for memory storage

        Returns:
            Stats dict with extraction outcome.
        """
        from ..memory.extraction.service import MemoryExtractionService, MemoryAction

        extraction_client, extraction_model = self._get_extraction_client()
        use_llm = extraction_client is not None

        if not use_llm:
            return {
                "summaries_processed": 0,
                "facts_extracted": 0,
                "facts_stored": 0,
                "extraction_method": "skipped",
            }

        extraction_service = MemoryExtractionService(llm_client=extraction_client)
        memory_client = self.get_memory_client(bot_id, user_id)

        if not memory_client:
            log.warning("No memory client available — skipping extraction")
            return {
                "summaries_processed": 0,
                "facts_extracted": 0,
                "facts_stored": 0,
                "extraction_method": "skipped",
            }

        min_importance = getattr(self.config, "MEMORY_EXTRACTION_MIN_IMPORTANCE", 0.5)
        profile_enabled = getattr(self.config, "MEMORY_PROFILE_ATTRIBUTE_ENABLED", True)

        total_facts = 0
        total_stored = 0
        summaries_processed = 0

        for result in summarization_results:
            if not result.get("created", False):
                continue

            summary_text = result.get("summary_text")
            summary_id = result.get("summary_id")
            session_start = result.get("session_start", 0)
            session_end = result.get("session_end", 0)

            if not summary_text or not summary_id:
                continue

            try:
                facts = extraction_service.extract_from_summary(
                    summary_text=summary_text,
                    session_start=session_start,
                    session_end=session_end,
                    summary_id=summary_id,
                    use_llm=use_llm,
                )

                summaries_processed += 1
                total_facts += len(facts)

                log.info(
                    f"[Extraction] Summary {summary_id[:8]}: "
                    f"{len(facts)} facts extracted from: {summary_text[:200]}"
                )

                if not facts:
                    log.info(f"[Extraction] Summary {summary_id[:8]}: no facts found — skipping")
                    self._mark_summary_extracted(bot_id, summary_id)
                    continue

                # Filter by importance
                pre_filter = len(facts)
                facts = [f for f in facts if f.importance >= min_importance]
                if not facts:
                    log.info(
                        f"[Extraction] Summary {summary_id[:8]}: "
                        f"all {pre_filter} facts below importance threshold ({min_importance})"
                    )
                    self._mark_summary_extracted(bot_id, summary_id)
                    continue

                # Deduplicate against existing memories
                existing_memories = memory_client.list_memories(limit=100, min_importance=0.0)

                if existing_memories:
                    actions = extraction_service.determine_memory_actions(
                        new_facts=facts,
                        existing_memories=existing_memories,
                    )
                else:
                    actions = [MemoryAction(action="ADD", fact=f) for f in facts]

                for action in actions:
                    fact = action.fact
                    if not fact:
                        continue

                    try:
                        tag_str = ",".join(fact.tags)
                        if action.action == "ADD":
                            log.info(
                                f"[Extraction] ADD: '{fact.content}' "
                                f"[{tag_str}] importance={fact.importance:.2f}"
                            )
                            memory_client.add_memory(
                                content=fact.content,
                                tags=fact.tags,
                                importance=fact.importance,
                                source_message_ids=fact.source_message_ids,
                            )
                            total_stored += 1
                        elif action.action == "UPDATE" and action.target_memory_id:
                            log.info(
                                f"[Extraction] UPDATE ({action.target_memory_id[:8]}): "
                                f"'{fact.content}' [{tag_str}] importance={fact.importance:.2f}"
                            )
                            memory_client.update_memory(
                                memory_id=action.target_memory_id,
                                content=fact.content,
                                importance=fact.importance,
                                tags=fact.tags,
                            )
                            total_stored += 1
                        elif action.action == "DELETE" and action.target_memory_id:
                            log.info(
                                f"[Extraction] DELETE: {action.target_memory_id[:8]} "
                                f"reason='{action.reason}'"
                            )
                            memory_client.delete_memory(memory_id=action.target_memory_id)

                        # Profile attribute extraction
                        if action.action in ("ADD", "UPDATE") and profile_enabled:
                            from ..memory_server.extraction import extract_profile_attributes_from_fact
                            extract_profile_attributes_from_fact(
                                fact=fact,
                                user_id=user_id,
                                config=self.config,
                            )
                    except Exception as e:
                        log.warning(f"Failed to process memory action {action.action}: {e}")

                # Mark summary as extracted (crash recovery marker)
                self._mark_summary_extracted(bot_id, summary_id)

            except Exception as e:
                log.error(f"Failed to extract from summary {summary_id[:8]}: {e}")
                continue

        log.info(
            f"[Extraction] Done: {summaries_processed} summaries processed, "
            f"{total_facts} facts extracted, {total_stored} stored"
        )

        return {
            "summaries_processed": summaries_processed,
            "facts_extracted": total_facts,
            "facts_stored": total_stored,
            "extraction_method": extraction_model or "heuristic",
        }

    def _mark_summary_extracted(self, bot_id: str, summary_id: str) -> None:
        """Set ``extracted_at`` in a summary's metadata for crash recovery."""
        try:
            memory_client = self.get_memory_client(bot_id, "system")
            if not memory_client:
                return
            backend = getattr(memory_client, "_backend", None) or getattr(memory_client, "backend", None)
            if not backend or not hasattr(backend, "engine"):
                return
            from sqlalchemy import text as sa_text
            table = f"{bot_id}_messages"
            with backend.engine.connect() as conn:
                conn.execute(
                    sa_text(f"""
                        UPDATE {table}
                        SET summary_metadata = jsonb_set(
                            COALESCE(summary_metadata::jsonb, '{{}}'::jsonb),
                            '{{extracted_at}}',
                            to_jsonb(:ts)
                        )
                        WHERE id = :sid AND role = 'summary'
                    """),
                    {"ts": datetime.utcnow().isoformat(), "sid": summary_id},
                )
                conn.commit()
        except Exception as e:
            log.debug(f"Failed to mark summary {summary_id[:8]} as extracted: {e}")

    async def _process_history_summarization(self, task: Task) -> dict:
        """Process proactive history summarization and extraction for a bot."""
        from ..memory.summarization import (
            HistorySummarizer,
            SUMMARIZATION_PROMPT,
            format_session_for_summarization,
        )
        from ..models.message import Message

        bot_id = task.bot_id or self._default_bot
        requested_model = (
            task.payload.get("model")
            or getattr(self.config, "MAINTENANCE_MODEL", None)
            or getattr(self.config, "SUMMARIZATION_MODEL", None)
        )
        use_heuristic_fallback = bool(task.payload.get("use_heuristic_fallback", True))
        max_tokens_per_chunk = int(task.payload.get("max_tokens_per_chunk", 4000))

        model_alias = None
        client = None

        if requested_model:
            try:
                model_alias, _ = self._resolve_request_model(
                    requested_model,
                    bot_id,
                    local_mode=False,
                )
                llm_bawt = self._get_llm_bawt(
                    model_alias=model_alias,
                    bot_id=bot_id,
                    user_id=task.user_id or "system",
                )
                client = llm_bawt.client
            except Exception as e:
                log.error(f"Failed to resolve/load model for history summarization: {e}")
                client = None
        elif self._client_cache:
            model_alias = next(iter(self._client_cache.keys()))
            client = self._client_cache[model_alias]
        else:
            try:
                model_alias, _ = self._resolve_request_model(
                    None,
                    bot_id,
                    local_mode=False,
                )
                llm_bawt = self._get_llm_bawt(
                    model_alias=model_alias,
                    bot_id=bot_id,
                    user_id=task.user_id or "system",
                )
                client = llm_bawt.client
            except Exception as e:
                log.error(f"Failed to resolve/load model for history summarization: {e}")
                client = None

        def summarize_with_loaded_client(session) -> str | None:
            if not client:
                return None

            conversation_text = format_session_for_summarization(session)
            prompt = SUMMARIZATION_PROMPT.format(messages=conversation_text)

            # Conservative budget to reduce context overflows on smaller models.
            if len(prompt) // 4 > 6000:
                return None

            try:
                messages = [
                    Message(
                        role="system",
                        content="You are a helpful assistant that summarizes conversations concisely.",
                    ),
                    Message(role="user", content=prompt),
                ]
                response = client.query(
                    messages=messages,
                    max_tokens=320,
                    temperature=0.3,
                    plaintext_output=True,
                    stream=False,
                )
                if not response:
                    return None

                lower = response.lower()
                error_indicators = ("error:", "exception occurred", "exceed context window", "tokens exceed")
                if any(ind in lower for ind in error_indicators):
                    return None
                return response.strip()
            except Exception as e:
                log.error(f"History summarization LLM call failed: {e}")
                return None

        # Create a settings resolver for per-bot summarization tunables
        from ..runtime_settings import RuntimeSettingsResolver
        from ..bots import BotManager
        bot_manager = BotManager(self.config)
        bot_obj = bot_manager.get_bot(bot_id) or bot_manager.get_default_bot()
        resolver = RuntimeSettingsResolver(config=self.config, bot=bot_obj, bot_id=bot_id)

        summarizer = HistorySummarizer(
            self.config,
            bot_id=bot_id,
            summarize_fn=summarize_with_loaded_client,
            settings_getter=resolver.resolve,
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._llm_executor,
            lambda: summarizer.summarize_eligible_sessions(
                use_heuristic_fallback=use_heuristic_fallback,
                max_tokens_per_chunk=max_tokens_per_chunk,
            ),
        )

        if model_alias:
            result["model"] = model_alias
        result["used_heuristic_fallback"] = use_heuristic_fallback

        # Phase 2: Extract memories from newly created summaries
        extraction_results = await self._extract_from_summaries(
            result.get("results", []),
            bot_id=bot_id,
            user_id=task.user_id or "system",
        )
        result["extraction"] = extraction_results

        # Summary writes change context composition; invalidate cached history views
        # for this bot so next request reloads from DB immediately.
        for (_model, cached_bot_id, _user), llm_bawt in self._llm_bawt_cache.items():
            if cached_bot_id != bot_id:
                continue
            invalidate = getattr(llm_bawt, "invalidate_history_cache", None)
            if callable(invalidate):
                invalidate()

        return result

    def submit_task(self, task: Task) -> str:
        """Submit a task to the processing queue."""
        from dataclasses import dataclass, field as dataclass_field
        
        @dataclass(order=True)
        class PrioritizedTask:
            priority: int
            timestamp: float
            task: Task = dataclass_field(compare=False)
        
        prioritized = PrioritizedTask(
            priority=-task.priority,
            timestamp=time.time(),
            task=task,
        )
        self._task_queue.put(prioritized)
        self._submitted_tasks[task.task_id] = task
        self._result_events[task.task_id] = asyncio.Event()
        return task.task_id
    
    def get_result(self, task_id: str) -> TaskResult | None:
        """Get the result of a completed task."""
        return self._results.get(task_id)
    
    async def wait_for_result(self, task_id: str, timeout: float = 30.0) -> TaskResult | None:
        """Wait for a task result with timeout."""
        event = self._result_events.get(task_id)
        if not event:
            return self._results.get(task_id)
        
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self._results.get(task_id)
        except asyncio.TimeoutError:
            return None
    
    async def worker_loop(self):
        """Main worker loop that processes tasks from the queue."""
        log.debug("Background task worker started")
        
        while not self._shutdown_event.is_set():
            try:
                try:
                    prioritized = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._task_queue.get(timeout=1.0)
                    )
                except Exception:
                    continue
                
                task = prioritized.task
                log.task_submitted(task.task_id, task.task_type.value, task.bot_id, task.payload)
                
                result = await self.process_task(task)
                self._results[task.task_id] = result
                self.tasks_processed += 1
                
                if task.task_id in self._result_events:
                    self._result_events[task.task_id].set()
                
                # Cleanup old results
                if len(self._results) > 1000:
                    oldest = sorted(self._results.keys())[:100]
                    for key in oldest:
                        self._results.pop(key, None)
                        self._submitted_tasks.pop(key, None)
                        self._result_events.pop(key, None)
                
            except Exception as e:
                log.exception(f"Worker loop error: {e}")
                await asyncio.sleep(1)
        
        log.debug("Background task worker stopped")
    
    def start_worker(self):
        """Start the background worker task."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self.worker_loop())
    
    async def shutdown(self):
        """Gracefully shutdown the service."""
        log.shutdown()
        self._shutdown_event.set()
        if self._worker_task:
            self._worker_task.cancel()
    
    def get_status(self) -> ServiceStatusResponse:
        """Get service status."""
        from ..core.status import _collect_mcp_info, _collect_memory_info
        from ..utils.config import has_database_credentials

        current = self._model_lifecycle.current_model
        memory_info = _collect_memory_info(self.config, self._default_bot)
        mcp_info = _collect_mcp_info(self.config, self._default_bot)

        db_required = has_database_credentials(self.config)
        db_ok = memory_info.postgres_connected if db_required else True
        mcp_ok = mcp_info.status == "up"
        healthy = db_ok and mcp_ok

        checks = {
            "service": "up",
            "database": "up" if db_ok else ("disabled" if not db_required else "down"),
            "mcp": mcp_info.status,
        }

        return ServiceStatusResponse(
            status="ok" if healthy else "degraded",
            healthy=healthy,
            uptime_seconds=self.uptime_seconds,
            tasks_processed=self.tasks_processed,
            tasks_pending=self._task_queue.qsize(),
            worker_running=bool(self._worker_task and not self._worker_task.done()),
            models_loaded=[current] if current else [],
            current_model=current,
            default_model=self._default_model,
            default_bot=self._default_bot,
            available_models=self._available_models,
            checks=checks,
            database_connected=memory_info.postgres_connected,
            database_host=memory_info.postgres_host,
            database_error=memory_info.postgres_error,
            messages_count=memory_info.messages_count,
            memories_count=memory_info.memories_count,
            pgvector_available=memory_info.pgvector_available,
            embeddings_available=memory_info.embeddings_available,
            mcp_mode=mcp_info.mode,
            mcp_status=mcp_info.status,
            mcp_url=mcp_info.url,
            mcp_http_status=mcp_info.http_status,
        )
