"""Background task service and chat execution runtime.

Core orchestrator that composes focused mixins:
- InstanceManagerMixin  — model/bot instance management
- TurnLifecycleMixin    — turn persistence + generation cancellation
- ChatStreamingMixin    — streaming chat completions + bridge
- BackgroundTasksMixin  — background task processing
"""

import asyncio
import json
import threading
import time
import uuid
from queue import PriorityQueue
from typing import Any, AsyncIterator
from urllib.parse import urlparse

from ..bots import get_bot, strip_emotes
from ..utils.config import Config
from .background_tasks import BackgroundTasksMixin
from .chat_streaming import ChatStreamingMixin
from .instance_manager import InstanceManagerMixin
from .logging import RequestContext, generate_request_id, get_service_logger
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ServiceStatusResponse,
    UsageInfo,
)
from .tasks import Task, TaskResult, TaskStatus
from .turn_lifecycle import TurnLifecycleMixin

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


class BackgroundService(
    InstanceManagerMixin,
    TurnLifecycleMixin,
    ChatStreamingMixin,
    BackgroundTasksMixin,
):
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

        # OpenClaw WS session bridge (set by api.py lifespan if enabled)
        self._session_bridge: Any | None = None

        # Background client for summarization/extraction (isolated from chat model lifecycle)
        self._bg_client: Any | None = None
        self._bg_client_model: str | None = None

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

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def model_lifecycle(self):
        """Get the model lifecycle manager for tool access."""
        return self._model_lifecycle

    # ---- Non-streaming chat completion ----

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse | AsyncIterator:
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

        # Start new generation (cancels previous for the SAME bot only)
        cancel_event, done_event = await self._start_generation(bot_id)
        turn_log_id = f"turn-{uuid.uuid4().hex}"

        # Persist turn log immediately so the user's prompt is recorded
        # even if the backend times out or errors before responding.
        self._persist_turn_log(
            turn_id=turn_log_id,
            request_id=ctx.request_id,
            path=ctx.path,
            stream=False,
            model=model_alias,
            bot_id=bot_id,
            user_id=user_id,
            status="pending",
            latency_ms=None,
            user_prompt=user_prompt,
            prepared_messages=[],
            response_text="",
        )

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
                llm_bawt._include_summaries = request.include_summaries
                llm_bawt._tts_mode = request.tts_mode

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

                # Check if cancelled during generation
                if cancel_event.is_set():
                    log.info("Generation cancelled - newer request received")
                    cancelled = True
                    return ""

                self._finalize_turn(
                    llm_bawt=llm_bawt,
                    turn_id=turn_log_id,
                    response_text=response,
                    tool_context=tool_context,
                    tool_call_details=tool_call_details,
                    prepared_messages=prepared_messages,
                    user_prompt=user_prompt,
                    model=model_alias,
                    bot_id=bot_id,
                    user_id=user_id,
                    elapsed_ms=(time.time() - llm_start_time) * 1000,
                    stream=False,
                )

                return response

            try:
                model_type = llm_bawt.client.model_definition.get("type", "")
                if model_type in ("openai", "grok", "agent_backend"):
                    response_text = await loop.run_in_executor(None, _do_query)
                else:
                    response_text = await loop.run_in_executor(self._llm_executor, _do_query)
            except Exception as e:
                elapsed_ms = (time.time() - llm_start_time) * 1000
                self._update_turn_log(
                    turn_id=turn_log_id,
                    status="error",
                    latency_ms=elapsed_ms,
                    error_text=str(e),
                )
                raise
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
            self._end_generation(cancel_event, done_event, bot_id)

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

    # ---- Task queue ----

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
