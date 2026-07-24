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


_mcp_thread: threading.Thread | None = None


def _ensure_mcp_server(config: Config) -> None:
    """Ensure the llm-bawt MCP server is running and configure the service to use it.

    This makes memory retrieval happen via MCP tool calls (e.g. tools/memory_search),
    which can be logged distinctly from embedded DB access.
    """
    global _mcp_thread

    # Default to local llm-bawt MCP server for llm-service if not configured.
    if not config.MCP_SERVER_URL:
        config.MCP_SERVER_URL = "http://127.0.0.1:8001"

    parsed = urlparse(config.MCP_SERVER_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8001

    # If something is already listening, assume it's the llm-bawt MCP server.
    if _is_tcp_listening(host, port):
        log.info("llm-bawt MCP server already listening at %s", config.MCP_SERVER_URL)
        return

    # Start the MCP server in-process (HTTP transport) on a daemon thread.
    def _run():
        try:
            from ..mcp_server.server import run_server
            run_server(transport="streamable-http", host=host, port=port)
        except Exception as e:
            log.error("Failed to start llm-bawt MCP server: %s", e)

    _mcp_thread = threading.Thread(target=_run, daemon=True, name="llm-bawt-mcp")
    _mcp_thread.start()
    log.info("Started llm-bawt MCP server at %s", config.MCP_SERVER_URL)


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

        # Bounded pool for background jobs (memory extraction, history
        # summarization, profile/consolidation maintenance, media GC). Kept
        # SEPARATE from the default executor so a burst of per-bot background
        # fan-out can never starve the workers that interactive agent-bridge
        # relays need — that contention stalled continuation turns by minutes
        # (TASK-283). Local GPU inference no longer runs in-process — it moved to
        # the standalone local_model_bridge (TASK-276/278) — so there is no
        # single-thread LLM executor here anymore; all chat streams on the
        # default pool.
        from concurrent.futures import ThreadPoolExecutor
        self._bg_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bg")

        # Cancellation support: when a new request comes in, cancel the current one
        # This handles the case where UI sends partial transcriptions that build up
        self._current_generation_cancel: threading.Event | None = None
        self._generation_done: threading.Event | None = None  # Signals when generation finishes
        self._cancel_lock = threading.Lock()

        # Memory client cache keyed by bot_id
        self._memory_clients: dict[tuple[str, str], Any] = {}
        from .turn_logs import TurnLogStore
        self._turn_log_store = TurnLogStore(config)
        # Persistent registry of in-flight AskUserQuestion pauses.  Lets the
        # chat UI hydrate open pickers on page load / second tab / after a
        # bridge restart instead of relying solely on the live SSE event.
        from .chat_pending_questions import PendingQuestionStore
        self._pending_question_store = PendingQuestionStore(config)

        # TASK-290: durable registry + audit of approval-gated tool calls. Used
        # by chat_streaming to persist a request row when the bridge emits
        # APPROVAL_REQUIRED, and by the resolve endpoint.
        from ..approval_policies import ToolApprovalPolicyStore
        self._tool_approval_policy_store = ToolApprovalPolicyStore(config)

        # OpenClaw WS session bridge (set by api.py lifespan if enabled)
        self._session_bridge: Any | None = None

        # Keyed pool of background-task API clients, one per resolved model
        # alias (TASK-281). Replaces the old single reassignable _bg_client slot
        # that thrashed/raced when concurrent background jobs resolved different
        # models — the loser got a torn-down/None client and silently produced
        # zero results. Created once per model, thereafter only read. Isolated
        # from the chat model lifecycle.
        self._bg_client_cache: dict[str, Any] = {}
        # Guards create-and-insert into _bg_client_cache. A plain threading.Lock
        # (NOT the asyncio _cache_lock) because _get_background_client is sync and
        # runs in the _bg_executor thread pool. Steady-state reads are lock-free.
        self._bg_client_lock = threading.Lock()

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

        # Start generation lifecycle.
        # Agent backends (openclaw / claude-code) can run independent requests
        # concurrently, so avoid the per-bot cancellation gate used for local
        # single-model execution.
        model_type = llm_bawt.client.model_definition.get("type", "")
        is_agent_backend = model_type in ("agent_backend", "claude-code")

        # TASK-646: intercept chat-bot /new HERE too — SAME shared helper as
        # the streaming path (summarize → rotate → confirm). Without this a
        # stream:false /new fell through to the LLM as plain text with full
        # history and the model hallucinated a fake "session reset"
        # confirmation. Bare /new short-circuits with the shared confirmation
        # and no LLM round-trip (mirrors streaming, which also skips the turn
        # log for the bare command).
        if not is_agent_backend:
            _confirm, user_prompt = self._maybe_handle_chat_new_command(
                llm_bawt, bot_id, user_prompt
            )
            if _confirm is not None:
                log.api_response(ctx, status=200, tokens=0)
                return ChatCompletionResponse(
                    model=model_alias,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=_confirm),
                            finish_reason="stop",
                        )
                    ],
                    usage=UsageInfo(
                        prompt_tokens=0, completion_tokens=0, total_tokens=0
                    ),
                )

        if is_agent_backend:
            cancel_event = threading.Event()
            done_event = threading.Event()
        else:
            # Cancels previous generation for the SAME bot only.
            cancel_event, done_event = await self._start_generation(bot_id)
        turn_log_id = f"turn-{uuid.uuid4().hex}"

        # TASK-303: Extract or generate a stable user-message id so the
        # persisted user message and the turn log share the same identity.
        # This mirrors the streaming path (chat_streaming.py:814).
        trigger_message_id = getattr(request, "user_message_id", None) or str(uuid.uuid4())
        # Canonical assistant-row id (frontend-minted) so live bubble == reloaded
        # row (single bubble). None on server-originated turns → server mints one.
        _amid = getattr(request, "assistant_message_id", None)
        assistant_message_id = _amid.strip() if isinstance(_amid, str) and _amid.strip() else None

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
            trigger_message_id=trigger_message_id,
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
                # Agent backends use a per-turn user-message prefix for
                # voice mode — bot.tts_mode is a chatbot-only default.
                _is_agent = llm_bawt.client.model_definition.get("type") in (
                    "agent_backend", "claude-code",
                )
                if _is_agent:
                    llm_bawt._tts_mode = request.tts_mode
                else:
                    llm_bawt._tts_mode = request.tts_mode or llm_bawt.bot.tts_mode
                llm_bawt._inject_user_prefix = bool(request.inject_user_prefix)
                # TASK-251: explicit thread selection — set FRESH every turn
                # (cached instance; a stale override must never leak into a
                # continuous request).
                _sid = getattr(request, "session_id", None)
                llm_bawt._session_id_override = (
                    _sid.strip() if isinstance(_sid, str) and _sid.strip() else None
                )

                # TASK-252: resolve the turn's explicit-thread binding (if
                # any) BEFORE the seed decision — the scoped seed branch
                # consumes it. REQUEST-LOCAL (kwarg channel, never shared
                # instance state). SAME shared helper as the streaming path.
                thread_binding = (
                    self._bind_agent_thread(llm_bawt, request) if _is_agent else None
                )

                # Prepare messages with history and memory context
                prepared_messages = llm_bawt.prepare_messages_for_query(user_prompt, message_id=trigger_message_id)

                # Log what we're sending to the LLM (verbose mode)
                log.llm_context(prepared_messages)

                # TASK-501: pre-assemble the fresh-session seed and push it to
                # the bridge (inject_messages) — SAME shared helper the streaming
                # path uses, so non-streaming stays consistent (no callback).
                # TASK-641: on /new, summarize the OUTGOING thread FIRST so
                # the seed below carries a fresh summary of the ending
                # conversation (bounded; no-op unless /new + scope carries
                # summaries). SAME shared helper as the streaming path.
                self._maybe_summarize_on_new(
                    llm_bawt, bot_id, user_prompt, thread_binding=thread_binding
                )
                from .routes.history import maybe_build_session_seed
                inject_seed_messages = maybe_build_session_seed(
                    llm_bawt, bot_id, model_alias, user_prompt, self,
                    thread_binding=thread_binding,
                )
                # TASK-284: an agent /new rotates the durable DB thread —
                # AFTER the seed is built so the seed captured the outgoing
                # thread's raw messages (session-scoped load reads the active
                # thread; rotating first emptied every inline-history seed).
                # SAME shared helper as the streaming path so the two
                # dispatch routes stay consistent.
                if _is_agent:
                    self._maybe_rotate_agent_session(
                        llm_bawt, bot_id, user_prompt, thread_binding=thread_binding
                    )

                # Execute the query with prepared messages
                response, tool_context, tool_call_details = llm_bawt.execute_llm_query(
                    prepared_messages,
                    plaintext_output=True,
                    stream=False,
                    inject_messages=inject_seed_messages,
                    thread_binding=thread_binding,
                )

                # Splice the injected seed into the logged prompt so the turn
                # log reflects what the harness session received: [system,
                # ...seed history..., user]. Additive — nothing dropped.
                _logged_messages = prepared_messages
                if inject_seed_messages:
                    _sys = [m for m in prepared_messages if getattr(m, "role", None) == "system"]
                    _rest = [m for m in prepared_messages if getattr(m, "role", None) != "system"]
                    _logged_messages = [*_sys, *inject_seed_messages, *_rest]

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
                    prepared_messages=_logged_messages,
                    user_prompt=user_prompt,
                    model=model_alias,
                    bot_id=bot_id,
                    user_id=user_id,
                    elapsed_ms=(time.time() - llm_start_time) * 1000,
                    stream=False,
                    assistant_message_id=assistant_message_id,
                )

                return response

            try:
                # In-process GPU inference is gone (local models run in
                # local_model_bridge, TASK-276/278), so every model now streams
                # on the default thread pool — no single-worker serialization.
                response_text = await loop.run_in_executor(None, _do_query)
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
        # Drop in-flight background jobs; don't block shutdown waiting on them.
        self._bg_executor.shutdown(wait=False, cancel_futures=True)

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
