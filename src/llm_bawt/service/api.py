"""
FastAPI-based background service for llm-bawt.

Provides:
- OpenAI-compatible chat completions API
- Background task processing (memory extraction, compaction)
- Health and status endpoints

Run with: python -m llm_bawt.service.server
Or: uvicorn llm_bawt.service.server:app --host 0.0.0.0 --port 8642
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import PriorityQueue
from typing import Any, AsyncIterator, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ..utils.config import Config
from ..utils.paths import resolve_log_dir
from ..bots import BotManager, get_bot, strip_emotes, StreamingEmoteFilter
from .tasks import Task, TaskResult, TaskStatus, TaskType
from .logging import (
    ServiceLogger,
    RequestContext,
    setup_service_logging,
    generate_request_id,
    get_service_logger,
)

log = get_service_logger(__name__)

# Configuration
DEFAULT_HTTP_PORT = 8642
SERVICE_VERSION = "0.1.0"


def _is_tcp_listening(host: str, port: int) -> bool:
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


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
                lines.append(f"")
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

        def msg_to_dict(msg):
            if hasattr(msg, 'to_dict'):
                return msg.to_dict()
            elif hasattr(msg, 'role'):
                return {"role": msg.role, "content": msg.content, "timestamp": getattr(msg, 'timestamp', 0)}
            return dict(msg) if isinstance(msg, dict) else str(msg)

        json_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "bot_id": bot_id,
            "user_id": user_id,
            "request": [msg_to_dict(msg) for msg in prepared_messages],
            "tool_calls": tool_calls or [],
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


# =============================================================================
# Pydantic Models for OpenAI-compatible API
# =============================================================================

class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str | None = None  # Optional, will use service default if not specified
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    user: str | None = None
    # llm-bawt extensions
    bot_id: str | None = Field(default=None, description="Bot personality to use")
    augment_memory: bool = Field(default=True, description="Whether to augment with memory context")
    extract_memory: bool = Field(default=True, description="Whether to extract memories from response")


class ChatCompletionChoice(BaseModel):
    """OpenAI-compatible chat completion choice."""
    index: int
    message: ChatMessage
    finish_reason: str | None = "stop"


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict]


class ModelInfo(BaseModel):
    """Model information for /v1/models endpoint."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "llm-bawt"


class ModelsResponse(BaseModel):
    """Response for /v1/models endpoint."""
    object: str = "list"
    data: list[ModelInfo]


class ModelSwitchRequest(BaseModel):
    """Request to switch the active model."""
    model: str = Field(..., description="Model alias to switch to")


class ModelSwitchResponse(BaseModel):
    """Response from model switch."""
    success: bool
    message: str
    previous_model: str | None = None
    new_model: str | None = None


class ModelDetail(BaseModel):
    """Detailed model information."""
    id: str
    type: str | None = None
    model_id: str | None = None
    description: str | None = None
    current: bool = False


class BotInfo(BaseModel):
    """Bot information for /v1/bots endpoint."""
    slug: str
    name: str
    description: str | None = None


class BotsResponse(BaseModel):
    """Response for /v1/bots endpoint."""
    object: str = "list"
    data: list[BotInfo]


class TaskSubmitRequest(BaseModel):
    """Request to submit a background task."""
    task_type: str
    payload: dict[str, Any]
    bot_id: str | None = None  # Will use config DEFAULT_BOT if not specified
    user_id: str  # Required - must be passed explicitly
    priority: int = 0


class TaskSubmitResponse(BaseModel):
    """Response after submitting a task."""
    task_id: str
    status: str = "pending"


class TaskStatusResponse(BaseModel):
    """Response for task status."""
    task_id: str
    status: str
    result: Any | None = None
    error: str | None = None
    processing_time_ms: float | None = None


class ServiceStatusResponse(BaseModel):
    """Service health and status."""
    status: str = "ok"
    version: str = SERVICE_VERSION
    uptime_seconds: float
    tasks_processed: int
    tasks_pending: int
    models_loaded: list[str] = []
    current_model: str | None = None
    available_models: list[str] = []


class HealthResponse(BaseModel):
    """Simple health check response."""
    status: str = "ok"
    version: str = SERVICE_VERSION


class HistoryMessage(BaseModel):
    """A message in the conversation history."""
    id: str | None = None
    role: str
    content: str
    timestamp: float


class HistoryResponse(BaseModel):
    """Response for conversation history."""
    bot_id: str
    messages: list[HistoryMessage]
    total_count: int


class HistorySearchResponse(BaseModel):
    """Response for history search."""
    bot_id: str
    query: str
    messages: list[HistoryMessage]
    total_count: int


class HistoryClearResponse(BaseModel):
    """Response for clearing history."""
    success: bool
    message: str
    deleted_count: int = 0


# Memory Management Models
class MemoryItem(BaseModel):
    """A memory item."""
    id: str | None = None
    content: str
    importance: float = 0.5
    relevance: float | None = None
    tags: list[str] = []
    created_at: float | None = None
    last_accessed: float | None = None
    access_count: int = 0
    source_message_ids: list[str] = []


class MemorySearchRequest(BaseModel):
    """Request for memory search."""
    query: str
    method: str = "all"  # text, embedding, high-importance, all
    limit: int = 10
    min_importance: float = 0.0
    bot_id: str | None = None


class MemorySearchResponse(BaseModel):
    """Response for memory search."""
    bot_id: str
    method: str
    query: str
    results: list[MemoryItem]
    total_count: int


class MemoryStatsResponse(BaseModel):
    """Memory statistics."""
    bot_id: str
    messages: dict
    memories: dict


class MemoryForgetRequest(BaseModel):
    """Request to forget messages."""
    count: int | None = None  # forget recent N
    minutes: int | None = None  # forget last N minutes
    message_id: str | None = None  # forget specific message by ID


class MemoryForgetResponse(BaseModel):
    """Response for forget operation."""
    success: bool
    messages_ignored: int
    memories_deleted: int
    message: str


class MemoryRestoreResponse(BaseModel):
    """Response for restore operation."""
    success: bool
    messages_restored: int
    message: str


class MemoryDeleteResponse(BaseModel):
    """Response for deleting a specific memory."""
    success: bool
    memory_id: str
    message: str


class MessagePreview(BaseModel):
    """Preview of a message for confirmation."""
    id: str  # UUID or int, stored as string
    role: str
    content: str
    timestamp: float | None = None


class MessagesPreviewResponse(BaseModel):
    """Response with message previews."""
    bot_id: str
    messages: list[MessagePreview]
    total_count: int


class RegenerateEmbeddingsResponse(BaseModel):
    """Response for regenerate embeddings operation."""
    success: bool
    updated: int
    failed: int
    embedding_dim: int | None = None
    message: str


class ConsolidateRequest(BaseModel):
    """Request for memory consolidation."""
    dry_run: bool = False
    similarity_threshold: float | None = None


class ConsolidateResponse(BaseModel):
    """Response for consolidation operation."""
    success: bool
    dry_run: bool
    clusters_found: int
    clusters_merged: int
    memories_consolidated: int
    new_memories_created: int
    errors: list[str] = []
    message: str


class RawCompletionRequest(BaseModel):
    """Request for raw LLM completion without bot/memory overhead.

    Use this for utility tasks like memory consolidation, summarization, etc.
    """
    prompt: str
    system: str | None = None
    model: str | None = None  # Uses service default if not specified
    max_tokens: int = 500
    temperature: float = 0.7


class RawCompletionResponse(BaseModel):
    """Response from raw LLM completion."""
    content: str
    model: str
    tokens: int | None = None
    elapsed_ms: float


# History Summarization Models
class SummarizableSession(BaseModel):
    """A session eligible for summarization."""
    start_timestamp: float
    end_timestamp: float
    start_time: str
    end_time: str
    message_count: int
    first_message: str
    last_message: str


class SummarizePreviewResponse(BaseModel):
    """Response for summarization preview."""
    bot_id: str
    sessions: list[SummarizableSession]
    total_messages: int


class SummarizeResponse(BaseModel):
    """Response for summarization operation."""
    success: bool
    sessions_summarized: int
    messages_summarized: int
    errors: list[str] = []


class SummaryInfo(BaseModel):
    """Information about a single summary."""
    id: str
    content: str
    timestamp: float
    session_start_time: str | None
    session_end_time: str | None
    message_count: int
    method: str


class ListSummariesResponse(BaseModel):
    """Response for listing summaries."""
    bot_id: str
    summaries: list[SummaryInfo]
    total_count: int


class DeleteSummaryResponse(BaseModel):
    """Response for deleting a summary."""
    success: bool
    summary_id: str | None = None
    messages_restored: int = 0
    detail: str | None = None


# User Profile Models
class UserProfileAttribute(BaseModel):
    """A single profile attribute."""
    id: int | None = None
    category: str
    key: str
    value: str
    confidence: float = 1.0
    source: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class UserProfileSummary(BaseModel):
    """Summary of a user profile."""
    user_id: str
    display_name: str | None = None
    description: str | None = None
    attribute_count: int = 0
    created_at: str | None = None


class UserProfileDetail(BaseModel):
    """Detailed user profile with attributes."""
    user_id: str
    display_name: str | None = None
    description: str | None = None
    attributes: list[UserProfileAttribute] = []
    created_at: str | None = None


class UserListResponse(BaseModel):
    """Response for listing users."""
    users: list[UserProfileSummary]
    total_count: int


# Unified Profile Models
class ProfileDetail(BaseModel):
    """Generic profile detail with attributes for any entity type (user or bot)."""
    entity_type: str  # "user" or "bot"
    entity_id: str
    display_name: str | None = None
    description: str | None = None
    summary: str | None = None
    attributes: list[UserProfileAttribute] = []
    created_at: str | None = None


class ProfileListResponse(BaseModel):
    """Response for listing profiles of a given type."""
    profiles: list[ProfileDetail]
    total_count: int


# =============================================================================
# Background Service
# =============================================================================

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
    
    def _load_available_models(self):
        """Load list of available models from config."""
        models = self.config.defined_models.get("models", {})
        self._available_models = list(models.keys())
        log.debug(f"Loaded {len(self._available_models)} models from config")
        
        # Set default model from bot/config selection or use first available
        bot_manager = BotManager(self.config)
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
        load_start = time.time()
        
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
        from ..models.message import Message
        
        # Create request context for logging
        ctx = RequestContext(
            request_id=generate_request_id(),
            method="POST",
            path="/v1/chat/completions",
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
                llm_bawt.finalize_response(user_prompt, response, tool_context)

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
        from ..models.message import Message
        
        # Create request context for logging
        ctx = RequestContext(
            request_id=generate_request_id(),
            method="POST",
            path="/v1/chat/completions",
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
        try:
            model_alias, model_warnings = self._resolve_request_model(request.model, bot_id, local_mode)
        except Exception as e:
            log.api_error(ctx, str(e), 400)
            raise

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
        
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        
        chunk_queue: asyncio.Queue = asyncio.Queue()
        full_response_holder = [""]  # Use list to allow mutation in nested function
        tool_context_holder = [""]  # Store tool context from native tool calls
        timing_holder = [0.0, 0.0]  # [start_time, end_time]
        cancelled_holder = [False]  # Track if we were cancelled
        
        # Capture the event loop before entering the thread
        loop = asyncio.get_running_loop()
        
        # Start new generation (cancels and waits for any previous one)
        cancel_event, done_event = await self._start_generation()
        
        def _stream_to_queue():
            """Run streaming in a thread and push chunks to the async queue."""
            try:
                # Check if already cancelled before starting
                if cancel_event.is_set():
                    cancelled_holder[0] = True
                    return
                
                # Use llm_bawt.prepare_messages_for_query to get full context
                # (history from DB + memory + system prompt)
                messages = llm_bawt.prepare_messages_for_query(user_prompt)
                
                # Log what we're sending to the LLM (verbose mode)
                log.llm_context(messages)
                
                # Track when first token arrives
                timing_holder[0] = time.time()
                
                # Choose streaming method based on whether bot uses tools
                if llm_bawt.bot.uses_tools and llm_bawt.memory:
                    # Check if client supports native streaming with tools (OpenAI)
                    use_native_streaming = (
                        llm_bawt.client.supports_native_tools()
                        and llm_bawt.tool_format in ("native", "NATIVE_OPENAI")
                        and hasattr(llm_bawt.client, "stream_with_tools")
                    )

                    if use_native_streaming:
                        # Native streaming with tools - streams content AND handles tool calls
                        from ..tools.executor import ToolExecutor
                        from ..tools.formats import get_format_handler
                        from ..models.message import Message as Msg

                        log.debug("Using native streaming with tools")

                        tool_definitions = llm_bawt._get_tool_definitions()
                        handler = get_format_handler(llm_bawt.tool_format)
                        tools_schema = handler.get_tools_schema(tool_definitions)

                        executor = ToolExecutor(
                            memory_client=llm_bawt.memory,
                            profile_manager=llm_bawt.profile_manager,
                            search_client=llm_bawt.search_client,
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
                            max_iterations = 5
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

                                            tool_call_obj = ToolCall(name=name, arguments=args, raw_text="")
                                            result = executor.execute(tool_call_obj)
                                            tool_results.append({
                                                "tool_call_id": tc.get("id", ""),
                                                "content": result,
                                            })

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
                            return llm_bawt.client.stream_raw(msgs, stop=stop_sequences)

                        stream_iter = stream_with_tools(
                            messages=messages,
                            stream_fn=stream_fn,
                            memory_client=llm_bawt.memory,
                            profile_manager=llm_bawt.profile_manager,
                            search_client=llm_bawt.search_client,
                            model_lifecycle=llm_bawt.model_lifecycle,
                            config=llm_bawt.config,
                            user_id=llm_bawt.user_id,
                            bot_id=llm_bawt.bot_id,
                            tool_format=llm_bawt.tool_format,
                            adapter=adapter,
                            history_manager=llm_bawt.history_manager,
                        )
                else:
                    # Pass adapter stop sequences even without tools
                    adapter = getattr(llm_bawt, 'adapter', None)
                    adapter_stops = adapter.get_stop_sequences() if adapter else []
                    stream_iter = llm_bawt.client.stream_raw(
                        messages, stop=adapter_stops or None
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
                    llm_bawt.finalize_response(user_prompt, full_response_holder[0], tool_context_holder[0])

                    # Write debug turn log if enabled (check config or env var)
                    if self.config.DEBUG_TURN_LOG or os.environ.get("LLM_BAWT_DEBUG_TURN_LOG"):
                        _write_debug_turn_log(
                            prepared_messages=messages,
                            user_prompt=user_prompt,
                            response=full_response_holder[0],
                            model=model_alias,
                            bot_id=bot_id,
                            user_id=user_id,
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
                    # Error occurred
                    raise chunk
                
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
            if task.task_type == TaskType.MEMORY_EXTRACTION:
                result = await self._process_extraction(task)
            elif task.task_type == TaskType.CONTEXT_COMPACTION:
                result = await self._process_compaction(task)
            elif task.task_type == TaskType.EMBEDDING_GENERATION:
                result = await self._process_embeddings(task)
            elif task.task_type == TaskType.MEANING_UPDATE:
                result = await self._process_meaning_update(task)
            elif task.task_type == TaskType.MEMORY_MAINTENANCE:
                result = await self._process_maintenance(task)
            elif task.task_type == TaskType.PROFILE_MAINTENANCE:
                result = await self._process_profile_maintenance(task)
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
    
    async def _process_extraction(self, task: Task) -> dict:
        """Process a memory extraction task.
        
        Uses the same model that was used for the chat to avoid loading
        multiple models into VRAM. The model is passed in the task payload.
        """
        from ..memory.extraction import MemoryExtractionService
        
        messages = task.payload.get("messages", [])
        bot_id = task.bot_id
        user_id = task.user_id
        
        log.debug(f"Extraction: {len(messages)} messages for {bot_id}/{user_id}")
        
        # Get the model from task payload (passed from chat request)
        # This ensures we use the same model that handled the chat
        model_to_use = task.payload.get("model")
        
        if not model_to_use:
            # Fallback: no model specified in task, skip LLM extraction
            log.debug("No model specified in extraction task - using non-LLM extraction")
            extraction_service = MemoryExtractionService(llm_client=None)
            use_llm = False
        else:
            # Get client from cache - should already be loaded from chat
            extraction_client = None
            if model_to_use in self._client_cache:
                extraction_client = self._client_cache[model_to_use]
                log.debug(f"Reusing cached client for extraction: {model_to_use}")
            else:
                # Check if any LLMBawt instance has this model loaded
                for (cached_model, _, _), llm_bawt in self._llm_bawt_cache.items():
                    if cached_model == model_to_use:
                        extraction_client = llm_bawt.client
                        self._client_cache[model_to_use] = extraction_client
                        log.debug(f"Reusing client from LLMBawt instance for extraction: {model_to_use}")
                        break
            
            if not extraction_client:
                log.warning(f"Model '{model_to_use}' not in cache - skipping LLM extraction to avoid reload")
                extraction_client = None
            
            extraction_service = MemoryExtractionService(llm_client=extraction_client)
            use_llm = extraction_client is not None
        
        # Run extraction in the SAME single-threaded executor as chat completions
        # This ensures extraction waits for any in-flight chat to complete
        # The executor has max_workers=1, so operations are serialized
        loop = asyncio.get_event_loop()
        
        try:
            facts = await loop.run_in_executor(
                self._llm_executor,  # Use the single-threaded LLM executor
                lambda: extraction_service.extract_from_conversation(messages, use_llm=use_llm)
            )
        except Exception as e:
            # Catch any llama.cpp state corruption errors
            log.warning(f"Extraction failed (will retry without LLM): {e}")
            # Fallback to non-LLM extraction
            extraction_service_fallback = MemoryExtractionService(llm_client=None)
            facts = await loop.run_in_executor(
                self._llm_executor,
                lambda: extraction_service_fallback.extract_from_conversation(messages, use_llm=False)
            )
            use_llm = False
        
        if not facts:
            log.debug("Extraction: no facts found")
            return {"facts_extracted": 0, "facts_stored": 0, "llm_used": use_llm}

        log.debug(f"Extraction: {len(facts)} facts, checking duplicates")

        memory_client = self.get_memory_client(bot_id, user_id)
        stored_count = 0
        profile_count = 0
        skipped_count = 0

        if memory_client:
            min_importance = getattr(self.config, "MEMORY_EXTRACTION_MIN_IMPORTANCE", 0.3)
            profile_enabled = getattr(self.config, "MEMORY_PROFILE_ATTRIBUTE_ENABLED", True)

            # Filter facts by minimum importance first
            facts = [f for f in facts if f.importance >= min_importance]

            if not facts:
                log.debug("Extraction: no facts above importance threshold")
                return {"facts_extracted": 0, "facts_stored": 0, "llm_used": use_llm}

            # Fetch existing memories to check for duplicates
            existing_memories = memory_client.list_memories(limit=100, min_importance=0.0)

            if existing_memories:
                # Use determine_memory_actions to filter out duplicates
                actions = extraction_service.determine_memory_actions(
                    new_facts=facts,
                    existing_memories=existing_memories,
                )

                # Count how many facts were skipped (duplicates)
                skipped_count = len(facts) - len(actions)
                if skipped_count > 0:
                    log.debug(f"Extraction: skipped {skipped_count} duplicates")

                # Process only the actions (ADD, UPDATE, DELETE)
                for action in actions:
                    fact = action.fact
                    if not fact:
                        continue

                    log.debug(f"[Extraction] {action.action}: '{fact.content[:50]}...' importance={fact.importance:.2f}")

                    try:
                        if action.action == "ADD":
                            memory_client.add_memory(
                                content=fact.content,
                                tags=fact.tags,
                                importance=fact.importance,
                                source_message_ids=fact.source_message_ids,
                            )
                            stored_count += 1
                        elif action.action == "UPDATE" and action.target_memory_id:
                            memory_client.update_memory(
                                memory_id=action.target_memory_id,
                                content=fact.content,
                                importance=fact.importance,
                                tags=fact.tags,
                            )
                            stored_count += 1
                        elif action.action == "DELETE" and action.target_memory_id:
                            memory_client.delete_memory(memory_id=action.target_memory_id)

                        # Extract profile attributes for ADD and UPDATE actions
                        if action.action in ("ADD", "UPDATE") and profile_enabled:
                            from ..memory_server.extraction import extract_profile_attributes_from_fact
                            if extract_profile_attributes_from_fact(
                                fact=fact,
                                user_id=user_id,
                                config=self.config,
                            ):
                                profile_count += 1
                    except Exception as e:
                        log.warning(f"Failed to process memory action {action.action}: {e}")
            else:
                # No existing memories - store all facts directly
                for fact in facts:
                    log.debug(f"[Extraction] ADD (no existing): '{fact.content[:50]}...' importance={fact.importance:.2f}")
                    try:
                        memory_client.add_memory(
                            content=fact.content,
                            tags=fact.tags,
                            importance=fact.importance,
                            source_message_ids=fact.source_message_ids,
                        )
                        stored_count += 1
                        if profile_enabled:
                            from ..memory_server.extraction import extract_profile_attributes_from_fact
                            if extract_profile_attributes_from_fact(
                                fact=fact,
                                user_id=user_id,
                                config=self.config,
                            ):
                                profile_count += 1
                    except Exception as e:
                        log.warning(f"Failed to store memory: {e}")
        
        if stored_count > 0 or profile_count > 0:
            log.info(f"💾 Stored {stored_count} memories" + (f", {profile_count} profile attrs" if profile_count > 0 else ""))
        log.memory_operation("extraction", bot_id, count=stored_count, details=f"extracted={len(facts)}, stored={stored_count}, skipped={skipped_count}, profiles={profile_count}, llm={use_llm}")
        return {"facts_extracted": len(facts), "facts_stored": stored_count, "facts_skipped": skipped_count, "profile_attrs": profile_count, "llm_used": use_llm}
    
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
        
        Prefers a loaded local model to avoid VRAM churn and remote providers.
        Falls back to configured/local defaults if no cached model available.
        """
        from ..memory.profile_maintenance import ProfileMaintenanceService
        from ..profiles import ProfileManager
        
        entity_id = task.payload.get("entity_id", task.user_id)
        entity_type = task.payload.get("entity_type", "user")
        dry_run = task.payload.get("dry_run", False)
        def is_openai_model(alias: str | None) -> bool:
            if not alias:
                return False
            model_def = self.config.defined_models.get("models", {}).get(alias, {})
            return model_def.get("type") == "openai"
        
        def first_local_cached() -> str | None:
            for alias in self._client_cache.keys():
                if not is_openai_model(alias):
                    return alias
            for (alias, _, _), _llm_bawt in self._llm_bawt_cache.items():
                if not is_openai_model(alias):
                    return alias
            return None
        
        def first_local_available() -> str | None:
            for alias in self._available_models:
                if not is_openai_model(alias):
                    return alias
            return None
        
        requested_model = task.payload.get("model") or (self.config.PROFILE_MAINTENANCE_MODEL or None)
        model_to_use: str | None = None
        
        if requested_model:
            try:
                model_to_use, _ = self._resolve_request_model(
                    requested_model,
                    task.bot_id or self._default_bot,
                    local_mode=True,
                )
            except Exception as e:
                log.error(f"Failed to resolve model for profile maintenance: {e}")
                return {"error": f"Failed to resolve model: {e}"}

        # Prefer currently loaded local model if no explicit model requested
        if not model_to_use:
            current_model = self._model_lifecycle.current_model
            if current_model and not is_openai_model(current_model):
                model_to_use = current_model

        # Fall back to any cached local model
        if not model_to_use:
            model_to_use = first_local_cached()

        # Last resort: resolve a local default
        if not model_to_use:
            try:
                model_to_use, _ = self._resolve_request_model(
                    None,
                    task.bot_id or self._default_bot,
                    local_mode=True,
                )
            except Exception as e:
                log.error(f"Failed to resolve model for profile maintenance: {e}")
                return {"error": f"Failed to resolve model: {e}"}
        
        # Never use OpenAI for profile maintenance; fall back to any local model
        if not model_to_use or is_openai_model(model_to_use):
            fallback_local = first_local_cached() or first_local_available()
            if fallback_local and not is_openai_model(fallback_local):
                if model_to_use:
                    log.warning(
                        f"Profile maintenance requested model '{model_to_use}' is openai; "
                        f"using local '{fallback_local}' instead"
                    )
                model_to_use = fallback_local
            else:
                err = "No local model available for profile maintenance"
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
        current = self._model_lifecycle.current_model
        return ServiceStatusResponse(
            uptime_seconds=self.uptime_seconds,
            tasks_processed=self.tasks_processed,
            tasks_pending=self._task_queue.qsize(),
            models_loaded=[current] if current else [],
            current_model=current,
            available_models=self._available_models,
        )


# =============================================================================
# FastAPI Application
# =============================================================================

# Global service instance
_service: BackgroundService | None = None


def get_service() -> BackgroundService:
    """Get the background service instance."""
    global _service
    if _service is None:
        raise RuntimeError("Service not initialized")
    return _service


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan handler for startup/shutdown."""
    global _service
    
    # Startup
    config = Config()

    # Prefer MCP tool-based memory retrieval for llm-service.
    # This ensures memory retrieval happens via MCP tools and can be logged clearly.
    _ensure_memory_mcp_server(config)

    _service = BackgroundService(config)
    _service.start_worker()
    
    # Start job scheduler if enabled
    scheduler = None
    if config.SCHEDULER_ENABLED:
        from .scheduler import JobScheduler, create_scheduler_tables, init_default_jobs
        from ..profiles import ProfileManager
        
        # Get engine from profile manager (reuse existing connection)
        pm = ProfileManager(config)
        create_scheduler_tables(pm.engine)
        init_default_jobs(pm.engine, config)
        
        scheduler = JobScheduler(
            engine=pm.engine,
            task_processor=_service,
            check_interval=config.SCHEDULER_CHECK_INTERVAL_SECONDS,
        )
        await scheduler.start()
        log.info(f"📅 Scheduler started (interval={config.SCHEDULER_CHECK_INTERVAL_SECONDS}s)")
    
    # Log startup with rich formatting
    log.startup(
        version=SERVICE_VERSION,
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        models=_service._available_models,
        default_model=_service._default_model,
    )

    log.info(
        "Memory mode: %s (%s)",
        "mcp" if getattr(config, "MEMORY_SERVER_URL", None) else "embedded",
        getattr(config, "MEMORY_SERVER_URL", ""),
    )
    
    yield
    
    # Shutdown
    if scheduler:
        await scheduler.stop()
    await _service.shutdown()


# Create FastAPI app
try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    
    app = FastAPI(
        title="llm-bawt API",
        description="OpenAI-compatible API with integrated memory system",
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )
    
    # -------------------------------------------------------------------------
    # Health & Status Endpoints
    # -------------------------------------------------------------------------
    
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse()
    
    @app.get("/status", response_model=ServiceStatusResponse, tags=["System"])
    async def get_status():
        """Get detailed service status."""
        return get_service().get_status()

    # -------------------------------------------------------------------------
    # Nextcloud Talk Webhook
    # -------------------------------------------------------------------------

    @app.post("/webhook/nextcloud", tags=["Webhooks"])
    async def nextcloud_talk_webhook(request: Request):
        """Handle incoming Nextcloud Talk webhooks."""
        from ..integrations.nextcloud.webhook import handle_nextcloud_webhook
        return await handle_nextcloud_webhook(request)

    # -------------------------------------------------------------------------
    # Admin: Nextcloud Talk Provisioning
    # -------------------------------------------------------------------------

    class NextcloudProvisionRequest(BaseModel):
        """Request to provision a Nextcloud Talk room and bot."""
        bot_id: str = Field(description="llm-bawt bot ID (e.g., nova, monika)")
        room_name: str | None = Field(default=None, description="Room name (default: bot name)")
        bot_name: str | None = Field(default=None, description="Bot display name (default: bot ID)")
        owner_user_id: str = Field(default="user", description="Room owner")

    class NextcloudProvisionResponse(BaseModel):
        """Response from provisioning."""
        bot_id: str
        room_token: str
        room_url: str
        nextcloud_bot_id: int
        nextcloud_bot_name: str

    @app.post("/admin/nextcloud-talk/provision", response_model=NextcloudProvisionResponse, tags=["Admin"])
    async def provision_nextcloud_talk(request: NextcloudProvisionRequest):
        """Provision a Nextcloud Talk room and bot for an llm_bawt bot."""
        from ..integrations.nextcloud.provisioner import get_provisioner_client
        from ..integrations.nextcloud.manager import get_nextcloud_manager

        manager = get_nextcloud_manager()

        # Check if already configured
        if manager.get_bot(request.bot_id):
            raise HTTPException(
                status_code=400,
                detail=f"Bot '{request.bot_id}' already has Nextcloud config"
            )

        # Defaults
        room_name = request.room_name or request.bot_id.title()
        bot_name = request.bot_name or request.bot_id.title()

        try:
            provisioner = get_provisioner_client()

            # Provision via service
            result = await provisioner.provision_talk_room_and_bot(
                room_name=room_name,
                bot_name=bot_name,
                owner_user_id=request.owner_user_id,
            )

            # Save config
            manager.add_bot(
                llm_bawt_bot=request.bot_id,
                nextcloud_bot_id=result.bot_id,
                secret=result.bot_secret,
                conversation_token=result.room_token,
            )

            return NextcloudProvisionResponse(
                bot_id=request.bot_id,
                room_token=result.room_token,
                room_url=result.room_url,
                nextcloud_bot_id=result.bot_id,
                nextcloud_bot_name=result.bot_name,
            )

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("Provisioning failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/admin/nextcloud-talk/reload", tags=["Admin"])
    async def reload_nextcloud_bots():
        """Force reload Nextcloud bot configuration from disk."""
        from ..integrations.nextcloud.manager import get_nextcloud_manager
        manager = get_nextcloud_manager()
        manager.reload()
        bots = manager.list_bots()
        log.info(f"🔄 Reloaded {len(bots)} Nextcloud bots: {[b.llm_bawt_bot for b in bots]}")
        return {
            "status": "reloaded",
            "bots_count": len(bots),
            "bots": [b.llm_bawt_bot for b in bots],
        }

    # -------------------------------------------------------------------------
    # OpenAI-Compatible Endpoints
    # -------------------------------------------------------------------------
    
    @app.get("/v1/models", response_model=ModelsResponse, tags=["OpenAI Compatible"])
    async def list_models():
        """List available models (OpenAI-compatible)."""
        service = get_service()
        models = [
            ModelInfo(id=alias)
            for alias in service._available_models
        ]
        return ModelsResponse(data=models)

    @app.get("/v1/models/current", tags=["Models"])
    async def get_current_model():
        """Get the currently active model."""
        service = get_service()
        current = service.model_lifecycle.current_model
        if not current:
            return {"model": None, "message": "No model currently loaded"}
        info = service.model_lifecycle.get_model_info(current)
        detail = ModelDetail(
            id=current,
            type=info.get("type") if info else None,
            model_id=info.get("model_id", info.get("repo_id")) if info else None,
            description=info.get("description") if info else None,
            current=True,
        )
        return {"model": detail}

    @app.post("/v1/models/switch", response_model=ModelSwitchResponse, tags=["Models"])
    async def switch_model(request: ModelSwitchRequest):
        """Switch to a different model. Takes effect on the next request."""
        service = get_service()
        previous = service.model_lifecycle.current_model
        success, message = service.model_lifecycle.switch_model(request.model)
        if not success:
            raise HTTPException(status_code=400, detail=message)
        return ModelSwitchResponse(
            success=True,
            message=message,
            previous_model=previous,
            new_model=request.model,
        )

    @app.get("/v1/bots", response_model=BotsResponse, tags=["System"])
    async def list_bots():
        """List available bots configured on the service."""
        service = get_service()
        bot_manager = BotManager(service.config)
        bots = [
            BotInfo(slug=bot.slug, name=bot.name, description=bot.description)
            for bot in bot_manager.list_bots()
        ]
        return BotsResponse(data=bots)
    
    @app.post("/v1/chat/completions", tags=["OpenAI Compatible"])
    async def chat_completions(request: ChatCompletionRequest):
        """
        Create a chat completion (OpenAI-compatible).
        
        Supports all standard OpenAI parameters plus llm_bawt extensions:
        - `bot_id`: Bot personality to use (default: nova)
        - `augment_memory`: Whether to include memory context (default: true)
        - `extract_memory`: Whether to extract memories from response (default: true)
        """
        from fastapi.responses import StreamingResponse
        
        service = get_service()
        
        # Log request BEFORE validation so we can debug failures
        log.debug(f"Request payload: {request.model_dump(exclude_none=True)}")
        
        if request.stream:
            # Streaming response
            try:
                return StreamingResponse(
                    service.chat_completion_stream(request),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                log.exception("Streaming chat completion failed")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Non-streaming
        try:
            response = await service.chat_completion(request)
            return response
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("Chat completion failed")
            raise HTTPException(status_code=500, detail=str(e))
    
    # -------------------------------------------------------------------------
    # Task Management Endpoints
    # -------------------------------------------------------------------------
    
    @app.post("/v1/tasks", response_model=TaskSubmitResponse, tags=["Tasks"])
    async def submit_task(request: TaskSubmitRequest):
        """Submit a background task for processing."""
        service = get_service()
        
        try:
            task_type = TaskType(request.task_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task type: {request.task_type}"
            )
        
        task = Task(
            task_type=task_type,
            payload=request.payload,
            bot_id=request.bot_id or service._default_bot,
            user_id=request.user_id,
            priority=request.priority,
        )
        
        task_id = service.submit_task(task)
        return TaskSubmitResponse(task_id=task_id)
    
    @app.get("/v1/tasks/{task_id}", response_model=TaskStatusResponse, tags=["Tasks"])
    async def get_task_status(
        task_id: str,
        wait: bool = Query(False, description="Wait for task completion"),
        timeout: float = Query(30.0, description="Wait timeout in seconds"),
    ):
        """Get the status of a submitted task."""
        service = get_service()
        
        if wait:
            result = await service.wait_for_result(task_id, timeout)
        else:
            result = service.get_result(task_id)
        
        if result:
            return TaskStatusResponse(
                task_id=task_id,
                status=result.status.value,
                result=result.result,
                error=result.error,
                processing_time_ms=result.processing_time_ms,
            )
        else:
            return TaskStatusResponse(task_id=task_id, status="pending")

    # -------------------------------------------------------------------------
    # History Endpoints
    # -------------------------------------------------------------------------

    @app.get("/v1/history", response_model=HistoryResponse, tags=["History"])
    async def get_history(
        bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
        limit: int = Query(50, description="Maximum number of messages to return"),
    ):
        """Get conversation history for a bot."""
        service = get_service()
        
        effective_bot_id = bot_id or service._default_bot
        
        try:
            # Use memory client to get messages from database
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")
            
            # Get messages directly from database (no time filter)
            messages = client.get_messages(since_seconds=None)  # Get all messages
            
            # Apply limit (from most recent)
            if limit > 0 and len(messages) > limit:
                messages = messages[-limit:]
            
            history_messages = [
                HistoryMessage(
                    id=msg.get("id"),
                    role=msg.get("role", ""),
                    content=msg.get("content", ""),
                    timestamp=msg.get("timestamp", 0.0)
                )
                for msg in messages
                if msg.get("role") != "system"  # Don't include system messages
            ]
            
            return HistoryResponse(
                bot_id=effective_bot_id,
                messages=history_messages,
                total_count=len(history_messages)
            )
        except Exception as e:
            log.error(f"Failed to get history: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/history/search", response_model=HistorySearchResponse, tags=["History"])
    async def search_history(
        query: str = Query(..., description="Search query"),
        bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
        limit: int = Query(50, description="Maximum number of messages to return"),
    ):
        """Search conversation history for a bot."""
        service = get_service()
        
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")
            
            # Get all messages and filter by query
            messages = client.get_messages(since_seconds=None)
            query_lower = query.lower()
            
            matching = [
                msg for msg in messages
                if msg.get("role") != "system" and query_lower in msg.get("content", "").lower()
            ]
            
            # Apply limit
            if limit > 0 and len(matching) > limit:
                matching = matching[-limit:]
            
            history_messages = [
                HistoryMessage(
                    id=msg.get("id"),
                    role=msg.get("role", ""),
                    content=msg.get("content", ""),
                    timestamp=msg.get("timestamp", 0.0)
                )
                for msg in matching
            ]
            
            return HistorySearchResponse(
                bot_id=effective_bot_id,
                query=query,
                messages=history_messages,
                total_count=len(history_messages)
            )
        except Exception as e:
            log.error(f"Failed to search history: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/v1/history", response_model=HistoryClearResponse, tags=["History"])
    async def clear_history(
        bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
    ):
        """Clear conversation history for a bot."""
        service = get_service()
        
        effective_bot_id = bot_id or service._default_bot
        
        try:
            model_alias = list(service._available_models)[0] if service._available_models else None
            if not model_alias:
                raise HTTPException(status_code=500, detail="No models available")
            
            llm_bawt = service._get_llm_bawt(model_alias, effective_bot_id, service.config.DEFAULT_USER)
            llm_bawt.history_manager.clear_history()
            
            # Also remove from cache to force fresh state
            cache_key = (model_alias, effective_bot_id, service.config.DEFAULT_USER)
            if cache_key in service._llm_bawt_cache:
                del service._llm_bawt_cache[cache_key]
            
            return HistoryClearResponse(
                success=True,
                message=f"History cleared for bot '{effective_bot_id}'"
            )
        except Exception as e:
            log.error(f"Failed to clear history: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # -------------------------------------------------------------------------
    # User Profile Endpoints
    # -------------------------------------------------------------------------

    @app.get("/v1/users", response_model=UserListResponse, tags=["Users"])
    async def list_users():
        """List all user profiles."""
        service = get_service()
        
        try:
            from ..profiles import ProfileManager, EntityType
            
            manager = ProfileManager(service.config)
            profiles = manager.list_profiles(EntityType.USER)
            
            users = []
            for profile in profiles:
                attr_count = len(manager.get_all_attributes(EntityType.USER, profile.entity_id))
                users.append(UserProfileSummary(
                    user_id=profile.entity_id,
                    display_name=profile.display_name,
                    description=profile.description,
                    attribute_count=attr_count,
                    created_at=profile.created_at.isoformat() if profile.created_at else None,
                ))
            
            return UserListResponse(
                users=users,
                total_count=len(users),
            )
        except Exception as e:
            log.error(f"Failed to list users: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/users/{user_id}", response_model=UserProfileDetail, tags=["Users"])
    async def get_user_profile(user_id: str):
        """Get detailed user profile with attributes."""
        service = get_service()
        
        try:
            from ..profiles import ProfileManager, EntityType
            
            manager = ProfileManager(service.config)
            profile = manager.get_profile(EntityType.USER, user_id)
            
            if not profile:
                raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
            
            attributes = manager.get_all_attributes(EntityType.USER, user_id)
            
            return UserProfileDetail(
                user_id=user_id,
                display_name=profile.display_name,
                description=profile.description,
                attributes=[
                    UserProfileAttribute(
                        id=attr.id,
                        category=attr.category.value if hasattr(attr.category, 'value') else str(attr.category),
                        key=attr.key,
                        value=attr.value,
                        confidence=attr.confidence,
                        source=attr.source,
                        created_at=attr.created_at.isoformat() if attr.created_at else None,
                        updated_at=attr.updated_at.isoformat() if attr.updated_at else None,
                    )
                    for attr in attributes
                ],
                created_at=profile.created_at.isoformat() if profile.created_at else None,
            )
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to get user profile: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/v1/users/attribute/{attribute_id}", tags=["Users"])
    async def delete_user_attribute(attribute_id: int):
        """Delete a user profile attribute by its ID."""
        service = get_service()
        
        try:
            from ..profiles import ProfileManager
            
            manager = ProfileManager(service.config)
            
            # First get the attribute to confirm it exists and show what we're deleting
            attr = manager.get_attribute_by_id(attribute_id)
            if not attr:
                raise HTTPException(status_code=404, detail=f"Attribute with ID {attribute_id} not found")
            
            success = manager.delete_attribute_by_id(attribute_id)
            if success:
                return {
                    "success": True,
                    "message": f"Deleted attribute {attr.category}.{attr.key} from {attr.entity_type}/{attr.entity_id}",
                    "deleted": {
                        "id": attribute_id,
                        "entity_type": str(attr.entity_type),
                        "entity_id": attr.entity_id,
                        "category": attr.category,
                        "key": attr.key,
                        "value": attr.value,
                    }
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to delete attribute")
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to delete user attribute: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # Unified Profile Endpoints
    # =========================================================================

    @app.get("/v1/profiles/{entity_id}", response_model=ProfileDetail, tags=["Profiles"])
    async def get_profile_auto(entity_id: str):
        """Get profile with attributes - auto-detects entity type (user or bot).

        Since entity IDs are unique across users and bots, this endpoint automatically
        determines whether the entity is a user or bot by checking both tables.
        The response includes the entity_type field to indicate what was found.

        Note: For listing all profiles of a type, use GET /v1/profiles/list/{entity_type}
        """
        # Reject reserved words that should use the list endpoint
        if entity_id in ("user", "bot"):
            raise HTTPException(
                status_code=400,
                detail=f"'{entity_id}' is a reserved word. To list all {entity_id} profiles, use GET /v1/profiles/list/{entity_id}"
            )

        service = get_service()

        try:
            from ..profiles import ProfileManager, EntityType

            manager = ProfileManager(service.config)

            # Try USER first, then BOT
            profile = manager.get_profile(EntityType.USER, entity_id)
            entity_type_str = "user"
            entity_type_enum = EntityType.USER

            if not profile:
                profile = manager.get_profile(EntityType.BOT, entity_id)
                entity_type_str = "bot"
                entity_type_enum = EntityType.BOT

            if not profile:
                raise HTTPException(
                    status_code=404,
                    detail=f"No profile found for entity '{entity_id}'"
                )

            attributes = manager.get_all_attributes(entity_type_enum, entity_id)

            return ProfileDetail(
                entity_type=entity_type_str,
                entity_id=entity_id,
                display_name=profile.display_name,
                description=profile.description,
                summary=profile.summary,
                attributes=[
                    UserProfileAttribute(
                        id=attr.id,
                        category=attr.category.value if hasattr(attr.category, 'value') else str(attr.category),
                        key=attr.key,
                        value=attr.value,
                        confidence=attr.confidence,
                        source=attr.source,
                        created_at=attr.created_at.isoformat() if attr.created_at else None,
                        updated_at=attr.updated_at.isoformat() if attr.updated_at else None,
                    )
                    for attr in attributes
                ],
                created_at=profile.created_at.isoformat() if profile.created_at else None,
            )
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to get profile for '{entity_id}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/profiles/list/{entity_type}", response_model=ProfileListResponse, tags=["Profiles"])
    async def list_profiles(entity_type: str):
        """List all profiles of a given type (user or bot).

        Use entity_type='user' for user profiles, 'bot' for bot profiles.
        """
        service = get_service()

        # Validate entity_type
        if entity_type not in ("user", "bot"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid entity_type '{entity_type}'. Must be 'user' or 'bot'."
            )

        try:
            from ..profiles import ProfileManager, EntityType

            # Convert string to EntityType enum
            entity_type_enum = EntityType.USER if entity_type == "user" else EntityType.BOT

            manager = ProfileManager(service.config)
            profiles = manager.list_profiles(entity_type_enum)

            result_profiles = []
            for profile in profiles:
                attributes = manager.get_all_attributes(entity_type_enum, profile.entity_id)
                result_profiles.append(ProfileDetail(
                    entity_type=entity_type,
                    entity_id=profile.entity_id,
                    display_name=profile.display_name,
                    description=profile.description,
                    summary=profile.summary,
                    attributes=[
                        UserProfileAttribute(
                            id=attr.id,
                            category=attr.category.value if hasattr(attr.category, 'value') else str(attr.category),
                            key=attr.key,
                            value=attr.value,
                            confidence=attr.confidence,
                            source=attr.source,
                            created_at=attr.created_at.isoformat() if attr.created_at else None,
                            updated_at=attr.updated_at.isoformat() if attr.updated_at else None,
                        )
                        for attr in attributes
                    ],
                    created_at=profile.created_at.isoformat() if profile.created_at else None,
                ))

            return ProfileListResponse(
                profiles=result_profiles,
                total_count=len(result_profiles),
            )
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to list {entity_type} profiles: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/profiles/{entity_type}/{entity_id}", response_model=ProfileDetail, tags=["Profiles"])
    async def get_profile(entity_type: str, entity_id: str):
        """Get profile with attributes for any entity type (user or bot).

        DEPRECATED: Use GET /v1/profiles/{entity_id} instead for auto-detection.
        This endpoint is maintained for backward compatibility.
        """
        service = get_service()

        # Validate entity_type
        if entity_type not in ("user", "bot"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid entity_type '{entity_type}'. Must be 'user' or 'bot'."
            )

        try:
            from ..profiles import ProfileManager, EntityType

            # Convert string to EntityType enum
            entity_type_enum = EntityType.USER if entity_type == "user" else EntityType.BOT

            manager = ProfileManager(service.config)
            profile = manager.get_profile(entity_type_enum, entity_id)

            if not profile:
                raise HTTPException(
                    status_code=404,
                    detail=f"{entity_type.capitalize()} '{entity_id}' not found"
                )

            attributes = manager.get_all_attributes(entity_type_enum, entity_id)

            return ProfileDetail(
                entity_type=entity_type,
                entity_id=entity_id,
                display_name=profile.display_name,
                description=profile.description,
                summary=profile.summary,
                attributes=[
                    UserProfileAttribute(
                        id=attr.id,
                        category=attr.category.value if hasattr(attr.category, 'value') else str(attr.category),
                        key=attr.key,
                        value=attr.value,
                        confidence=attr.confidence,
                        source=attr.source,
                        created_at=attr.created_at.isoformat() if attr.created_at else None,
                        updated_at=attr.updated_at.isoformat() if attr.updated_at else None,
                    )
                    for attr in attributes
                ],
                created_at=profile.created_at.isoformat() if profile.created_at else None,
            )
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to get {entity_type} profile: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # -------------------------------------------------------------------------
    # Memory Management Endpoints
    # -------------------------------------------------------------------------

    @app.get("/v1/memory/stats", response_model=MemoryStatsResponse, tags=["Memory"])
    async def get_memory_stats(
        bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
    ):
        """Get memory statistics for a bot."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")
            stats = client.stats()
            return MemoryStatsResponse(
                bot_id=effective_bot_id,
                messages=stats.get("messages", {}),
                memories=stats.get("memories", {})
            )
        except Exception as e:
            log.error(f"Failed to get memory stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/memory/search", response_model=MemorySearchResponse, tags=["Memory"])
    async def search_memory(request: MemorySearchRequest):
        """Search memories."""
        service = get_service()
        effective_bot_id = request.bot_id or service._default_bot

        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")
            
            results = []
            if request.method in ("embedding", "all"):
                # Semantic search
                memories = client.search(
                    request.query,
                    n_results=request.limit,
                    min_relevance=request.min_importance,
                )
                for mem in memories:
                    results.append(MemoryItem(
                        id=str(getattr(mem, "id", "")),
                        content=str(getattr(mem, "content", "")),
                        importance=float(getattr(mem, "importance", 0.5)),
                        relevance=getattr(mem, "relevance", None),
                        tags=list(getattr(mem, "tags", []) or []),
                        created_at=getattr(mem, "created_at", None),
                        access_count=0,
                    ))
            elif request.method == "high-importance":
                # Get high importance memories
                memories = client.list_memories(
                    limit=request.limit,
                    min_importance=request.min_importance or 0.7,
                )
                for mem in memories:
                    results.append(MemoryItem(
                        id=str(mem.get("id", "")),
                        content=mem.get("content", ""),
                        importance=mem.get("importance", 0.5),
                        tags=mem.get("tags", []),
                        created_at=mem.get("created_at"),
                    ))
            
            return MemorySearchResponse(
                bot_id=effective_bot_id,
                method=request.method,
                query=request.query,
                results=results,
                total_count=len(results)
            )
        except Exception as e:
            log.error(f"Failed to search memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/memory", response_model=MemorySearchResponse, tags=["Memory"])
    async def list_memories(
        bot_id: str = Query(None, description="Bot ID"),
        limit: int = Query(20, description="Max results"),
    ):
        """List all memories for a bot (ordered by importance)."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            memories = client.list_memories(limit=limit, min_importance=0.0)
            
            results = [
                MemoryItem(
                    id=str(mem.get("id", "")),
                    content=mem.get("content", ""),
                    importance=mem.get("importance", 0.5),
                    tags=mem.get("tags", []),
                    created_at=mem.get("created_at"),
                    access_count=mem.get("access_count", 0),
                )
                for mem in memories
            ]
            
            return MemorySearchResponse(
                bot_id=effective_bot_id,
                method="list",
                query="",
                results=results,
                total_count=len(results)
            )
        except Exception as e:
            log.error(f"Failed to list memories: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/memory/message", tags=["Memory"])
    async def get_message_by_id(
        message_id: str = Query(..., description="Message ID (supports prefix match)"),
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Get a specific message by ID."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            # Get the message
            message = client.get_message_by_id(message_id)
            
            if message:
                return {"message": message}
            else:
                return {"error": f"Message '{message_id}' not found"}
        except Exception as e:
            log.error(f"Failed to get message: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/v1/memory/{memory_id}", response_model=MemoryDeleteResponse, tags=["Memory"])
    async def delete_memory(
        memory_id: str,
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Delete a specific memory by ID."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            # The memory_id might be a prefix - try to find the full ID
            success = client.delete_memory(memory_id)
            
            if success:
                return MemoryDeleteResponse(
                    success=True,
                    memory_id=memory_id,
                    message=f"Memory '{memory_id}' deleted"
                )
            else:
                raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to delete memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/memory/forget", response_model=MemoryForgetResponse, tags=["Memory"])
    async def forget_messages(
        request: MemoryForgetRequest,
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Forget recent messages (soft delete)."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            if request.message_id:
                success = client.ignore_message_by_id(request.message_id)
                if success:
                    return MemoryForgetResponse(
                        success=True,
                        messages_ignored=1,
                        memories_deleted=0,
                        message=f"Ignored message {request.message_id[:8]}..."
                    )
                else:
                    raise HTTPException(status_code=404, detail=f"Message {request.message_id} not found")
            elif request.count:
                result = client.forget_recent_messages(request.count)
            elif request.minutes:
                result = client.forget_messages_since_minutes(request.minutes)
            else:
                raise HTTPException(status_code=400, detail="Must specify count, minutes, or message_id")

            messages_ignored = int(result.get("messages_ignored", 0))
            memories_deleted = int(result.get("memories_deleted", 0))
            
            return MemoryForgetResponse(
                success=True,
                messages_ignored=messages_ignored,
                memories_deleted=memories_deleted,
                message=f"Ignored {messages_ignored} messages, deleted {memories_deleted} memories"
            )
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to forget messages: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/memory/restore", response_model=MemoryRestoreResponse, tags=["Memory"])
    async def restore_messages(
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Restore ignored messages."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            restored = client.restore_ignored_messages()
            
            return MemoryRestoreResponse(
                success=True,
                messages_restored=restored,
                message=f"Restored {restored} messages"
            )
        except Exception as e:
            log.error(f"Failed to restore messages: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/memory/preview/recent", response_model=MessagesPreviewResponse, tags=["Memory"])
    async def preview_recent_messages(
        count: int = Query(10, description="Number of recent messages to preview"),
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Preview recent messages before forgetting."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            messages = client.preview_recent_messages(count)
            
            return MessagesPreviewResponse(
                bot_id=effective_bot_id,
                messages=[
                    MessagePreview(
                        id=msg["id"],
                        role=msg.get("role", "?"),
                        content=msg.get("content", ""),
                        timestamp=msg.get("timestamp"),
                    )
                    for msg in messages
                ],
                total_count=len(messages),
            )
        except Exception as e:
            log.error(f"Failed to preview messages: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/memory/preview/minutes", response_model=MessagesPreviewResponse, tags=["Memory"])
    async def preview_messages_since_minutes(
        minutes: int = Query(..., description="Number of minutes to look back"),
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Preview messages from last N minutes before forgetting."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            messages = client.preview_messages_since_minutes(minutes)
            
            return MessagesPreviewResponse(
                bot_id=effective_bot_id,
                messages=[
                    MessagePreview(
                        id=msg["id"],
                        role=msg.get("role", "?"),
                        content=msg.get("content", ""),
                        timestamp=msg.get("timestamp"),
                    )
                    for msg in messages
                ],
                total_count=len(messages),
            )
        except Exception as e:
            log.error(f"Failed to preview messages: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/memory/preview/ignored", response_model=MessagesPreviewResponse, tags=["Memory"])
    async def preview_ignored_messages(
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Preview ignored messages before restoring."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            messages = client.preview_ignored_messages()
            
            return MessagesPreviewResponse(
                bot_id=effective_bot_id,
                messages=[
                    MessagePreview(
                        id=msg["id"],
                        role=msg.get("role", "?"),
                        content=msg.get("content", ""),
                        timestamp=msg.get("timestamp"),
                    )
                    for msg in messages
                ],
                total_count=len(messages),
            )
        except Exception as e:
            log.error(f"Failed to preview ignored messages: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/memory/regenerate-embeddings", response_model=RegenerateEmbeddingsResponse, tags=["Memory"])
    async def regenerate_embeddings(
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Regenerate embeddings for all memories."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            result = client.regenerate_embeddings()
            
            if "error" in result:
                return RegenerateEmbeddingsResponse(
                    success=False,
                    updated=0,
                    failed=0,
                    message=result["error"]
                )
            
            return RegenerateEmbeddingsResponse(
                success=True,
                updated=result.get("updated", 0),
                failed=result.get("failed", 0),
                embedding_dim=result.get("embedding_dim"),
                message=f"Updated {result.get('updated', 0)} embeddings"
            )
        except Exception as e:
            log.error(f"Failed to regenerate embeddings: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/memory/consolidate", response_model=ConsolidateResponse, tags=["Memory"])
    async def consolidate_memories(
        request: ConsolidateRequest,
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Find and merge redundant memories."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            result = client.consolidate_memories(
                dry_run=request.dry_run,
                similarity_threshold=request.similarity_threshold,
            )
            
            return ConsolidateResponse(
                success=True,
                dry_run=bool(result.get("dry_run", request.dry_run)),
                clusters_found=int(result.get("clusters_found", 0)),
                clusters_merged=int(result.get("clusters_merged", 0)),
                memories_consolidated=int(result.get("memories_consolidated", 0)),
                new_memories_created=int(result.get("new_memories_created", 0)),
                errors=list(result.get("errors", [])),
                message=f"{'Would merge' if request.dry_run else 'Merged'} {int(result.get('clusters_merged', 0))} clusters",
            )
        except Exception as e:
            log.error(f"Failed to consolidate memories: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/llm/complete", response_model=RawCompletionResponse, tags=["LLM"])
    async def raw_completion(request: RawCompletionRequest):
        """Raw LLM completion using the currently loaded model.
        
        Use this for utility tasks like:
        - Memory consolidation (merging similar memories)
        - Summarization
        - Classification
        - Any task that needs LLM but not the full chat pipeline
        
        This endpoint uses the already-loaded model in the service.
        It will NOT load a new model - if no model is loaded, it returns 503.
        Only one LLM model can be loaded at a time (embedding model is separate).
        """
        import time
        service = get_service()
        
        # Check if we have any loaded client
        if not service._client_cache:
            raise HTTPException(
                status_code=503, 
                detail="No model loaded. Make a chat request first to load a model."
            )
        
        # Use the currently loaded model (there should only be one)
        loaded_models = list(service._client_cache.keys())
        model_alias = loaded_models[0]  # Use whatever is loaded
        
        # If caller specified a model, warn if it doesn't match
        if request.model and request.model != model_alias:
            log.debug(f"Requested model '{request.model}' but using loaded model '{model_alias}'")
        
        try:
            start = time.perf_counter()
            client = service._client_cache[model_alias]
            
            # Build messages as Message objects (required by client.query)
            from ..models.message import Message
            messages = []
            if request.system:
                messages.append(Message(role="system", content=request.system))
            messages.append(Message(role="user", content=request.prompt))
            
            # Query the model directly
            response = client.query(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # Estimate tokens
            tokens = len(response) // 4 if response else 0
            
            return RawCompletionResponse(
                content=response,
                model=model_alias,
                tokens=tokens,
                elapsed_ms=elapsed_ms,
            )
            
        except Exception as e:
            log.error(f"Raw completion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # History Summarization Endpoints
    # =========================================================================

    @app.get("/v1/history/summarize/preview", response_model=SummarizePreviewResponse, tags=["History"])
    async def preview_summarizable_sessions(
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Preview sessions that would be summarized (dry run)."""
        from datetime import datetime
        from ..memory.summarization import HistorySummarizer

        service = get_service()
        effective_bot_id = bot_id or service._default_bot

        try:
            summarizer = HistorySummarizer(service.config, effective_bot_id)
            sessions = summarizer.preview_summarizable_sessions()

            session_infos = []
            total_messages = 0

            for session in sessions:
                total_messages += session.message_count

                # Get first and last user messages for preview
                user_msgs = [m for m in session.messages if m.get("role") == "user"]
                first_msg = user_msgs[0].get("content", "")[:100] if user_msgs else ""
                last_msg = user_msgs[-1].get("content", "")[:100] if len(user_msgs) > 1 else first_msg

                session_infos.append(SummarizableSession(
                    start_timestamp=session.start_timestamp,
                    end_timestamp=session.end_timestamp,
                    start_time=datetime.fromtimestamp(session.start_timestamp).strftime("%Y-%m-%d %H:%M"),
                    end_time=datetime.fromtimestamp(session.end_timestamp).strftime("%H:%M"),
                    message_count=session.message_count,
                    first_message=first_msg,
                    last_message=last_msg,
                ))

            return SummarizePreviewResponse(
                bot_id=effective_bot_id,
                sessions=session_infos,
                total_messages=total_messages,
            )
        except Exception as e:
            log.error(f"Failed to preview summarizable sessions: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/history/summarize", response_model=SummarizeResponse, tags=["History"])
    async def summarize_history(
        bot_id: str = Query(None, description="Bot ID"),
        use_heuristic: bool = Query(False, description="Fall back to heuristic if LLM fails"),
    ):
        """Summarize eligible history sessions."""
        from ..memory.summarization import HistorySummarizer, format_session_for_summarization, SUMMARIZATION_PROMPT
        from ..models.message import Message

        service = get_service()
        effective_bot_id = bot_id or service._default_bot

        # Create a summarize function that uses the loaded client directly
        # This avoids the HTTP self-call deadlock
        def summarize_with_loaded_client(session) -> str | None:
            """Summarize using the already-loaded LLM client."""
            if not service._client_cache:
                log.warning("No model loaded for summarization")
                return None
            
            # Get the loaded client
            model_alias = list(service._client_cache.keys())[0]
            client = service._client_cache[model_alias]
            
            # Build the prompt
            conversation_text = format_session_for_summarization(session)
            prompt = SUMMARIZATION_PROMPT.format(messages=conversation_text)
            
            # Check for token limits (rough estimate: 4 chars per token)
            estimated_tokens = len(prompt) // 4
            if estimated_tokens > 6000:  # Leave room for response
                log.warning(f"Session too large ({estimated_tokens} estimated tokens), needs chunking")
                return None  # Will trigger chunked summarization
            
            try:
                messages = [
                    Message(role="system", content="You are a helpful assistant that summarizes conversations concisely."),
                    Message(role="user", content=prompt),
                ]
                response = client.query(
                    messages=messages,
                    max_tokens=200,
                    temperature=0.3,
                    plaintext_output=True,  # No rich formatting for background tasks
                    stream=False,  # Don't stream for summarization
                )
                
                # Check for error responses in content
                if response:
                    error_indicators = ["Error:", "exception occurred", "exceed context window", "tokens exceed"]
                    for indicator in error_indicators:
                        if indicator.lower() in response.lower():
                            log.error(f"LLM returned error: {response[:100]}...")
                            return None
                
                return response.strip() if response else None
            except Exception as e:
                log.error(f"LLM summarization failed: {e}")
                return None

        try:
            summarizer = HistorySummarizer(
                service.config, 
                effective_bot_id,
                summarize_fn=summarize_with_loaded_client,
            )
            result = summarizer.summarize_eligible_sessions(use_heuristic_fallback=use_heuristic)

            return SummarizeResponse(
                success=True,
                sessions_summarized=result.get("sessions_summarized", 0),
                messages_summarized=result.get("messages_summarized", 0),
                errors=result.get("errors", []),
            )
        except Exception as e:
            log.error(f"Failed to summarize history: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/history/summaries", response_model=ListSummariesResponse, tags=["History"])
    async def list_summaries(
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """List existing history summaries."""
        from datetime import datetime
        from ..memory.summarization import HistorySummarizer

        service = get_service()
        effective_bot_id = bot_id or service._default_bot

        try:
            summarizer = HistorySummarizer(service.config, effective_bot_id)
            summaries = summarizer.list_summaries()

            summary_infos = []
            for summ in summaries:
                start_ts = summ.get("session_start")
                end_ts = summ.get("session_end")

                summary_infos.append(SummaryInfo(
                    id=summ.get("id", ""),
                    content=summ.get("content", ""),
                    timestamp=summ.get("timestamp", 0),
                    session_start_time=datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M") if start_ts else None,
                    session_end_time=datetime.fromtimestamp(end_ts).strftime("%H:%M") if end_ts else None,
                    message_count=summ.get("message_count", 0),
                    method=summ.get("method", "unknown"),
                ))

            return ListSummariesResponse(
                bot_id=effective_bot_id,
                summaries=summary_infos,
                total_count=len(summary_infos),
            )
        except Exception as e:
            log.error(f"Failed to list summaries: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/v1/history/summary/{summary_id}", response_model=DeleteSummaryResponse, tags=["History"])
    async def delete_summary(
        summary_id: str,
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Delete a summary and restore the original messages."""
        from ..memory.summarization import HistorySummarizer

        service = get_service()
        effective_bot_id = bot_id or service._default_bot

        try:
            summarizer = HistorySummarizer(service.config, effective_bot_id)
            result = summarizer.delete_summary(summary_id)

            if result.get("success"):
                return DeleteSummaryResponse(
                    success=True,
                    summary_id=result.get("summary_id"),
                    messages_restored=result.get("messages_restored", 0),
                )
            else:
                return DeleteSummaryResponse(
                    success=False,
                    detail=result.get("error", "Failed to delete summary"),
                )
        except Exception as e:
            log.error(f"Failed to delete summary: {e}")
            raise HTTPException(status_code=500, detail=str(e))

except ImportError:
    # FastAPI not installed - create stub
    app = None
    log.warning("FastAPI not installed. Install with: pip install fastapi uvicorn")


# =============================================================================
# CLI Entry Point
# =============================================================================

def _find_service_pid(port: int) -> int | None:
    """Find the PID of a process listening on the given port."""
    import subprocess
    try:
        # Use lsof to find the process listening on the port
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            # May return multiple PIDs (parent/child), get the first one
            pids = result.stdout.strip().split("\n")
            return int(pids[0])
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass
    return None


def _is_service_running(host: str, port: int) -> bool:
    """Check if the service is already running by attempting to connect."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            # Use 127.0.0.1 for 0.0.0.0 since we can't connect to 0.0.0.0
            check_host = "127.0.0.1" if host == "0.0.0.0" else host
            result = sock.connect_ex((check_host, port))
            return result == 0
    except (OSError, socket.error):
        return False


def _kill_service(port: int) -> bool:
    """Kill the service running on the given port. Returns True if successful."""
    import signal
    import os
    
    pid = _find_service_pid(port)
    if pid is None:
        return False
    
    try:
        # Send SIGTERM for graceful shutdown
        os.kill(pid, signal.SIGTERM)
        
        # Wait briefly for the process to terminate
        import time
        for _ in range(10):  # Wait up to 1 second
            time.sleep(0.1)
            try:
                os.kill(pid, 0)  # Check if process still exists
            except OSError:
                return True  # Process terminated
        
        # If still running, send SIGKILL
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.1)
        return True
    except OSError:
        return False


def main():
    """Entry point for the background service."""
    import argparse
    
    # Load config for defaults
    config = Config()
    
    parser = argparse.ArgumentParser(description="llm-bawt background service")
    parser.add_argument("--host", default=config.SERVICE_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.SERVICE_PORT, help="Port to listen on")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show additional detail (payloads, timing)")
    parser.add_argument("--debug", action="store_true", help="Enable low-level DEBUG messages (unformatted)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument("--restart", action="store_true", help="Kill existing service and start a new one")
    parser.add_argument("--stop", action="store_true", help="Stop the running service and exit")
    args = parser.parse_args()
    
    # Setup logging with the new rich-formatted logger
    # Note: --verbose enables payload logging, --debug enables low-level DEBUG
    setup_service_logging(verbose=args.verbose, debug=args.debug)
    
    # Also update config.VERBOSE so BackgroundService can use it
    config.VERBOSE = args.verbose
    
    # Handle --stop: kill the service and exit
    if args.stop:
        if _is_service_running(args.host, args.port):
            print(f"Stopping service on port {args.port}...")
            if _kill_service(args.port):
                print("Service stopped.")
                return 0
            else:
                print("Failed to stop service.")
                return 1
        else:
            print(f"No service running on port {args.port}.")
            return 0
    
    # Check if service is already running
    if _is_service_running(args.host, args.port):
        if args.restart:
            print(f"Restarting service on port {args.port}...")
            if not _kill_service(args.port):
                print("Warning: Could not kill existing service, attempting to start anyway...")
            # Brief pause to ensure port is released
            import time
            time.sleep(0.5)
        else:
            print(f"Service is already running on port {args.port}.")
            print("Use --restart to restart the service, or --stop to stop it.")
            return 0
    
    if app is None:
        print("Error: FastAPI not installed. Install with: pip install fastapi uvicorn")
        return 1
    
    try:
        import uvicorn
        
        # Configure uvicorn log level
        # When using our rich logging, set uvicorn to warning to reduce noise
        uvicorn_log_level = "debug" if args.debug else "warning"
        
        # Exclude generated/data files from reload watching to prevent feedback loops
        reload_excludes = [
            "__pycache__", "*.pyc", ".git",
            ".logs", ".run", "models",
            "*.log", "*.pid",
        ] if args.reload else None
        
        uvicorn.run(
            "llm_bawt.service.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            reload_excludes=reload_excludes,
            log_level=uvicorn_log_level,
        )
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install uvicorn")
        return 1


if __name__ == "__main__":
    main()
