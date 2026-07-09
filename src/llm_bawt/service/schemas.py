"""Pydantic schemas for the llm-bawt service API."""

import time
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SERVICE_VERSION = "0.1.0"
BotKind = Literal["chat", "agent"]

# =============================================================================
# OpenAI-Compatible Schemas
# =============================================================================

class ChatMessage(BaseModel):
    """OpenAI-compatible chat message.

    TASK-225: extended with ``attachment_ids`` so callers can attach
    images by reference to a ``media_assets`` row instead of inlining
    base64 in the content array. The server resolves each id at the
    LLM-call boundary via :class:`~llm_bawt.media.store.MediaStore` and
    persists a tiny ``{asset_id, kind}`` ref on the chat-history row.

    Backwards compatibility: the legacy multimodal shape
    ``content: [{type:"text"}, {type:"image_url"}]`` keeps working —
    Claude Code and other OpenAI-multimodal clients still send inline
    base64 image_url parts, and the server auto-uploads those to
    MediaStore so they too get a persistent asset_id and show up in
    history just like new-style uploads.
    """
    role: Literal["system", "user", "assistant", "function", "tool"]
    # list[dict] before str so Pydantic v2 left-to-right union matching
    # tries the list variant first for OpenAI content-array payloads.
    content: list[dict] | str | None = None
    name: str | None = None
    # TASK-225: optional list of ``ma_<ulid>`` ids referencing rows in the
    # ``media_assets`` table. Only meaningful on ``role=="user"`` — other
    # roles are ignored. Resolved server-side; clients never send base64.
    attachment_ids: list[str] | None = Field(
        default=None,
        description="Optional media_assets ids to attach (resolved server-side).",
    )


class ChatRequestAnimation(BaseModel):
    """One animation entry passed in by the caller (TASK-214).

    Mirrors the bawthub Prisma `AvatarAnimation` shape but only carries the
    fields llm-bawt needs at request time: name + description. Caller is
    expected to filter to enabled rows before sending.
    """
    name: str
    description: str | None = None


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
    include_summaries: bool = Field(default=True, description="Whether to inject conversation summary records into context")
    tts_mode: bool = Field(default=False, description="Whether to append TTS output formatting instructions to the system prompt")
    inject_user_prefix: bool = Field(default=False, description="For agent backends only: prepend chat.agent_user_prefix body to every user message. Survives agent-SDK session resume (unlike system prompt edits). Stacks independently with the voice-mode prefix.")
    client_system_context: str | None = Field(default=None, description="System context extracted from client messages (set by routes, not by callers)", exclude=True)
    ha_mode: bool = Field(default=False, description="HA-mode: cap history, force tool_choice=required on first call (set by routes)", exclude=True)
    user_message_id: str | None = Field(default=None, description="Frontend-generated UUID for the user message (used as trigger_message_id in turn logs)")
    assistant_message_id: str | None = Field(default=None, description="Frontend-generated UUID for the ASSISTANT reply row. Persisted as the assistant message id so the live streaming bubble and the reloaded history row share one id (single bubble). None → server mints one. Closes the EPIC TASK-217 assistant-identity gap.")
    # TASK-269: continuation-turn linkage.  Set when this turn is answering a
    # prior deferred AskUserQuestion.  parent_turn_id threads the chain (and
    # drives cross-tab resolution via turn_start{parent_turn_id}); when
    # answered_question_id is set, the new turn is recorded as the one that
    # carried the answer back to the agent.
    parent_turn_id: str | None = Field(default=None, description="TASK-269: the awaiting turn this continuation answers")
    answered_question_id: str | None = Field(default=None, description="TASK-269: tool_use_id of the question this continuation answers")
    # TASK-214: animations + avatar visibility now flow on each request.
    # The avatar catalog is owned by the bawthub frontend; llm-bawt is stateless
    # w.r.t. animations. `avatar_visible` is currently informational (consumed
    # by the embedding classifier added in TASK-215).
    animations: list[ChatRequestAnimation] | None = Field(
        default=None,
        description="Available avatar animations the model can pick from when tts_mode=true. "
                    "When None or empty, no animation tool is injected. "
                    "Owned by bawthub frontend (Prisma).",
    )
    avatar_visible: bool | None = Field(
        default=None,
        description="Whether an avatar is currently being rendered on the client. "
                    "TASK-215 will gate animation work on this so we don't waste "
                    "compute when no one can see the animation.",
    )


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


class ModelPricing(BaseModel):
    """Per-1M-token USD rates for client-side cost estimation.

    All optional — a model with no pricing simply shows no computed cost.
    Stored in ``ModelDefinition.extra['pricing']`` (no schema migration) and
    surfaced here so the chat context badge can estimate turn cost as
    ``sum(tokens_of_kind * rate_of_kind) / 1_000_000``.
    """
    input: float | None = None
    output: float | None = None
    cache_read: float | None = None
    cache_write: float | None = None


class ModelInfo(BaseModel):
    """Model information for /v1/models endpoint."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "llm-bawt"
    type: str | None = None
    model_id: str | None = None
    description: str | None = None
    pricing: ModelPricing | None = None


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


# =============================================================================
# Model Definition CRUD Schemas
# =============================================================================

class ModelDefinitionResponse(BaseModel):
    """Response payload for a single model definition."""
    alias: str
    type: str
    model_id: str | None = None
    repo_id: str | None = None
    filename: str | None = None
    description: str | None = None
    extra: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ModelDefinitionListResponse(BaseModel):
    """Response for listing model definitions."""
    models: list[ModelDefinitionResponse]
    total_count: int


class ModelDefinitionUpsertRequest(BaseModel):
    """Request to create or update a model definition."""
    type: str = Field(..., description="Model type: openai, codex, ollama, gguf, huggingface, grok, openclaw")
    model_id: str | None = Field(default=None, description="Model ID (for openai/codex/ollama/grok/openclaw)")
    repo_id: str | None = Field(default=None, description="HuggingFace repo ID (for gguf/huggingface)")
    filename: str | None = Field(default=None, description="GGUF filename")
    description: str | None = None
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Optional fields: chat_format, context_window, max_tokens, n_gpu_layers, tool_support, tool_format",
    )


class ModelDefinitionDeleteResponse(BaseModel):
    """Response from deleting a model definition."""
    success: bool
    alias: str
    message: str


class ModelDefinitionSeedRequest(BaseModel):
    """Request to seed DB models from current YAML config."""
    overwrite: bool = Field(default=False, description="If true, overwrite existing DB entries with YAML values")


class ModelDefinitionSeedResponse(BaseModel):
    """Response from seeding model definitions."""
    seeded: int
    total_yaml: int
    message: str


class BotInfo(BaseModel):
    """Bot information for /v1/bots endpoint."""
    slug: str
    name: str
    description: str | None = None
    system_prompt: str = ""
    requires_memory: bool = True
    voice_optimized: bool = False
    tts_mode: bool = False
    include_summaries: bool = True
    include_in_global_search: bool = True
    default_voice: str | None = None
    uses_tools: bool = False
    uses_search: bool = False
    uses_home_assistant: bool = False
    default_model: str | None = None
    color: str | None = None
    avatar: str | None = None
    avatar_render: str | None = None
    bot_type: BotKind = "chat"
    agent_backend: str | None = None
    agent_backend_config: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)


class BotsResponse(BaseModel):
    """Response for /v1/bots endpoint."""
    object: str = "list"
    data: list[BotInfo]


class BotProfileResponse(BaseModel):
    """Response payload for a bot profile."""

    slug: str
    name: str
    description: str
    system_prompt: str
    requires_memory: bool = True
    voice_optimized: bool = False
    tts_mode: bool = False
    include_summaries: bool = True
    include_in_global_search: bool = True
    uses_tools: bool = False
    uses_search: bool = False
    uses_home_assistant: bool = False
    default_model: str | None = None
    color: str | None = None
    avatar: str | None = None
    # Server-derived, self-hosted render of ``avatar`` as a data: URL
    # (Twemoji SVG for emoji, small WebP for images). Read-only for clients.
    avatar_render: str | None = None
    default_voice: str | None = None
    nextcloud_config: dict[str, Any] | None = None
    bot_type: BotKind = "chat"
    agent_backend: str | None = None
    agent_backend_config: dict[str, Any] | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class BotProfileUpsertRequest(BaseModel):
    """Request payload for upserting a bot profile."""

    name: str
    description: str = ""
    system_prompt: str
    requires_memory: bool = True
    voice_optimized: bool = False
    tts_mode: bool = False
    include_summaries: bool = True
    include_in_global_search: bool = True
    uses_tools: bool = False
    uses_search: bool = False
    uses_home_assistant: bool = False
    default_model: str | None = None
    color: str | None = None
    avatar: str | None = None
    default_voice: str | None = None
    nextcloud_config: dict[str, Any] | None = None
    bot_type: BotKind | None = None
    agent_backend: str | None = None
    agent_backend_config: dict[str, Any] | None = None


class BotProfilePatchRequest(BaseModel):
    """Partial update — only provided fields are changed."""

    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    requires_memory: bool | None = None
    voice_optimized: bool | None = None
    tts_mode: bool | None = None
    include_summaries: bool | None = None
    include_in_global_search: bool | None = None
    uses_tools: bool | None = None
    uses_search: bool | None = None
    uses_home_assistant: bool | None = None
    default_model: str | None = None
    color: str | None = None
    avatar: str | None = None
    default_voice: str | None = None
    nextcloud_config: dict[str, Any] | None = None
    bot_type: BotKind | None = None
    agent_backend: str | None = None
    agent_backend_config: dict[str, Any] | None = None


class BotCreateRequest(BotProfileUpsertRequest):
    """Request payload for creating a new bot profile."""

    slug: str


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


class TaskListItem(BaseModel):
    """A task entry for task listing."""
    task_id: str
    status: str
    task_type: str | None = None
    bot_id: str | None = None
    user_id: str | None = None
    priority: int | None = None
    created_at: str | None = None
    completed_at: str | None = None
    processing_time_ms: float | None = None
    error: str | None = None


class TaskListResponse(BaseModel):
    """Response for listing tasks."""
    tasks: list[TaskListItem]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class ScheduledJobInfo(BaseModel):
    """A scheduled job with latest run summary."""
    id: str
    job_type: str
    bot_id: str
    enabled: bool
    interval_minutes: int
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    created_at: datetime | None = None
    last_status: str | None = None
    last_duration_ms: int | None = None
    last_error: str | None = None


class ScheduledJobsResponse(BaseModel):
    """Response for listing scheduled jobs."""
    jobs: list[ScheduledJobInfo]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class JobRunInfo(BaseModel):
    """A single scheduler job run record."""
    id: str
    job_id: str
    job_type: str | None = None
    bot_id: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = None
    error_message: str | None = None
    result: Any | None = None


class JobRunsResponse(BaseModel):
    """Response for listing scheduler job run history."""
    runs: list[JobRunInfo]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class TurnLogListItem(BaseModel):
    """Summary row for one persisted turn log."""
    id: str
    created_at: datetime
    request_id: str | None = None
    path: str
    stream: bool = False
    model: str | None = None
    bot_id: str | None = None
    user_id: str | None = None
    status: str
    latency_ms: float | None = None
    user_prompt: str | None = None
    response_preview: str | None = None
    response_tail: str | None = None
    response_chars: int = 0
    response_preview_truncated: bool = False
    tool_call_count: int = 0
    error_text: str | None = None
    animation: str | None = None
    agent_session_key: str | None = None
    agent_request_id: str | None = None
    trigger_message_id: str | None = None
    # Per-turn token accounting from the upstream SDK (claude_code, etc.).
    # Shape: {input_tokens, cache_read_tokens, cache_creation_tokens,
    #         output_tokens, context_window, max_output_tokens, total_cost_usd}.
    # None when the upstream backend doesn't expose usage info.
    token_usage: dict[str, Any] | None = None
    # TASK-269 turn lifecycle / continuation chain.
    end_reason: str | None = None
    question_id: str | None = None
    parent_turn_id: str | None = None
    # Wall-clock completion time. None = turn still in progress (the single
    # path-agnostic in-flight signal; supersedes the status="streaming" proxy).
    ended_at: datetime | None = None
    # Embedded question row (chat_pending_questions.row_to_dict) when this turn
    # ended with end_reason="question" — lets the UI render a QuestionMessage
    # straight from history hydration without a second fetch.
    question: dict[str, Any] | None = None


class TurnLogListResponse(BaseModel):
    """Response for listing persisted turn logs."""
    turns: list[TurnLogListItem]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class TurnLogDetail(BaseModel):
    """Detailed persisted turn log."""
    id: str
    created_at: datetime
    request_id: str | None = None
    path: str
    stream: bool = False
    model: str | None = None
    bot_id: str | None = None
    user_id: str | None = None
    status: str
    latency_ms: float | None = None
    user_prompt: str | None = None
    request: Any | None = None
    response: str | None = None
    # TASK-360 (P4): partial/persisted model reasoning ("thinking") so a cold
    # reload mid-turn can recover already-produced reasoning.
    reasoning: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    error_text: str | None = None
    animation: str | None = None
    agent_session_key: str | None = None
    agent_request_id: str | None = None
    token_usage: dict[str, Any] | None = None
    # TASK-269 turn lifecycle / continuation chain.
    end_reason: str | None = None
    question_id: str | None = None
    parent_turn_id: str | None = None
    question: dict[str, Any] | None = None


class RecentBotTurn(BaseModel):
    """Compact summary of the most recent turn for one bot.

    Returned by ``GET /v1/turn-logs/recent-by-bot``. Designed for dashboard
    "last activity per bot" views: enough information to render a tile
    (who, when, what was said, how many tools fired, did it succeed) without
    a follow-up fetch.
    """
    bot_id: str
    turn_id: str
    created_at: datetime
    model: str | None = None
    status: str
    latency_ms: float | None = None
    user_prompt_preview: str | None = None
    response_preview: str | None = None
    response_tail: str | None = None
    response_chars: int = 0
    response_preview_truncated: bool = False
    tool_call_count: int = 0
    token_usage: dict[str, Any] | None = None
    trigger_message_id: str | None = None


class RecentByBotsResponse(BaseModel):
    """Single response carrying one ``RecentBotTurn`` per bot.

    Sorted by ``created_at`` descending so the most recently active bot is
    first. Bots in the request's ``bot_ids`` filter that have no turn in the
    window are simply absent from ``turns`` — the caller can detect "no
    activity" by checking which slugs are missing.
    """
    turns: list[RecentBotTurn]


class ToolCallEvent(BaseModel):
    """Tool-call event linked to one trigger history message."""
    turn_id: str
    created_at: datetime
    request_id: str | None = None
    model: str | None = None
    bot_id: str | None = None
    user_id: str | None = None
    message_id: str
    message_role: str = "user"
    message_timestamp: float | None = None
    tool_call_count: int = 0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class ToolCallEventsResponse(BaseModel):
    """Tool-call events for history annotation in UI/CLI."""
    events: list[ToolCallEvent]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class ServiceStatusResponse(BaseModel):
    """Service health and status."""
    status: str = "ok"
    version: str = SERVICE_VERSION
    healthy: bool = True
    uptime_seconds: float
    tasks_processed: int
    tasks_pending: int
    worker_running: bool = False
    models_loaded: list[str] = []
    current_model: str | None = None
    default_model: str | None = None
    default_bot: str | None = None
    available_models: list[str] = []
    # Health checks
    checks: dict[str, str] = Field(default_factory=dict)
    # Database / memory
    database_connected: bool = False
    database_host: str | None = None
    database_error: str | None = None
    messages_count: int = 0
    memories_count: int = 0
    pgvector_available: bool = False
    embeddings_available: bool = False
    # llm-bawt MCP server
    mcp_mode: str = "embedded"
    mcp_status: str = "up"
    mcp_url: str | None = None
    mcp_http_status: int | None = None


# =============================================================================
# Full System Status (mirrors core.status.SystemStatus)
# =============================================================================

class ServiceInfoSchema(BaseModel):
    """LLM service status."""
    available: bool = False
    healthy: bool = False
    uptime_seconds: float | None = None
    current_model: str | None = None
    default_model: str | None = None
    default_bot: str | None = None
    tasks_processed: int = 0
    tasks_pending: int = 0


class MemoryInfoSchema(BaseModel):
    """Database and memory subsystem status."""
    postgres_connected: bool = False
    postgres_host: str | None = None
    postgres_error: str | None = None
    messages_count: int = 0
    memories_count: int = 0
    pgvector_available: bool = False
    embeddings_available: bool = False


class ModelStatusInfoSchema(BaseModel):
    """Current model configuration details."""
    alias: str
    type: str = "unknown"
    max_tokens: int = 0
    max_tokens_source: str = "global"
    context_window: int | None = None
    context_source: str | None = None
    gpu_name: str | None = None
    vram_total_gb: float | None = None
    vram_free_gb: float | None = None
    vram_detection_method: str | None = None
    n_gpu_layers: str | None = None
    gpu_layers_source: str | None = None
    native_context_limit: int | None = None


class DependencyInfoSchema(BaseModel):
    """Optional dependency availability."""
    cuda_version: str | None = None
    llama_cpp_available: bool = False
    llama_cpp_gpu: bool | None = None
    hf_hub_available: bool = False
    torch_available: bool = False
    openai_key_set: bool = False
    newsapi_key_set: bool = False
    search_provider: str | None = None
    embeddings_available: bool = False


class McpInfoSchema(BaseModel):
    """llm-bawt MCP server status."""
    mode: str = "embedded"
    status: str = "up"
    url: str | None = None
    http_status: int | None = None


class BotSummarySchema(BaseModel):
    """Minimal bot info for the status display."""
    slug: str
    name: str
    is_default: bool = False


class ConfigInfoSchema(BaseModel):
    """System configuration summary."""
    version: str = SERVICE_VERSION
    mode: str = "direct"
    service_url: str | None = None
    environment: str = "local"
    bot_name: str = ""
    bot_slug: str = ""
    model_alias: str | None = None
    model_source: str | None = None
    user_id: str | None = None
    all_bots: list[BotSummarySchema] = []
    models_defined: int = 0
    models_service: int | None = None
    scheduler_enabled: bool = False
    scheduler_interval: int = 0
    ha_mcp_enabled: bool = False
    ha_mcp_url: str | None = None
    ha_native_mcp_url: str | None = None
    ha_native_mcp_tools: int = 0
    bind_host: str = "0.0.0.0"


class SystemStatusResponse(BaseModel):
    """Full system status — mirrors ``core.status.SystemStatus``."""
    config: ConfigInfoSchema
    service: ServiceInfoSchema
    mcp: McpInfoSchema
    model: ModelStatusInfoSchema | None = None
    memory: MemoryInfoSchema
    dependencies: DependencyInfoSchema


class HealthResponse(BaseModel):
    """Simple health check response."""
    status: str = "ok"
    version: str = SERVICE_VERSION


class HistoryMessage(BaseModel):
    """A message in the conversation history.

    ``attachments`` (TASK-226) carries the resolved media-asset envelopes
    described in :mod:`llm_bawt.media.serializers`. Always present so
    frontend renderers can iterate unconditionally; an empty list means
    the row has no media. The list is hydrated by the
    ``/v1/history`` route only — DB layers strip it from the canonical
    ``get_messages`` path because LLM-prep code doesn't want it.
    """
    id: str | None = None
    role: str
    content: str
    timestamp: float
    attachments: list[dict] = []
    # TASK-301: persisted model reasoning ("thinking"), hydrated by the
    # /v1/history route only (display-only; never in LLM context). None when the
    # row carried no reasoning (user rows, pre-feature assistant rows).
    reasoning: str | None = None


class HistoryResponse(BaseModel):
    """Response for conversation history.

    Pagination flags describe the loaded window's boundaries against the
    bot's full timeline:

    - ``has_more`` / ``has_older``: more messages exist *before* the
      oldest row returned. The two aliases are kept in sync — ``has_more``
      pre-dates the deep-link work and is the field every legacy reader
      checks; ``has_older`` is the explicit name used by the deep-link
      ``/v1/history/around`` window endpoint and `?after=` forward paging.
    - ``has_newer``: more messages exist *after* the newest row returned.
      Newly added for deep-link windows and forward pagination; defaults
      ``False`` so legacy backward-only pagination behaves unchanged.
    - ``oldest_timestamp`` / ``newest_timestamp``: window boundaries used
      by the frontend as cursors for paginating in either direction.
    """
    bot_id: str
    messages: list[HistoryMessage]
    total_count: int
    has_more: bool = False
    has_older: bool = False
    has_newer: bool = False
    oldest_timestamp: float | None = None
    newest_timestamp: float | None = None
    anchor_id: str | None = None


class HistorySearchResponse(BaseModel):
    """Response for per-bot history search."""
    bot_id: str
    query: str
    messages: list[HistoryMessage]
    total_count: int
    has_more: bool = False
    has_older: bool = False
    oldest_timestamp: float | None = None
    newest_timestamp: float | None = None


class HistorySearchAllMessage(BaseModel):
    """A single hit from a cross-bot full-text message search.

    Mirrors :class:`HistoryMessage` plus the ``bot_id`` source attribution
    (so the frontend knows which bot's chat to deep-link into) and the FTS
    ``rank`` score (so the dropdown can re-rank or filter low-confidence
    matches). Carries no attachments — cross-bot search is content-only by
    design; the attachment hydration round-trip only fires after the user
    follows the link into the per-bot chat surface.
    """
    id: str
    role: str
    content: str
    timestamp: float
    bot_id: str
    rank: float


class HistorySearchAllResponse(BaseModel):
    """Response for cross-bot history search.

    ``messages`` are pre-sorted by FTS rank descending then timestamp
    descending (most recent breaks ties). ``total_count`` matches the
    length of the returned list; pagination cursors are not used because
    the storage layer applies the limit before merging across bots.
    """
    query: str
    messages: list[HistorySearchAllMessage]
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
    created_at: float | str | None = None
    last_accessed: float | str | None = None
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


class MemoryUpdateRequest(BaseModel):
    """Request payload for updating a memory."""
    content: str | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    tags: list[str] | None = None


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
    sessions_targeted: int | None = None
    summaries_replaced: int | None = None
    summaries_purged: int | None = None
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


# Profile Attribute Models
class UserProfileAttribute(BaseModel):
    """A single profile attribute."""
    id: int | None = None
    category: str
    key: str
    value: Any
    confidence: float = 1.0
    source: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ProfileAttributeUpdateRequest(BaseModel):
    """Request payload for updating an existing profile attribute."""
    value: Any | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None


class ProfileAttributeUpsertRequest(BaseModel):
    """Request payload for creating/updating a profile attribute by identity."""

    entity_type: Literal["user", "bot"]
    entity_id: str
    category: str
    key: str
    value: Any
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "explicit"


# Unified Profile Models
class ProfileDetail(BaseModel):
    """Generic profile detail with attributes for any entity type (user or bot)."""
    entity_type: str  # "user" or "bot"
    entity_id: str
    email: str | None = None
    display_name: str | None = None
    description: str | None = None
    summary: str | None = None
    summary_updated_at: str | None = None
    attributes: list[UserProfileAttribute] = []
    created_at: str | None = None


class ProfileUpdateRequest(BaseModel):
    """Request payload for updating core profile fields."""
    display_name: str | None = None
    description: str | None = None
    summary: str | None = None
    email: str | None = None


class ProfileListResponse(BaseModel):
    """Response for listing profiles of a given type."""
    profiles: list[ProfileDetail]
    total_count: int


class ProfileAttributeListResponse(BaseModel):
    """Response for listing profile attributes."""
    attributes: list[UserProfileAttribute]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)

# =============================================================================
# Nextcloud Admin Schemas
# =============================================================================

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


# =============================================================================
# Runtime Settings Schemas
# =============================================================================

class RuntimeSettingItem(BaseModel):
    """Runtime setting key/value item."""
    key: str
    value: Any


class RuntimeSettingsResponse(BaseModel):
    """List runtime settings for a scope."""
    scope_type: str
    scope_id: str
    settings: list[RuntimeSettingItem]


class RuntimeSettingRecord(BaseModel):
    """One runtime setting row with scope metadata."""
    scope_type: str
    scope_id: str
    key: str
    value: Any
    updated_at: datetime | None = None


class RuntimeSettingsListResponse(BaseModel):
    """List runtime settings across scopes."""
    settings: list[RuntimeSettingRecord]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class BotProfileListResponse(BaseModel):
    """Response for listing bot profiles."""
    profiles: list[BotProfileResponse]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class RuntimeSettingUpsertRequest(BaseModel):
    """Upsert one runtime setting."""
    scope_type: Literal["global", "bot"]
    scope_id: str | None = None
    key: str
    value: Any


class RuntimeSettingBatchItem(BaseModel):
    """One runtime setting upsert in a batch request."""
    scope_type: Literal["global", "bot"]
    scope_id: str | None = None
    key: str
    value: Any


class RuntimeSettingBatchUpsertRequest(BaseModel):
    """Batch upsert runtime settings."""
    items: list[RuntimeSettingBatchItem]


class PromptTemplateResponse(BaseModel):
    """Resolved or exact prompt template payload."""

    key: str
    title: str
    category: str
    format: str = "plain_text"
    body: str
    scope_type: str
    scope_id: str
    source: str
    required_vars: list[str] = Field(default_factory=list)
    placeholders: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime | None = None


class PromptTemplateListResponse(BaseModel):
    """Response for listing prompt templates.

    Returns the FULL body for every prompt. For agent-context-safe lookups
    (just names, scope, source) use PromptTemplateIndexResponse via
    `GET /v1/prompts/index` — that route omits body/placeholders/metadata.
    """

    prompts: list[PromptTemplateResponse]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class PromptTemplateSummary(BaseModel):
    """Lightweight prompt-template entry — names and locators, no body.

    Use this when an agent (or a UI) needs to enumerate available prompts to
    decide which one to fetch in full. `body_length` is the only signal about
    the body itself — useful for "is this empty" / "is this huge" decisions
    without paying the context cost of the actual text.
    """

    key: str
    title: str
    category: str
    format: str = "plain_text"
    scope_type: str
    scope_id: str
    source: str
    body_length: int = 0
    updated_at: datetime | None = None


class PromptTemplateIndexResponse(BaseModel):
    """Compact prompt listing — names and metadata only, no bodies.

    Same filters as PromptTemplateListResponse, but each entry is a
    PromptTemplateSummary. Safe for agents to enumerate without dragging
    every prompt body into their context window.
    """

    prompts: list[PromptTemplateSummary]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class PromptTemplateUpsertRequest(BaseModel):
    """Create or update one prompt template."""

    body: str
    title: str | None = None
    category: str | None = None
    format: str = "plain_text"
    scope_type: Literal["global", "bot"] = "global"
    scope_id: str | None = None
    required_vars: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_by: str | None = None
    change_note: str | None = None


class PromptTemplateValidateRequest(BaseModel):
    """Validate a candidate prompt body."""

    body: str | None = None
    required_vars: list[str] | None = None
    variables: dict[str, Any] = Field(default_factory=dict)


class PromptTemplateValidateResponse(BaseModel):
    """Validation result for a prompt template."""

    valid: bool
    required_vars: list[str] = Field(default_factory=list)
    placeholders: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    unknown_placeholders: list[str] = Field(default_factory=list)
    rendered_preview: str | None = None
    errors: list[str] = Field(default_factory=list)


class PromptTemplateVersionResponse(BaseModel):
    """One stored prompt template version."""

    version: int
    body: str
    change_note: str | None = None
    created_by: str | None = None
    created_at: datetime


class PromptTemplateVersionsResponse(BaseModel):
    """Version history for a prompt key/scope."""

    key: str
    scope_type: str
    scope_id: str
    versions: list[PromptTemplateVersionResponse]
    total_count: int


class PromptTemplateSeedResponse(BaseModel):
    """Result of seeding built-in prompt templates into the DB."""

    created: int
    skipped: int
    total: int
    seeded_keys: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Avatar animation schemas
# ---------------------------------------------------------------------------
# TASK-214: the CRUD schemas (AvatarAnimationBase / Create / Update / Response)
# were used only by the deleted /v1/avatar/animations routes. The catalog now
# lives in bawthub Prisma. Per-request animation entries use the lighter
# ChatRequestAnimation defined near the top of this module.
