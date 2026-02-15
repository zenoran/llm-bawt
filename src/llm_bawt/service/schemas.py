"""Pydantic schemas for the llm-bawt service API."""

import time
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SERVICE_VERSION = "0.1.0"

# =============================================================================
# OpenAI-Compatible Schemas
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
    system_prompt: str = ""
    requires_memory: bool = True
    voice_optimized: bool = False
    uses_tools: bool = False
    uses_search: bool = False
    uses_home_assistant: bool = False
    default_model: str | None = None
    color: str | None = None
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
    uses_tools: bool = False
    uses_search: bool = False
    uses_home_assistant: bool = False
    default_model: str | None = None
    nextcloud_config: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class BotProfileUpsertRequest(BaseModel):
    """Request payload for upserting a bot profile."""

    name: str
    description: str = ""
    system_prompt: str
    requires_memory: bool = True
    voice_optimized: bool = False
    uses_tools: bool = False
    uses_search: bool = False
    uses_home_assistant: bool = False
    default_model: str | None = None
    nextcloud_config: dict[str, Any] | None = None


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


# =============================================================================
# Full System Status (mirrors core.status.SystemStatus)
# =============================================================================

class ServiceInfoSchema(BaseModel):
    """LLM service status."""
    available: bool = False
    healthy: bool = False
    uptime_seconds: float | None = None
    current_model: str | None = None
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
    """MCP memory server status."""
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
    user_id: str | None = None
    all_bots: list[BotSummarySchema] = []
    models_defined: int = 0
    models_service: int | None = None
    scheduler_enabled: bool = False
    scheduler_interval: int = 0
    ha_mcp_enabled: bool = False
    ha_mcp_url: str | None = None
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
    has_more: bool = False
    oldest_timestamp: float | None = None


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


class ProfileAttributeUpdateRequest(BaseModel):
    """Request payload for updating an existing profile attribute."""
    value: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None


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
