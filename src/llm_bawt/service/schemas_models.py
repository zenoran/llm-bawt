"""Model catalog + model-definition request/response schemas.

Split out of ``service/schemas.py`` (TASK-557). ``schemas.py`` re-imports every
name here so ``from ..schemas import X`` across the service is unchanged.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


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
