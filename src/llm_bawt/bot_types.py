"""Helpers for normalizing bot kind metadata."""

from __future__ import annotations

from typing import Any


def agent_backend_for_model_def(model_def: dict[str, Any] | None) -> str | None:
    """Map a model-catalog definition to the agent backend it belongs to.

    Single source of truth for "does catalog entry X belong to agent
    backend Y" — used by model resolution (instance_manager), bridge
    config injection (service/core), profile validation (routes/settings)
    and the CLI direct path.

    Catalog shapes (historical accretion):
      * ``type='claude-code'``                          → ``"claude-code"``
      * ``type='codex'`` (pre-normalization)            → ``"codex"``
      * ``type='agent_backend'`` + ``backend=X``        → ``X``
        (``backend`` may live top-level in the merged config dict or
        inside ``extra`` when read straight from a DB row)

    Returns the backend slug or ``None`` for ordinary chat models.
    """
    if not model_def:
        return None
    model_type = str(model_def.get("type") or "").strip().lower()
    if model_type == "claude-code":
        return "claude-code"
    if model_type == "codex":
        return "codex"
    if model_type == "agent_backend":
        backend = model_def.get("backend")
        if not backend:
            extra = model_def.get("extra")
            if isinstance(extra, dict):
                backend = extra.get("backend")
        backend = str(backend or "").strip().lower()
        return backend or None
    return None


def normalize_bot_type(bot_type: str | None = None, agent_backend: str | None = None) -> str:
    """Resolve the effective bot type from explicit type + backend fields."""
    normalized_backend = (agent_backend or "").strip().lower()
    normalized_type = (bot_type or "").strip().lower()
    if normalized_type == "agent" or normalized_backend:
        return "agent"
    return "chat"


def format_bot_type(bot_type: str | None = None, agent_backend: str | None = None) -> str:
    """Render a short display label for a bot type."""
    resolved = normalize_bot_type(bot_type, agent_backend)
    normalized_backend = (agent_backend or "").strip().lower()
    if resolved == "agent" and normalized_backend:
        return f"agent/{normalized_backend}"
    return resolved
