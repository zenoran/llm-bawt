"""Helpers for normalizing bot kind metadata."""

from __future__ import annotations


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
