"""Provider adapter registry.

Maps a provider id -> adapter instance (constructed with the app Config). Phase 2
adds the LLM api-key adapters (openai/anthropic/grok); GitHub ships in Phase 1.
"""

from __future__ import annotations

from ...utils.config import Config
from .base import ProviderAdapter
from .claude_sub import ClaudeSubAdapter
from .github import GitHubAdapter

# Adapter classes keyed by provider id. Extend here as providers are added.
_ADAPTER_CLASSES: dict[str, type[ProviderAdapter]] = {
    GitHubAdapter.id: GitHubAdapter,
    ClaudeSubAdapter.id: ClaudeSubAdapter,
}


def get_adapter(config: Config, provider_id: str) -> ProviderAdapter | None:
    cls = _ADAPTER_CLASSES.get(provider_id)
    if cls is None:
        return None
    return cls(config)


def all_adapters(config: Config) -> list[ProviderAdapter]:
    return [cls(config) for cls in _ADAPTER_CLASSES.values()]
