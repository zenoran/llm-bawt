"""Provider adapter registry.

Adapters declare a ``name`` class var; ``register`` indexes them by that name
and ``lookup`` resolves a name → adapter instance. Auto-registers the
OpenAI ChatGPT adapter so callers only need to import this package.
"""

from __future__ import annotations

from .base import ProviderAdapter
from .openai_chatgpt import OpenAIChatGPTAdapter
from .zai import ZaiAdapter

REGISTRY: dict[str, ProviderAdapter] = {}


def register(adapter: ProviderAdapter) -> None:
    REGISTRY[adapter.name] = adapter


def lookup(name: str) -> ProviderAdapter | None:
    return REGISTRY.get(name)


# Default registrations. Adding a new provider = create the adapter file,
# import it here, and register an instance.
register(OpenAIChatGPTAdapter())
register(ZaiAdapter())

__all__ = ["ProviderAdapter", "REGISTRY", "register", "lookup"]
