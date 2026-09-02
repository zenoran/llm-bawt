"""Tenant seed data (TASK-350 / TASK-355).

Design of record (2026-09-02): **no bots and no model catalog rows are
seeded.** A bot is dead weight until a provider credential exists, so the
setup wizard walks the operator through connecting a provider FIRST and then
creates the first bot against a model discovered live from that provider.

What remains seed data:

- the ``ModelSeed`` / ``EndpointSeed`` shapes the engine uses to register the
  operator's chosen model (insert-if-missing);
- ``FIRST_BOT_PROVIDERS`` — which provider connections can back the first
  (``chat``-harness) bot, and how they map to the standard access paths;
- ``DEFAULT_FIRST_BOT_PROMPT`` — the editable starting prompt the wizard
  offers. Deliberately generic: no owner, no personal traits.

Standard ``access_paths`` rows (``xai-chat``, ``openai-api``, ...) are owned by
:data:`llm_bawt.memory.model_catalog_migration.STANDARD_ACCESS_PATHS`, which
the seed engine installs first. API keys are never seed data — they live in
the encrypted CredentialStore behind the provider adapters (TASK-825).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSeed:
    """One row for the ``models`` table (insert-if-missing)."""

    key: str
    vendor: str
    display_name: str
    default_context_window: int
    description: str | None = None
    default_tool_support: str | None = None


@dataclass(frozen=True)
class EndpointSeed:
    """One row for ``model_endpoints``, referenced by (model key, access-path key).

    ``upstream_model_id`` is what the provider API is actually called with.
    """

    model_key: str
    access_path_key: str
    upstream_model_id: str
    serving_config: dict[str, Any] = field(default_factory=dict)
    context_window_override: int | None = None
    tool_support_override: str | None = None
    pricing: dict[str, Any] | None = None


@dataclass(frozen=True)
class FirstBotProvider:
    """A provider connection that can back the tenant's first ``chat`` bot.

    ``provider_id`` is the adapter id (``/v1/providers/{id}``);
    ``access_path_key`` is the standard access path the bot's endpoint is
    registered on; ``discovery`` is the alias ``/v1/models/upstream`` accepts.
    """

    provider_id: str
    vendor: str
    access_path_key: str
    discovery: str


#: Providers the wizard offers for the first bot. The first bot is a ``chat``
#: harness bot, which speaks chat-completions only — so Anthropic API keys
#: (anthropic-messages protocol → claude-code harness → agent bridge) are a
#: Phase-2 addition, not a first-bot option.
FIRST_BOT_PROVIDERS: tuple[FirstBotProvider, ...] = (
    FirstBotProvider(
        provider_id="xai",
        vendor="xai",
        access_path_key="xai-chat",
        discovery="xai",
    ),
    FirstBotProvider(
        provider_id="openai-api",
        vendor="openai",
        access_path_key="openai-api",
        discovery="openai",
    ),
)


#: Context window used when the provider's catalog doesn't report one. The
#: ``models`` table requires a positive window; the wizard lets the operator
#: override it.
DEFAULT_CONTEXT_WINDOW = 128_000


DEFAULT_FIRST_BOT_PROMPT = """\
You are a helpful assistant running on this BawtHub instance.

- Lead with the answer. Keep replies tight and in plain language.
- Use the tools you have (memory, web search) when they make the answer better.
- Remember what the user tells you about themselves and their preferences.
- Be honest about uncertainty instead of guessing."""


def first_bot_provider(provider_id: str) -> FirstBotProvider | None:
    """Look up a first-bot provider route by adapter id."""
    wanted = (provider_id or "").strip().lower()
    return next((p for p in FIRST_BOT_PROVIDERS if p.provider_id == wanted), None)
