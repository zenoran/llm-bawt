"""Unified config resolution (TASK-488).

ONE resolution API over the two scoped-override stores that previously each
re-implemented the identical ``(key, scope_type, scope_id)`` bot -> global ->
code-default precedence:

- ``prompt_templates`` (bodies)  -> :class:`PromptResolver`
- ``runtime_settings`` (scalars) -> :class:`RuntimeSettingsResolver`

The stores stay separate — bodies and scalars are genuinely different value
types with different tables — but resolution, scope normalization, and
provenance are unified behind :class:`ConfigResolver`. Every lookup returns the
value AND the layer that supplied it, so "why is this what it is" is answerable
without re-deriving the precedence chain by hand (this is what the effective
-config inspector reads through).

Migration is incremental and reversible: existing ``RuntimeSettingsResolver`` /
``PromptResolver`` call sites keep working unchanged; ConfigResolver wraps them
rather than replacing their storage.
"""

from dataclasses import dataclass
from typing import Any

from .prompt_registry import (
    PromptResolver,
    ResolvedPrompt,
    _normalize_scope,
    get_prompt_resolver,
)
from .runtime_settings import RuntimeSettingsResolver

# Provenance labels, shared across scalars and bodies so callers see one
# vocabulary regardless of which store answered.
SOURCE_REQUEST = "request_override"
SOURCE_BOT = "bot_override"
SOURCE_GLOBAL = "global_override"
SOURCE_DEFAULT = "code_default"
SOURCE_UNSET = "unset"

# PromptResolver reports "db_override" / "code_default"; map to the shared
# vocabulary, refining db_override into bot/global using the resolved scope.
_PROMPT_SOURCE_MAP = {
    "code_default": SOURCE_DEFAULT,
}


@dataclass(frozen=True)
class ResolvedValue:
    """A resolved config value with its provenance."""

    key: str
    value: Any
    source: str  # one of the SOURCE_* labels above
    scope_type: str  # "global" | "bot"
    scope_id: str  # "*" for global, the bot slug for bot scope


class ConfigResolver:
    """Single entry point for resolving both scalar settings and prompt bodies.

    Wraps the two existing resolvers, sharing ONE ``_normalize_scope`` and ONE
    provenance vocabulary. Constructed per bot (like ``RuntimeSettingsResolver``)
    so scalar bot-scope resolution has its subject; body resolution takes an
    explicit scope per call (bodies are looked up global-first with per-key bot
    overrides, matching the existing ``PromptResolver`` contract).
    """

    def __init__(
        self,
        config,
        bot=None,
        bot_id: str | None = None,
        settings: RuntimeSettingsResolver | None = None,
        prompts: PromptResolver | None = None,
    ):
        self.config = config
        self.bot_id = (bot_id or getattr(bot, "slug", "") or "").strip().lower() or None
        # Reuse the caller's RuntimeSettingsResolver when provided (shares its
        # 5s cache); otherwise build one for this bot.
        self._settings = settings or RuntimeSettingsResolver(
            config=config, bot=bot, bot_id=self.bot_id
        )
        self._prompts = prompts or get_prompt_resolver(config)

    # -- shared scope normalization (the ONE implementation) -----------------
    @staticmethod
    def normalize_scope(scope_type: str = "global", scope_id: str | None = None) -> tuple[str, str]:
        return _normalize_scope(scope_type, scope_id)

    # -- scalars (runtime_settings) -----------------------------------------
    def resolve_scalar(
        self,
        key: str,
        fallback: Any = None,
        request_overrides: dict[str, Any] | None = None,
    ) -> ResolvedValue:
        """Resolve a scalar setting with provenance.

        Precedence: request override -> bot -> global -> code-default -> unset.
        """
        value, source = self._settings.resolve_with_source(
            key, fallback=fallback, request_overrides=request_overrides
        )
        if source == SOURCE_BOT:
            scope_type, scope_id = "bot", (self.bot_id or "")
        else:
            # request/global/default/unset all report against global scope —
            # request overrides are per-turn and not bot-scoped storage.
            scope_type, scope_id = "global", "*"
        return ResolvedValue(
            key=key, value=value, source=source, scope_type=scope_type, scope_id=scope_id
        )

    def resolve_scalar_value(
        self,
        key: str,
        fallback: Any = None,
        request_overrides: dict[str, Any] | None = None,
    ) -> Any:
        """Convenience: the value only (back-compat with ``.resolve``)."""
        return self.resolve_scalar(key, fallback=fallback, request_overrides=request_overrides).value

    # -- bodies (prompt_templates) ------------------------------------------
    def resolve_body(
        self,
        key: str,
        scope_type: str = "global",
        scope_id: str | None = None,
    ) -> ResolvedPrompt | None:
        """Resolve a prompt body. Returns the ``ResolvedPrompt`` (with ``.body``,
        ``.source``, ``.scope_type``/``.scope_id``) or None for unknown keys."""
        return self._prompts.resolve(key, scope_type=scope_type, scope_id=scope_id)

    def resolve_body_with_provenance(
        self,
        key: str,
        scope_type: str = "global",
        scope_id: str | None = None,
    ) -> ResolvedValue | None:
        """Resolve a body but return it in the unified ``ResolvedValue`` shape,
        translating PromptResolver's source vocabulary into the shared one."""
        resolved = self.resolve_body(key, scope_type=scope_type, scope_id=scope_id)
        if resolved is None:
            return None
        source = _PROMPT_SOURCE_MAP.get(resolved.source)
        if source is None:
            # db_override — refine into bot/global from the resolved scope.
            source = SOURCE_BOT if resolved.scope_type == "bot" else SOURCE_GLOBAL
        return ResolvedValue(
            key=key,
            value=resolved.body,
            source=source,
            scope_type=resolved.scope_type,
            scope_id=resolved.scope_id,
        )
