"""Typed setting definitions with bot-type applicability (TASK-491/492).

The counterpart to ``DEFAULT_PROMPT_DEFINITIONS`` (bodies) for scalar/flag
settings. Every tunable declares its type, default, which bot_type(s) it applies
to, and where it is stored. This metadata is the single source of truth the
effective-config inspector and the frontend consume so they can render/mark only
the settings that actually apply to a given bot — killing the class of confusion
where an inert flag (e.g. ``include_summaries`` on an agent bot) is shown as if
it did something.

This registry does NOT itself resolve values — resolution stays in
``ConfigResolver``. It only declares the shape and applicability.
"""

from dataclasses import dataclass, field


# Storage backends a setting can live in.
STORAGE_RUNTIME_SETTING = "runtime_setting"        # runtime_settings table (scoped)
STORAGE_BOT_COLUMN = "bot_profiles_column"          # a first-class bot_profiles column
STORAGE_AGENT_BACKEND_CONFIG = "agent_backend_config"  # legacy JSON blob (being retired)
STORAGE_REQUEST_FLAG = "request_flag"               # per-turn request-only flag

BOT_TYPES_ALL = ("chat", "agent")


@dataclass(frozen=True)
class SettingDefinition:
    """One typed setting."""

    key: str
    type: str  # "bool" | "int" | "float" | "str"
    default: object
    applies_to: tuple[str, ...]  # bot types this setting affects
    storage: str
    label: str = ""
    help: str = ""
    # If this setting supersedes/absorbs a legacy key, name it here so the
    # migration and compat shims know the provenance.
    legacy_keys: tuple[str, ...] = field(default_factory=tuple)


# Canonical typed settings. Keyed by setting key.
SETTING_DEFINITIONS: dict[str, SettingDefinition] = {
    # --- session-memory continuity (TASK-492: unifies the two flags) ---------
    "session_memory_continuity": SettingDefinition(
        key="session_memory_continuity",
        type="bool",
        default=True,
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Carry prior conversation into context",
        help=(
            "Coarse master gate — a MIRROR of history_scope (TASK-518): true iff "
            "history_scope != 'none' (i.e. at least one of inline history / "
            "summaries is on). Kept as its own key so legacy readers (bridge seed "
            "gate, migrations) need no change. UI writes it derived; history_scope "
            "is the real source of truth for what gets carried."
        ),
        legacy_keys=("include_summaries", "seed_summary_on_new_session"),
    ),
    # --- history scope: the two independent carry-bits (TASK-493/518) ---------
    # NOTE: modelled as a constrained str until the registry grows a real "enum"
    # type. It encodes TWO INDEPENDENT bits, substring-tested: include_history =
    # "inline" in scope, include_summaries = "summaries" in scope. The four
    # canonical values are the full cross-product; there is no coupling between
    # the two axes (see utils.history.scope_flags).
    "history_scope": SettingDefinition(
        key="history_scope",
        type="str",
        default="inline+summaries",
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="History scope",
        help=(
            "What prior context is carried, as two independent bits: "
            "'inline+summaries' (recent messages + rolling summaries), "
            "'inline' (recent messages only), "
            "'summaries' (dense summary-only, no raw messages), or "
            "'none' (carry nothing). Absorbs the legacy include_summaries flag."
        ),
        legacy_keys=("include_summaries",),
    ),
    # --- promoted from agent_backend_config JSON blob (TASK-491) --------------
    "seed_summary_on_new_session": SettingDefinition(
        key="seed_summary_on_new_session",
        type="bool",
        default=False,
        applies_to=("agent",),
        storage=STORAGE_RUNTIME_SETTING,
        label="Seed new session with summary",
        help="Agent bots only: on new SDK-session creation, seed it with a chat-history summary.",
        legacy_keys=("seed_summary_on_new_session",),
    ),
    "timeout_seconds": SettingDefinition(
        key="timeout_seconds",
        type="int",
        default=None,
        applies_to=("agent",),
        storage=STORAGE_RUNTIME_SETTING,
        label="Agent turn timeout (s)",
        help="Agent bots only: max seconds for a single agent turn before timeout.",
        legacy_keys=("timeout_seconds",),
    ),
    "session_model": SettingDefinition(
        key="session_model",
        type="str",
        default=None,
        applies_to=("agent",),
        storage=STORAGE_RUNTIME_SETTING,
        label="Session model (raw SDK id)",
        help="Agent bots only: the raw backend/SDK model id the session runs.",
        legacy_keys=("session_model",),
    ),
    # --- identity: keep in the blob/column, but typed so the UI knows it ------
    "session_key": SettingDefinition(
        key="session_key",
        type="str",
        default=None,
        applies_to=("agent",),
        storage=STORAGE_AGENT_BACKEND_CONFIG,
        label="Session key",
        help="Agent bots only: stable session identity. Not a tunable — do not edit casually.",
        legacy_keys=("session_key",),
    ),
    # --- chat-only legacy flag (kept for compat; superseded by continuity) ----
    "include_summaries": SettingDefinition(
        key="include_summaries",
        type="bool",
        default=True,
        applies_to=("chat",),
        storage=STORAGE_REQUEST_FLAG,
        label="Include summaries",
        help="Chat bots only: include summary rows when assembling history. "
             "Superseded by session_memory_continuity.",
        legacy_keys=("include_summaries",),
    ),
}


def definitions_for_bot_type(bot_type: str) -> list[SettingDefinition]:
    """Settings that apply to a given bot_type."""
    return [d for d in SETTING_DEFINITIONS.values() if bot_type in d.applies_to]


def applies_to_bot_type(key: str, bot_type: str) -> bool:
    """Whether a setting applies to a bot_type. Unknown keys default True (be
    permissive — an undeclared setting is not something we should hide)."""
    d = SETTING_DEFINITIONS.get(key)
    return True if d is None else (bot_type in d.applies_to)
