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
    type: str  # "bool" | "int" | "float" | "str" | "string_list"
    default: object
    applies_to: tuple[str, ...]  # bot types this setting affects
    storage: str
    label: str = ""
    help: str = ""
    # If this setting supersedes/absorbs a legacy key, name it here so the
    # migration and compat shims know the provenance.
    legacy_keys: tuple[str, ...] = field(default_factory=tuple)
    # Optional frontend render hint. Blank = render by ``type``; "model" tells
    # the UI to render a model-alias picker (dropdown of /v1/models) instead of
    # a free-text box. Surfaced verbatim in /v1/config/schema (TASK-522).
    ui_widget: str = ""


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
    "agent_global_prompt_enabled": SettingDefinition(
        key="agent_global_prompt_enabled",
        type="bool",
        default=False,
        applies_to=("agent",),
        storage=STORAGE_RUNTIME_SETTING,
        label="Inject shared agent global prompt",
        help=(
            "Agent bots only: inject the shared 'agents.global_prompt' block into "
            "the cacheable system-prompt prefix. Steers planning to the BawtHub "
            "task system (observable) instead of the harness plan mode. Resolves "
            "global-then-bot, so a global=true baseline applies to every agent bot "
            "unless a bot overrides it. Chat bots never see it (manifest-gated)."
        ),
    ),
    "claude_code_disallowed_tools": SettingDefinition(
        key="claude_code_disallowed_tools",
        type="string_list",
        default=[
            "EnterPlanMode",
            "ExitPlanMode",
            "EnterWorktree",
            "ExitWorktree",
        ],
        applies_to=("agent",),
        storage=STORAGE_RUNTIME_SETTING,
        label="Claude Code disabled SDK tools",
        help=(
            "Global Claude Code harness policy: tool names withheld from every "
            "Claude Agent SDK turn. Changes are delivered with the next command; "
            "proxy turns additionally disable WebSearch and WebFetch."
        ),
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
    # --- background-job models (TASK-522): global-only, data-driven ----------
    # System-wide model selection for the 3 LLM-using background jobs
    # (summarization, extraction, profile maintenance). Resolved at GLOBAL scope
    # only — no per-bot override — via resolve_job_model(). Cascade at every call
    # site is JOB-SPECIFIC FIRST, then the shared maintenance_model fallback:
    #     <job>_model -> maintenance_model
    # so each job's own picker actually controls that job, and maintenance_model
    # is the single "default for all jobs" knob. The terminal default lives on
    # maintenance_model (dolphin-qwen-3b), so an all-unset instance still runs.
    # These retire the env/config vars SUMMARIZATION_MODEL / EXTRACTION_MODEL /
    # MAINTENANCE_MODEL / PROFILE_MAINTENANCE_MODEL (option (a): keep the shared
    # maintenance knob).
    "maintenance_model": SettingDefinition(
        key="maintenance_model",
        type="str",
        default="dolphin-qwen-3b",
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Default job model (all jobs)",
        help="Default model for every background job. Each job (summarization, "
             "extraction, profile maintenance) inherits this unless it has its "
             "own model set below. This is the last resort, so it always has a "
             "value — an all-unset instance runs on it.",
        legacy_keys=("MAINTENANCE_MODEL",),
        ui_widget="model",
    ),
    "summarization_model": SettingDefinition(
        key="summarization_model",
        type="str",
        default=None,
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="History summarization model",
        help="Model the history-summarization job runs. Unset = inherit the "
             "default job model (maintenance_model).",
        legacy_keys=("SUMMARIZATION_MODEL",),
        ui_widget="model",
    ),
    "extraction_model": SettingDefinition(
        key="extraction_model",
        type="str",
        default=None,
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Memory extraction model",
        help="Model the memory-extraction job runs. Unset = inherit the default "
             "job model (maintenance_model).",
        legacy_keys=("EXTRACTION_MODEL",),
        ui_widget="model",
    ),
    "profile_maintenance_model": SettingDefinition(
        key="profile_maintenance_model",
        type="str",
        default=None,
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Profile maintenance model",
        help="Model the profile-maintenance job runs. Unset = inherit the "
             "default job model (maintenance_model).",
        legacy_keys=("PROFILE_MAINTENANCE_MODEL",),
        ui_widget="model",
    ),
    # --- subagent/small-fast model (TASK-546): proxy-compatible override -------
    # When the claude-code bridge routes through the Anthropic-compat proxy
    # (non-Anthropic providers like openai_chatgpt/zai/xai), Claude Code's
    # internal background Haiku calls and subagent model resolution send bare
    # Anthropic model IDs that the proxy rejects (400). This setting overrides
    # BOTH ANTHROPIC_SMALL_FAST_MODEL and CLAUDE_CODE_SUBAGENT_MODEL env vars
    # for proxy-routed turns, so background tasks and subagents use a
    # provider-qualified model the proxy accepts. Unset = inherit the bot's
    # main model (default_model) on proxy turns; ignored on Anthropic-direct
    # turns where the built-in Haiku path works natively.
    "subagent_model": SettingDefinition(
        key="subagent_model",
        type="str",
        default=None,
        applies_to=("agent",),
        storage=STORAGE_AGENT_BACKEND_CONFIG,
        label="Subagent / background model",
        help=(
            "Agent bots only: model used for Claude Code's internal subagents "
            "and background tasks (title generation, tool-use summaries, etc.) "
            "when running through the proxy. Unset = inherit the bot's main "
            "model. Must be a provider-qualified model the proxy accepts "
            "(e.g. openai_chatgpt/gpt-5.4-mini)."
        ),
        legacy_keys=("subagent_model",),
        ui_widget="model",
    ),
    # --- context window / output budget (TASK-602: retire env fallbacks) -------
    # These are resolved at GLOBAL scope via resolve_global_runtime_setting()
    # (global runtime_settings row -> this declared default), NEVER from a Config
    # BaseSettings env field. Same pattern resolve_job_model() used to retire the
    # SUMMARIZATION_MODEL / EXTRACTION_MODEL env vars.
    "model_context_window_default": SettingDefinition(
        key="model_context_window_default",
        type="int",
        default=128000,
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Default model context window",
        help=(
            "Fallback context window (tokens) used ONLY when a model has no "
            "per-model window in the catalog (models.default_context_window / "
            "model_endpoints.context_window_override). The catalog is the source "
            "of truth for per-model windows; this global default just catches "
            "NULLs. Not a per-model or per-bot value — set the window on the "
            "model in the catalog instead."
        ),
    ),
    "max_output_tokens": SettingDefinition(
        key="max_output_tokens",
        type="int",
        default=4096,
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Max output tokens",
        help=(
            "Max tokens the model may generate per reply. Also subtracted from "
            "the resolved context window to derive the context token budget "
            "(budget = window - max_output). Retires the LLM_BAWT_MAX_OUTPUT_TOKENS "
            "env var."
        ),
        legacy_keys=("max_output_tokens",),
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
