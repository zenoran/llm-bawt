"""TASK-611: Tier-3 context-policy settings (registration, resolution, cutover).

Proves the three canonical Tier-3 settings are registered, that the assembler
reads them by their canonical keys (not the retired summarization_* env attrs),
and that summary_count=0 yields a raw-only bot (no summaries) — the slice guard.
"""

from types import SimpleNamespace

from llm_bawt.models.message import Message
from llm_bawt.utils.history import HistoryManager
from llm_bawt.setting_definitions import SETTING_DEFINITIONS, setting_default


class _DummyConsole:
    def print(self, *args, **kwargs):
        return None


class _DummyClient:
    console = _DummyConsole()


def _history(settings_getter=None, config=None) -> HistoryManager:
    # NOTE: config deliberately lacks the retired SUMMARIZATION_* attrs to prove
    # the assembler no longer reads them (registry defaults only).
    config = config or SimpleNamespace(HISTORY_FILE=None, SYSTEM_MESSAGE="system")
    return HistoryManager(
        client=_DummyClient(),
        config=config,
        bot_id="nova",
        settings_getter=settings_getter,
    )


def _mixed_messages(n_summaries: int, n_regular: int) -> list[Message]:
    msgs: list[Message] = []
    for i in range(n_summaries):
        msgs.append(Message("summary", f"summary {i}", timestamp=float(i)))
    for i in range(n_regular):
        role = "user" if i % 2 else "assistant"
        msgs.append(Message(role, f"message {i}", timestamp=float(100 + i)))
    return msgs


# --- registration -----------------------------------------------------------

def test_tier3_settings_registered_global_and_bot():
    for key, default in (("history_tokens", 12000), ("summary_count", 5), ("compact_context", True)):
        d = SETTING_DEFINITIONS[key]
        assert d.default == default
        # Global + per-bot override: applies to BOTH bot types.
        assert set(d.applies_to) == {"chat", "agent"}


def test_history_tokens_default_is_bounded_not_zero():
    # The footgun TASK-602 forbids: 0 = fill the whole window.
    assert setting_default("history_tokens") == 12000
    assert setting_default("history_tokens") != 0


# --- canonical-key resolution (no env reads) --------------------------------

def test_summary_count_read_by_canonical_key():
    # settings_getter returns 2 ONLY for the canonical key; the retired key
    # would return the fallback. Two summaries carried proves the canonical read.
    def getter(key, fallback):
        return 2 if key == "summary_count" else fallback

    history = _history(getter)
    history.messages = _mixed_messages(n_summaries=5, n_regular=2)
    ctx = history.get_context_messages(max_tokens=10_000)
    summaries = [m for m in ctx if m.role == "summary"]
    assert len(summaries) == 2


def test_summary_count_zero_carries_no_summaries():
    # Raw-only bot: summary_count=0 must select NONE (the [-0:] slice guard).
    def getter(key, fallback):
        return 0 if key == "summary_count" else fallback

    history = _history(getter)
    history.messages = _mixed_messages(n_summaries=5, n_regular=2)
    ctx = history.get_context_messages(max_tokens=10_000)
    assert [m for m in ctx if m.role == "summary"] == []
    # Regular history still flows.
    assert [m for m in ctx if m.role in ("user", "assistant")]


def test_compact_context_read_by_canonical_key():
    seen = {}

    def getter(key, fallback):
        seen[key] = seen.get(key, 0) + 1
        return fallback

    history = _history(getter)
    history.messages = [Message("summary", "line one\nline two", timestamp=1.0)]
    history.get_context_messages(max_tokens=10_000)
    # The compaction path consulted the canonical key, never the retired one.
    assert "compact_context" in seen
    assert "summarization_compact_context" not in seen


def test_assembler_survives_config_without_retired_attrs():
    # No SUMMARIZATION_* attrs on config, no settings_getter -> registry defaults.
    history = _history(settings_getter=None)
    history.messages = _mixed_messages(n_summaries=3, n_regular=2)
    ctx = history.get_context_messages(max_tokens=10_000)
    # Default summary_count=5 >= 3 available -> all three summaries carried.
    assert len([m for m in ctx if m.role == "summary"]) == 3
