"""Tests for prompt history context selection."""

from types import SimpleNamespace

from llm_bawt.models.message import Message
from llm_bawt.utils.history import HistoryManager


class _DummyConsole:
    def print(self, *args, **kwargs):
        return None


class _DummyClient:
    console = _DummyConsole()


def _history(settings_getter=None) -> HistoryManager:
    config = SimpleNamespace(
        HISTORY_FILE=None,
        SYSTEM_MESSAGE="system",
        MAX_CONTEXT_MESSAGES=0,
        MEMORY_PROTECTED_RECENT_TURNS=3,
        SUMMARIZATION_MAX_IN_CONTEXT=5,
        SUMMARIZATION_COMPACT_CONTEXT=True,
    )
    return HistoryManager(
        client=_DummyClient(),
        config=config,
        bot_id="nova",
        settings_getter=settings_getter,
    )


def test_max_context_messages_caps_raw_history_messages() -> None:
    history = _history(lambda key, fallback: 4 if key == "max_context_messages" else fallback)
    history.messages = [
        Message("user" if i % 2 else "assistant", f"message {i}", timestamp=float(i))
        for i in range(1, 11)
    ]

    context = history.get_context_messages(max_tokens=10_000)

    regular = [msg for msg in context if msg.role in ("user", "assistant")]
    assert [msg.content for msg in regular] == ["message 7", "message 8", "message 9", "message 10"]


def test_zero_max_context_messages_keeps_token_only_behavior() -> None:
    history = _history(lambda key, fallback: 0 if key == "max_context_messages" else fallback)
    history.messages = [
        Message("user" if i % 2 else "assistant", f"message {i}", timestamp=float(i))
        for i in range(1, 7)
    ]

    context = history.get_context_messages(max_tokens=10_000)

    regular = [msg for msg in context if msg.role in ("user", "assistant")]
    assert len(regular) == 6
