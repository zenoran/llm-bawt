"""Tests for tool-call event correlation helpers."""

from types import SimpleNamespace

from llm_bawt.service.tool_call_events import (
    extract_trigger_message,
    message_id_matches,
    parse_message_filters,
)
from llm_bawt.utils.history import HistoryManager


class _DummyConsole:
    def print(self, *args, **kwargs):
        return None


class _DummyClient:
    def __init__(self):
        self.console = _DummyConsole()


class _DummyBackend:
    def add_message(self, role: str, content: str, timestamp: float | None = None) -> str:
        return "db-msg-123"


def test_extract_trigger_message_uses_last_user_message_id():
    payload = {
        "messages": [
            {"role": "system", "content": "x"},
            {"role": "user", "id": "msg-1", "timestamp": 1.0},
            {"role": "assistant", "content": "y"},
            {"role": "user", "db_id": "msg-2", "timestamp": 2.0},
        ]
    }
    trigger = extract_trigger_message(payload)
    assert trigger == ("msg-2", "user", 2.0)


def test_parse_message_filters_supports_single_and_csv():
    result = parse_message_filters("abc", ["def,ghi", "jkl"])
    assert result == {"abc", "def", "ghi", "jkl"}


def test_message_id_matches_supports_prefix_targets():
    assert message_id_matches("12345678-aaaa-bbbb-cccc-ddddeeeeffff", {"12345678"})
    assert not message_id_matches("abcdef", {"123"})


def test_history_manager_uses_backend_message_id():
    config = SimpleNamespace(
        HISTORY_FILE=None,
        SYSTEM_MESSAGE="",
        HISTORY_DURATION_SECONDS=3600,
        SUMMARIZATION_COMPACT_CONTEXT=True,
    )
    history = HistoryManager(
        client=_DummyClient(),
        config=config,
        db_backend=_DummyBackend(),
        bot_id="nova",
    )
    history.add_message("user", "hello")
    assert history.messages[-1].db_id == "db-msg-123"
