"""Tests for aggregate cross-bot search visibility policy (TASK-571 layout)."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from llm_bawt import bots as bots_module
from llm_bawt.mcp_server.storage import MemoryStorage
from llm_bawt.memory.postgresql import MESSAGES_PARENT


@dataclass
class _FakeBot:
    slug: str
    include_in_global_search: bool | None = True


class _FakeBotManager:
    def __init__(self, config):
        self.config = config

    def list_bots(self):
        return [
            _FakeBot("nova", include_in_global_search=False),
            _FakeBot("byte", include_in_global_search=True),
            _FakeBot("caid", include_in_global_search=None),
        ]


def test_global_search_bots_exclude_opted_out_bots(monkeypatch) -> None:
    """Aggregate search should skip only configured opt-out bots."""

    storage = MemoryStorage(config=SimpleNamespace())
    monkeypatch.setattr(bots_module, "BotManager", _FakeBotManager)
    monkeypatch.setattr(
        storage,
        "_list_partition_bots",
        lambda parent: ["byte", "caid", "nova", "orphan"],
    )

    # Only the explicit ``include_in_global_search=False`` bot drops out;
    # True, unset (None counts as visible), and partition-only orphans stay.
    assert storage._global_search_bots(MESSAGES_PARENT) == [
        "byte",
        "caid",
        "orphan",
    ]


def test_global_search_bots_no_exclusions(monkeypatch) -> None:
    """With no opt-outs the partition enumeration passes through untouched."""

    class AllVisibleBotManager(_FakeBotManager):
        def list_bots(self):
            return [_FakeBot("nova"), _FakeBot("byte")]

    storage = MemoryStorage(config=SimpleNamespace())
    monkeypatch.setattr(bots_module, "BotManager", AllVisibleBotManager)
    monkeypatch.setattr(
        storage,
        "_list_partition_bots",
        lambda parent: ["byte", "nova"],
    )

    assert storage._global_search_bots(MESSAGES_PARENT) == ["byte", "nova"]


def test_global_search_bots_empty_partitions(monkeypatch) -> None:
    """No partitions → no sources, and the bot policy isn't even consulted."""

    storage = MemoryStorage(config=SimpleNamespace())
    monkeypatch.setattr(storage, "_list_partition_bots", lambda parent: [])

    assert storage._global_search_bots(MESSAGES_PARENT) == []
