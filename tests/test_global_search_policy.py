"""Tests for aggregate cross-bot search visibility policy."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from llm_bawt import bots as bots_module
from llm_bawt.mcp_server.storage import MemoryStorage


@dataclass
class _FakeBot:
    slug: str
    include_in_global_search: bool | None = True


def test_global_search_tables_exclude_opted_out_bots(monkeypatch) -> None:
    """Aggregate search should skip only configured opt-out bot tables."""

    class FakeBotManager:
        def __init__(self, config):
            self.config = config

        def list_bots(self):
            return [
                _FakeBot("nova", include_in_global_search=False),
                _FakeBot("byte", include_in_global_search=True),
                _FakeBot("caid", include_in_global_search=None),
            ]

    storage = MemoryStorage(config=SimpleNamespace())
    monkeypatch.setattr(bots_module, "BotManager", FakeBotManager)
    monkeypatch.setattr(
        storage,
        "_discover_tables",
        lambda suffix: [
            ("byte", "byte_messages"),
            ("caid", "caid_messages"),
            ("nova", "nova_messages"),
            ("orphan", "orphan_messages"),
        ],
    )

    assert storage._discover_global_search_tables("_messages") == [
        ("byte", "byte_messages"),
        ("caid", "caid_messages"),
        ("orphan", "orphan_messages"),
    ]
