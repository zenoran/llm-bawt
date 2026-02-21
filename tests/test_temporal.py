"""Tests for temporal prompt context helpers."""

from __future__ import annotations

import time

from llm_bawt.models.message import Message
from llm_bawt.utils.temporal import build_temporal_context


def test_temporal_context_includes_datetime() -> None:
    block = build_temporal_context([])
    assert "Current date/time:" in block


def test_temporal_context_uses_previous_turn_for_last_contact() -> None:
    now = time.time()
    msgs = [
        Message(role="assistant", content="hello", timestamp=now - 120),
        Message(role="user", content="new prompt", timestamp=now),
    ]
    block = build_temporal_context(msgs)
    # Should not resolve to "just now" from the current user message.
    assert "Last conversation: 2 minutes ago" in block
