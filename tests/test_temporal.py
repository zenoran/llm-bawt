"""Tests for temporal prompt context helpers."""

from __future__ import annotations

import time

from llm_bawt.models.message import Message
from llm_bawt.utils.temporal import build_temporal_context


def test_temporal_context_includes_core_fields() -> None:
    block = build_temporal_context([])
    assert "## Temporal Context" in block
    assert "NOW_UTC:" in block
    assert "NOW_LOCAL:" in block
    assert "TODAY_LOCAL:" in block
    assert "YESTERDAY_LOCAL:" in block
    assert "LAST_CONTACT_AT_LOCAL:" in block


def test_temporal_context_uses_previous_turn_for_last_contact() -> None:
    now = time.time()
    msgs = [
        Message(role="assistant", content="hello", timestamp=now - 120),
        Message(role="user", content="new prompt", timestamp=now),
    ]
    block = build_temporal_context(msgs)
    # Should not resolve to "just now" from the current user message.
    assert "LAST_CONTACT_RELATIVE: 2 minutes ago" in block
