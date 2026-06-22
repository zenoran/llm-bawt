"""Tests for the path-agnostic turn-completion signal (turn_logs.ended_at).

The `ended_at` stamp is the single source of truth for "is this turn still
running?" — set exactly once on the first terminal transition by
save_turn/update_turn.  Both writers delegate the decision to `_is_terminal`,
so testing that pure function (plus the terminal-status set) validates the
stamping behavior without needing a live Postgres engine.
"""

from llm_bawt.service.turn_logs import TERMINAL_TURN_STATUSES, _is_terminal


def test_in_progress_statuses_are_not_terminal():
    # The two non-terminal statuses: streaming (streaming path) and pending
    # (non-streaming / bot-to-bot path). Neither should stamp ended_at.
    assert _is_terminal("streaming", None) is False
    assert _is_terminal("pending", None) is False
    assert "streaming" not in TERMINAL_TURN_STATUSES
    assert "pending" not in TERMINAL_TURN_STATUSES


def test_all_terminal_statuses_stamp():
    for status in ("ok", "completed", "error", "timeout", "cancelled", "aborted"):
        assert _is_terminal(status, None) is True, status


def test_end_reason_alone_is_terminal():
    # Streaming finalize stamps end_reason without a terminal status — that
    # path must still count as done (covers end_reason="stop"/"question"/etc).
    assert _is_terminal(None, "stop") is True
    assert _is_terminal(None, "question") is True
    assert _is_terminal("streaming", "stop") is True


def test_no_signal_is_not_terminal():
    assert _is_terminal(None, None) is False
    assert _is_terminal("", None) is False


def test_unknown_status_without_end_reason_is_not_terminal():
    # Defensive: an unrecognized status shouldn't accidentally mark done.
    assert _is_terminal("weird_new_status", None) is False
