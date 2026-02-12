"""Tests for history summarization helpers."""

import time

from llm_bawt.memory.summarization import (
    Session,
    estimate_session_token_savings,
    find_summarizable_sessions,
    prioritize_summarizable_sessions,
)


def _session(start: float, end: float, messages: list[dict]) -> Session:
    return Session(
        start_timestamp=start,
        end_timestamp=end,
        messages=messages,
        message_ids=[m["id"] for m in messages],
    )


def test_find_summarizable_sessions_filters_by_age_and_size() -> None:
    now = time.time()
    old = now - 7200
    recent = now - 60

    eligible = _session(
        old,
        old + 10,
        [
            {"id": f"a{i}", "role": "user", "content": "hello", "timestamp": old + i}
            for i in range(1, 5)
        ],
    )
    too_small = _session(
        old,
        old + 10,
        [
            {"id": f"b{i}", "role": "user", "content": "hi", "timestamp": old + i}
            for i in range(1, 3)
        ],
    )
    too_new = _session(
        recent,
        recent + 10,
        [
            {"id": f"c{i}", "role": "user", "content": "now", "timestamp": recent + i}
            for i in range(1, 5)
        ],
    )

    result = find_summarizable_sessions(
        [eligible, too_small, too_new],
        history_duration_seconds=300,
        min_messages=4,
    )

    assert len(result) == 1
    assert result[0].start_timestamp == eligible.start_timestamp


def test_prioritize_summarizable_sessions_orders_by_estimated_savings() -> None:
    now = time.time() - 7200
    short_session = _session(
        now,
        now + 10,
        [
            {"id": "s1", "role": "user", "content": "short", "timestamp": now + 1},
            {"id": "s2", "role": "assistant", "content": "ok", "timestamp": now + 2},
            {"id": "s3", "role": "user", "content": "tiny", "timestamp": now + 3},
            {"id": "s4", "role": "assistant", "content": "done", "timestamp": now + 4},
        ],
    )
    long_text = "very long message " * 120
    long_session = _session(
        now + 20,
        now + 40,
        [
            {"id": "l1", "role": "user", "content": long_text, "timestamp": now + 21},
            {"id": "l2", "role": "assistant", "content": long_text, "timestamp": now + 22},
            {"id": "l3", "role": "user", "content": long_text, "timestamp": now + 23},
            {"id": "l4", "role": "assistant", "content": long_text, "timestamp": now + 24},
        ],
    )

    savings_short = estimate_session_token_savings(short_session)
    savings_long = estimate_session_token_savings(long_session)
    assert savings_long > savings_short

    prioritized = prioritize_summarizable_sessions([short_session, long_session])
    assert prioritized[0].start_timestamp == long_session.start_timestamp
