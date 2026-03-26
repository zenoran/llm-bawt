"""Tests for history summarization helpers."""

import time

from llm_bawt.memory.summarization import (
    Session,
    SUMMARIZATION_PROMPT,
    compress_structured_summary_text,
    extract_summary_sections,
    normalize_structured_summary_text,
    estimate_session_token_savings,
    find_budget_overflow_sessions,
    prioritize_summarizable_sessions,
)


def _session(start: float, end: float, messages: list[dict]) -> Session:
    return Session(
        start_timestamp=start,
        end_timestamp=end,
        messages=messages,
        message_ids=[m["id"] for m in messages],
    )


def _msg(id: str, role: str, content: str, timestamp: float) -> dict:
    return {"id": id, "role": role, "content": content, "timestamp": timestamp}


def test_find_budget_overflow_sessions_returns_empty_when_everything_fits() -> None:
    """When all messages fit in the token budget, nothing should be summarized."""
    now = time.time()
    messages = [
        _msg(f"a{i}", "user", "hello world", now + i)
        for i in range(1, 5)
    ]
    result = find_budget_overflow_sessions(
        messages,
        session_gap_seconds=3600,
        max_context_tokens=50000,  # huge budget
        protected_recent_turns=3,
        min_messages_per_session=2,
    )
    assert len(result) == 0


def test_find_budget_overflow_sessions_returns_empty_with_zero_budget() -> None:
    """With zero budget, nothing should be summarized (budget-driven means no budget = no action)."""
    now = time.time()
    messages = [
        _msg(f"a{i}", "user", "hello world", now + i)
        for i in range(1, 5)
    ]
    result = find_budget_overflow_sessions(
        messages,
        max_context_tokens=0,
    )
    assert len(result) == 0


def test_find_budget_overflow_sessions_identifies_overflow() -> None:
    """When messages exceed budget, oldest sessions should be returned for summarization."""
    now = time.time()
    # Create two sessions with a gap > 3600s
    old_session_ts = now - 7200
    recent_session_ts = now - 60

    old_msgs = [
        _msg(f"old{i}", "user" if i % 2 else "assistant", "A" * 200, old_session_ts + i)
        for i in range(1, 9)
    ]
    recent_msgs = [
        _msg(f"new{i}", "user" if i % 2 else "assistant", "B" * 200, recent_session_ts + i)
        for i in range(1, 5)
    ]

    all_msgs = old_msgs + recent_msgs

    # Set a tight budget that can't fit everything
    # Each message is ~200/4 + 4 = 54 tokens, 12 messages = ~648 tokens
    # With a budget of 400, the old session should overflow
    result = find_budget_overflow_sessions(
        all_msgs,
        session_gap_seconds=3600,
        max_context_tokens=400,
        protected_recent_turns=2,
        min_messages_per_session=2,
    )
    assert len(result) >= 1
    # The overflow sessions should be from the old timestamps
    for session in result:
        assert session.end_timestamp < recent_session_ts


def test_find_budget_overflow_sessions_respects_min_messages() -> None:
    """Sessions with fewer messages than min_messages_per_session should be excluded."""
    now = time.time()
    old_ts = now - 7200

    # One message in old session (below min_messages_per_session=2)
    messages = [
        _msg("old1", "user", "A" * 1000, old_ts),
        _msg("new1", "user", "B" * 200, now - 10),
        _msg("new2", "assistant", "C" * 200, now - 5),
    ]

    result = find_budget_overflow_sessions(
        messages,
        session_gap_seconds=3600,
        max_context_tokens=300,
        protected_recent_turns=1,
        min_messages_per_session=2,
    )
    assert len(result) == 0


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


def test_summarization_prompt_includes_intent_and_tone_sections() -> None:
    prompt = SUMMARIZATION_PROMPT.format(messages="User: hi\nAssistant: hello")
    assert "Intent:" in prompt
    assert "Tone:" in prompt
    assert "Open Loops:" in prompt


def test_extract_summary_sections_parses_structured_output() -> None:
    text = (
        "Summary: User asked about deploys.\n"
        "Intent: Fix a broken production pipeline.\n"
        "Tone: Frustrated but focused.\n"
        "Open Loops: Confirm rollback policy."
    )
    parsed = extract_summary_sections(text)
    assert parsed["summary"] == "User asked about deploys."
    assert parsed["intent"] == "Fix a broken production pipeline."
    assert parsed["tone"] == "Frustrated but focused."
    assert parsed["open_loops"] == "Confirm rollback policy."


def test_normalize_structured_summary_text_fills_missing_sections() -> None:
    normalized = normalize_structured_summary_text("User asked about DNS propagation.")
    assert normalized.startswith("Summary: User asked about DNS propagation.")
    assert "Intent:" not in normalized
    assert "Tone:" not in normalized
    assert "Open Loops:" not in normalized


def test_compress_structured_summary_text_preserves_overlong_sections() -> None:
    source_text = (
        "Summary: " + ("A" * 500) + "\n"
        "Key Details: " + ("B" * 1200) + "\n"
        "Intent: " + ("C" * 300) + "\n"
        "Tone: " + ("D" * 260) + "\n"
        "Open Loops: " + ("E" * 400)
    )
    compact = compress_structured_summary_text(source_text)
    parsed = extract_summary_sections(compact)

    assert parsed["summary"] == "A" * 500
    assert parsed["key_details"] == "B" * 1200
    assert parsed["intent"] == "C" * 300
    assert parsed["tone"] == "D" * 260
    assert parsed["open_loops"] == "E" * 400


def test_compress_structured_summary_text_ignores_source_session_for_storage() -> None:
    now = time.time() - 7200
    session = _session(
        now,
        now + 10,
        [
            {"id": "m1", "role": "user", "content": "x" * 300, "timestamp": now + 1},
            {"id": "m2", "role": "assistant", "content": "y" * 300, "timestamp": now + 2},
        ],
    )
    source_text = (
        "Summary: " + ("A " * 200) + "\n"
        "Key Details: " + ("B " * 400) + "\n"
        "Intent: " + ("C " * 120) + "\n"
        "Tone: " + ("D " * 120) + "\n"
        "Open Loops: " + ("E " * 180)
    )
    compact = compress_structured_summary_text(source_text, source_session=session)
    parsed = extract_summary_sections(compact)
    assert parsed["summary"].startswith("A A A")
    assert "..." not in compact


def test_compress_structured_summary_text_strips_summary_boilerplate() -> None:
    compact = compress_structured_summary_text(
        "Summary: The conversation begins with a debugging pass over flaky deploys.\n"
        "Open Loops: Confirm rollback threshold."
    )
    parsed = extract_summary_sections(compact)
    assert parsed["summary"] == "A debugging pass over flaky deploys."
