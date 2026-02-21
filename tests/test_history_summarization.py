"""Tests for history summarization helpers."""

import time

from llm_bawt.memory.summarization import (
    Session,
    SUMMARIZATION_PROMPT,
    compress_structured_summary_text,
    extract_summary_sections,
    is_low_signal_session,
    normalize_structured_summary_text,
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
        skip_low_signal=False,
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
    assert "Intent:" in normalized
    assert "Tone:" in normalized
    assert "Open Loops:" in normalized


def test_compress_structured_summary_text_caps_overlong_sections() -> None:
    source_text = (
        "Summary: " + ("A" * 500) + "\n"
        "Key Details: " + ("B" * 1200) + "\n"
        "Intent: " + ("C" * 300) + "\n"
        "Tone: " + ("D" * 260) + "\n"
        "Open Loops: " + ("E" * 400)
    )
    compact = compress_structured_summary_text(source_text)
    parsed = extract_summary_sections(compact)

    assert len(parsed["summary"]) <= 223
    assert len(parsed["key_details"]) <= 363
    assert len(parsed["intent"]) <= 143
    assert len(parsed["tone"]) <= 103
    assert len(parsed["open_loops"]) <= 183


def test_compress_structured_summary_text_scales_with_source_session() -> None:
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
    # 55% of 600 chars ~= 330 cap (minimum floor is 260), so this should be aggressively trimmed.
    assert len(compact) < 420


def test_is_low_signal_session_flags_greeting_noise() -> None:
    now = time.time() - 7200
    session = _session(
        now,
        now + 5,
        [
            {"id": "n1", "role": "user", "content": "hi", "timestamp": now + 1},
            {"id": "n2", "role": "assistant", "content": "hey", "timestamp": now + 2},
            {"id": "n3", "role": "user", "content": "test", "timestamp": now + 3},
            {"id": "n4", "role": "assistant", "content": "ok", "timestamp": now + 4},
        ],
    )
    assert is_low_signal_session(session) is True


def test_is_low_signal_session_allows_substantive_exchange() -> None:
    now = time.time() - 7200
    session = _session(
        now,
        now + 50,
        [
            {
                "id": "s1",
                "role": "user",
                "content": "The deployment is failing after migration and I need help tracing DB auth issues.",
                "timestamp": now + 1,
            },
            {
                "id": "s2",
                "role": "assistant",
                "content": "Share exact error lines and we can isolate whether this is token, permission, or network config.",
                "timestamp": now + 2,
            },
            {
                "id": "s3",
                "role": "user",
                "content": "Error says role cannot read table, but token refresh also failed right before it.",
                "timestamp": now + 3,
            },
            {
                "id": "s4",
                "role": "assistant",
                "content": "Got it. Check migration grants first, then rotate token and retry webhook callback validation.",
                "timestamp": now + 4,
            },
        ],
    )
    assert is_low_signal_session(session) is False
