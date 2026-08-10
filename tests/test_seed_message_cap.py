"""TASK-785: per-message char cap on fresh-session continuity seeds.

Hermetic — exercises the pure ``cap_seed_contents`` helper plus the setting
registration. The cap exists because inter-bot task-report deliveries
(12-16k chars each) were seeded verbatim, tripling fresh-session base cost.
"""

from llm_bawt.service.routes.history_seed import cap_seed_contents
from llm_bawt.setting_definitions import setting_default


def test_under_cap_passthrough_untouched() -> None:
    contents, trimmed = cap_seed_contents(
        [("user", "short"), ("assistant", "also short")], cap=3000
    )
    assert contents == ["short", "also short"]
    assert trimmed == 0


def test_exactly_at_cap_untouched() -> None:
    body = "x" * 100
    contents, trimmed = cap_seed_contents([("user", body)], cap=100)
    assert contents == [body]
    assert trimmed == 0


def test_oversized_user_message_trimmed_with_marker() -> None:
    body = "A" * 5000
    contents, trimmed = cap_seed_contents([("user", body)], cap=3000)
    assert trimmed == 1
    assert contents[0].startswith("A" * 3000)
    assert "trimmed 2000 chars for session seed" in contents[0]
    assert "messages_get" in contents[0]
    # Trimmed result must actually be smaller than the original.
    assert len(contents[0]) < len(body)


def test_oversized_assistant_message_trimmed() -> None:
    contents, trimmed = cap_seed_contents([("assistant", "B" * 4000)], cap=1000)
    assert trimmed == 1
    assert len(contents[0]) < 4000


def test_summary_rows_exempt_from_cap() -> None:
    body = "S" * 5000
    contents, trimmed = cap_seed_contents([("summary", body)], cap=100)
    assert contents == [body]
    assert trimmed == 0


def test_cap_zero_disables_pre_785_behavior() -> None:
    body = "C" * 50000
    contents, trimmed = cap_seed_contents(
        [("user", body), ("assistant", body)], cap=0
    )
    assert contents == [body, body]
    assert trimmed == 0


def test_mixed_stream_counts_only_trimmed_rows() -> None:
    msgs = [
        ("summary", "s" * 4000),   # exempt
        ("user", "u" * 4000),      # trimmed
        ("assistant", "ok"),       # under cap
        ("user", "w" * 3500),      # trimmed
    ]
    contents, trimmed = cap_seed_contents(msgs, cap=3000)
    assert trimmed == 2
    assert contents[0] == "s" * 4000
    assert contents[2] == "ok"


def test_whale_delivery_scenario_reduces_seed_size() -> None:
    """The motivating case: three 15k-char inter-bot reports in one seed."""
    whale = "TASK-714 checkpoint report " * 600  # ~16k chars
    msgs = [("user", whale)] * 3 + [("assistant", "ack")] * 3
    contents, trimmed = cap_seed_contents(msgs, cap=3000)
    assert trimmed == 3
    before = sum(len(c) for _, c in msgs)
    after = sum(len(c) for c in contents)
    assert after < before * 0.25  # ~48k -> under 12k


def test_setting_registered_with_expected_default() -> None:
    assert setting_default("seed_message_max_chars", None) == 3000
