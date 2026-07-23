"""Tests for prompt history context selection."""

from types import SimpleNamespace

import pytest

from llm_bawt.models.message import Message
from llm_bawt.utils.history import ContextBudgetError, HistoryManager


def _getter(overrides: dict):
    return lambda key, fallback: overrides.get(key, fallback)


class _DummyConsole:
    def print(self, *args, **kwargs):
        return None


class _DummyClient:
    console = _DummyConsole()


def _history(settings_getter=None) -> HistoryManager:
    config = SimpleNamespace(
        HISTORY_FILE=None,
        SYSTEM_MESSAGE="system",
        MAX_CONTEXT_MESSAGES=0,
    )
    return HistoryManager(
        client=_DummyClient(),
        config=config,
        bot_id="nova",
        settings_getter=settings_getter,
    )


# ── TASK-612: allocation ladder (system → raw@history_tokens → summaries) ──


def _msgs(n: int):
    return [
        Message("user" if i % 2 else "assistant", f"message {i}", timestamp=float(i))
        for i in range(1, n + 1)
    ]


def test_raw_is_bounded_by_history_tokens_then_summaries_fill_remainder() -> None:
    # history_tokens=15 fits only ~2 raw msgs (~6 tokens each) even though the
    # physical budget (10k) could hold all 10 — proves raw is capped by the
    # Tier-3 policy knob, not the physical budget. Summaries then fill the
    # large remaining physical budget.
    history = _history(_getter({"history_tokens": 15, "summary_count": 5}))
    history.messages = (
        [Message("summary", f"summary {c}", timestamp=float(i)) for i, c in enumerate("ABC")]
        + _msgs(10)
    )

    context = history.get_context_messages(max_tokens=10_000)

    regular = [m.content for m in context if m.role in ("user", "assistant")]
    summaries = [m for m in context if m.role == "summary"]
    # newest raw survive (raw bounded, newest-first)
    assert regular == ["message 9", "message 10"]
    # all 3 summaries fit the remaining physical budget
    assert len(summaries) == 3
    # output order is chronological: summaries precede raw
    roles = [m.role for m in context]
    assert roles.index("summary") < roles.index("user")


def test_summary_count_zero_carries_no_summaries() -> None:
    history = _history(_getter({"history_tokens": 12000, "summary_count": 0}))
    history.messages = (
        [Message("summary", "summary A", timestamp=1.0)] + _msgs(4)
    )

    context = history.get_context_messages(max_tokens=10_000)

    assert not [m for m in context if m.role == "summary"]
    assert [m.content for m in context if m.role in ("user", "assistant")] == [
        "message 1", "message 2", "message 3", "message 4",
    ]


def test_newest_complete_turn_floored_when_history_tokens_tiny() -> None:
    # history_tokens=1 can't fit even one ~6-token message, but the newest turn
    # must survive because it fits the physical budget.
    history = _history(_getter({"history_tokens": 1, "summary_count": 5}))
    history.messages = _msgs(5)

    context = history.get_context_messages(max_tokens=10_000)

    regular = [m.content for m in context if m.role in ("user", "assistant")]
    assert regular == ["message 5"]


def test_system_only_overflow_is_the_hard_fail() -> None:
    # Physical budget below the system prompt cost -> ContextBudgetError.
    history = _history(_getter({"history_tokens": 12000, "summary_count": 5}))
    history.messages = _msgs(3)

    with pytest.raises(ContextBudgetError):
        history.get_context_messages(max_tokens=1)


def test_newest_turn_over_physical_budget_degrades_without_raising() -> None:
    # system("system")=5 tokens fits budget=8, but system+newest(6)=11 > 8, so
    # the newest turn is dropped (degrade-and-log) — NOT a hard fail.
    history = _history(_getter({"history_tokens": 12000, "summary_count": 5}))
    history.messages = _msgs(3)

    context = history.get_context_messages(max_tokens=8)

    assert any(m.role == "system" for m in context)
    assert not [m for m in context if m.role in ("user", "assistant")]


def test_newest_complete_turn_preserves_user_and_assistant() -> None:
    # TASK-612 finding 2: when the newest message is an assistant reply, the
    # floor must preserve the WHOLE turn (its user prompt too), not just the
    # trailing assistant. history_tokens=1 fits nothing in the bounded fill, so
    # the complete-turn floor is what carries the pair.
    history = _history(_getter({"history_tokens": 1, "summary_count": 5}))
    history.messages = [
        Message("user", "u1", timestamp=1.0),
        Message("assistant", "a1", timestamp=2.0),
    ]

    context = history.get_context_messages(max_tokens=10_000)

    regular = [m.content for m in context if m.role in ("user", "assistant")]
    assert regular == ["u1", "a1"]


def test_summary_only_scope_not_starved_by_raw() -> None:
    # TASK-612 finding 1: under a tight budget, a summary-only caller
    # (include_history=False) must not have its summaries starved by raw that
    # gets allocated and then discarded. want_history=False skips raw entirely,
    # so the summary fills the whole physical budget.
    history = _history(_getter({"history_tokens": 12000, "summary_count": 5}))
    history.messages = [
        Message("summary", "S", timestamp=0.0),
        Message("user", "u1", timestamp=1.0),
        Message("assistant", "a1", timestamp=2.0),
    ]

    payload = history.build_context_payload(
        include_history=False, include_summaries=True, max_tokens=20
    )

    assert len(payload.summary_messages) == 1
    assert len(payload.regular_messages) == 0


# ── TASK-612 follow-up: tool-evidence rows ride RAW history, not system ──


def _tool(content_ish: str, ts: float) -> Message:
    return Message("system", f"[Tool Results] {content_ish}", timestamp=ts)


def test_tool_evidence_rides_raw_and_survives_floor_with_its_turn() -> None:
    # user -> [Tool Results] -> assistant with history_tokens=1: the whole turn
    # (including tool evidence) is floored in under the physical budget, and the
    # tool row never rode the non-negotiable system budget.
    history = _history(_getter({"history_tokens": 1, "summary_count": 5}))
    history.messages = [
        Message("user", "u1", timestamp=1.0),
        _tool("ran ls", 2.0),
        Message("assistant", "a1", timestamp=3.0),
    ]

    context = history.get_context_messages(max_tokens=10_000, charge_system=True)

    contents = [m.content for m in context]
    assert "u1" in contents and "a1" in contents
    assert any(c.startswith("[Tool Results]") for c in contents)


def test_summary_only_excludes_tool_evidence() -> None:
    # Summary-only inline scope (include_history=False): tool-evidence must not
    # sneak through as raw, and must not starve the summary.
    history = _history(_getter({"history_tokens": 12000, "summary_count": 5}))
    history.messages = [
        Message("summary", "S", timestamp=0.0),
        Message("user", "u1", timestamp=1.0),
        _tool("big", 2.0),
        Message("assistant", "a1", timestamp=3.0),
    ]

    payload = history.build_context_payload(
        include_history=False, include_summaries=True, max_tokens=20
    )

    assert len(payload.summary_messages) == 1
    assert len(payload.tool_result_messages) == 0
    assert len(payload.regular_messages) == 0


def test_seed_delivery_does_not_budget_tool_evidence() -> None:
    # A large tool-evidence row would eat the seed budget if charged. Seed
    # delivery strips ALL system rows, so it must not be budgeted — both
    # conversation rows still fit.
    history = _history(_getter({"history_tokens": 1000, "summary_count": 5}))
    history.messages = [
        Message("user", "aaaaa", timestamp=1.0),
        _tool("x" * 400, 2.0),
        Message("assistant", "bbbbb", timestamp=3.0),
    ]

    seed = history.build_context_payload(delivery="seed", max_tokens=15)

    assert all(m.role != "system" for m in seed.seed_messages)
    assert len(seed.regular_messages) == 2


def test_oversized_tool_evidence_degrades_not_hard_fail() -> None:
    # Physical budget fits the persona system prompt but not the giant tool turn:
    # degrade-and-log, never the hard fail (only the persona prompt can raise).
    history = _history(_getter({"history_tokens": 12000, "summary_count": 5}))
    history.messages = [
        Message("user", "u1", timestamp=1.0),
        _tool("x" * 400, 2.0),
        Message("assistant", "a1", timestamp=3.0),
    ]

    context = history.get_context_messages(max_tokens=8, charge_system=True)

    # persona system prompt survives; no ContextBudgetError raised
    assert any(
        m.role == "system" and not m.content.startswith("[Tool") for m in context
    )


# ── TASK-613: seed delivery must not charge stripped system rows ──


def _uniform(n: int, content: str = "aaaaa"):
    # each message estimates to len//4+4 = 5 tokens; "system" is also 5 tokens
    return [
        Message("user" if i % 2 else "assistant", content, timestamp=float(i))
        for i in range(1, n + 1)
    ]


def test_seed_delivery_does_not_reserve_system_tokens() -> None:
    # budget=15, system=5 tokens, messages=5 tokens each, history_tokens huge.
    # charge_system=True (inline): headroom 15-5=10 -> 2 msgs.
    # charge_system=False (seed):  headroom 15     -> 3 msgs.
    history = _history(_getter({"history_tokens": 1000, "summary_count": 5}))
    history.messages = _uniform(20)

    inline = history.get_context_messages(max_tokens=15, charge_system=True)
    seed = history.get_context_messages(max_tokens=15, charge_system=False)

    assert len([m for m in inline if m.role in ("user", "assistant")]) == 2
    assert len([m for m in seed if m.role in ("user", "assistant")]) == 3


def test_build_context_payload_seed_vs_inline_accounting() -> None:
    history = _history(_getter({"history_tokens": 1000, "summary_count": 5}))
    history.messages = _uniform(20)

    inline = history.build_context_payload(delivery="inline", max_tokens=15)
    seed = history.build_context_payload(delivery="seed", max_tokens=15)

    # inline charges system -> 2 delivered raw msgs; seed doesn't -> 3
    assert len(inline.regular_messages) == 2
    assert len(seed.regular_messages) == 3
    # seed delivery view carries no system rows
    assert all(m.role != "system" for m in seed.seed_messages)


def test_seed_over_system_budget_does_not_hard_fail() -> None:
    # budget below system cost: inline hard-fails, seed does not (system stripped)
    history = _history(_getter({"history_tokens": 1000, "summary_count": 5}))
    history.messages = _uniform(3)

    with pytest.raises(ContextBudgetError):
        history.get_context_messages(max_tokens=3, charge_system=True)

    # no raise; system not delivered so it can't overflow the seed budget
    seed = history.get_context_messages(max_tokens=3, charge_system=False)
    assert seed is not None


# ── TASK-647: history_max_age_hours bounds the raw bucket by age ──


def test_max_age_drops_old_raw_keeps_recent_and_summaries() -> None:
    import time as _time

    now = _time.time()
    history = _history(_getter({
        "history_tokens": 12000,
        "summary_count": 5,
        "history_max_age_hours": 2,
    }))
    history.messages = [
        Message("summary", "summary A", timestamp=now - 90000),
        Message("user", "old question", timestamp=now - 10800),      # 3h ago
        Message("assistant", "old answer", timestamp=now - 10700),
        Message("user", "fresh question", timestamp=now - 60),
        Message("assistant", "fresh answer", timestamp=now - 30),
    ]

    context = history.get_context_messages(max_tokens=10_000)

    regular = [m.content for m in context if m.role in ("user", "assistant")]
    assert regular == ["fresh question", "fresh answer"]
    # summaries are NOT age-bounded
    assert len([m for m in context if m.role == "summary"]) == 1


def test_max_age_zero_is_unlimited() -> None:
    import time as _time

    now = _time.time()
    history = _history(_getter({"history_max_age_hours": 0}))
    history.messages = [
        Message("user", "ancient", timestamp=now - 10 * 86400),
        Message("assistant", "reply", timestamp=now - 10 * 86400 + 5),
    ]

    context = history.get_context_messages(max_tokens=10_000)
    regular = [m.content for m in context if m.role in ("user", "assistant")]
    assert regular == ["ancient", "reply"]


def test_max_age_applies_on_no_budget_path_and_keeps_untimestamped() -> None:
    import time as _time

    now = _time.time()
    history = _history(_getter({"history_max_age_hours": 1}))
    stale = Message("user", "stale", timestamp=now - 7200)
    no_ts = Message("assistant", "no timestamp", timestamp=None)
    fresh = Message("user", "fresh", timestamp=now - 10)
    history.messages = [stale, no_ts, fresh]

    context = history.get_context_messages(max_tokens=0)  # no-budget path
    regular = [m.content for m in context if m.role in ("user", "assistant")]
    assert regular == ["no timestamp", "fresh"]
