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


def test_max_context_messages_caps_raw_history_messages() -> None:
    history = _history(lambda key, fallback: 4 if key == "max_context_messages" else fallback)
    history.messages = [
        Message("user" if i % 2 else "assistant", f"message {i}", timestamp=float(i))
        for i in range(1, 11)
    ]

    context = history.get_context_messages(max_tokens=10_000)

    regular = [msg for msg in context if msg.role in ("user", "assistant")]
    assert [msg.content for msg in regular] == ["message 7", "message 8", "message 9", "message 10"]


def test_zero_max_context_messages_keeps_token_only_behavior() -> None:
    history = _history(lambda key, fallback: 0 if key == "max_context_messages" else fallback)
    history.messages = [
        Message("user" if i % 2 else "assistant", f"message {i}", timestamp=float(i))
        for i in range(1, 7)
    ]

    context = history.get_context_messages(max_tokens=10_000)

    regular = [msg for msg in context if msg.role in ("user", "assistant")]
    assert len(regular) == 6


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
