"""Hermetic timing contracts; no database, network, or model calls."""
from datetime import datetime

import pytest

from llm_bawt.service.prompt_timing import PromptTiming, resolve_local, validate_cron


def dt(value):
    return datetime.fromisoformat(value)


def cron(expression, zone="America/New_York", **kwargs):
    return PromptTiming(kind="cron", cron_expression=expression, timezone=zone, **kwargs)


def test_once_and_preview():
    timing = PromptTiming(kind="once", once_at=dt("2026-09-16T09:00:00-04:00"))
    assert timing.next_after(dt("2026-09-15T00:00:00Z")) == dt("2026-09-16T13:00:00Z")
    assert timing.next_after(dt("2026-09-16T13:00:00Z")) is None
    assert len(timing.preview(dt("2026-09-15T00:00:00Z"))) == 1


def test_intervals_are_anchored_not_completion_relative():
    timing = PromptTiming(kind="interval", interval_seconds=3600,
                          anchor_at=dt("2026-09-15T12:15:00Z"))
    assert timing.next_after(dt("2026-09-15T14:57:00Z")) == dt("2026-09-15T15:15:00Z")
    assert timing.latest_at(dt("2026-09-15T14:57:00Z")) == dt("2026-09-15T14:15:00Z")
    assert timing.latest_at(dt("2026-09-15T12:14:59Z")) is None
    assert timing.next_after(dt("2026-09-01T00:00:00Z")) == timing.anchor_at


def test_inclusive_start_exclusive_end():
    timing = cron("0 9 * * *", starts_at=dt("2026-09-16T13:00:00Z"),
                  ends_at=dt("2026-09-17T13:00:00Z"))
    assert timing.next_after(dt("2026-09-01T00:00:00Z")) == dt("2026-09-16T13:00:00Z")
    assert timing.next_after(dt("2026-09-16T13:00:00Z")) is None
    assert timing.latest_at(dt("2026-09-20T00:00:00Z")) == dt("2026-09-16T13:00:00Z")
    assert timing.latest_at(dt("2026-09-16T12:59:00Z")) is None


@pytest.mark.parametrize("expression", ["0 0 31 2 *", "* * * * * *", "@daily",
    "0 9 ? * mon", "0 9 L * *", "H * * * *", "0 0 * * fri#2", "0 25 * * *",
    "0 0 * * * TZ=UTC", "0 0 * janx *", "*/0 * * * *"])
def test_invalid_or_non_unix_cron_rejected(expression):
    with pytest.raises(ValueError):
        validate_cron(expression)


@pytest.mark.parametrize("expression", ["*/5 * * * *", "0 9 * * MON-FRI",
    "0 0 1 * wed", "0 0 * jan,mar sun", "0 0 * * 7"])
def test_unix_expressions(expression):
    assert validate_cron(expression) == expression.lower()


def test_unix_day_of_month_or_weekday():
    timing = cron("0 9 1 * wed", "UTC")
    assert timing.next_after(dt("2026-09-01T09:00:00Z")) == dt("2026-09-02T09:00:00Z")


def test_spring_gap_skipped_not_shifted():
    timing = cron("30 2 * * *")
    assert timing.next_after(dt("2027-03-13T08:00:00Z")) == dt("2027-03-15T06:30:00Z")
    assert timing.latest_at(dt("2027-03-14T08:00:00Z")) == dt("2027-03-13T07:30:00Z")


def test_fall_repeated_time_only_first_fold():
    timing = cron("30 1 * * *")
    assert timing.next_after(dt("2026-11-01T04:00:00Z")) == dt("2026-11-01T05:30:00Z")
    assert timing.next_after(dt("2026-11-01T05:30:00Z")) == dt("2026-11-02T06:30:00Z")
    assert timing.next_after(dt("2026-11-01T06:15:00Z")) == dt("2026-11-02T06:30:00Z")
    assert timing.latest_at(dt("2026-11-01T06:15:00Z")) == dt("2026-11-01T05:30:00Z")


def test_latest_every_minute_during_second_fold():
    timing = cron("* * * * *")
    assert timing.latest_at(dt("2026-11-01T06:15:00Z")) == dt("2026-11-01T05:59:00Z")
    assert timing.next_after(dt("2026-11-01T06:15:00Z")) == dt("2026-11-01T07:00:00Z")


def test_non_hour_dst_transition():
    timing = cron("45 1 * * *", "Australia/Lord_Howe")
    first = dt("2027-04-03T14:45:00Z")
    assert timing.next_after(dt("2027-04-03T13:00:00Z")) == first
    assert timing.next_after(first) == dt("2027-04-04T15:15:00Z")
    assert timing.latest_at(dt("2027-04-03T15:05:00Z")) == first


def test_leap_day_with_bounded_search():
    timing = cron("0 0 29 2 *", "UTC")
    assert timing.next_after(dt("2096-03-01T00:00:00Z")) == dt("2104-02-29T00:00:00Z")


def test_local_editor_ambiguity_gap_and_offset():
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_local(dt("2026-11-01T01:30:00"), "America/New_York")
    assert resolve_local(dt("2026-11-01T01:30:00"), "America/New_York", -240) == dt("2026-11-01T05:30:00Z")
    assert resolve_local(dt("2026-11-01T01:30:00"), "America/New_York", -300) == dt("2026-11-01T06:30:00Z")
    with pytest.raises(ValueError, match="does not exist"):
        resolve_local(dt("2027-03-14T02:30:00"), "America/New_York")
    with pytest.raises(ValueError, match="offset"):
        resolve_local(dt("2026-09-15T09:00:00"), "America/New_York", 0)


@pytest.mark.parametrize("kwargs", [
    {"kind": "once", "once_at": "2026-09-15T09:00:00"},
    {"kind": "once", "once_at": "2026-09-15T09:00:00Z", "cron_expression": "* * * * *"},
    {"kind": "cron", "cron_expression": "* * * * *", "timezone": "fake/zone"},
    {"kind": "interval", "interval_seconds": 61, "anchor_at": "2026-09-15T09:00:00Z"},
    {"kind": "interval", "interval_seconds": 30, "anchor_at": "2026-09-15T09:00:00Z"},
    {"kind": "interval", "interval_seconds": 60},
])
def test_invalid_timing_definitions(kwargs):
    with pytest.raises(ValueError):
        PromptTiming(**kwargs)
