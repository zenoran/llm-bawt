"""Pure prompt schedule timing. Cron parsing belongs to croniter, not us.

Calendar candidates are generated in wall time, then resolved with zoneinfo.
This deliberately avoids croniter's implicit DST adjustment (02:30 -> 03:00)
and repeated-hour execution. Stored/returned instants are always aware UTC.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import CroniterBadDateError, croniter
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

UTC = timezone.utc
EPSILON = timedelta(microseconds=1)
MAX_CANDIDATES = 4096  # Bounds DST-gap filtering, even historical date-line jumps.


def utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Execution instants must include a timezone offset")
    return value.astimezone(UTC)


def validate_cron(expression: str) -> str:
    """Restrict the public grammar to five-field Unix cron; library validates values."""
    fields = expression.lower().split()
    if len(fields) != 5 or len(expression) > 256:
        raise ValueError("Use five-field Unix cron: minute hour day month weekday")
    for index, field in enumerate(fields):
        names = {3: "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec",
                 4: "sun|mon|tue|wed|thu|fri|sat"}.get(index)
        atom = rf"(?:[0-9]+|{names})" if names else r"[0-9]+"
        term = rf"(?:\*|{atom}(?:-{atom})?)(?:/[1-9][0-9]*)?"
        if not re.fullmatch(rf"{term}(?:,{term})*", field):
            raise ValueError("Only Unix cron names, numbers, *, lists, ranges and steps are supported")
    normalized = " ".join(fields)
    if not croniter.is_valid(normalized, strict=True):
        raise ValueError("Invalid or impossible cron expression")
    return normalized


def resolve_local(local: datetime, zone: str, offset_minutes: int | None = None) -> datetime:
    """Resolve editor wall time; require a choice for ambiguous one-time instants."""
    if local.tzinfo is not None:
        raise ValueError("Local date/time must not include an offset; supply offset_minutes separately")
    tz = ZoneInfo(zone)
    choices = {}
    for fold in (0, 1):
        candidate = local.replace(tzinfo=tz, fold=fold)
        instant = candidate.astimezone(UTC)
        if instant.astimezone(tz).replace(tzinfo=None) == local:
            choices[instant] = int(candidate.utcoffset().total_seconds() // 60)
    if not choices:
        raise ValueError("This local time does not exist in the selected timezone")
    if offset_minutes is not None:
        for instant, offset in choices.items():
            if offset == offset_minutes:
                return instant
        raise ValueError("The supplied offset is not valid for this local time")
    if len(choices) > 1:
        raise ValueError("This local time is ambiguous; choose an explicit UTC offset")
    return next(iter(choices))


class PromptTiming(BaseModel):
    """Shared API/storage timing contract. Bounds are inclusive start, exclusive end."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["once", "interval", "cron"]
    timezone: str = "UTC"
    once_at: AwareDatetime | None = None
    interval_seconds: int | None = Field(default=None, ge=60, le=31536000, strict=True)
    anchor_at: AwareDatetime | None = None
    cron_expression: str | None = None
    starts_at: AwareDatetime | None = None
    ends_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_definition(self) -> "PromptTiming":
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("Use a valid IANA timezone") from exc
        present = (self.once_at is not None, self.interval_seconds is not None,
                   self.anchor_at is not None, self.cron_expression is not None)
        required = {"once": (True, False, False, False),
                    "interval": (False, True, True, False),
                    "cron": (False, False, False, True)}[self.kind]
        if present != required:
            raise ValueError(f"Timing fields do not match schedule kind {self.kind}")
        if self.interval_seconds is not None and self.interval_seconds % 60:
            raise ValueError("Intervals must be whole minutes")
        if self.starts_at and self.ends_at and utc(self.ends_at) <= utc(self.starts_at):
            raise ValueError("End must be later than start")
        if self.once_at:
            if self.starts_at or self.ends_at:
                raise ValueError("One-time schedules do not accept recurring bounds")
        if self.cron_expression:
            object.__setattr__(self, "cron_expression", validate_cron(self.cron_expression))
        return self

    def _cron_candidate(self, bound: datetime, *, previous: bool) -> datetime | None:
        tz = ZoneInfo(self.timezone)
        local = bound.astimezone(tz)
        base = local.replace(tzinfo=None)
        if previous and local.fold:
            # During the second fold, the latest eligible first-fold minute can
            # be later on the wall clock (01:59 EDT precedes 01:15 EST).
            base += local.replace(fold=0).utcoffset() - local.utcoffset()
        iterator = croniter(self.cron_expression, base, day_or=True,
                            max_years_between_matches=8)
        for _ in range(MAX_CANDIDATES):
            try:
                wall = iterator.get_prev(datetime) if previous else iterator.get_next(datetime)
            except CroniterBadDateError:
                return None
            candidate = wall.replace(tzinfo=tz, fold=0).astimezone(UTC)
            # A gap round-trips to a different wall time. Always choose fold=0.
            if candidate.astimezone(tz).replace(tzinfo=None) != wall:
                continue
            if (candidate < bound) if previous else (candidate > bound):
                return candidate
        raise ValueError("Cron candidate search exceeded its bounded DST window")

    def next_after(self, after: datetime) -> datetime | None:
        """Next occurrence strictly after an instant, within configured bounds."""
        after = utc(after)
        if self.starts_at:
            after = max(after, utc(self.starts_at) - EPSILON)
        if self.kind == "once":
            result = utc(self.once_at)
            if result <= after:
                return None
        elif self.kind == "interval":
            anchor = utc(self.anchor_at)
            interval = timedelta(seconds=self.interval_seconds)
            index = max(0, (after - anchor) // interval + 1)
            result = anchor + index * interval
        else:
            result = self._cron_candidate(after, previous=False)
        if result is not None and self.ends_at and result >= utc(self.ends_at):
            return None
        return result

    def latest_at(self, at: datetime) -> datetime | None:
        """Latest occurrence at/before an instant, without iterating missed slots."""
        at = utc(at)
        if self.ends_at:
            at = min(at, utc(self.ends_at) - EPSILON)
        if self.kind == "once":
            result = utc(self.once_at)
        elif self.kind == "interval":
            anchor = utc(self.anchor_at)
            interval = timedelta(seconds=self.interval_seconds)
            index = (at - anchor) // interval
            if index < 0:
                return None
            result = anchor + index * interval
        else:
            result = self._cron_candidate(at + EPSILON, previous=True)
        if result is None or result > at:
            return None
        if self.starts_at and result < utc(self.starts_at):
            return None
        return result

    def preview(self, after: datetime, count: int = 5) -> list[dict[str, str]]:
        if not 1 <= count <= 10:
            raise ValueError("Preview count must be between 1 and 10")
        result = []
        cursor = utc(after)
        for _ in range(count):
            cursor = self.next_after(cursor)
            if cursor is None:
                break
            result.append({"utc": cursor.isoformat(),
                           "local": cursor.astimezone(ZoneInfo(self.timezone)).isoformat()})
        return result
