"""OpenAI ChatGPT (codex subscription) usage adapter.

The ChatGPT codex backend has **no standalone usage endpoint** — the OAuth
bearer only authorizes ``/responses``. Plan-usage instead rides on every
``/responses`` call as ``x-codex-*`` response headers. The claude-code bridge's
proxy harvests those headers on each codex turn and publishes a canonical
snapshot to Redis (see
``src/claude_code_bridge/proxy/usage_capture.py``). This adapter just reads
that snapshot — it owns no credential and makes no upstream call.

Mapping:
- ``primary``   window → ``session_5h``  (the rolling 5-hour limit)
- ``secondary`` window → ``weekly_all``  (the weekly all-models limit)
- ``plan_type`` → tiered display name, e.g. ``ChatGPT · codex · Plus``
"""

from __future__ import annotations

import json
import logging
import os

from ..base import UsageAdapter
from ..canonical import (
    ProviderUsage,
    UsageLimit,
    STATUS_ERROR,
    STATUS_OK,
)

logger = logging.getLogger(__name__)

# Shared contract with the bridge-side writer. Keep both in lockstep.
REDIS_KEY = "llm_bawt:usage:openai_chatgpt:snapshot"

_PLAN_LABEL = {
    "free": "Free",
    "plus": "Plus",
    "pro": "Pro",
    "team": "Team",
    "business": "Business",
    "enterprise": "Enterprise",
}


def _display_name(plan_type) -> str:
    base = OpenAIChatGPTUsageAdapter.display_name
    if not plan_type:
        return base
    raw = str(plan_type).strip()
    if not raw:
        return base
    label = _PLAN_LABEL.get(raw.lower(), raw.replace("_", " ").title())
    return f"{base} · {label}"


def _window_label(minutes) -> str | None:
    try:
        m = int(minutes)
    except (TypeError, ValueError):
        return None
    if m <= 0:
        return None
    if m % 1440 == 0:
        return f"{m // 1440}d"
    if m % 60 == 0:
        return f"{m // 60}h"
    return f"{m}m"


def _limit_from_window(window: dict | None, cid: str, label: str, fallback_window: str) -> UsageLimit | None:
    if not isinstance(window, dict):
        return None
    used = window.get("used_percent")
    if used is None:
        return None
    try:
        used_pct = float(used)
    except (TypeError, ValueError):
        return None
    return UsageLimit(
        id=cid,
        label=label,
        used_pct=used_pct,
        resets_at=window.get("reset_at"),
        window=_window_label(window.get("window_minutes")) or fallback_window,
        active=True,
    )


class OpenAIChatGPTUsageAdapter(UsageAdapter):
    provider = "openai_chatgpt"
    display_name = "ChatGPT · codex"
    backend = "claude-code"

    async def _read_snapshot(self) -> dict | None:
        url = (
            os.getenv("LLM_BAWT_REDIS_URL")
            or os.getenv("REDIS_URL")
            or "redis://redis:6379/0"
        )
        try:
            import redis.asyncio as redis

            client = redis.from_url(
                url,
                decode_responses=True,
                socket_timeout=2,
                socket_connect_timeout=2,
            )
            try:
                raw = await client.get(REDIS_KEY)
            finally:
                await client.aclose()
        except Exception as e:  # noqa: BLE001
            logger.warning("codex usage snapshot read failed: %s", e)
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except (ValueError, TypeError):
            return None

    async def fetch(self) -> ProviderUsage:
        snap = await self._read_snapshot()
        if snap is None:
            return self._base(
                available=False,
                status=STATUS_ERROR,
                error=(
                    "No codex usage observed yet — plan-usage is harvested from "
                    "/responses headers and populates after the next codex turn."
                ),
            )

        limits: list[UsageLimit] = []
        primary = _limit_from_window(snap.get("primary"), "session_5h", "5-hour limit", "5h")
        if primary is not None:
            limits.append(primary)
        secondary = _limit_from_window(snap.get("secondary"), "weekly_all", "Weekly · all models", "7d")
        if secondary is not None:
            limits.append(secondary)

        display = _display_name(snap.get("plan_type"))
        if not limits:
            return self._base(
                available=False,
                status=STATUS_OK,
                display_name=display,
                error="Codex snapshot present but carried no usable limit windows.",
                raw=snap,
            )

        return self._base(
            available=True,
            status=STATUS_OK,
            display_name=display,
            fetched_at=snap.get("captured_at"),
            limits=limits,
            raw=snap,
        )
