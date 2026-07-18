"""Shared claude-code-bridge constants + helpers (TASK-555).

Relocated from ``bridge.py`` so the split-out mixin modules can import them
without a cycle back through ``bridge``. ``bridge.py`` re-imports every name
here (preserving ``from .bridge import _get_fresh_oauth_token`` in __main__).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

logger = logging.getLogger("claude_code_bridge.bridge")

SESSION_PREFIX = "claude-code:"

# TASK-445: session-seed constants.
# agent_backend_config flag that gates history-summary seeding on new sessions.
SEED_SETTING_KEY = "seed_summary_on_new_session"
# TASK-492: unified session-memory-continuity key (supersedes SEED_SETTING_KEY).
CONTINUITY_SETTING_KEY = "session_memory_continuity"

# TASK-490: MCP tool context body. Canonical default lives in the app registry
# (agents.mcp_tool_context_template); the bridge cannot import llm_bawt, so it
# fetches the (overridable) body via GET /v1/prompts/{key} and falls back to
# this BYTE-IDENTICAL copy. The leading "\n\n" separator is added at append time.
MCP_TOOL_CONTEXT_KEY = "agents.mcp_tool_context_template"
_MCP_TOOL_CONTEXT_FALLBACK = (
    "## MCP Tool Context\n"
    "Your bot_id is \"{bot_slug}\". When using bawthub MCP tools:\n"
    "- Memory/message tools: always pass bot_id=\"{bot_slug}\"\n"
    "- Profile tool with entity_type=\"user\": use entity_id=\"nick\" (the user)\n"
    "- Profile tool with entity_type=\"bot\": use entity_id=\"{bot_slug}\" (yourself)"
)
# Stamped on synthetic seed transcript entries. Not load-bearing for resume
# (the CLI reads message content, not this), just keeps entries well-formed.
_SEED_CLI_VERSION = "2.1.191"
# Mirrors the SDK's _sanitize_path: every non-alphanumeric char -> '-'.
_SEED_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9]")


def _bot_slug_from_session_key(session_key: str) -> str:
    sk = (session_key or "").strip()
    if not sk:
        return ""
    return sk.split(":", 1)[0]


def _fmt_tokens(n: int | None) -> str:
    """Human-format a token count: 67288 -> '67.3k', 1486 -> '1.5k'."""
    if n is None:
        return "?"
    n = int(n)
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k"


def _usage_input_total(usage: dict | None) -> int:
    """Total prompt tokens in an Anthropic-shaped usage dict."""
    if not isinstance(usage, dict):
        return 0
    return (
        int(usage.get("input_tokens", 0) or 0)
        + int(usage.get("cache_read_input_tokens", 0) or 0)
        + int(usage.get("cache_creation_input_tokens", 0) or 0)
    )


def _proxy_context_window(model: str | None) -> int | None:
    """Context window for proxy-routed models the Claude Code CLI doesn't know.

    The CLI's ``getContextWindowForModel`` defaults unknown models to 200k
    (Claude's window). Grok-via-bridge is ``xai/<model>`` and would otherwise
    report 200k + Claude-tier costs. Keep this table small and explicit.
    """
    if not model:
        return None
    m = str(model).strip().lower()
    # Exact / prefix matches for known xAI models (docs.x.ai, July 2026).
    if m.startswith("xai/grok-4.5"):
        return 500_000
    if m.startswith("xai/grok-4.3") or m.startswith("xai/grok-4.20"):
        return 1_000_000
    if m.startswith("xai/grok-build"):
        return 256_000
    if m.startswith("xai/"):
        return 500_000
    return None


# Per-1M-token USD rates for proxy providers the CLI prices as "unknown"
# (which falls through to Opus-tier $5/$25 and wildly overstates Grok cost).
_XAI_RATES = {
    # model substring -> (input, output, cache_read, cache_write)
    "grok-4.5": (2.0, 6.0, 0.5, 2.0),
    "grok-4.3": (1.25, 2.50, 0.20, 1.25),
    "grok-4.20": (1.25, 2.50, 0.20, 1.25),
    "grok-build": (1.0, 2.0, 0.20, 1.0),
}
_XAI_DEFAULT_RATES = (2.0, 6.0, 0.5, 2.0)


def _estimate_proxy_cost_usd(model: str | None, usage: dict | None) -> float | None:
    """Estimate turn cost for proxy-routed models from token counts.

    Returns None when we can't price the model (leave SDK total_cost_usd).
    """
    if not model or not isinstance(usage, dict):
        return None
    m = str(model).strip().lower()
    if not m.startswith("xai/"):
        return None
    rates = _XAI_DEFAULT_RATES
    for key, r in _XAI_RATES.items():
        if key in m:
            rates = r
            break
    in_rate, out_rate, cr_rate, cw_rate = rates
    inp = int(usage.get("input_tokens", 0) or 0)
    cr = int(usage.get("cache_read_input_tokens", 0) or usage.get("cache_read_tokens", 0) or 0)
    cw = int(
        usage.get("cache_creation_input_tokens", 0)
        or usage.get("cache_creation_tokens", 0)
        or 0
    )
    out = int(usage.get("output_tokens", 0) or 0)
    cost = (inp * in_rate + cr * cr_rate + cw * cw_rate + out * out_rate) / 1_000_000
    return round(cost, 6) if cost > 0 else 0.0


def _pick_iteration_usage(
    latest_assistant_usage: dict | None,
    latest_stream_usage: dict | None,
    cumulative_usage: dict | None,
    *,
    proxy_model: bool,
) -> dict:
    """Choose the usage snapshot that best represents final context size.

    Preference:
      1. For proxy streams (xAI / ChatGPT / z.ai): prefer the last
         ``message_delta`` when it reports a higher total input than the
         AssistantMessage snapshot. The synthetic Anthropic stream only
         finalizes real usage on message_delta; AssistantMessage can keep
         message_start zeros or a partial merge that under-counts.
      2. Otherwise last AssistantMessage usage (true final API view).
      3. Stream usage, then cumulative ResultMessage.usage.
    """
    am = latest_assistant_usage if isinstance(latest_assistant_usage, dict) else None
    sm = latest_stream_usage if isinstance(latest_stream_usage, dict) else None
    cu = cumulative_usage if isinstance(cumulative_usage, dict) else None

    if proxy_model and sm and _usage_input_total(sm) > 0:
        if not am or _usage_input_total(sm) >= _usage_input_total(am):
            return sm
    if am and _usage_input_total(am) > 0:
        return am
    if sm and _usage_input_total(sm) > 0:
        return sm
    if am:
        return am
    if sm:
        return sm
    return cu or {}



def _read_latest_compact_metadata(session_id: str | None) -> dict | None:
    """Return the most recent ``compact_boundary`` metadata for a session.

    The Claude Agent SDK does NOT emit ``compact_boundary`` on the wire — a
    ``/compact`` turn surfaces only as ``SystemMessage(subtype="status")`` with
    a ``compact_result`` string ("success"/"failed"); the actual pre/post token
    counts are written solely to the on-disk transcript. So to report the new
    resident context size we read it back from the session's ``.jsonl`` (located
    by ``<session_id>.jsonl`` under ``~/.claude/projects/``) and return the last
    ``compactMetadata`` dict ({trigger, preTokens, postTokens, durationMs, ...}).
    Best-effort: returns ``None`` if the file or marker can't be found.
    """
    if not session_id:
        return None
    try:
        base = Path.home() / ".claude" / "projects"
        matches = list(base.glob(f"*/{session_id}.jsonl")) or list(
            base.glob(f"**/{session_id}.jsonl")
        )
        if not matches:
            return None
        meta: dict | None = None
        with open(matches[0], "r") as fh:
            for line in fh:
                if "compact_boundary" not in line:
                    continue
                try:
                    entry = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if entry.get("subtype") == "compact_boundary":
                    cm = entry.get("compactMetadata")
                    if isinstance(cm, dict):
                        meta = cm  # keep the LAST boundary in file order
        return meta
    except Exception:
        logger.debug("compact metadata read failed for %s", session_id, exc_info=True)
        return None


_CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"
_OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_REFRESH_BUFFER_MS = 5 * 60 * 1000


def _load_oauth_bundle() -> tuple[dict, dict | None]:
    """Return (raw_credentials, claudeAiOauth bundle)."""
    if not _CREDENTIALS_PATH.exists():
        return {}, None
    data = json.loads(_CREDENTIALS_PATH.read_text())
    oauth = data.get("claudeAiOauth") or None
    return data, oauth


def _save_oauth_bundle(raw_credentials: dict, oauth: dict) -> None:
    raw_credentials["claudeAiOauth"] = oauth
    _CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CREDENTIALS_PATH.write_text(json.dumps(raw_credentials, indent=2))


def _token_expired_or_stale(expires_at: int | None) -> bool:
    if not expires_at:
        return False
    now_ms = int(time.time() * 1000)
    return (now_ms + _REFRESH_BUFFER_MS) >= int(expires_at)


def _refresh_oauth_bundle(oauth: dict, *, raw_credentials: dict | None = None) -> dict:
    refresh_token = oauth.get("refreshToken")
    if not refresh_token:
        raise RuntimeError("Claude OAuth refresh token missing")

    scopes = oauth.get("scopes") or []
    resp = httpx.post(
        _OAUTH_TOKEN_URL,
        json={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": _OAUTH_CLIENT_ID,
            "scope": " ".join(scopes),
        },
        headers={"Content-Type": "application/json"},
        timeout=15.0,
    )
    if resp.is_error:
        detail = (resp.text or "").strip().replace("\n", " ")[:500]
        raise RuntimeError(f"Claude OAuth refresh failed ({resp.status_code}): {detail}")
    payload = resp.json()

    refreshed = {
        **oauth,
        "accessToken": payload["access_token"],
        "refreshToken": payload.get("refresh_token", refresh_token),
        "expiresAt": int(time.time() * 1000) + int(payload["expires_in"]) * 1000,
        "scopes": payload.get("scope", "").split() if payload.get("scope") else scopes,
    }
    if raw_credentials is not None:
        try:
            _save_oauth_bundle(raw_credentials, refreshed)
        except Exception as e:
            logger.warning("Refreshed Claude OAuth token but could not persist credentials file: %s", e)
    return refreshed


def _get_fresh_oauth_token(*, force_refresh: bool = False) -> str | None:
    """Return a valid Claude OAuth token, refreshing file-backed creds when needed."""
    try:
        raw_credentials, oauth = _load_oauth_bundle()
        if oauth:
            if force_refresh or _token_expired_or_stale(oauth.get("expiresAt")):
                oauth = _refresh_oauth_bundle(oauth, raw_credentials=raw_credentials)
                logger.info("Refreshed Claude OAuth token from credentials file")
            token = oauth.get("accessToken")
            if token:
                return token
    except Exception as e:
        logger.warning("Failed to load/refresh Claude OAuth token from credentials file: %s", e)
    # Fall back to env var
    return os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")


def _is_cli_crash(exc: Exception) -> bool:
    """Return True if the exception looks like a CLI subprocess crash (exit code 1)."""
    msg = str(exc).lower()
    return "exit code: 1" in msg or "exit code 1" in msg


def _is_auth_failure(exc: Exception, stderr_lines: list[str]) -> bool:
    haystack = "\n".join([str(exc), *stderr_lines]).lower()
    auth_markers = (
        "oauth token has expired",
        "failed to authenticate",
        "authentication_error",
        "401",
        "unauthorized",
        "invalid api key",
        "invalid token",
        "token has expired",
        "invalid_grant",
        "refresh token not found or invalid",
        "claude oauth refresh failed",
    )
    return any(marker in haystack for marker in auth_markers)
