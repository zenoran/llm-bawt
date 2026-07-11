"""Claude Code bridge: Redis command listener + Agent SDK event translator."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterable
from datetime import datetime, timezone
from pathlib import Path

import httpx
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, StreamEvent
from claude_agent_sdk.types import (
    AssistantMessage,
    HookMatcher,
    MirrorErrorMessage,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from agent_bridge.approval import (
    ApprovalDecision,
    PolicyAction,
    PolicyBundle,
    evaluate as evaluate_policies,
)
from agent_bridge.events import AgentEvent, AgentEventKind, synthesize_event_id
from agent_bridge.publisher import COMMANDS_STREAM, RedisPublisher
from agent_bridge.session_queue import SessionQueue

logger = logging.getLogger(__name__)

SESSION_PREFIX = "claude-code:"

# TASK-445: session-seed constants.
# agent_backend_config flag that gates history-summary seeding on new sessions.
SEED_SETTING_KEY = "seed_summary_on_new_session"

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


class ClaudeCodeBridge:
    """Reads chat.send commands from Redis, runs them through the Claude Agent
    SDK, and publishes AgentEvent-formatted results back to Redis."""

    # Default timeout for a single query() call (seconds).
    # The CLI's internal API_TIMEOUT_MS is 600s; we cut shorter to fail fast.
    DEFAULT_REQUEST_TIMEOUT = 300

    # TASK-269: synthetic tool_result fed back to the model when it calls
    # AskUserQuestion.  Returned via PermissionResultDeny.message (an ALLOW
    # would make the SDK actually run the built-in tool, which crashes headless
    # with "undefined is not an object" — there's no interactive widget here).
    # The model reads this as the tool's output, acknowledges, and ends its
    # turn; the user's real answer arrives later as a continuation turn.
    _DEFERRED_ACK = (
        "[The question has been delivered to the user, who will answer it in a "
        "separate later message. Do not wait and do not guess an answer. Briefly "
        "acknowledge that you've asked, then end your turn — you'll receive the "
        "user's answer as a new message and can act on it then.]"
    )

    def __init__(
        self,
        publisher: RedisPublisher,
        *,
        backend_name: str = "claude-code",
        app_api_url: str = "",
        cwd: str = "/app",
        permission_mode: str = "bypassPermissions",
        add_dirs: list[str] | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self._publisher = publisher
        self._backend_name = backend_name
        self._app_api_url = app_api_url
        self._cwd = cwd
        self._permission_mode = permission_mode
        self._add_dirs = add_dirs or []
        self._request_timeout = request_timeout or self.DEFAULT_REQUEST_TIMEOUT
        self._command_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._redis = None  # set in _command_listener
        # Shared session queue — serializes sends per session and tracks
        # active tasks for abort support.
        self._session_queue = SessionQueue()
        # request_id → frontend user-message UUID, populated in _handle_send
        # and read by _publish_event so every emitted AgentEvent (tool_*,
        # assistant_*, etc.) carries the originating message id.  Cleared on
        # _handle_send finally.
        self._trigger_message_ids: dict[str, str] = {}
        # TASK-269: AskUserQuestion no longer blocks the SDK turn.  can_use_tool
        # emits an AWAIT_TOOL_RESULT event and returns a synthetic "deferred"
        # ack immediately, so the turn ends cleanly and the user's answer comes
        # back as a separate continuation turn (SDK resume + answer message) —
        # no in-process Future to hold, no chat.tool_result round-trip.
        # Load MCP servers from settings file
        self._mcp_servers = self._load_mcp_servers()
        # TASK-270: Anthropic-compat proxy URL set by __main__ after the
        # ProxyServer binds. None when proxy is disabled. When set and the
        # request's model starts with a known provider prefix (see
        # ``proxy.adapters.REGISTRY``), the SDK subprocess is spawned with
        # ANTHROPIC_BASE_URL pointing here so its outbound /v1/messages
        # call lands on the proxy instead of api.anthropic.com.
        self._proxy_base_url: str | None = None
        # ---- Approval-gated tool policies (TASK-291 / TASK-292) ----
        # Compiled bundle fetched from the app, TTL-cached. Grants are one-shot
        # allows keyed by grant_key, populated by approval.grant commands when a
        # user approves a gated call, consumed when the model re-issues it.
        self._policy_bundle: PolicyBundle | None = None
        self._policy_bundle_fetched_at: float = 0.0
        self._policy_bundle_ttl: float = float(
            os.getenv("CLAUDE_CODE_APPROVAL_BUNDLE_TTL", "15")
        )
        # When the app is unreachable, fail OPEN by default (allow tools) so a
        # transient app blip never halts every agent. Set 1/true to fail CLOSED
        # (deny gated tools when policies can't be fetched). Non-gated tools are
        # unaffected either way — only tools a (cached) policy gates are denied.
        self._approval_fail_closed: bool = os.getenv(
            "CLAUDE_CODE_APPROVAL_FAIL_CLOSED", ""
        ).strip().lower() in ("1", "true", "yes")
        self._approval_grants: dict[str, float] = {}  # grant_key -> expiry (monotonic)
        self._approval_reload_task: asyncio.Task | None = None
        # Tracks whether the last bundle fetch reached the app. Only consulted
        # when fail-closed is on: a failing fetch then denies all tools.
        self._policy_fetch_ok: bool = True

    def set_proxy_base_url(self, url: str) -> None:
        """Wire up the in-process Anthropic-compat proxy. Called by __main__
        after ``ProxyServer.start()`` binds its ephemeral port."""
        self._proxy_base_url = url
        logger.info("Bridge proxy base_url set: %s", url)

    @staticmethod
    def _model_provider_prefix(model: str) -> str | None:
        """Return the provider prefix if ``model`` is a proxy-routed name
        (``"<provider>/<upstream>"`` where provider is registered), else None."""
        if not model or "/" not in model:
            return None
        provider, _, upstream = model.partition("/")
        if not provider or not upstream:
            return None
        # Local import keeps the proxy subpackage off the bridge's import
        # graph until needed (it pulls in openai, fastapi, etc.).
        from .proxy.adapters import REGISTRY

        return provider if provider in REGISTRY else None

    def _load_mcp_servers(self) -> dict:
        """Load MCP server configs from ~/.claude/settings.json."""
        settings_path = Path.home() / ".claude" / "settings.json"
        try:
            if settings_path.exists():
                data = json.loads(settings_path.read_text())
                servers = data.get("mcpServers", {})
                if servers:
                    logger.info("Loaded MCP servers from settings: %s", list(servers.keys()))
                    return servers
        except Exception as e:
            logger.warning("Failed to load MCP servers from %s: %s", settings_path, e)
        return {}

    # ----- Periodic cache cleanup -----

    _CLEANUP_INTERVAL = 6 * 3600  # every 6 hours
    _CACHE_DIRS = ("shell-snapshots", "file-history", "debug", "paste-cache")
    _CACHE_MAX_AGE = 24 * 3600  # delete files older than 24h

    async def _periodic_cache_cleanup(self) -> None:
        """Periodically prune ~/.claude/ cache dirs to prevent unbounded growth."""
        import shutil

        claude_dir = Path.home() / ".claude"
        while True:
            try:
                await asyncio.sleep(self._CLEANUP_INTERVAL)
                now = time.time()
                total_removed = 0

                for dirname in self._CACHE_DIRS:
                    cache_path = claude_dir / dirname
                    if not cache_path.is_dir():
                        continue
                    for entry in cache_path.iterdir():
                        try:
                            age = now - entry.stat().st_mtime
                            if age > self._CACHE_MAX_AGE:
                                if entry.is_dir():
                                    shutil.rmtree(entry)
                                else:
                                    entry.unlink()
                                total_removed += 1
                        except OSError:
                            pass

                # Prune old session JSONL files (can grow very large)
                sessions_dir = claude_dir / "projects"
                if sessions_dir.is_dir():
                    for jsonl in sessions_dir.rglob("*.jsonl"):
                        try:
                            age = now - jsonl.stat().st_mtime
                            if age > self._CACHE_MAX_AGE:
                                jsonl.unlink()
                                total_removed += 1
                        except OSError:
                            pass

                if total_removed:
                    logger.info("Cache cleanup: removed %d stale entries", total_removed)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Cache cleanup error")

    # ----- Session persistence via bot profile API -----

    async def _get_session(self, bot_id: str) -> tuple[str, str] | None:
        """Get (sdk_session_id, model) from bot's agent_backend_config."""
        if not self._app_api_url:
            return None
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._app_api_url}/v1/bots")
                resp.raise_for_status()
                for bot in resp.json().get("data", []):
                    if bot.get("slug") == bot_id:
                        bc = bot.get("agent_backend_config") or {}
                        sk = bc.get("session_key")
                        # session_model = bridge-owned record of which model
                        # the persisted SDK session was created with (drives
                        # resume-vs-reset). "model" is the pre-migration key.
                        model = bc.get("session_model") or bc.get("model", "")
                        if sk:
                            sk = str(sk).strip()
                            # Guard against legacy bug where routing keys
                            # like "snark:nick" were stored as SDK session ids.
                            if ":" in sk:
                                logger.warning("Ignoring invalid persisted session_key for %s: %s", bot_id, sk)
                                return None
                            return (sk, model)
                        return None
        except Exception as e:
            logger.warning("Failed to get session for %s: %s", bot_id, e)
        return None

    async def _set_session(self, bot_id: str, sdk_session_id: str, model: str) -> None:
        """Write SDK session_id back to bot's agent_backend_config via PATCH."""
        if not self._app_api_url:
            logger.warning("No API URL — session not persisted for %s", bot_id)
            return
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # Fetch current config to merge session_key in
                resp = await client.get(f"{self._app_api_url}/v1/bots")
                resp.raise_for_status()
                bc = {}
                for bot in resp.json().get("data", []):
                    if bot.get("slug") == bot_id:
                        bc = dict(bot.get("agent_backend_config") or {})
                        break
                bc["session_key"] = sdk_session_id
                # Bridge-owned session metadata. The user-facing model lives
                # on the bot's default_model (catalog alias); "model" is no
                # longer accepted in agent_backend_config by the profile API.
                bc.pop("model", None)
                bc["session_model"] = model

                await client.patch(
                    f"{self._app_api_url}/v1/bots/{bot_id}/profile",
                    json={"agent_backend_config": bc},
                )
            logger.info("Session persisted: %s -> %s", bot_id, sdk_session_id)
        except Exception as e:
            logger.warning("Failed to persist session for %s: %s", bot_id, e)

    async def _clear_session(self, bot_id: str) -> bool:
        """Remove session_key from bot's agent_backend_config via PATCH."""
        if not self._app_api_url:
            return False
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._app_api_url}/v1/bots")
                resp.raise_for_status()
                bc = {}
                had_session = False
                for bot in resp.json().get("data", []):
                    if bot.get("slug") == bot_id:
                        bc = dict(bot.get("agent_backend_config") or {})
                        had_session = "session_key" in bc
                        bc.pop("session_key", None)
                        break

                await client.patch(
                    f"{self._app_api_url}/v1/bots/{bot_id}/profile",
                    json={"agent_backend_config": bc},
                )
                logger.info("Session cleared: %s (had_session=%s)", bot_id, had_session)
                return had_session
        except Exception as e:
            logger.warning("Failed to clear session for %s: %s", bot_id, e)
            return False

    # ----- TASK-445: seed a fresh SDK session with chat summary history -----

    async def _get_mcp_tool_context(self, bot_slug: str) -> str:
        """Return the MCP tool context block for a bot (TASK-490).

        Fetches the (bot-overridable) template body from the app registry via
        GET /v1/prompts/{key}, cached per bot for the process, and falls back to
        a byte-identical local copy if the app is unreachable — so behavior is
        preserved regardless. Returns the block WITHOUT the leading separator;
        the caller prepends ``\\n\\n``.
        """
        cache = getattr(self, "_mcp_ctx_cache", None)
        if cache is None:
            cache = {}
            self._mcp_ctx_cache = cache
        if bot_slug in cache:
            return cache[bot_slug]

        body = _MCP_TOOL_CONTEXT_FALLBACK
        if self._app_api_url:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(
                        f"{self._app_api_url}/v1/prompts/{MCP_TOOL_CONTEXT_KEY}",
                        params={"scope_type": "bot", "scope_id": bot_slug},
                    )
                    resp.raise_for_status()
                    fetched = (resp.json() or {}).get("body")
                    if fetched:
                        body = fetched
            except Exception as e:
                logger.warning(
                    "MCP tool context fetch failed for %s (%s); using local fallback",
                    bot_slug, e,
                )
        # Bridge-side substitution (robust against stray braces in overrides).
        rendered = body.replace("{bot_slug}", bot_slug)
        cache[bot_slug] = rendered
        return rendered

    async def _get_seed_setting(self, bot_id: str) -> bool:
        """Read the per-bot ``seed_summary_on_new_session`` flag from
        agent_backend_config. Defaults to False (off) when absent or on error."""
        if not self._app_api_url or not bot_id:
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._app_api_url}/v1/bots")
                resp.raise_for_status()
                for bot in resp.json().get("data", []):
                    if bot.get("slug") == bot_id:
                        bc = bot.get("agent_backend_config") or {}
                        return bool(bc.get(SEED_SETTING_KEY, False))
        except Exception as e:
            logger.warning("Failed to read seed setting for %s: %s", bot_id, e)
        return False

    async def _fetch_context_seed(self, bot_id: str, model: str) -> dict | None:
        """Fetch the chatbot-style context payload (summaries + budgeted recent
        turns) from the app's /v1/history/context-seed endpoint."""
        if not self._app_api_url or not bot_id:
            return None
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{self._app_api_url}/v1/history/context-seed",
                    params={"bot_id": bot_id, "model": model},
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning("Failed to fetch context seed for %s: %s", bot_id, e)
            return None

    def _project_slug(self, cwd: str) -> str:
        """Reproduce the SDK's project-dir sanitization (non-alnum -> '-')."""
        return _SEED_SANITIZE_RE.sub("-", cwd or "")

    def _render_seed_briefing(self, messages: list[dict]) -> str:
        """Flatten the context messages into a single labeled briefing block.

        Summaries keep their ``[Previous conversation X ago]`` headers; recent
        turns are rendered as a readable transcript. Text-only by construction —
        no tool_use blocks — so the resumed session never wedges."""
        lines = [
            "[Session continuity seed — prior context restored from chat history]",
            "Below is our earlier conversation for continuity: rolling summaries "
            "of older sessions, then the most recent messages. Treat it as "
            "background context; you don't need to repeat it back.",
            "",
        ]
        for m in messages:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "summary":
                lines.append(content)
            elif role == "user":
                lines.append(f"Nick: {content}")
            elif role == "assistant":
                lines.append(f"Assistant (you): {content}")
            else:
                lines.append(content)
            lines.append("")
        return "\n".join(lines).strip()

    def _write_seed_transcript(self, session_id: str, messages: list[dict]) -> Path:
        """Write a synthetic two-entry Claude Code transcript (user briefing +
        assistant ack) so the SDK can ``resume`` it into a fresh session.

        Entry shape verified against real on-disk transcripts. The assistant
        entry uses ``model: "<synthetic>"`` — Claude Code's own marker for a
        fabricated turn (the SDK writes these for injected errors and resumes
        past them cleanly). Ends on the assistant so the leaf is clean."""
        slug = self._project_slug(self._cwd)
        proj_dir = Path.home() / ".claude" / "projects" / slug
        proj_dir.mkdir(parents=True, exist_ok=True)
        path = proj_dir / f"{session_id}.jsonl"

        briefing = self._render_seed_briefing(messages)
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        user_uuid = str(uuid.uuid4())
        asst_uuid = str(uuid.uuid4())

        common = {
            "isSidechain": False,
            "sessionId": session_id,
            "cwd": self._cwd,
            "version": _SEED_CLI_VERSION,
            "gitBranch": "",
            "userType": "external",
        }
        user_entry = {
            "parentUuid": None,
            "type": "user",
            "message": {"role": "user", "content": briefing},
            "uuid": user_uuid,
            "timestamp": ts,
            **common,
        }
        asst_entry = {
            "parentUuid": user_uuid,
            "type": "assistant",
            "uuid": asst_uuid,
            "timestamp": ts,
            "message": {
                "role": "assistant",
                "model": "<synthetic>",
                "type": "message",
                "stop_reason": "stop_sequence",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Context restored — I've reviewed our prior "
                            "summaries and recent messages and I'm caught up."
                        ),
                    }
                ],
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
            **common,
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(user_entry) + "\n")
            f.write(json.dumps(asst_entry) + "\n")
        logger.info(
            "Seed transcript written: %s (%d context msgs)",
            path, len(messages),
        )
        return path

    async def _seed_new_session(self, bot_id: str, model: str) -> dict | None:
        """Seed a brand-new SDK session for ``bot_id`` from chat summary history,
        if the per-bot setting is on. Writes the synthetic transcript, persists
        the minted session id, and returns a stats dict for the /new ack.

        Returns:
            None      -> seeding disabled for this bot (caller behaves as before)
            {seeded: False, reason} -> enabled but nothing to seed / error
            {seeded: True, session_id, summary_count, message_count,
             approx_tokens, oldest_timestamp, newest_timestamp}
        """
        if not await self._get_seed_setting(bot_id):
            return None
        seed = await self._fetch_context_seed(bot_id, model)
        if not seed:
            return {"seeded": False, "reason": "seed fetch failed"}
        messages = seed.get("messages") or []
        if not messages:
            return {"seeded": False, "reason": "no history to seed"}
        try:
            session_id = str(uuid.uuid4())
            self._write_seed_transcript(session_id, messages)
            await self._set_session(bot_id, session_id, model)
        except Exception as e:
            logger.warning("Seed write/persist failed for %s: %s", bot_id, e)
            return {"seeded": False, "reason": f"seed write failed: {e}"}
        stats = dict(seed.get("stats") or {})
        stats["seeded"] = True
        stats["session_id"] = session_id
        return stats

    @staticmethod
    def _format_seed_ack(stats: dict | None) -> str:
        """Human-facing /new acknowledgement, reporting seed stats when present."""
        base = "Session reset."
        if stats is None:
            return f"{base} Ready for a new conversation."
        if not stats.get("seeded"):
            reason = stats.get("reason", "nothing to seed")
            return f"{base} History seeding on, but {reason}. Fresh start."
        summ = stats.get("summary_count", 0)
        msgs = stats.get("message_count", 0)
        toks = stats.get("approx_tokens", 0)
        span = ""
        oldest = stats.get("oldest_timestamp")
        newest = stats.get("newest_timestamp")
        if oldest and newest:
            try:
                o = datetime.fromtimestamp(oldest, timezone.utc).strftime("%Y-%m-%d")
                n = datetime.fromtimestamp(newest, timezone.utc).strftime("%Y-%m-%d")
                span = f", spanning {o} → {n}" if o != n else f", from {o}"
            except Exception:
                span = ""
        return (
            f"{base} Seeded {summ} summary record{'s' if summ != 1 else ''} + "
            f"{msgs} recent message{'s' if msgs != 1 else ''} "
            f"(~{_fmt_tokens(toks)} tokens){span}. Continuity restored."
        )

    async def start(self) -> None:
        self._command_task = asyncio.create_task(self._command_listener())
        self._cleanup_task = asyncio.create_task(self._periodic_cache_cleanup())
        logger.info(
            "ClaudeCodeBridge started (backend=%s)",
            self._backend_name,
        )

    async def stop(self) -> None:
        for task in (self._command_task, self._cleanup_task, self._approval_reload_task):
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._publisher.close()
        logger.info("ClaudeCodeBridge stopped")

    async def run_forever(self) -> None:
        await self.start()
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    # ----- Redis command listener -----

    async def _command_listener(self) -> None:
        import redis.asyncio as aioredis

        conn_kwargs = self._publisher._redis.connection_pool.connection_kwargs
        host = conn_kwargs.get("host", "localhost")
        port = conn_kwargs.get("port", 6379)
        db = conn_kwargs.get("db", 0)
        # socket_timeout=None: redis-py 8.0 defaults to 5s, which races our
        # blocking XREADGROUP(block=5000) reads. Bound only the connect.
        async_redis = aioredis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_timeout=None,
            socket_connect_timeout=5,
        )
        await async_redis.ping()
        self._redis = async_redis

        try:
            await async_redis.xgroup_create(
                COMMANDS_STREAM, "claude-code-bridge", id="0", mkstream=True,
            )
        except Exception:
            pass

        logger.info("Command listener started on %s (group=claude-code-bridge)", COMMANDS_STREAM)

        # TASK-291: subscribe to approval-policy reload broadcasts so admin
        # edits drop the cached bundle without a restart. Best-effort.
        if self._approval_reload_task is None or self._approval_reload_task.done():
            reload_url = f"redis://{host}:{port}/{db}"
            self._approval_reload_task = asyncio.create_task(
                self._approval_reload_listener(reload_url)
            )

        while True:
            try:
                results = await async_redis.xreadgroup(
                    "claude-code-bridge",
                    "worker-0",
                    {COMMANDS_STREAM: ">"},
                    count=1,
                    block=5000,
                )
                if not results:
                    continue

                for _stream, messages in results:
                    for msg_id, fields in messages:
                        action = fields.get("action", "")
                        backend = fields.get("backend", "")

                        if action == "chat.send" and backend != self._backend_name:
                            # Not our chat command — ACK so it doesn't pile up
                            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
                            continue

                        if action == "chat.send":
                            asyncio.create_task(
                                self._handle_send(fields, msg_id, async_redis)
                            )
                        elif action == "rpc.call":
                            asyncio.create_task(
                                self._handle_rpc(fields, msg_id, async_redis)
                            )
                        elif action == "chat.tool_result":
                            # Resolves a pending AskUserQuestion Future so the
                            # paused SDK turn can continue.  Deliberately NOT
                            # wrapped in the session lock — _handle_send is
                            # holding that lock while awaiting the Future, so
                            # taking it here would deadlock the run forever.
                            asyncio.create_task(
                                self._handle_tool_result(fields, msg_id, async_redis)
                            )
                        elif action == "approval.grant":
                            # TASK-292: user approved a gated tool. Store a
                            # one-shot allow so the re-issued call on the
                            # continuation turn sails through.
                            if backend and backend != self._backend_name:
                                await async_redis.xack(
                                    COMMANDS_STREAM, "claude-code-bridge", msg_id
                                )
                                continue
                            asyncio.create_task(
                                self._handle_approval_grant(fields, msg_id, async_redis)
                            )
                        else:
                            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)

            except asyncio.CancelledError:
                await async_redis.aclose()
                raise
            except Exception:
                logger.exception("Command listener error")
                await asyncio.sleep(2)

    # ----- Handle a single chat.send command -----

    async def _handle_send(
        self, fields: dict, msg_id: str, async_redis,
    ) -> None:
        request_id = fields.get("request_id", "")
        session_key = fields.get("session_key", "")
        bot_slug = (fields.get("bot_id", "") or "").strip() or _bot_slug_from_session_key(session_key)
        message = fields.get("message", "")
        system_prompt = fields.get("system_prompt") or None
        model = (fields.get("model") or "").strip()
        # Frontend-supplied user-message UUID; stamped on every emitted event
        # so the frontend can bucket tool activity under the originating user
        # message without falling back to turn_id heuristics.
        trigger_message_id = (fields.get("trigger_message_id") or "").strip() or None

        # Per-bot ClaudeAgentOptions tuning (TASK: bot config -> SDK).
        # ``effort`` constrains thinking depth; ``max_turns`` caps the
        # autonomous tool-loop length per dispatch. Both default to None
        # (SDK default) when the bot doesn't override.
        _allowed_effort = {"low", "medium", "high", "xhigh", "max"}
        effort_raw = (fields.get("effort") or "").strip().lower() or None
        bot_effort = effort_raw if effort_raw in _allowed_effort else None
        if effort_raw and bot_effort is None:
            logger.warning(
                "Ignoring invalid effort=%r for %s (allowed: %s)",
                effort_raw, bot_slug, sorted(_allowed_effort),
            )
        max_turns_raw = (fields.get("max_turns") or "").strip()
        bot_max_turns: int | None = None
        if max_turns_raw:
            try:
                mt = int(max_turns_raw)
                bot_max_turns = mt if mt > 0 else None
            except ValueError:
                logger.warning(
                    "Ignoring invalid max_turns=%r for %s (must be positive int)",
                    max_turns_raw, bot_slug,
                )

        attachments_raw = fields.get("attachments", "")
        attachments: list[dict] = []
        if attachments_raw:
            try:
                attachments = json.loads(attachments_raw)
            except json.JSONDecodeError:
                pass

        if not request_id or not message:
            logger.warning("Invalid send command: missing request_id or message")
            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
            return

        if not model:
            # No silent fallback. The caller MUST pass an explicit model. Surface
            # the failure both to the log and to the originating chat so the user
            # immediately sees which bot's config is missing a model.
            err = (
                f"Claude Code bridge: missing 'model' field for bot={bot_slug or '?'} "
                f"session={session_key}. Set the bot's Model (default_model) to a "
                f"claude-code catalog entry on the bot's profile."
            )
            logger.error(err)
            self._publish_event(
                request_id, session_key, 1,
                kind=AgentEventKind.ERROR,
                text=err,
            )
            self._publisher.publish_run_done(request_id)
            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
            return

        if trigger_message_id:
            self._trigger_message_ids[request_id] = trigger_message_id

        # /new resets the session — strip it and start fresh
        if message.lstrip().startswith("/new"):
            cleared = await self._clear_session(bot_slug or session_key)
            logger.info("Session reset via /new: %s (had_session=%s)", bot_slug or session_key, cleared)
            # Publish a deterministic SESSION_RESET unified event so the
            # frontend can clear its visible buffer without racing
            # turn_complete timing.  See TASK-249.
            self._publish_session_reset_unified(
                bot_slug or session_key, session_key, had_session=cleared,
            )
            # TASK-445: optionally seed the fresh session from chat summary
            # history. Persists the minted session id so a trailing message
            # below resumes the seeded transcript (no double-seed).
            seed_stats = await self._seed_new_session(bot_slug, model)
            message = message.lstrip().removeprefix("/new").strip()
            if not message:
                # Just "/new" with no follow-up — acknowledge and done
                self._publish_event(
                    request_id, session_key, 1,
                    kind=AgentEventKind.ASSISTANT_DONE,
                    text=self._format_seed_ack(seed_stats),
                    model=model,
                )
                self._publisher.publish_run_done(request_id)
                await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
                return

        if self._session_queue.is_busy(session_key):
            logger.info(
                "Session %s busy — queuing send request_id=%s",
                session_key, request_id,
            )

        async with self._session_queue.active(session_key):
            logger.info(
                "Handling send: request_id=%s session=%s model=%s system_prompt=%s msg=%.60s...",
                request_id, session_key, model,
                f"{len(system_prompt)} chars" if system_prompt else "none",
                message,
            )

            seq = 0
            text_parts: list[str] = []
            current_tool_name: str | None = None
            current_tool_input: str = ""
            actual_model: str = model  # updated from SystemMessage if available
            # Map tool_use_id -> tool name (the SDK's ToolResultBlock doesn't echo
            # the name, only the id) so we can recognise a Playwright screenshot
            # result and persist its image instead of letting the inline base64
            # ride in the model context forever.
            tool_names_by_id: dict[str, str] = {}
            # {asset_id, kind} refs for screenshots persisted to the media store
            # during this turn; stamped onto the terminal ASSISTANT_DONE event so
            # the app can attach them to the bot's reply message.
            turn_screenshot_assets: list[dict] = []

            try:
                # Inject MCP tool context so Claude passes the right identifiers.
                # Body comes from the registry (TASK-490) with a byte-identical
                # local fallback; separator added here.
                if system_prompt and self._mcp_servers and bot_slug:
                    mcp_ctx = await self._get_mcp_tool_context(bot_slug)
                    system_prompt += f"\n\n{mcp_ctx}"

                # Reuse SDK session for conversation continuity.
                # If the model changed, start a fresh session.
                existing = await self._get_session(bot_slug)
                resume_id = None
                if existing:
                    prev_sid, prev_model = existing
                    if prev_model == model:
                        resume_id = prev_sid
                    else:
                        logger.info(
                            "Model changed (%s -> %s), starting new session for %s",
                            prev_model, model, bot_slug or session_key,
                        )
                        await self._clear_session(bot_slug or session_key)

                # TASK-445: cold start with no session to resume — first-ever
                # run or post-model-switch. Seed from summary history so the new
                # SDK session opens with continuity. A /new above that already
                # seeded will have persisted a session, so _get_session found it
                # and resume_id is set — this block is skipped (no double-seed).
                if resume_id is None:
                    cold_seed = await self._seed_new_session(bot_slug, model)
                    if cold_seed and cold_seed.get("seeded"):
                        resume_id = cold_seed["session_id"]
                        logger.info(
                            "Cold-start seeded session for %s: %s (%s summaries, %s msgs)",
                            bot_slug, resume_id,
                            cold_seed.get("summary_count"), cold_seed.get("message_count"),
                        )

                # Resolve settings file path
                settings_path = str(Path.home() / ".claude" / "settings.json")
                if not Path(settings_path).exists():
                    settings_path = None

                # Build prompt — multimodal if attachments present
                if attachments:
                    content: list[dict] = [{"type": "text", "text": message}]
                    for att in attachments:
                        mime = att.get("mimeType", "image/png")
                        data = att.get("content", "")
                        if data:
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime,
                                    "data": data,
                                },
                            })
                    logger.info("Multimodal prompt: %d text + %d images", 1, len(attachments))
                    user_content: str | list[dict] = content
                else:
                    user_content = message

                # The prompt MUST be an AsyncIterable: can_use_tool only works in
                # the SDK's streaming-input mode (a plain str raises "can_use_tool
                # callback requires streaming mode").
                #
                # Critically, the generator must stay OPEN for the whole turn.
                # Returning right after the single yield closes the subprocess
                # input stream, which tears down the bidirectional control channel
                # that can_use_tool (AskUserQuestion) rides on — the pending
                # permission request then dies with "Tool permission request
                # failed: Error: Stream closed", and the half-finished turn leaves
                # a dangling tool_use that makes the resumed session un-replayable
                # (it wedges every subsequent message on that session).
                #
                # So: yield the user message, then block on `done_event` until the
                # response loop signals the turn is complete (ResultMessage), and
                # only then return — letting the SDK close input and end the
                # output stream via StopAsyncIteration cleanly.
                def _make_prompt_input(done_event: asyncio.Event) -> AsyncIterable:
                    async def _prompt():
                        yield {
                            "type": "user",
                            "message": {"role": "user", "content": user_content},
                            "parent_tool_use_id": None,
                            "session_id": "default",
                        }
                        await done_event.wait()

                    return _prompt()

                auth_retry_attempted = False
                fresh_session_retry = False
                # Pull (or create) the cooperative cancel event for this session so a
                # chat.abort that arrives mid-loop is observed at the next message
                # boundary — without waiting for `task.cancel()` to fire CancelledError
                # at the next `await` (which can be tens of seconds inside a tool call).
                cancel_event = (
                    self._session_queue.cancel_event(session_key) if session_key else None
                )
                if cancel_event is not None and cancel_event.is_set():
                    # Stale signal from a previous run — clear so this turn can proceed.
                    cancel_event.clear()
                while True:
                    # An async generator is single-use and the auth/session retry
                    # paths below re-enter this loop, so build a fresh prompt +
                    # completion gate per attempt.  turn_done releases the prompt
                    # generator (closing SDK input) only once the turn finishes —
                    # see _make_prompt_input for why it must stay open until then.
                    turn_done = asyncio.Event()
                    prompt_input = _make_prompt_input(turn_done)
                    stderr_lines: list[str] = []
                    # Per-attempt: the auth/session retry paths re-run the turn
                    # from scratch, so reset screenshot tracking to avoid double-
                    # counting an earlier attempt's uploads on the final DONE event.
                    tool_names_by_id.clear()
                    turn_screenshot_assets.clear()

                    def _log_stderr(line: str) -> None:
                        line = line.rstrip()
                        stderr_lines.append(line)
                        logger.warning("CLI stderr: %s", line)

                    # TASK-270: route this turn to the in-process Anthropic-compat
                    # proxy when the model name carries a known provider prefix
                    # (e.g. "openai_chatgpt/gpt-5.4"). The proxy reads ChatGPT
                    # OAuth from ~/.codex/auth.json and forwards to OpenAI's
                    # Responses API. Otherwise fall through to Anthropic-direct.
                    use_proxy = (
                        self._proxy_base_url is not None
                        and self._model_provider_prefix(model) is not None
                    )

                    # The CLI's WebSearch/WebFetch are Anthropic *server-side*
                    # tools — they only execute against api.anthropic.com. On the
                    # proxy path (grok/openai) they collapse into parameter-less
                    # function stubs the upstream can't run, so the tool_use hangs
                    # with no result (verified: TOOLMAP start, no end). Disable
                    # them for proxied turns; those bots get local web coverage
                    # via the bawthub `web_search` MCP tool + crawl4ai fetch
                    # instead. Anthropic-direct bots keep native server search.
                    proxy_disallowed_tools = (
                        ["WebSearch", "WebFetch"] if use_proxy else []
                    )

                    sdk_env = {}
                    # Force Task/Agent subagents to run SYNCHRONOUSLY. CLI 2.1.x
                    # backgrounds agents by default: the tool returns immediately,
                    # the model ends its turn ("agents launched, standing by"),
                    # and the CLI re-invokes the model via task-notification when
                    # they finish — emitting a SECOND ResultMessage. This bridge
                    # is one-send-one-turn: it finalizes on the FIRST ResultMessage
                    # and disconnect()s the client, which kills the subprocess and
                    # orphans every still-running subagent, so their results are
                    # lost and the user sees a dead-end turn. With this flag the
                    # CLI awaits agent completion in-turn (verified live on
                    # 2.1.191: flag on → tool_result carries the agent's answer,
                    # single ResultMessage; flag off → "running in background"
                    # stub + task-notification + second ResultMessage).
                    sdk_env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] = "1"
                    if use_proxy:
                        sdk_env["ANTHROPIC_BASE_URL"] = self._proxy_base_url  # type: ignore[assignment]
                        # The SDK still requires *some* auth token to send; the
                        # proxy ignores it. Use a sentinel that obviously isn't
                        # a real Anthropic key so logs don't confuse anyone.
                        sdk_env["ANTHROPIC_AUTH_TOKEN"] = "proxy-routed"
                        # CLAUDE_CODE_OAUTH_TOKEN takes precedence over
                        # ANTHROPIC_AUTH_TOKEN inside the CLI; clear it so
                        # the SDK doesn't fall back to api.anthropic.com.
                        sdk_env["CLAUDE_CODE_OAUTH_TOKEN"] = ""
                        logger.debug(
                            "Routing turn through proxy: model=%s base=%s",
                            model, self._proxy_base_url,
                        )
                    else:
                        # Read fresh token on each request (auto-refresh from credentials file)
                        fresh_token = _get_fresh_oauth_token(force_refresh=auth_retry_attempted)
                        if fresh_token:
                            sdk_env["CLAUDE_CODE_OAUTH_TOKEN"] = fresh_token

                    # Pass `seq` to the can_use_tool factory by reference so it
                    # can keep the AWAIT_TOOL_RESULT event ordered in the same
                    # sequence as the surrounding ASSISTANT_DELTA / TOOL_START
                    # events.  Tuple-wrapped in a single-element list so the
                    # closure can mutate it without rebinding.
                    seq_holder = [seq]
                    can_use_tool_cb = self._make_can_use_tool(
                        request_id=request_id,
                        session_key=session_key,
                        seq_holder=seq_holder,
                    )
                    # TASK-292: the approval gate lives in a PreToolUse hook, NOT
                    # can_use_tool. Under permission_mode="bypassPermissions" (our
                    # standing config) the SDK auto-approves regular tools and
                    # never calls can_use_tool for them, so a can_use_tool-based
                    # gate is dead code. PreToolUse hooks are a separate control
                    # plane that fires regardless of permission_mode (verified
                    # live: hook fires + "deny" blocks under bypass). Shares
                    # seq_holder with can_use_tool — turns are sequential, so no
                    # concurrent mutation.
                    pre_tool_use_cb = self._make_pre_tool_use_hook(
                        request_id=request_id,
                        session_key=session_key,
                        seq_holder=seq_holder,
                    )

                    # TASK-288 observability: log the system_prompt value AS SENT
                    # to the SDK, paired with resume state. This is the only place
                    # the resume-gate decision is visible — the earlier "Handling
                    # send" log prints the pre-gate request value and cannot prove
                    # whether persona actually reached the agent on a resumed turn.
                    logger.info(
                        "SDK call: resume=%s system_prompt_sent=%s",
                        bool(resume_id),
                        f"{len(system_prompt)} chars" if system_prompt else "none",
                    )

                    options = ClaudeAgentOptions(
                        model=model,
                        # TASK-288: send the system prompt on EVERY turn, resume
                        # included. The SDK rebuilds and re-sends systemPrompt on
                        # every query() (it is NOT locked at session start), and
                        # the prompt is now byte-stable (temporal + response-style
                        # moved off it), so re-sending reads the full prefix from
                        # cache (~10% cost, POC-confirmed) while keeping persona
                        # alive on resume instead of decaying to the stock default.
                        system_prompt=system_prompt,
                        cwd=self._cwd,
                        disallowed_tools=proxy_disallowed_tools,
                        permission_mode=self._permission_mode,
                        include_partial_messages=True,
                        resume=resume_id,
                        add_dirs=self._add_dirs if self._add_dirs else [],
                        stderr=_log_stderr,
                        env=sdk_env,
                        settings=settings_path,
                        effort=bot_effort,
                        # Opt into summarized reasoning text. On Opus 4.7+ the
                        # thinking display defaults to "omitted" — the model
                        # thinks but Anthropic streams empty thinking blocks
                        # (signature only), so the UI reasoning lane (TASK-301)
                        # had nothing to render on the Anthropic-direct path.
                        # "summarized" returns a readable summary as
                        # thinking_delta text (raw CoT is never exposed on
                        # Opus). Proxy-routed models synthesize their own
                        # thinking and are unaffected by this flag.
                        thinking={"type": "adaptive", "display": "summarized"},
                        max_turns=bot_max_turns,
                        mcp_servers=self._mcp_servers if self._mcp_servers else {},
                        can_use_tool=can_use_tool_cb,
                        # TASK-292: matcher=None → fires for every tool. The hook
                        # (not can_use_tool) is the sole approval gate.
                        hooks={
                            "PreToolUse": [
                                HookMatcher(matcher=None, hooks=[pre_tool_use_cb]),
                            ],
                        },
                        # The SDK's stdio reader defaults to a 1 MiB JSON buffer
                        # (claude_agent_sdk subprocess_cli _DEFAULT_MAX_BUFFER_SIZE).
                        # A single Playwright screenshot or large browser_snapshot
                        # tool-result blows past that and the reader raises
                        # SDKJSONDecodeError mid-stream, which the bridge surfaces
                        # as an ERROR event and the app turns into a hard
                        # RuntimeError — i.e. the bot's turn silently dies. Raise
                        # the ceiling so those results flow; screenshots are then
                        # offloaded to the media store below.
                        max_buffer_size=32 * 1024 * 1024,
                    )

                    session_persisted = False
                    aborted = False
                    # Track the latest AssistantMessage.usage to surface "current
                    # context fullness" to the UI. ResultMessage.usage is
                    # cumulative across all internal API iterations in a turn,
                    # so cache_read_input_tokens can exceed contextWindow on
                    # multi-tool-use turns and produce >100% counters. The last
                    # AssistantMessage's usage reflects the actual final API
                    # call's view of the context.
                    latest_assistant_usage: dict | None = None
                    # Some synthetic Anthropic streams (notably the
                    # Responses-backed ChatGPT proxy) surface final input/cache
                    # usage only on StreamEvent(message_delta), while the
                    # AssistantMessage snapshot can remain at message_start's
                    # zeroed input fields. Track the last stream-level usage as
                    # a fallback so turn_logs get the real pill numbers.
                    latest_stream_usage: dict | None = None
                    # /compact lifecycle tracking for this turn. The SDK reports
                    # compaction via SystemMessage(subtype="status") — never a
                    # compact_boundary on the wire — so we watch the status
                    # payload to (a) give immediate feedback (a /compact can be
                    # ~50s of otherwise-silent work the UI reads as "hung") and
                    # (b) report the new resident size, since the /compact
                    # ResultMessage.usage is all-zeros.
                    compact_announced = False
                    compact_status: str | None = None  # None | "success" | "failed"
                    compact_error_msg: str | None = None
                    turn_session_id: str | None = resume_id
                    # Some providers / SDK paths terminate after an
                    # AssistantMessage snapshot with text/tool_use content and
                    # NEVER emit a trailing ResultMessage. If we only finalize
                    # on ResultMessage, the bridge logs "Send completed" but
                    # the app receives no ASSISTANT_DONE and the turn is saved
                    # as an empty timeout. Capture the latest assistant text
                    # snapshot so we can publish a fallback DONE on clean EOF.
                    assistant_snapshot_text: str = ""
                    assistant_done_emitted = False
                    # Track upstream API retries so we can (a) show live
                    # status in the UI ("z.ai overloaded, retrying…") and
                    # (b) include the error in the final DONE when all
                    # retries are exhausted and the turn ends empty.
                    api_retry_count = 0
                    api_last_error: str | None = None
                    api_retry_surfaced = False  # True once we've pushed a delta

                    sdk_client = None
                    msg_stream = None
                    try:
                        sdk_client = ClaudeSDKClient(options=options)
                        await sdk_client.connect(prompt_input)
                        msg_stream = sdk_client.receive_messages()
                        # Register the live client so `chat.abort` can call
                        # `disconnect()` on it — that closes the SDK Query and
                        # drives the subprocess transport through EOF/SIGTERM/
                        # SIGKILL teardown even mid-tool-call. `task.cancel()`
                        # alone is insufficient because CancelledError only fires
                        # at the next `await`, and the SDK is awaiting on subprocess
                        # output that doesn't arrive until the running tool exits.
                        if session_key:
                            self._session_queue.set_active_client(session_key, sdk_client)
                        while True:
                            # Cooperative abort check — runs before every SDK
                            # `__anext__`, so an abort signalled by chat.abort is
                            # observed even if the previous `await` was already
                            # past the cancel injection point.
                            if cancel_event is not None and cancel_event.is_set():
                                logger.info(
                                    "chat.abort signalled, halting SDK iteration: session=%s request=%s",
                                    session_key, request_id,
                                )
                                aborted = True
                                break
                            try:
                                msg = await asyncio.wait_for(
                                    msg_stream.__anext__(),
                                    timeout=self._request_timeout,
                                )
                            except StopAsyncIteration:
                                    break
                            except TimeoutError:
                                raise TimeoutError(
                                    f"No SDK messages for {self._request_timeout}s — CLI may be hung"
                                )

                            # Session-mirror write failure. MirrorErrorMessage is a
                            # SystemMessage subclass the SDK emits when its
                            # SessionStore.append fails — i.e. a turn's frame did not
                            # get persisted to the on-disk transcript. Left unhandled
                            # it slips into the generic SystemMessage branch below,
                            # matches none of its data conditions, and vanishes — so a
                            # persistence failure that can later wedge resume/replay
                            # goes completely unsignalled. Surface it as a structured
                            # warning (operational, not user-facing — no chat bubble).
                            if isinstance(msg, MirrorErrorMessage):
                                logger.warning(
                                    "SDK session-mirror append failed: key=%s error=%s session=%s",
                                    getattr(msg, "key", None),
                                    getattr(msg, "error", None),
                                    session_key,
                                )
                            if isinstance(msg, SystemMessage):
                                data = getattr(msg, "data", {}) or {}
                                # Capture session_id + actual model from the first
                                # SystemMessage (the init), then persist once.
                                if not session_persisted:
                                    if data.get("model"):
                                        actual_model = data["model"]
                                        logger.info("Actual model: %s", actual_model)
                                    if not resume_id:
                                        sid = data.get("session_id")
                                        if sid:
                                            await self._set_session(bot_slug or session_key, sid, model)
                                    session_persisted = True
                                # Track the session_id for this turn regardless of
                                # resume state — used to read the compaction result
                                # back from the transcript below.
                                if data.get("session_id"):
                                    turn_session_id = data["session_id"]
                                # Compaction lifecycle. A /compact turn emits
                                # SystemMessage(subtype="status"): first
                                # status="compacting", then a payload carrying
                                # compact_result ("success"/"failed") and, on
                                # failure, compact_error. There is NO
                                # compact_boundary on the wire. Surface the start
                                # immediately so the turn doesn't read as "hung",
                                # and record the outcome for the ResultMessage.
                                if data.get("status") == "compacting" and not compact_announced:
                                    compact_announced = True
                                    seq += 1
                                    note = "🗜️ Compacting conversation to free up context…"
                                    text_parts.append(note)
                                    self._publish_event(
                                        request_id, session_key, seq,
                                        kind=AgentEventKind.ASSISTANT_DELTA,
                                        text=note,
                                    )
                                cr = data.get("compact_result")
                                if cr == "success":
                                    compact_status = "success"
                                elif cr == "failed":
                                    compact_status = "failed"
                                    compact_error_msg = (
                                        data.get("compact_error") or "unknown error"
                                    )
                                # API retry lifecycle. The SDK CLI retries on
                                # upstream errors (429, 500, 529, etc.). Surface
                                # the retry to the UI as a live status delta so
                                # the user sees feedback instead of a dead bubble.
                                if data.get("subtype") == "api_retry":
                                    attempt = data.get("attempt", 0)
                                    max_retries = data.get("max_retries", 10)
                                    err_status = data.get("error_status", "?")
                                    err_text = data.get("error", "unknown")
                                    api_retry_count = attempt
                                    api_last_error = f"HTTP {err_status}: {err_text}"
                                    logger.warning(
                                        "API retry %d/%d: status=%s error=%s session=%s",
                                        attempt, max_retries, err_status, err_text, session_key,
                                    )
                                    # Push a live status on first retry so the
                                    # user immediately sees something.
                                    if not api_retry_surfaced:
                                        api_retry_surfaced = True
                                        seq += 1
                                        note = f"⏳ Upstream unavailable ({err_text}), retrying…"
                                        text_parts.append(note)
                                        self._publish_event(
                                            request_id, session_key, seq,
                                            kind=AgentEventKind.ASSISTANT_DELTA,
                                            text=note,
                                        )
                                # ── Sub-agent task lifecycle (TASK-344) ──
                                # The SDK emits TaskStarted/Progress/Updated/
                                # Notification messages (SystemMessage subclasses)
                                # when the Agent or Workflow tool spawns sub-agents.
                                # Detect them and publish structured events so the
                                # app→frontend pipeline can show live progress.
                                if isinstance(msg, TaskStartedMessage):
                                    seq += 1
                                    logger.info(
                                        "Sub-agent started: task_id=%s desc=%s tool_use_id=%s",
                                        msg.task_id, msg.description, msg.tool_use_id,
                                    )
                                    self._publish_event(
                                        request_id, session_key, seq,
                                        kind=AgentEventKind.SUBAGENT_STARTED,
                                        tool_use_id=msg.tool_use_id,
                                        extra_raw={
                                            "task_id": msg.task_id,
                                            "description": msg.description or "",
                                            "task_type": getattr(msg, "task_type", None),
                                            "uuid": getattr(msg, "uuid", ""),
                                        },
                                    )
                                elif isinstance(msg, TaskProgressMessage):
                                    seq += 1
                                    usage = msg.usage if isinstance(msg.usage, dict) else {}
                                    self._publish_event(
                                        request_id, session_key, seq,
                                        kind=AgentEventKind.SUBAGENT_PROGRESS,
                                        tool_use_id=msg.tool_use_id,
                                        tool_name=getattr(msg, "last_tool_name", None),
                                        extra_raw={
                                            "task_id": msg.task_id,
                                            "description": msg.description or "",
                                            "usage": {
                                                "total_tokens": usage.get("total_tokens", 0),
                                                "tool_uses": usage.get("tool_uses", 0),
                                                "duration_ms": usage.get("duration_ms", 0),
                                            },
                                        },
                                    )
                                elif isinstance(msg, TaskUpdatedMessage):
                                    # Status updates (running, paused, etc.) —
                                    # surface as SUBAGENT_PROGRESS with the status.
                                    if msg.status:
                                        seq += 1
                                        self._publish_event(
                                            request_id, session_key, seq,
                                            kind=AgentEventKind.SUBAGENT_PROGRESS,
                                            extra_raw={
                                                "task_id": msg.task_id,
                                                "status": msg.status,
                                            },
                                        )
                                elif isinstance(msg, TaskNotificationMessage):
                                    seq += 1
                                    usage = msg.usage if isinstance(msg.usage, dict) else {}
                                    logger.info(
                                        "Sub-agent done: task_id=%s status=%s tool_use_id=%s",
                                        msg.task_id, msg.status, msg.tool_use_id,
                                    )
                                    self._publish_event(
                                        request_id, session_key, seq,
                                        kind=AgentEventKind.SUBAGENT_DONE,
                                        tool_use_id=msg.tool_use_id,
                                        text=getattr(msg, "summary", "") or "",
                                        extra_raw={
                                            "task_id": msg.task_id,
                                            "status": msg.status,
                                            "output_file": getattr(msg, "output_file", ""),
                                            "usage": {
                                                "total_tokens": usage.get("total_tokens", 0),
                                                "tool_uses": usage.get("tool_uses", 0),
                                                "duration_ms": usage.get("duration_ms", 0),
                                            } if usage else None,
                                        },
                                    )
                            msg_type = type(msg).__name__
                            if not isinstance(msg, (StreamEvent, SystemMessage)):
                                content = getattr(msg, "content", [])
                                logger.debug(
                                    "SDK msg: %s blocks=%d content_types=%s",
                                    msg_type, len(content) if isinstance(content, list) else 0,
                                    [getattr(b, "type", type(b).__name__) for b in content] if isinstance(content, list) else "n/a",
                                )

                            if isinstance(msg, StreamEvent):
                                event = msg.event
                                event_type = event.get("type", "")
                                if event_type == "message_delta":
                                    ev_usage = event.get("usage")
                                    if isinstance(ev_usage, dict) and ev_usage:
                                        latest_stream_usage = ev_usage

                                if event_type == "content_block_delta":
                                    delta = event.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text = delta.get("text", "")
                                        if text:
                                            seq += 1
                                            text_parts.append(text)
                                            self._publish_event(
                                                request_id, session_key, seq,
                                                kind=AgentEventKind.ASSISTANT_DELTA,
                                                text=text,
                                            )
                                    elif delta.get("type") == "thinking_delta":
                                        # Model reasoning ("thinking"). Surface on
                                        # the REASONING_DELTA channel for the UI's
                                        # collapsible lane. Deliberately NOT
                                        # appended to text_parts — reasoning must
                                        # never enter the final assistant message
                                        # body (TASK-301).
                                        thinking = delta.get("thinking", "")
                                        if thinking:
                                            seq += 1
                                            self._publish_event(
                                                request_id, session_key, seq,
                                                kind=AgentEventKind.REASONING_DELTA,
                                                text=thinking,
                                            )
                                    elif delta.get("type") == "signature_delta":
                                        # Opaque reasoning signature — no display
                                        # value; drop it.
                                        pass
                                    elif delta.get("type") == "input_json_delta":
                                        current_tool_input += delta.get("partial_json", "")

                                elif event_type == "content_block_start":
                                    block = event.get("content_block", {})
                                    if block.get("type") == "tool_use":
                                        current_tool_name = block.get("name", "unknown")
                                        current_tool_input = ""

                                elif event_type == "content_block_stop":
                                    if current_tool_name:
                                        current_tool_name = None
                                        current_tool_input = ""

                            elif isinstance(msg, AssistantMessage):
                                # Capture per-iteration usage — overwrites on
                                # each internal API call so the LAST one wins,
                                # giving the UI the model's true final view
                                # of the context (not a cumulative sum).
                                am_usage = getattr(msg, "usage", None)
                                if isinstance(am_usage, dict) and am_usage:
                                    latest_assistant_usage = am_usage
                                # When this AssistantMessage is a sub-agent's inner
                                # activity, the SDK stamps parent_tool_use_id with the
                                # spawning Agent/Workflow tool's id. Thread it onto the
                                # TOOL_START so the UI can nest this tool card under the
                                # parent Agent card. None for top-level calls (TASK-344).
                                parent_tuid = getattr(msg, "parent_tool_use_id", None)
                                snapshot_parts: list[str] = []
                                for block in getattr(msg, "content", []):
                                    if isinstance(block, ToolUseBlock):
                                        seq += 1
                                        tu_id = getattr(block, "id", None)
                                        if tu_id:
                                            tool_names_by_id[tu_id] = block.name
                                        # QDIAG (TASK-413): record when the MODEL
                                        # actually emits an AskUserQuestion tool_use
                                        # block. Pair this with the "QDIAG can_use_tool
                                        # ENTER" line below: block-seen WITHOUT a
                                        # matching ENTER == the bridge/SDK dropped the
                                        # question (control-channel wedge on a resumed
                                        # turn); block NOT seen at all == the model
                                        # narrated the question in prose without calling
                                        # the tool. Split model-vs-pipeline for real.
                                        if self._is_ask_user_question(block.name):
                                            logger.info(
                                                "QDIAG model-emitted AskUserQuestion "
                                                "tool_use_id=%s session=%s parent=%s — "
                                                "expect a matching 'QDIAG can_use_tool "
                                                "ENTER' next",
                                                tu_id, session_key, parent_tuid,
                                            )
                                        # TOOLMAP (TASK-414): the id the bridge stamps
                                        # on TOOL_START. Must equal the TOOL_END
                                        # tool_use_id for the same call; if the
                                        # frontend can't heal a card, compare this
                                        # id against the "TOOLMAP end" line and the
                                        # frontend "TOOLMAP" orphan line.
                                        logger.info(
                                            "TOOLMAP start tool=%s tool_use_id=%s parent=%s session=%s",
                                            block.name, tu_id, parent_tuid, session_key,
                                        )
                                        self._publish_event(
                                            request_id, session_key, seq,
                                            kind=AgentEventKind.TOOL_START,
                                            tool_name=block.name,
                                            tool_arguments=block.input if isinstance(block.input, dict) else {},
                                            tool_use_id=tu_id,
                                            parent_tool_use_id=parent_tuid,
                                        )
                                        continue
                                    btype = getattr(block, "type", None)
                                    if isinstance(block, dict):
                                        btype = block.get("type")
                                    if btype == "text":
                                        if isinstance(block, dict):
                                            btext = block.get("text", "")
                                        else:
                                            btext = getattr(block, "text", "") or ""
                                        if btext:
                                            snapshot_parts.append(str(btext))
                                if snapshot_parts:
                                    assistant_snapshot_text = "".join(snapshot_parts)

                            elif isinstance(msg, UserMessage):
                                # Mirror of the AssistantMessage path: a sub-agent's
                                # tool *result* arrives on a UserMessage carrying the
                                # same parent_tool_use_id, so the TOOL_END nests under
                                # the same parent Agent card as its TOOL_START.
                                parent_tuid = getattr(msg, "parent_tool_use_id", None)
                                for block in getattr(msg, "content", []):
                                    if isinstance(block, ToolResultBlock):
                                        seq += 1
                                        result_content = block.content or ""
                                        # Persist Playwright screenshots to the
                                        # media store and collect a {asset_id,
                                        # kind} ref so the app can attach them to
                                        # the bot's reply (browsable per turn).
                                        # Never let an upload failure break the
                                        # turn — the inline image still reaches
                                        # the model regardless.
                                        if isinstance(result_content, list) and self._is_screenshot_tool(
                                            tool_names_by_id.get(block.tool_use_id or "")
                                        ):
                                            try:
                                                refs = await self._persist_screenshot_blocks(
                                                    result_content, session_key, block.tool_use_id,
                                                )
                                                turn_screenshot_assets.extend(refs)
                                            except Exception:
                                                logger.warning(
                                                    "Screenshot persist failed (tool_use_id=%s)",
                                                    block.tool_use_id, exc_info=True,
                                                )
                                        if isinstance(result_content, list):
                                            result_content = "\n".join(
                                                b.get("text", "") if isinstance(b, dict) else str(b)
                                                for b in result_content
                                            )
                                        # TOOLMAP (TASK-414): log the id the bridge
                                        # stamps on TOOL_END. The frontend heals a
                                        # running card by call_id; the bridge only
                                        # emits tool_use_id here (no call_id — the SDK
                                        # ToolResultBlock has none). If a start/end
                                        # tool_use_id pair diverges, or the app fails
                                        # to translate id→call_id, the card orphans on
                                        # "running". Pair this with the frontend
                                        # "TOOLMAP" console line and the TOOL_START log
                                        # below (grep both for the same tool_use_id).
                                        logger.info(
                                            "TOOLMAP end tool_use_id=%s parent=%s session=%s is_error=%s",
                                            block.tool_use_id,
                                            parent_tuid,
                                            session_key,
                                            getattr(block, "is_error", None),
                                        )
                                        self._publish_event(
                                            request_id, session_key, seq,
                                            kind=AgentEventKind.TOOL_END,
                                            # NB: the SDK's ToolResultBlock does
                                            # not echo the tool *name* — only the
                                            # tool_use_id linking back to the
                                            # originating ToolUseBlock.  Surface
                                            # both: tool_use_id in its proper
                                            # field, and a placeholder name so
                                            # legacy consumers still see something.
                                            tool_name=block.tool_use_id or "unknown",
                                            tool_use_id=block.tool_use_id,
                                            parent_tool_use_id=parent_tuid,
                                            tool_result=str(result_content)[:2000],
                                            # The SDK marks failed tool runs with
                                            # is_error on the ToolResultBlock — the
                                            # single authoritative failure signal.
                                            # Thread it so the UI can tint the card.
                                            tool_error=bool(getattr(block, "is_error", False)),
                                        )

                            elif isinstance(msg, ResultMessage):
                                full_text = "".join(text_parts)
                                if not full_text:
                                    result_text = getattr(msg, "text", "") or ""
                                    if not result_text:
                                        for block in getattr(msg, "content", []):
                                            if isinstance(block, dict) and block.get("type") == "text":
                                                result_text += block.get("text", "")
                                    full_text = result_text
                                    if not full_text and assistant_snapshot_text:
                                        full_text = assistant_snapshot_text
                                # If no text and retries happened, surface the error
                                if not full_text and api_retry_count > 0:
                                    error_note = (
                                        f"\n\n❌ Upstream error after {api_retry_count} "
                                        f"retries: {api_last_error or 'unknown'}. "
                                        f"Try again in a moment."
                                    )
                                    if api_retry_surfaced:
                                        # Already pushed "⏳ retrying" — append the outcome
                                        full_text = "".join(text_parts) + error_note
                                    else:
                                        full_text = error_note.lstrip()

                                # Extract token usage + context window for UI surfacing.
                                #
                                # IMPORTANT: ResultMessage.usage is CUMULATIVE across all
                                # internal API iterations in the turn — for a multi-tool-use
                                # turn that re-reads cached context on each call, the summed
                                # cache_read_input_tokens can exceed the context_window itself
                                # and produce nonsense >100% counters in the UI. We instead
                                # use the LAST AssistantMessage's per-iteration usage, which
                                # represents the model's final view of the context (what the
                                # user actually wants to see as "context fullness"). Cumulative
                                # output_tokens and total_cost_usd still come from ResultMessage
                                # since those genuinely accumulate across the turn.
                                #
                                # ResultMessage.model_usage is keyed by model id and exposes
                                # the model's contextWindow + maxOutputTokens.
                                token_usage_payload: dict | None = None
                                ctx_window = None
                                max_output = None
                                try:
                                    cumulative_usage = getattr(msg, "usage", None) or {}
                                    proxy_model = self._model_provider_prefix(
                                        actual_model or model
                                    ) is not None
                                    # Prefer stream message_delta for proxy providers
                                    # (xAI/ChatGPT/z.ai) — AssistantMessage often keeps
                                    # message_start zeros or a partial merge.
                                    iter_usage = _pick_iteration_usage(
                                        latest_assistant_usage,
                                        latest_stream_usage,
                                        cumulative_usage,
                                        proxy_model=proxy_model,
                                    )
                                    model_usage = getattr(msg, "model_usage", None) or {}
                                    ctx_window = None
                                    max_output = None
                                    if isinstance(model_usage, dict):
                                        # Prefer the actual model we ran on; fall back to any entry.
                                        mu_entry = (
                                            model_usage.get(actual_model)
                                            if actual_model
                                            else None
                                        )
                                        if mu_entry is None and model_usage:
                                            mu_entry = next(iter(model_usage.values()), None)
                                        if isinstance(mu_entry, dict):
                                            ctx_window = mu_entry.get("contextWindow")
                                            max_output = mu_entry.get("maxOutputTokens")
                                    # Claude Code defaults unknown models to 200k. Override
                                    # with the real window for proxy-routed providers.
                                    proxy_ctx = _proxy_context_window(actual_model or model)
                                    if proxy_ctx:
                                        # Always trust our table over the CLI default for
                                        # known proxy models (CLI never knows xAI windows).
                                        if not ctx_window or int(ctx_window) in (200_000, 0):
                                            ctx_window = proxy_ctx
                                        elif int(ctx_window) < proxy_ctx:
                                            # e.g. CLI said 200k for a 500k/1M model
                                            ctx_window = proxy_ctx
                                    if iter_usage or ctx_window:
                                        # z.ai reports input_tokens only in message_delta, so the
                                        # per-iteration AssistantMessage.usage (iter_usage) carries
                                        # the message_start value (0). Fall back to the cumulative
                                        # ResultMessage.usage, which via the SDK's last-non-zero merge
                                        # holds the real final-context input. No-op for Anthropic,
                                        # where iter_usage.input_tokens is already >0 (its
                                        # message_delta sends explicit 0s that updateUsage ignores).
                                        _input_tokens = int(iter_usage.get("input_tokens", 0) or 0)
                                        if _input_tokens == 0:
                                            _input_tokens = int(
                                                cumulative_usage.get("input_tokens", 0) or 0
                                            )
                                        _cache_read = int(
                                            iter_usage.get("cache_read_input_tokens", 0) or 0
                                        )
                                        _cache_create = int(
                                            iter_usage.get("cache_creation_input_tokens", 0) or 0
                                        )
                                        # If the chosen snapshot still has zero total input but
                                        # cumulative does not, take cache fields from cumulative too.
                                        if (
                                            _input_tokens + _cache_read + _cache_create
                                        ) == 0 and isinstance(cumulative_usage, dict):
                                            _cache_read = int(
                                                cumulative_usage.get(
                                                    "cache_read_input_tokens", 0
                                                )
                                                or 0
                                            )
                                            _cache_create = int(
                                                cumulative_usage.get(
                                                    "cache_creation_input_tokens", 0
                                                )
                                                or 0
                                            )
                                        _out_tokens = int(
                                            cumulative_usage.get("output_tokens", 0) or 0
                                        )
                                        if _out_tokens == 0:
                                            _out_tokens = int(
                                                iter_usage.get("output_tokens", 0) or 0
                                            )
                                        # Cost must be CUMULATIVE across the turn
                                        # (every internal API iteration billed). Context
                                        # fullness above is last-iteration only.
                                        if isinstance(cumulative_usage, dict) and (
                                            _usage_input_total(cumulative_usage) > 0
                                            or int(cumulative_usage.get("output_tokens", 0) or 0) > 0
                                        ):
                                            usage_for_cost = {
                                                "input_tokens": int(
                                                    cumulative_usage.get("input_tokens", 0) or 0
                                                ),
                                                "cache_read_input_tokens": int(
                                                    cumulative_usage.get(
                                                        "cache_read_input_tokens", 0
                                                    )
                                                    or 0
                                                ),
                                                "cache_creation_input_tokens": int(
                                                    cumulative_usage.get(
                                                        "cache_creation_input_tokens", 0
                                                    )
                                                    or 0
                                                ),
                                                "output_tokens": int(
                                                    cumulative_usage.get("output_tokens", 0)
                                                    or 0
                                                ),
                                            }
                                        else:
                                            usage_for_cost = {
                                                "input_tokens": _input_tokens,
                                                "cache_read_input_tokens": _cache_read,
                                                "cache_creation_input_tokens": _cache_create,
                                                "output_tokens": _out_tokens,
                                            }
                                        # Prefer provider-accurate cost for proxy models;
                                        # CLI prices unknowns as Opus-tier ($5/$25).
                                        cost = _estimate_proxy_cost_usd(
                                            actual_model or model, usage_for_cost
                                        )
                                        if cost is None:
                                            cost = getattr(msg, "total_cost_usd", None)
                                        token_usage_payload = {
                                            "input_tokens": _input_tokens,
                                            "cache_read_tokens": _cache_read,
                                            "cache_creation_tokens": _cache_create,
                                            # Output is still the cumulative turn total — that's
                                            # what the user generated overall, regardless of how
                                            # many internal iterations produced it.
                                            "output_tokens": _out_tokens,
                                            "context_window": ctx_window,
                                            "max_output_tokens": max_output,
                                            "total_cost_usd": cost,
                                        }
                                        if actual_model and (
                                            str(actual_model).startswith("zai/")
                                            or str(actual_model).startswith("xai/")
                                        ):
                                            logger.info(
                                                "proxy usage: model=%s iter_in=%s stream_in=%s "
                                                "cum_in=%s out=%s ctx=%s cost=%s",
                                                actual_model,
                                                (latest_assistant_usage or {}).get(
                                                    "input_tokens"
                                                )
                                                if isinstance(
                                                    latest_assistant_usage, dict
                                                )
                                                else None,
                                                (latest_stream_usage or {}).get(
                                                    "input_tokens"
                                                )
                                                if isinstance(latest_stream_usage, dict)
                                                else None,
                                                cumulative_usage.get("input_tokens")
                                                if isinstance(cumulative_usage, dict)
                                                else None,
                                                _out_tokens,
                                                ctx_window,
                                                cost,
                                            )
                                except Exception as _usage_err:
                                    logger.debug("Failed to extract token usage: %s", _usage_err)

                                # Compaction outcome. The /compact ResultMessage
                                # usage is all-zeros and the new resident size lives
                                # only in the transcript, so on success we read back
                                # compactMetadata.postTokens to (a) append a human
                                # summary to the reply and (b) OVERRIDE the usage
                                # gauge so the UI drops to the post-compaction size
                                # immediately instead of showing the stale
                                # pre-compaction number (the "reported context is
                                # exactly the same" symptom). On failure we explain
                                # why (e.g. "Not enough messages to compact.") so a
                                # no-op /compact isn't silent.
                                if compact_status == "success":
                                    cm = await asyncio.to_thread(
                                        _read_latest_compact_metadata, turn_session_id
                                    )
                                    pre = (cm or {}).get("preTokens")
                                    post = (cm or {}).get("postTokens")
                                    if post is not None:
                                        freed = (
                                            f" ({round(100 * (pre - post) / pre)}% freed)"
                                            if pre
                                            else ""
                                        )
                                        note = (
                                            f"\n\n✅ Compacted: {_fmt_tokens(pre)} → "
                                            f"{_fmt_tokens(post)} tokens{freed}."
                                        )
                                        token_usage_payload = {
                                            "input_tokens": int(post),
                                            "cache_read_tokens": 0,
                                            "cache_creation_tokens": 0,
                                            "output_tokens": 0,
                                            "context_window": ctx_window,
                                            "max_output_tokens": max_output,
                                            "total_cost_usd": getattr(msg, "total_cost_usd", None),
                                        }
                                    else:
                                        note = "\n\n✅ Conversation compacted."
                                    seq += 1
                                    text_parts.append(note)
                                    self._publish_event(
                                        request_id, session_key, seq,
                                        kind=AgentEventKind.ASSISTANT_DELTA,
                                        text=note,
                                    )
                                    full_text = "".join(text_parts)
                                elif compact_status == "failed":
                                    note = f"\n\nℹ️ Nothing to compact — {compact_error_msg}"
                                    seq += 1
                                    text_parts.append(note)
                                    self._publish_event(
                                        request_id, session_key, seq,
                                        kind=AgentEventKind.ASSISTANT_DELTA,
                                        text=note,
                                    )
                                    full_text = "".join(text_parts)

                                seq += 1
                                self._publish_event(
                                    request_id, session_key, seq,
                                    kind=AgentEventKind.ASSISTANT_DONE,
                                    text=full_text,
                                    model=actual_model,
                                    token_usage=token_usage_payload,
                                    attachments=turn_screenshot_assets or None,
                                )
                                assistant_done_emitted = True
                                # Turn complete — release the prompt generator so
                                # the SDK closes its input stream.  Kept open until
                                # now so the can_use_tool control channel survived
                                # any AskUserQuestion pause earlier in the turn.
                                turn_done.set()
                                # ResultMessage is terminal for this send (one user
                                # message -> one assistant turn), so stop iterating
                                # NOW instead of looping back to await a trailing
                                # StopAsyncIteration. After a deferred
                                # AskUserQuestion the streaming-input session stays
                                # alive — heartbeat/stream events keep re-arming the
                                # per-message timeout — so that await can block
                                # indefinitely while STILL holding the per-session
                                # lock, deadlocking the next continuation turn on the
                                # same session (TASK-269). The `finally` below closes
                                # the stream and kills the subprocess cleanly.
                                break
                        if aborted:
                            # Cooperative abort fired — fall straight through to
                            # publish_run_done without retry. We deliberately drop
                            # any partial response: the user explicitly cancelled.
                            seq += 1
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=AgentEventKind.ASSISTANT_DONE,
                                text="".join(text_parts),
                                model=actual_model,
                            )
                            assistant_done_emitted = True
                        elif not assistant_done_emitted:
                            # Clean EOF without a ResultMessage. z.ai / GLM via
                            # the Claude SDK can end a turn after an
                            # AssistantMessage snapshot only (text and/or tool
                            # uses already captured above). Publish a fallback
                            # terminal DONE so the app persists the reply instead
                            # of timing out with response_chars=0.
                            full_text = "".join(text_parts)
                            if assistant_snapshot_text:
                                if not full_text:
                                    full_text = assistant_snapshot_text
                                elif assistant_snapshot_text.startswith(full_text):
                                    full_text = assistant_snapshot_text
                            # Surface retry errors when the turn ends empty
                            if not full_text and api_retry_count > 0:
                                error_note = (
                                    f"\n\n❌ Upstream error after {api_retry_count} "
                                    f"retries: {api_last_error or 'unknown'}. "
                                    f"Try again in a moment."
                                )
                                if api_retry_surfaced:
                                    full_text = "".join(text_parts) + error_note
                                else:
                                    full_text = error_note.lstrip()
                            # Always publish — even if full_text is empty.
                            # An empty ASSISTANT_DONE is far better than no
                            # DONE at all, which causes timeout + vanishing bubble.
                            seq += 1
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=AgentEventKind.ASSISTANT_DONE,
                                text=full_text,
                                model=actual_model,
                                attachments=turn_screenshot_assets or None,
                            )
                            assistant_done_emitted = True
                            logger.info(
                                "EOF fallback ASSISTANT_DONE: chars=%d request_id=%s session=%s",
                                len(full_text), request_id, session_key,
                            )
                        break
                    except asyncio.CancelledError:
                        # task.cancel() arrived from elsewhere (legacy path /
                        # belt-and-suspenders fallback). Make sure the run is
                        # finalized before we re-raise so the frontend doesn't
                        # see a stuck `streaming` turn.
                        logger.info(
                            "Send cancelled via task.cancel: request_id=%s session=%s",
                            request_id, session_key,
                        )
                        seq += 1
                        try:
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=AgentEventKind.ASSISTANT_DONE,
                                text="".join(text_parts),
                                model=actual_model,
                            )
                            assistant_done_emitted = True
                        except Exception:
                            logger.debug("Failed to publish ASSISTANT_DONE on cancel", exc_info=True)
                        try:
                            self._publisher.publish_run_done(request_id)
                        except Exception:
                            logger.debug("Failed to publish run_done on cancel", exc_info=True)
                        raise
                    except Exception as e:
                        if cancel_event is not None and cancel_event.is_set():
                            logger.info(
                                "Abort teardown surfaced %r; treating as abort: session=%s",
                                e, session_key,
                            )
                            aborted = True
                            seq += 1
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=AgentEventKind.ASSISTANT_DONE,
                                text="".join(text_parts),
                                model=actual_model,
                            )
                            assistant_done_emitted = True
                            break

                        # 1) Auth failure → refresh token and retry once
                        if (
                            not auth_retry_attempted
                            and not text_parts
                            and _is_auth_failure(e, stderr_lines)
                        ):
                            auth_retry_attempted = True
                            logger.warning(
                                "Auth failure for %s; refreshing token and retrying",
                                request_id,
                            )
                            continue

                        # 2) CLI crash or timeout before any text streamed →
                        #    clear stale session and retry once fresh
                        if (
                            not fresh_session_retry
                            and not text_parts
                            and (_is_cli_crash(e) or isinstance(e, TimeoutError))
                        ):
                            fresh_session_retry = True
                            reason = "timeout" if isinstance(e, TimeoutError) else "CLI crash (exit code 1)"
                            logger.warning(
                                "%s for %s; clearing session and retrying fresh",
                                reason, request_id,
                            )
                            if stderr_lines:
                                logger.warning("Captured stderr before retry: %s", stderr_lines)
                            await self._clear_session(bot_slug or session_key)
                            resume_id = None
                            continue

                        raise
                    finally:
                        # Guarantee the prompt generator is released on EVERY exit
                        # path (StopAsyncIteration, the ResultMessage break above,
                        # abort, or exception). If it stays parked on
                        # `await done_event.wait()` the SDK input stream never closes
                        # and this session's lock can never be reacquired — the
                        # TASK-269 deadlock. Event.set() is idempotent.
                        turn_done.set()
                        # Always deregister this iteration's client so a
                        # subsequent chat.abort doesn't try to disconnect()
                        # something that's already finished. We pop only if
                        # the registry still points at our client — concurrent
                        # aborts may have already popped it to disconnect().
                        if session_key and sdk_client is not None:
                            current = self._session_queue.get_active_client(session_key)
                            if current is sdk_client:
                                self._session_queue.pop_active_client(session_key)
                            try:
                                await asyncio.wait_for(sdk_client.disconnect(), timeout=20.0)
                            except (asyncio.TimeoutError, asyncio.CancelledError):
                                pass
                            except Exception:
                                logger.debug(
                                    "sdk_client.disconnect() raised", exc_info=True,
                                )

                self._publisher.publish_run_done(request_id)
                if aborted:
                    logger.info(
                        "Send aborted via chat.abort: request_id=%s session=%s",
                        request_id, session_key,
                    )
                else:
                    logger.info("Send completed: request_id=%s session=%s", request_id, session_key)

            except asyncio.CancelledError:
                # Already handled inside the inner try (we published run_done
                # before re-raising). Suppress here so the asyncio task ends
                # cleanly without the "Task was destroyed but it is pending"
                # noise.
                logger.debug("Send cancellation propagated past inner handler")
                raise
            except Exception as e:
                logger.exception("Send failed: request_id=%s", request_id)
                seq += 1
                self._publish_event(
                    request_id, session_key, seq,
                    kind=AgentEventKind.ERROR,
                    text=str(e),
                )
                self._publisher.publish_run_done(request_id)
            finally:
                # Drop the per-run trigger_message_id mapping so we don't leak.
                self._trigger_message_ids.pop(request_id, None)
                await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)

    # ----- Event publishing -----

    async def _handle_tool_result(
        self, fields: dict, msg_id: str, async_redis,
    ) -> None:
        """Deprecated no-op (TASK-269).

        AskUserQuestion no longer blocks on an in-process Future, so there is
        nothing to resolve here.  The answer now arrives as a brand-new
        continuation turn (chat.send carrying the user's answer).  We keep this
        handler only to drain/ACK any stray chat.tool_result commands a stale
        client might still emit during a deploy window.
        """
        tool_use_id = (fields.get("tool_use_id") or "").strip()
        logger.info(
            "chat.tool_result is deprecated and ignored (tool_use_id=%s) — "
            "answers are delivered as continuation turns now",
            tool_use_id,
        )
        try:
            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
        except Exception:
            pass

    # ----- Approval-gated tool policies (TASK-291 / TASK-292) -----

    # Synthetic tool result fed back to the model when a call is gated. Mirrors
    # _DEFERRED_ACK: the turn ends cleanly and the decision returns as a brand
    # new continuation turn (approve → re-issue; deny → drop it).
    _APPROVAL_PENDING_ACK = (
        "[This action requires user approval and has been sent to the user. Do "
        "NOT retry it now and do not work around it. Briefly acknowledge that "
        "you've requested approval, then end your turn — you'll receive the "
        "user's decision as a new message and can act on it then.]"
    )

    async def _get_policy_bundle(self) -> PolicyBundle:
        """Return the approval-policy bundle, TTL-cached, fetched from the app.

        Conditional on the cached etag so an unchanged bundle costs one cheap
        round-trip and no re-parse. On any fetch error the cached bundle is
        kept; if there's no cache yet, an empty bundle is returned (no policies
        → default-allow), which is the fail-open posture.
        """
        now = time.monotonic()
        if (
            self._policy_bundle is not None
            and (now - self._policy_bundle_fetched_at) < self._policy_bundle_ttl
        ):
            return self._policy_bundle
        if not self._app_api_url:
            self._policy_bundle = self._policy_bundle or PolicyBundle(version=1, etag="", policies=[])
            self._policy_bundle_fetched_at = now
            return self._policy_bundle
        try:
            params = {}
            if self._policy_bundle is not None and self._policy_bundle.etag:
                params["etag"] = self._policy_bundle.etag
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{self._app_api_url}/v1/tool-approval-policies/bundle", params=params
                )
                resp.raise_for_status()
                data = resp.json()
            self._policy_bundle_fetched_at = now
            self._policy_fetch_ok = True
            if data.get("unchanged"):
                return self._policy_bundle  # type: ignore[return-value]
            self._policy_bundle = PolicyBundle.from_dict(data)
            logger.debug(
                "Fetched approval bundle: etag=%s policies=%d",
                self._policy_bundle.etag, len(self._policy_bundle.policies),
            )
            return self._policy_bundle
        except Exception as e:  # noqa: BLE001
            # Keep the last-known bundle; only warn occasionally to avoid spam.
            logger.warning("Approval bundle fetch failed (%s); using cached/empty", e)
            self._policy_bundle_fetched_at = now  # back off until next TTL
            self._policy_fetch_ok = False
            return self._policy_bundle or PolicyBundle(version=1, etag="", policies=[])

    def _grant_approval(self, grant_key: str, ttl_seconds: float) -> None:
        if not grant_key:
            return
        self._approval_grants[grant_key] = time.monotonic() + max(1.0, ttl_seconds)
        logger.info("Approval grant stored: %s (ttl=%ss)", grant_key[:12], int(ttl_seconds))

    def _consume_grant(self, grant_key: str) -> bool:
        """Pop a live grant for this key. Prunes expired grants as a side effect."""
        now = time.monotonic()
        # prune
        expired = [k for k, exp in self._approval_grants.items() if exp <= now]
        for k in expired:
            self._approval_grants.pop(k, None)
        exp = self._approval_grants.get(grant_key)
        if exp is None or exp <= now:
            return False
        self._approval_grants.pop(grant_key, None)
        return True

    def _decide_approval(self, tool_name: str, tool_input: dict) -> ApprovalDecision:
        """Pure policy decision for the current cached bundle (TASK-292).

        Isolated from the SDK/HTTP glue so it's unit-testable: inject a bundle
        and grants, assert the action. Does NOT consume grants (the caller does,
        only when it's actually going to allow).
        """
        bundle = self._policy_bundle or PolicyBundle(version=1, etag="", policies=[])
        return evaluate_policies(
            bundle.policies, self._backend_name, tool_name,
            tool_input if isinstance(tool_input, dict) else {},
        )

    async def _evaluate_tool_gate(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ToolPermissionContext,
        request_id: str,
        session_key: str,
        seq_holder: list[int],
    ):
        """Permission decision for a non-question tool (TASK-292).

        Returns a PermissionResult. On require_approval (no live grant) emits an
        APPROVAL_REQUIRED event and returns a deferred deny so the turn ends
        cleanly — the user's decision arrives as a continuation turn.
        """
        await self._get_policy_bundle()

        # Fail-closed: the app is unreachable AND the operator chose safety over
        # availability → deny every tool until policies can be fetched again.
        if self._approval_fail_closed and not self._policy_fetch_ok:
            logger.warning(
                "Approval fail-closed: policies unreachable, denying %s", tool_name
            )
            return PermissionResultDeny(
                message=(
                    "[Tool execution is paused: the approval-policy service is "
                    "unreachable and this bridge is configured fail-closed. "
                    "Acknowledge and end your turn.]"
                ),
                interrupt=False,
            )

        decision = self._decide_approval(tool_name, tool_input)

        if decision.action is PolicyAction.ALLOW:
            return PermissionResultAllow()

        if decision.action is PolicyAction.DENY:
            logger.info(
                "Tool DENIED by policy %s: %s %r",
                getattr(decision.policy, "id", "?"), tool_name, decision.subject[:80],
            )
            return PermissionResultDeny(
                message=(
                    f"[This action is blocked by an approval policy and cannot be "
                    f"run: {decision.subject[:200]}. Do not retry it. Continue "
                    f"without it or explain what you need.]"
                ),
                interrupt=False,
            )

        # ---- require_approval ----
        if self._consume_grant(decision.grant_key):
            logger.info(
                "Tool ALLOWED by prior approval grant: %s %r",
                tool_name, decision.subject[:80],
            )
            # TASK-305: mark this re-attempt as pre-approved so the UI can show
            # the gold/lock affordance on the exact card that ran with prior
            # authorization. The bridge is the single source of truth for this.
            self._emit_tool_preapproved(
                decision, tool_name, (ctx.tool_use_id or "").strip(),
                request_id, session_key, seq_holder,
            )
            return PermissionResultAllow()

        tool_use_id = (ctx.tool_use_id or "").strip()
        if not tool_use_id:
            # No id → app can't key a persistent request; defer cleanly anyway.
            logger.warning(
                "Approval-gated %s with no tool_use_id — deferring without persistence",
                tool_name,
            )
            return PermissionResultDeny(message=self._APPROVAL_PENDING_ACK, interrupt=False)

        self._emit_approval_required(
            decision, tool_name, tool_input, tool_use_id,
            request_id, session_key, seq_holder,
        )
        return PermissionResultDeny(message=self._APPROVAL_PENDING_ACK, interrupt=False)

    def _emit_approval_required(
        self,
        decision: ApprovalDecision,
        tool_name: str,
        tool_input: dict,
        tool_use_id: str,
        request_id: str,
        session_key: str,
        seq_holder: list[int],
    ) -> None:
        """Publish an APPROVAL_REQUIRED event for a gated tool (TASK-292).

        Single source of truth for the approval-request event payload, shared by
        the can_use_tool gate (_evaluate_tool_gate) and the PreToolUse hook gate
        (_evaluate_tool_gate_hook). Best-effort: a publish failure is logged,
        never raised — the caller still ends the turn with the pending-ack.
        """
        try:
            seq_holder[0] += 1
            self._publish_event(
                request_id, session_key, seq_holder[0],
                kind=AgentEventKind.APPROVAL_REQUIRED,
                tool_name=tool_name,
                tool_arguments=tool_input if isinstance(tool_input, dict) else {},
                tool_use_id=tool_use_id,
                extra_raw={
                    "policy_id": getattr(decision.policy, "id", None),
                    "severity": decision.severity.value,
                    "category": getattr(decision.policy, "category", None),
                    "subject": decision.subject,
                    "label": decision.label,
                    "prompt": decision.prompt,
                    "grant_key": decision.grant_key,
                    "action": decision.action.value,
                },
            )
        except Exception:
            logger.exception(
                "Failed to publish APPROVAL_REQUIRED for tool_use_id=%s", tool_use_id,
            )

        logger.info(
            "Tool gated (approval required): %s tool_use_id=%s policy=%s — turn ends",
            tool_name, tool_use_id, getattr(decision.policy, "id", "?"),
        )

    def _emit_tool_preapproved(
        self,
        decision: ApprovalDecision,
        tool_name: str,
        tool_use_id: str,
        request_id: str,
        session_key: str,
        seq_holder: list[int],
    ) -> None:
        """Publish a TOOL_PREAPPROVED event for a re-attempt that consumed a grant.

        Single source of truth for the pre-approved marker (TASK-305), shared by
        both gate paths (can_use_tool and the PreToolUse hook). The bridge is the
        only place that knows a tool ran because a one-shot grant was consumed —
        the client must not re-derive it. Best-effort: a publish failure is
        logged, never raised — the tool still runs.
        """
        if not tool_use_id:
            return
        try:
            seq_holder[0] += 1
            self._publish_event(
                request_id, session_key, seq_holder[0],
                kind=AgentEventKind.TOOL_PREAPPROVED,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                extra_raw={
                    "policy_id": getattr(decision.policy, "id", None),
                    "severity": decision.severity.value,
                    "grant_key": decision.grant_key,
                },
            )
        except Exception:
            logger.exception(
                "Failed to publish TOOL_PREAPPROVED for tool_use_id=%s", tool_use_id,
            )

    @staticmethod
    def _hook_deny(reason: str) -> dict:
        """PreToolUse hook output that blocks a tool with a model-visible reason.

        The reason is surfaced to the model as the tool's failure text (verified
        live: the model reports the permissionDecisionReason as the block error).
        """
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    async def _evaluate_tool_gate_hook(
        self,
        tool_name: str,
        tool_input: dict,
        tool_use_id: str,
        request_id: str,
        session_key: str,
        seq_holder: list[int],
    ) -> dict:
        """Approval gate as a PreToolUse hook decision (TASK-292).

        The live gate under permission_mode="bypassPermissions": the SDK only
        consults PreToolUse hooks (never can_use_tool) for regular tools in that
        mode. Mirrors _evaluate_tool_gate's policy logic but speaks the hook
        control plane. Returns a hook JSON output dict:

          ALLOW            → {} (no decision; tool proceeds normally)
          DENY             → "deny" + block reason
          REQUIRE_APPROVAL → consume a live one-shot grant → {} (allow); else
                             emit APPROVAL_REQUIRED and "deny" with the
                             pending-ack reason so the turn ends cleanly and the
                             user's decision arrives as a continuation turn
                             (same model as TASK-269 / the can_use_tool gate).
        """
        await self._get_policy_bundle()

        # Fail-closed: app unreachable AND operator chose safety → deny all.
        if self._approval_fail_closed and not self._policy_fetch_ok:
            logger.warning(
                "Approval fail-closed: policies unreachable, denying %s", tool_name
            )
            return self._hook_deny(
                "[Tool execution is paused: the approval-policy service is "
                "unreachable and this bridge is configured fail-closed. "
                "Acknowledge and end your turn.]"
            )

        decision = self._decide_approval(tool_name, tool_input)

        if decision.action is PolicyAction.ALLOW:
            return {}

        if decision.action is PolicyAction.DENY:
            logger.info(
                "Tool DENIED by policy %s: %s %r",
                getattr(decision.policy, "id", "?"), tool_name, decision.subject[:80],
            )
            return self._hook_deny(
                f"[This action is blocked by an approval policy and cannot be "
                f"run: {decision.subject[:200]}. Do not retry it. Continue "
                f"without it or explain what you need.]"
            )

        # ---- require_approval ----
        if self._consume_grant(decision.grant_key):
            logger.info(
                "Tool ALLOWED by prior approval grant: %s %r",
                tool_name, decision.subject[:80],
            )
            # TASK-305: mark this re-attempt as pre-approved (gold/lock card).
            self._emit_tool_preapproved(
                decision, tool_name, (tool_use_id or "").strip(),
                request_id, session_key, seq_holder,
            )
            return {}

        tool_use_id = (tool_use_id or "").strip()
        if not tool_use_id:
            # No id → app can't key a persistent request; defer cleanly anyway.
            logger.warning(
                "Approval-gated %s with no tool_use_id — deferring without persistence",
                tool_name,
            )
            return self._hook_deny(self._APPROVAL_PENDING_ACK)

        self._emit_approval_required(
            decision, tool_name, tool_input, tool_use_id,
            request_id, session_key, seq_holder,
        )
        return self._hook_deny(self._APPROVAL_PENDING_ACK)

    async def _handle_approval_grant(self, fields: dict, msg_id: str, async_redis) -> None:
        """Store a one-shot allow from an approval.grant command, then ACK."""
        try:
            grant_key = (fields.get("grant_key") or "").strip()
            ttl = float(fields.get("ttl_seconds") or "600")
            if grant_key:
                self._grant_approval(grant_key, ttl)
        except Exception:
            logger.exception("Failed to handle approval.grant")
        finally:
            try:
                await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
            except Exception:
                pass

    async def _approval_reload_listener(self, redis_url: str) -> None:
        """Drop the cached bundle on an approval:policies:reload broadcast.

        Best-effort: any failure is logged and retried; it never crashes the
        bridge. Lets an admin edit take effect without waiting out the TTL.
        """
        import redis.asyncio as aioredis

        while True:
            client = None
            pubsub = None
            try:
                # socket_timeout=None: pubsub.listen() is a long-lived blocking
                # read; redis-py 8.0's default 5s socket timeout would fire on
                # every idle interval and spin this loop. Bound only the connect.
                # (Mirrors _command_listener's XREADGROUP handling.)
                client = aioredis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_timeout=None,
                    socket_connect_timeout=5,
                    health_check_interval=30,
                )
                pubsub = client.pubsub()
                await pubsub.subscribe("approval:policies:reload")
                logger.info("Approval reload listener subscribed")
                async for msg in pubsub.listen():
                    if msg.get("type") != "message":
                        continue
                    self._policy_bundle_fetched_at = 0.0  # force refetch next call
                    logger.info("Approval bundle cache invalidated by reload broadcast")
            except asyncio.CancelledError:
                # Clean shutdown — release the pubsub connection.
                if pubsub is not None:
                    try:
                        await pubsub.aclose()
                    except Exception:
                        pass
                if client is not None:
                    try:
                        await client.aclose()
                    except Exception:
                        pass
                raise
            except Exception:
                logger.warning("Approval reload listener error; retrying in 5s", exc_info=True)
                # Close the failed client so retries don't leak connections.
                if pubsub is not None:
                    try:
                        await pubsub.aclose()
                    except Exception:
                        pass
                if client is not None:
                    try:
                        await client.aclose()
                    except Exception:
                        pass
                await asyncio.sleep(5)

    # ----- SDK can_use_tool callback -----

    @staticmethod
    def _is_ask_user_question(tool_name: str) -> bool:
        """Match AskUserQuestion regardless of MCP namespacing.

        The SDK built-in is just ``AskUserQuestion``, but if it ever shows up
        prefixed by an MCP server (``mcp__<server>__AskUserQuestion``) we want
        to intercept that too.
        """
        if not tool_name:
            return False
        tail = tool_name.rsplit("__", 1)[-1]
        return tail == "AskUserQuestion"

    def _make_can_use_tool(
        self,
        *,
        request_id: str,
        session_key: str,
        seq_holder: list[int],
    ):
        """Build the per-run can_use_tool callback bound to this turn.

        TASK-269 — deferred/continuation model.  AskUserQuestion does NOT
        block the SDK turn.  We emit an AWAIT_TOOL_RESULT event (the app
        persists the question and marks the turn end_reason="question") and
        immediately return a synthetic "deferred" ack via
        PermissionResultDeny.message.  The model reads that as the tool's
        output, acknowledges, and ends its turn cleanly — no Future, no
        wait_for, no 30-minute ceiling, no session lock held open.  The user's
        real answer arrives later as a brand-new continuation turn.

        Why DENY and not ALLOW: an ALLOW makes the SDK actually execute the
        built-in AskUserQuestion, which crashes in this headless context
        ("undefined is not an object (evaluating 'H.map')" — no interactive
        widget renderer) and sends the model into a retry loop.  A DENY with a
        message is the clean channel to feed text back as the tool result.
        Everything else gets the SDK's default allow.
        """

        async def can_use_tool(
            tool_name: str,
            tool_input: dict,
            ctx: ToolPermissionContext,
        ):
            if not self._is_ask_user_question(tool_name):
                # TASK-292: the approval gate moved to the PreToolUse hook
                # (_make_pre_tool_use_hook). Under bypassPermissions the SDK
                # doesn't call can_use_tool for regular tools anyway, and we want
                # EXACTLY ONE gate to avoid double-emitting APPROVAL_REQUIRED.
                # So allow here; the hook is the sole policy enforcement point.
                return PermissionResultAllow()

            # QDIAG (TASK-413): the callback fired for an AskUserQuestion. If the
            # model emitted the block (see "QDIAG model-emitted") but this line is
            # ABSENT, the SDK/control-channel dropped the question before reaching
            # us — a pipeline bug, not the model. Logged at ENTRY so it survives
            # even the no-tool_use_id / exception paths below.
            logger.info(
                "QDIAG can_use_tool ENTER AskUserQuestion tool_use_id=%s session=%s",
                (ctx.tool_use_id or "").strip() or "<none>", session_key,
            )

            tool_use_id = (ctx.tool_use_id or "").strip()
            if not tool_use_id:
                # Without a tool_use_id the app can't key a persistent question
                # row, but we can still defer cleanly so the turn ends instead
                # of hanging.  The model just won't get a follow-up answer.
                logger.warning(
                    "AskUserQuestion intercepted with no tool_use_id — deferring without persistence"
                )
                return PermissionResultDeny(message=self._DEFERRED_ACK, interrupt=False)

            # Emit AWAIT_TOOL_RESULT so the app persists the question, records
            # it on the turn (end_reason="question", question_id), and fans it
            # out to the UI.  Ordered in the same seq as surrounding deltas.
            try:
                seq_holder[0] += 1
                self._publish_event(
                    request_id, session_key, seq_holder[0],
                    kind=AgentEventKind.AWAIT_TOOL_RESULT,
                    tool_name=tool_name,
                    tool_arguments=tool_input if isinstance(tool_input, dict) else {},
                    tool_use_id=tool_use_id,
                )
            except Exception:
                logger.exception(
                    "Failed to publish AWAIT_TOOL_RESULT for tool_use_id=%s", tool_use_id,
                )

            logger.info(
                "AskUserQuestion deferred: tool_use_id=%s session=%s — turn ends, "
                "answer arrives as a continuation turn",
                tool_use_id, session_key,
            )

            # Immediate synthetic ack — no await.  The model ends its turn.
            return PermissionResultDeny(message=self._DEFERRED_ACK, interrupt=False)

        return can_use_tool

    def _make_pre_tool_use_hook(
        self,
        *,
        request_id: str,
        session_key: str,
        seq_holder: list[int],
    ):
        """Build the per-run PreToolUse hook bound to this turn (TASK-292).

        This is the live approval gate. It fires for every tool regardless of
        permission_mode (verified live under bypassPermissions). AskUserQuestion
        is handed back to the SDK (return {}) so its dedicated can_use_tool
        deferral (TASK-269) keeps owning it — the policy gate must not
        double-handle the question flow. Everything else goes through the policy
        engine via _evaluate_tool_gate_hook.
        """

        async def pre_tool_use(input_data, tool_use_id, context):
            tool_name = ""
            try:
                tool_name = input_data.get("tool_name") or ""
                if self._is_ask_user_question(tool_name):
                    return {}
                tool_input = input_data.get("tool_input")
                if not isinstance(tool_input, dict):
                    tool_input = {}
                tuid = (tool_use_id or input_data.get("tool_use_id") or "")
                return await self._evaluate_tool_gate_hook(
                    tool_name, tool_input, tuid,
                    request_id, session_key, seq_holder,
                )
            except Exception:
                logger.exception(
                    "PreToolUse approval gate errored for %s", tool_name or "?"
                )
                # Match the bundle-fetch posture: fail-closed denies on error,
                # otherwise fail-open so a gate bug can't wedge every tool.
                if self._approval_fail_closed:
                    return self._hook_deny(
                        "[Approval gate error and bridge is fail-closed; "
                        "tool blocked. Acknowledge and end your turn.]"
                    )
                return {}

        return pre_tool_use

    async def _handle_rpc(
        self, fields: dict, msg_id: str, async_redis,
    ) -> None:
        """Handle RPC commands (e.g. session.reset)."""
        import json as _json

        request_id = fields.get("request_id", "")
        method = fields.get("method", "")
        params_raw = fields.get("params", "{}")
        try:
            params = _json.loads(params_raw) if isinstance(params_raw, str) else params_raw
        except (_json.JSONDecodeError, TypeError):
            params = {}

        try:
            if method == "session.reset":
                session_key = params.get("sessionKey", "")
                target = _bot_slug_from_session_key(session_key) or session_key
                if target:
                    cleared = await self._clear_session(target)
                    logger.info("Session reset: %s (had_session=%s)", target, cleared)
                    # Emit a deterministic SESSION_RESET unified event so any
                    # active frontend SSE consumer for this bot can clear its
                    # visible buffer.  See TASK-249.
                    self._publish_session_reset_unified(
                        target, session_key or target, had_session=cleared,
                    )
                    self._publisher.publish_rpc_result(
                        request_id, {"ok": True, "reset": target, "had_session": cleared}
                    )
                else:
                    self._publisher.publish_rpc_result(
                        request_id, {"ok": False, "error": "missing sessionKey"}
                    )
            elif method == "chat.abort":
                session_key = params.get("sessionKey", "")
                # Three-layer abort:
                #   1. Set the cooperative cancel event so the SDK message loop
                #      breaks out at the next iteration boundary.
                #   2. Pop and disconnect() the active ClaudeSDKClient — this
                #      closes Query/transport and kills the underlying `claude`
                #      CLI subprocess mid-tool-call. Without (2), task.cancel()
                #      only raises CancelledError at the next `await` point,
                #      which can be tens of seconds away inside a long Bash/Read
                #      tool call.
                #   3. Fall back to task.cancel() so a runaway task that didn't
                #      respect (1) and (2) still gets torn down.
                self._session_queue.signal_cancel(session_key)
                # DIAGNOSTIC (abort-not-killing-subprocess): dump the live
                # registry keys so we can see whether the active stream is
                # registered under a different key than the abort target.
                logger.info(
                    "chat.abort registry probe: target=%r stream_keys=%r task_keys=%r",
                    session_key,
                    list(self._session_queue._active_clients.keys()),
                    list(self._session_queue._active_tasks.keys()),
                )
                client = self._session_queue.pop_active_client(session_key)
                client_disconnected = False
                if client is not None:
                    try:
                        await asyncio.wait_for(client.disconnect(), timeout=20.0)
                        client_disconnected = True
                    except asyncio.TimeoutError:
                        logger.warning(
                            "disconnect() timed out for session %s — subprocess may be stuck",
                            session_key,
                        )
                    except Exception:
                        logger.debug(
                            "disconnect() raised on chat.abort for session %s",
                            session_key,
                            exc_info=True,
                        )
                cancelled = self._session_queue.cancel_active(session_key)
                detail_parts: list[str] = []
                if cancelled:
                    detail_parts.append("task_cancelled")
                if client_disconnected:
                    detail_parts.append("client_disconnected")
                if not detail_parts:
                    detail_parts.append("no_active_task")
                logger.info(
                    "chat.abort: session=%s detail=%s",
                    session_key,
                    ",".join(detail_parts),
                )
                self._publisher.publish_rpc_result(
                    request_id,
                    {
                        "ok": True,
                        "aborted": session_key,
                        "detail": ",".join(detail_parts),
                    },
                )
            else:
                # Unknown RPC — return ok so callers don't hang
                self._publisher.publish_rpc_result(
                    request_id, {"ok": False, "error": f"unknown method: {method}"}
                )
        except Exception as e:
            logger.warning("RPC %s failed: %s", method, e)
            self._publisher.publish_rpc_result(
                request_id, {"ok": False, "error": str(e)}
            )
        finally:
            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)

    # ----- Event publishing -----

    @staticmethod
    def _shorten_paths(s: str | None) -> str | None:
        """Replace container home dir with ~ to save space in persisted logs."""
        if s is None:
            return None
        return s.replace("/home/bridge/", "~/").replace("/home/bridge", "~")

    @classmethod
    def _shorten_paths_in_dict(cls, d: dict | None) -> dict | None:
        if d is None:
            return None
        out = {}
        for k, v in d.items():
            if isinstance(v, str):
                out[k] = cls._shorten_paths(v)
            elif isinstance(v, dict):
                out[k] = cls._shorten_paths_in_dict(v)
            else:
                out[k] = v
        return out

    # ----- Screenshot persistence (Playwright -> media store) -----

    @staticmethod
    def _is_screenshot_tool(tool_name: str | None) -> bool:
        """True for the Playwright MCP screenshot tool (namespaced or bare)."""
        if not tool_name:
            return False
        return tool_name.split("__")[-1] == "browser_take_screenshot"

    @staticmethod
    def _extract_image_block(block: object) -> tuple[str, str] | None:
        """Pull (base64_data, mime) from a tool-result image content block.

        Handles both the MCP shape ``{type:image, data, mimeType}`` and the
        Anthropic API shape ``{type:image, source:{data, media_type}}``.
        """
        if not isinstance(block, dict) or block.get("type") != "image":
            return None
        data = block.get("data")
        mime = block.get("mimeType") or block.get("mime_type")
        if not data:
            src = block.get("source")
            if isinstance(src, dict):
                data = src.get("data")
                mime = mime or src.get("media_type")
        if not data:
            return None
        return data, (mime or "image/png")

    async def _persist_screenshot_blocks(
        self, content: list, session_key: str, tool_use_id: str | None,
    ) -> list[dict]:
        """Upload each image block in a screenshot tool-result to the media
        store; return ``{asset_id, kind}`` refs. Best-effort — logs and skips
        anything it can't parse or upload."""
        refs: list[dict] = []
        if not self._app_api_url:
            return refs
        user_id = session_key.split(":", 1)[1] if ":" in session_key else "nick"
        for block in content:
            img = self._extract_image_block(block)
            if img is None:
                if isinstance(block, dict) and block.get("type") == "image":
                    logger.warning(
                        "Screenshot image block in unrecognised shape (keys=%s)",
                        list(block.keys()),
                    )
                continue
            data_b64, mime = img
            asset_id = await self._upload_data_url(
                f"data:{mime};base64,{data_b64}", user_id, tool_use_id,
            )
            if asset_id:
                refs.append({"asset_id": asset_id, "kind": "image"})
        return refs

    async def _upload_data_url(
        self, data_url: str, user_id: str, tool_use_id: str | None,
    ) -> str | None:
        """POST a ``data:`` URL to /v1/uploads as an agent attachment; return asset_id."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{self._app_api_url}/v1/uploads",
                    params={"source": "agent_attachment"},
                    headers={"X-Entity-Id": user_id},
                    json={
                        "data_url": data_url,
                        "filename": f"screenshot-{tool_use_id or 'shot'}.png",
                    },
                )
                if resp.status_code >= 400:
                    logger.warning(
                        "Screenshot upload failed: %s %s",
                        resp.status_code, resp.text[:200],
                    )
                    return None
                return (resp.json() or {}).get("asset_id")
        except Exception:
            logger.warning("Screenshot upload error", exc_info=True)
            return None

    def _publish_event(
        self,
        request_id: str,
        session_key: str,
        seq: int,
        *,
        kind: AgentEventKind,
        text: str | None = None,
        tool_name: str | None = None,
        tool_arguments: dict | None = None,
        tool_result: str | None = None,
        tool_error: bool | None = None,
        model: str | None = None,
        token_usage: dict | None = None,
        tool_use_id: str | None = None,
        parent_tool_use_id: str | None = None,
        attachments: list[dict] | None = None,
        extra_raw: dict | None = None,
    ) -> None:
        text = self._shorten_paths(text)
        tool_result = self._shorten_paths(tool_result)
        tool_arguments = self._shorten_paths_in_dict(tool_arguments)
        event_id = synthesize_event_id(
            session_key, kind.value,
            {"text": text, "tool": tool_name, "seq": seq},
            seq,
        )
        event = AgentEvent(
            event_id=event_id,
            session_key=session_key,
            run_id=request_id,
            kind=kind,
            origin="system",
            text=text,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            tool_result=tool_result,
            tool_error=tool_error,
            model=model,
            seq=seq,
            timestamp=datetime.now(timezone.utc),
            # APPROVAL_REQUIRED stashes policy metadata (policy_id, severity,
            # subject, prompt, grant_key) in raw so the app can persist the
            # request row and the UI can render the Approve/Deny card.
            raw=dict(extra_raw) if extra_raw else {},
            token_usage=token_usage,
            provider=self._backend_name,
            # Inherit from the active run so every event for this request
            # (TOOL_START, TOOL_END, ASSISTANT_*, RUN_*) carries the same
            # originating user-message UUID without each call site needing
            # to remember to thread it through.
            trigger_message_id=self._trigger_message_ids.get(request_id),
            tool_use_id=tool_use_id,
            parent_tool_use_id=parent_tool_use_id,
            attachments=attachments,
        )
        self._publisher.publish_run_event(request_id, event)

    def _publish_session_reset_unified(
        self,
        bot_id: str,
        session_key: str,
        *,
        had_session: bool = False,
        user_id: str = "nick",
    ) -> None:
        """Publish a SESSION_RESET event on the unified SSE stream.

        Deterministic signal for the frontend to clear its visible message
        buffer for ``bot_id``.  Carries the confirmation text so the UI can
        render it as a synthetic assistant message in one place instead of
        racing the HTTP run stream's ASSISTANT_DONE.  See TASK-249.
        """
        if not bot_id:
            return
        try:
            self._publisher.publish_unified_event(bot_id, user_id, {
                "_type": "session_reset",
                "bot_id": bot_id,
                "user_id": user_id,
                "session_key": session_key,
                "had_session": bool(had_session),
                "text": "Session reset. Ready for a new conversation.",
                "provider": self._backend_name,
                "ts": datetime.now(timezone.utc).timestamp(),
            })
        except Exception:
            logger.exception(
                "Failed to publish unified session_reset for bot=%s session=%s",
                bot_id, session_key,
            )
