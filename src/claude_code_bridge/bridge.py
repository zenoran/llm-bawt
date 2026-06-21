"""Claude Code bridge: Redis command listener + Agent SDK event translator."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterable
from datetime import datetime, timezone
from pathlib import Path

import httpx
from claude_agent_sdk import ClaudeAgentOptions, StreamEvent, query
from claude_agent_sdk.types import (
    AssistantMessage,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from agent_bridge.events import AgentEvent, AgentEventKind, synthesize_event_id
from agent_bridge.publisher import COMMANDS_STREAM, RedisPublisher
from agent_bridge.session_queue import SessionQueue

logger = logging.getLogger(__name__)

SESSION_PREFIX = "claude-code:"


def _bot_slug_from_session_key(session_key: str) -> str:
    sk = (session_key or "").strip()
    if not sk:
        return ""
    return sk.split(":", 1)[0]


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

    async def start(self) -> None:
        self._command_task = asyncio.create_task(self._command_listener())
        self._cleanup_task = asyncio.create_task(self._periodic_cache_cleanup())
        logger.info(
            "ClaudeCodeBridge started (backend=%s)",
            self._backend_name,
        )

    async def stop(self) -> None:
        for task in (self._command_task, self._cleanup_task):
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
            message = message.lstrip().removeprefix("/new").strip()
            if not message:
                # Just "/new" with no follow-up — acknowledge and done
                self._publish_event(
                    request_id, session_key, 1,
                    kind=AgentEventKind.ASSISTANT_DONE,
                    text="Session reset. Ready for a new conversation.",
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
                # Inject MCP tool context so Claude passes the right identifiers
                if system_prompt and self._mcp_servers and bot_slug:
                    system_prompt += (
                        f"\n\n## MCP Tool Context\n"
                        f"Your bot_id is \"{bot_slug}\". When using bawthub MCP tools:\n"
                        f"- Memory/message tools: always pass bot_id=\"{bot_slug}\"\n"
                        f"- Profile tool with entity_type=\"user\": use entity_id=\"nick\" (the user)\n"
                        f"- Profile tool with entity_type=\"bot\": use entity_id=\"{bot_slug}\" (yourself)"
                    )

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

                    sdk_env = {}
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
                        permission_mode=self._permission_mode,
                        include_partial_messages=True,
                        resume=resume_id,
                        add_dirs=self._add_dirs if self._add_dirs else [],
                        stderr=_log_stderr,
                        env=sdk_env,
                        settings=settings_path,
                        effort=bot_effort,
                        max_turns=bot_max_turns,
                        mcp_servers=self._mcp_servers if self._mcp_servers else {},
                        can_use_tool=can_use_tool_cb,
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

                    msg_stream = None
                    try:
                        msg_stream = query(prompt=prompt_input, options=options).__aiter__()
                        # Register the live stream so `chat.abort` can call
                        # `aclose()` on it — that's what actually kills the
                        # underlying CLI subprocess mid-tool-call. `task.cancel()`
                        # alone is insufficient because CancelledError only fires
                        # at the next `await`, and the SDK is awaiting on subprocess
                        # output that doesn't arrive until the running tool exits.
                        if session_key:
                            self._session_queue.set_active_stream(session_key, msg_stream)
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

                            # Capture session_id and actual model from first SystemMessage only
                            if isinstance(msg, SystemMessage) and not session_persisted:
                                data = getattr(msg, "data", {}) or {}
                                if data.get("model"):
                                    actual_model = data["model"]
                                    logger.info("Actual model: %s", actual_model)
                                if not resume_id:
                                    sid = data.get("session_id")
                                    if sid:
                                        await self._set_session(bot_slug or session_key, sid, model)
                                session_persisted = True
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
                                for block in getattr(msg, "content", []):
                                    if isinstance(block, ToolUseBlock):
                                        seq += 1
                                        tu_id = getattr(block, "id", None)
                                        if tu_id:
                                            tool_names_by_id[tu_id] = block.name
                                        self._publish_event(
                                            request_id, session_key, seq,
                                            kind=AgentEventKind.TOOL_START,
                                            tool_name=block.name,
                                            tool_arguments=block.input if isinstance(block.input, dict) else {},
                                            tool_use_id=tu_id,
                                        )

                            elif isinstance(msg, UserMessage):
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
                                            tool_result=str(result_content)[:2000],
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
                                try:
                                    cumulative_usage = getattr(msg, "usage", None) or {}
                                    # Per-iteration view (preferred) — falls back to the
                                    # cumulative usage when no AssistantMessage was seen
                                    # (e.g., single-API-call turns where they're equal anyway).
                                    iter_usage = latest_assistant_usage or cumulative_usage
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
                                    if iter_usage or ctx_window:
                                        token_usage_payload = {
                                            "input_tokens": int(iter_usage.get("input_tokens", 0) or 0),
                                            "cache_read_tokens": int(
                                                iter_usage.get("cache_read_input_tokens", 0) or 0
                                            ),
                                            "cache_creation_tokens": int(
                                                iter_usage.get("cache_creation_input_tokens", 0) or 0
                                            ),
                                            # Output is still the cumulative turn total — that's
                                            # what the user generated overall, regardless of how
                                            # many internal iterations produced it.
                                            "output_tokens": int(
                                                cumulative_usage.get("output_tokens", 0) or 0
                                            ),
                                            "context_window": ctx_window,
                                            "max_output_tokens": max_output,
                                            "total_cost_usd": getattr(msg, "total_cost_usd", None),
                                        }
                                except Exception as _usage_err:
                                    logger.debug("Failed to extract token usage: %s", _usage_err)

                                seq += 1
                                self._publish_event(
                                    request_id, session_key, seq,
                                    kind=AgentEventKind.ASSISTANT_DONE,
                                    text=full_text,
                                    model=actual_model,
                                    token_usage=token_usage_payload,
                                    attachments=turn_screenshot_assets or None,
                                )
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
                        except Exception:
                            logger.debug("Failed to publish ASSISTANT_DONE on cancel", exc_info=True)
                        try:
                            self._publisher.publish_run_done(request_id)
                        except Exception:
                            logger.debug("Failed to publish run_done on cancel", exc_info=True)
                        raise
                    except Exception as e:
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
                        # Always deregister this iteration's stream so a
                        # subsequent chat.abort doesn't try to aclose() a
                        # generator that's already finished. We pop only if
                        # the registry still points at our stream — concurrent
                        # aborts may have already popped it to call aclose().
                        if session_key and msg_stream is not None:
                            current = self._session_queue.get_active_stream(session_key)
                            if current is msg_stream:
                                self._session_queue.pop_active_stream(session_key)
                            # Best-effort close — kills the SDK subprocess if
                            # it's still running. Bounded so a stuck SIGKILL
                            # can't wedge the bridge.
                            try:
                                await asyncio.wait_for(msg_stream.aclose(), timeout=10.0)
                            except (asyncio.TimeoutError, asyncio.CancelledError):
                                pass
                            except Exception:
                                logger.debug(
                                    "msg_stream.aclose() raised", exc_info=True,
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
                # Default-allow for everything else.  permission_mode on the
                # ClaudeAgentOptions still governs the SDK-side prompt flow;
                # this callback only short-circuits AskUserQuestion.
                return PermissionResultAllow()

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
                #   2. Pop and aclose() the active SDK message stream — this is
                #      what actually kills the underlying `claude` CLI subprocess
                #      mid-tool-call. Without (2), `task.cancel()` only raises
                #      CancelledError at the next `await` point, which can be
                #      tens of seconds away inside a long Bash/Read tool call.
                #   3. Fall back to task.cancel() so a runaway task that didn't
                #      respect (1) and (2) still gets torn down.
                self._session_queue.signal_cancel(session_key)
                # DIAGNOSTIC (abort-not-killing-subprocess): dump the live
                # registry keys so we can see whether the active stream is
                # registered under a different key than the abort target.
                logger.info(
                    "chat.abort registry probe: target=%r stream_keys=%r task_keys=%r",
                    session_key,
                    list(self._session_queue._active_streams.keys()),
                    list(self._session_queue._active_tasks.keys()),
                )
                stream = self._session_queue.pop_active_stream(session_key)
                stream_closed = False
                if stream is not None:
                    try:
                        await asyncio.wait_for(stream.aclose(), timeout=10.0)
                        stream_closed = True
                    except asyncio.TimeoutError:
                        logger.warning(
                            "aclose() timed out for session %s — subprocess may be stuck",
                            session_key,
                        )
                    except Exception:
                        logger.debug(
                            "aclose() raised on chat.abort for session %s",
                            session_key,
                            exc_info=True,
                        )
                cancelled = self._session_queue.cancel_active(session_key)
                detail_parts: list[str] = []
                if cancelled:
                    detail_parts.append("task_cancelled")
                if stream_closed:
                    detail_parts.append("stream_closed")
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
        model: str | None = None,
        token_usage: dict | None = None,
        tool_use_id: str | None = None,
        attachments: list[dict] | None = None,
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
            model=model,
            seq=seq,
            timestamp=datetime.now(timezone.utc),
            raw={},
            token_usage=token_usage,
            provider=self._backend_name,
            # Inherit from the active run so every event for this request
            # (TOOL_START, TOOL_END, ASSISTANT_*, RUN_*) carries the same
            # originating user-message UUID without each call site needing
            # to remember to thread it through.
            trigger_message_id=self._trigger_message_ids.get(request_id),
            tool_use_id=tool_use_id,
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
