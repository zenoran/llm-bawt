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
    ResultMessage,
    SystemMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from openclaw_bridge.events import OpenClawEvent, OpenClawEventKind, synthesize_event_id
from openclaw_bridge.publisher import COMMANDS_STREAM, RedisPublisher
from openclaw_bridge.session_queue import SessionQueue

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
    SDK, and publishes OpenClawEvent-formatted results back to Redis."""

    # Default timeout for a single query() call (seconds).
    # The CLI's internal API_TIMEOUT_MS is 600s; we cut shorter to fail fast.
    DEFAULT_REQUEST_TIMEOUT = 300

    def __init__(
        self,
        publisher: RedisPublisher,
        *,
        backend_name: str = "claude-code",
        app_api_url: str = "",
        default_model: str = "claude-sonnet-4-20250514",
        cwd: str = "/app",
        permission_mode: str = "bypassPermissions",
        add_dirs: list[str] | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self._publisher = publisher
        self._backend_name = backend_name
        self._app_api_url = app_api_url
        self._default_model = default_model
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
        # Load MCP servers from settings file
        self._mcp_servers = self._load_mcp_servers()

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
                        model = bc.get("model", "")
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
                bc["model"] = model

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
            "ClaudeCodeBridge started (backend=%s, model=%s)",
            self._backend_name, self._default_model,
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
        async_redis = aioredis.Redis(host=host, port=port, db=db, decode_responses=True)
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
                            session_key = fields.get("session_key", "")
                            task = asyncio.create_task(
                                self._handle_send(fields, msg_id, async_redis)
                            )
                            if session_key:
                                self._session_queue.set_active_task(session_key, task)
                        elif action == "rpc.call":
                            asyncio.create_task(
                                self._handle_rpc(fields, msg_id, async_redis)
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
        model = fields.get("model") or self._default_model
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

        # /new resets the session — strip it and start fresh
        if message.lstrip().startswith("/new"):
            cleared = await self._clear_session(bot_slug or session_key)
            logger.info("Session reset via /new: %s (had_session=%s)", bot_slug or session_key, cleared)
            message = message.lstrip().removeprefix("/new").strip()
            if not message:
                # Just "/new" with no follow-up — acknowledge and done
                self._publish_event(
                    request_id, session_key, 1,
                    kind=OpenClawEventKind.ASSISTANT_DONE,
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

        async with self._session_queue.lock(session_key):
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
            current_tool_id: str | None = None
            actual_model: str = model  # updated from SystemMessage if available

            try:
                # Inject MCP tool context so Claude passes the right identifiers
                if system_prompt and self._mcp_servers and bot_slug:
                    system_prompt += (
                        f"\n\n## MCP Tool Context\n"
                        f"Your bot_id is \"{bot_slug}\". When using llm-bawt-memory MCP tools:\n"
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

                    async def _image_prompt():
                        yield {
                            "type": "user",
                            "message": {"role": "user", "content": content},
                            "parent_tool_use_id": None,
                            "session_id": "default",
                        }

                    prompt_input: str | AsyncIterable = _image_prompt()
                else:
                    prompt_input = message

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
                    stderr_lines: list[str] = []

                    def _log_stderr(line: str) -> None:
                        line = line.rstrip()
                        stderr_lines.append(line)
                        logger.warning("CLI stderr: %s", line)

                    # Read fresh token on each request (auto-refresh from credentials file)
                    fresh_token = _get_fresh_oauth_token(force_refresh=auth_retry_attempted)
                    sdk_env = {}
                    if fresh_token:
                        sdk_env["CLAUDE_CODE_OAUTH_TOKEN"] = fresh_token

                    options = ClaudeAgentOptions(
                        model=model,
                        system_prompt=system_prompt if not resume_id else None,
                        cwd=self._cwd,
                        permission_mode=self._permission_mode,
                        include_partial_messages=True,
                        resume=resume_id,
                        add_dirs=self._add_dirs if self._add_dirs else [],
                        stderr=_log_stderr,
                        env=sdk_env,
                        settings=settings_path,
                        mcp_servers=self._mcp_servers if self._mcp_servers else {},
                    )

                    session_persisted = False
                    aborted = False

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
                                                kind=OpenClawEventKind.ASSISTANT_DELTA,
                                                text=text,
                                            )
                                    elif delta.get("type") == "input_json_delta":
                                        current_tool_input += delta.get("partial_json", "")

                                elif event_type == "content_block_start":
                                    block = event.get("content_block", {})
                                    if block.get("type") == "tool_use":
                                        current_tool_name = block.get("name", "unknown")
                                        current_tool_id = block.get("id", "")
                                        current_tool_input = ""

                                elif event_type == "content_block_stop":
                                    if current_tool_name:
                                        tool_args = {}
                                        if current_tool_input:
                                            try:
                                                tool_args = json.loads(current_tool_input)
                                            except json.JSONDecodeError:
                                                tool_args = {"raw": current_tool_input}
                                        current_tool_name = None
                                        current_tool_input = ""
                                        current_tool_id = None

                            elif isinstance(msg, AssistantMessage):
                                for block in getattr(msg, "content", []):
                                    if isinstance(block, ToolUseBlock):
                                        seq += 1
                                        self._publish_event(
                                            request_id, session_key, seq,
                                            kind=OpenClawEventKind.TOOL_START,
                                            tool_name=block.name,
                                            tool_arguments=block.input if isinstance(block.input, dict) else {},
                                        )

                            elif isinstance(msg, UserMessage):
                                for block in getattr(msg, "content", []):
                                    if isinstance(block, ToolResultBlock):
                                        seq += 1
                                        result_content = block.content or ""
                                        if isinstance(result_content, list):
                                            result_content = "\n".join(
                                                b.get("text", "") if isinstance(b, dict) else str(b)
                                                for b in result_content
                                            )
                                        self._publish_event(
                                            request_id, session_key, seq,
                                            kind=OpenClawEventKind.TOOL_END,
                                            tool_name=block.tool_use_id or "unknown",
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
                                # ResultMessage.usage carries per-turn billed tokens; the
                                # full input_tokens count = input + cache_creation + cache_read
                                # since the model sees all of it as context. ResultMessage.model_usage
                                # is keyed by model id and exposes the model's contextWindow so
                                # consumers can render % used.
                                token_usage_payload: dict | None = None
                                try:
                                    usage = getattr(msg, "usage", None) or {}
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
                                    if usage or ctx_window:
                                        token_usage_payload = {
                                            "input_tokens": int(usage.get("input_tokens", 0) or 0),
                                            "cache_read_tokens": int(
                                                usage.get("cache_read_input_tokens", 0) or 0
                                            ),
                                            "cache_creation_tokens": int(
                                                usage.get("cache_creation_input_tokens", 0) or 0
                                            ),
                                            "output_tokens": int(usage.get("output_tokens", 0) or 0),
                                            "context_window": ctx_window,
                                            "max_output_tokens": max_output,
                                            "total_cost_usd": getattr(msg, "total_cost_usd", None),
                                        }
                                except Exception as _usage_err:
                                    logger.debug("Failed to extract token usage: %s", _usage_err)

                                seq += 1
                                self._publish_event(
                                    request_id, session_key, seq,
                                    kind=OpenClawEventKind.ASSISTANT_DONE,
                                    text=full_text,
                                    model=actual_model,
                                    token_usage=token_usage_payload,
                                )
                        if aborted:
                            # Cooperative abort fired — fall straight through to
                            # publish_run_done without retry. We deliberately drop
                            # any partial response: the user explicitly cancelled.
                            seq += 1
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=OpenClawEventKind.ASSISTANT_DONE,
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
                                kind=OpenClawEventKind.ASSISTANT_DONE,
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
                    kind=OpenClawEventKind.ERROR,
                    text=str(e),
                )
                self._publisher.publish_run_done(request_id)
            finally:
                await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)

    # ----- Event publishing -----

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

    def _publish_event(
        self,
        request_id: str,
        session_key: str,
        seq: int,
        *,
        kind: OpenClawEventKind,
        text: str | None = None,
        tool_name: str | None = None,
        tool_arguments: dict | None = None,
        tool_result: str | None = None,
        model: str | None = None,
        token_usage: dict | None = None,
    ) -> None:
        text = self._shorten_paths(text)
        tool_result = self._shorten_paths(tool_result)
        tool_arguments = self._shorten_paths_in_dict(tool_arguments)
        event_id = synthesize_event_id(
            session_key, kind.value,
            {"text": text, "tool": tool_name, "seq": seq},
            seq,
        )
        event = OpenClawEvent(
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
        )
        self._publisher.publish_run_event(request_id, event)
