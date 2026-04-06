"""Claude Code bridge: Redis command listener + Agent SDK event translator."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterable
from datetime import datetime, timezone
from pathlib import Path

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

logger = logging.getLogger(__name__)

SESSION_PREFIX = "claude-code:"

_CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"


def _get_fresh_oauth_token() -> str | None:
    """Read the current OAuth token from the credentials file.

    The VS Code extension / CLI keeps this file refreshed. Reading it
    on each request ensures we always use a valid token.
    """
    try:
        if _CREDENTIALS_PATH.exists():
            data = json.loads(_CREDENTIALS_PATH.read_text())
            token = (data.get("claudeAiOauth") or {}).get("accessToken")
            if token:
                return token
    except Exception:
        pass
    # Fall back to env var
    return os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")


class ClaudeCodeBridge:
    """Reads chat.send commands from Redis, runs them through the Claude Agent
    SDK, and publishes OpenClawEvent-formatted results back to Redis."""

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
    ) -> None:
        self._publisher = publisher
        self._backend_name = backend_name
        self._app_api_url = app_api_url
        self._default_model = default_model
        self._cwd = cwd
        self._permission_mode = permission_mode
        self._add_dirs = add_dirs or []
        self._command_task: asyncio.Task | None = None
        self._redis = None  # set in _command_listener
        # Track active send tasks per session_key for abort support
        self._active_tasks: dict[str, asyncio.Task] = {}
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
        logger.info(
            "ClaudeCodeBridge started (backend=%s, model=%s)",
            self._backend_name, self._default_model,
        )

    async def stop(self) -> None:
        if self._command_task:
            self._command_task.cancel()
            try:
                await self._command_task
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
                                self._active_tasks[session_key] = task
                                task.add_done_callback(
                                    lambda t, sk=session_key: self._active_tasks.pop(sk, None)
                                )
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
            cleared = await self._clear_session(session_key)
            logger.info("Session reset via /new: %s (had_session=%s)", session_key, cleared)
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
            if system_prompt and self._mcp_servers and session_key:
                system_prompt += (
                    f"\n\n## MCP Tool Context\n"
                    f"Your bot_id is \"{session_key}\". When using llm-bawt-memory MCP tools:\n"
                    f"- Memory/message tools: always pass bot_id=\"{session_key}\"\n"
                    f"- Profile tool with entity_type=\"user\": use entity_id=\"nick\" (the user)\n"
                    f"- Profile tool with entity_type=\"bot\": use entity_id=\"{session_key}\" (yourself)"
                )

            # Reuse SDK session for conversation continuity.
            # If the model changed, start a fresh session.
            existing = await self._get_session(session_key)
            resume_id = None
            if existing:
                prev_sid, prev_model = existing
                if prev_model == model:
                    resume_id = prev_sid
                else:
                    logger.info(
                        "Model changed (%s -> %s), starting new session for %s",
                        prev_model, model, session_key,
                    )
                    await self._clear_session(session_key)

            def _log_stderr(line: str) -> None:
                logger.warning("CLI stderr: %s", line.rstrip())

            # Read fresh token on each request (auto-refresh from credentials file)
            fresh_token = _get_fresh_oauth_token()
            sdk_env = {}
            if fresh_token:
                sdk_env["CLAUDE_CODE_OAUTH_TOKEN"] = fresh_token

            # Resolve settings file path
            settings_path = str(Path.home() / ".claude" / "settings.json")
            if not Path(settings_path).exists():
                settings_path = None

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

            session_persisted = False

            async for msg in query(prompt=prompt_input, options=options):
                # Capture session_id and actual model from first SystemMessage only
                if isinstance(msg, SystemMessage) and not session_persisted:
                    data = getattr(msg, "data", {}) or {}
                    if data.get("model"):
                        actual_model = data["model"]
                        logger.info("Actual model: %s", actual_model)
                    if not resume_id:
                        sid = data.get("session_id")
                        if sid:
                            await self._set_session(session_key, sid, model)
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

                    seq += 1
                    self._publish_event(
                        request_id, session_key, seq,
                        kind=OpenClawEventKind.ASSISTANT_DONE,
                        text=full_text,
                        model=actual_model,
                    )

            self._publisher.publish_run_done(request_id)
            logger.info("Send completed: request_id=%s session=%s", request_id, session_key)

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
                if session_key:
                    cleared = await self._clear_session(session_key)
                    logger.info("Session reset: %s (had_session=%s)", session_key, cleared)
                    self._publisher.publish_rpc_result(
                        request_id, {"ok": True, "reset": session_key, "had_session": cleared}
                    )
                else:
                    self._publisher.publish_rpc_result(
                        request_id, {"ok": False, "error": "missing sessionKey"}
                    )
            elif method == "chat.abort":
                session_key = params.get("sessionKey", "")
                task = self._active_tasks.get(session_key)
                if task and not task.done():
                    task.cancel()
                    logger.info("Aborted active task for session: %s", session_key)
                    self._publisher.publish_rpc_result(
                        request_id, {"ok": True, "aborted": session_key}
                    )
                else:
                    logger.info("No active task to abort for session: %s", session_key)
                    self._publisher.publish_rpc_result(
                        request_id, {"ok": True, "aborted": session_key, "detail": "no_active_task"}
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
    ) -> None:
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
        )
        self._publisher.publish_run_event(request_id, event)
