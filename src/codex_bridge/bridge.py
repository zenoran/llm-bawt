"""Codex bridge: Redis command listener + Codex SDK event translator.

Mirrors ``src/claude_code_bridge/bridge.py`` but talks to the OpenAI Codex
SDK (``codex_app_server``) instead of the Claude Agent SDK. Architectural
decisions are documented in the project's contextPrompt and the section
headers below.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from openclaw_bridge.events import OpenClawEvent, OpenClawEventKind, synthesize_event_id
from openclaw_bridge.publisher import COMMANDS_STREAM, RedisPublisher
from openclaw_bridge.session_queue import SessionQueue

from .transport import CodexTransport, validate_auth_json

logger = logging.getLogger(__name__)

CONSUMER_GROUP = "codex-bridge"
CONSUMER_NAME = "worker-0"


def _bot_slug_from_session_key(session_key: str) -> str:
    sk = (session_key or "").strip()
    if not sk:
        return ""
    return sk.split(":", 1)[0]


# --- TASK-204: failure detection helpers -----------------------------------


_AUTH_MARKERS = (
    "401",
    "403",
    "unauthorized",
    "authentication failed",
    "auth failure",
    "auth_failure",
    "invalid_grant",
    "invalid token",
    "refresh_token",
    "token has expired",
    "auth_mode=null",
    "authmode=null",
    "chatgpt token expired",
    "please run codex login",
)


def _is_auth_failure(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _AUTH_MARKERS)


_PROCESS_DEAD_MARKERS = (
    "broken pipe",
    "connection lost",
    "connection closed",
    "process exited",
    "subprocess died",
    "stream closed",
    "rpc transport closed",
)


def _is_process_dead(exc: BaseException) -> bool:
    if isinstance(exc, (ProcessLookupError, BrokenPipeError, ConnectionResetError)):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _PROCESS_DEAD_MARKERS)


# --- TASK-206: recoverable session errors ----------------------------------


_SESSION_ERROR_MARKERS = (
    "thread not found",
    "no rollout found",
    "missing required parameter: input",
    "encrypted_content",
    "thread_id is invalid",
)


def _is_codex_session_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _SESSION_ERROR_MARKERS)


# --- TASK-207: unified-diff old/new synthesis -----------------------------


def _synthesize_old_new(diff_text: str) -> tuple[str, str]:
    """Walk a unified diff and reconstruct the pre/post hunk content.

    '-' lines → old; '+' lines → new; ' '/context lines → both. Header lines
    (---, +++, @@) are dropped. This produces a best-effort old_string and
    new_string that ClaudeToolCallCard renders as a red/green diff card.
    """
    old_lines: list[str] = []
    new_lines: list[str] = []
    for line in (diff_text or "").splitlines():
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        if not line:
            old_lines.append("")
            new_lines.append("")
            continue
        marker, body = line[0], line[1:]
        if marker == "-":
            old_lines.append(body)
        elif marker == "+":
            new_lines.append(body)
        elif marker == " ":
            old_lines.append(body)
            new_lines.append(body)
        else:
            old_lines.append(line)
            new_lines.append(line)
    return ("\n".join(old_lines), "\n".join(new_lines))


# --- TASK-208: model -> context window/max_output cache --------------------


class _ModelInfoCache:
    """One-shot lookup of context window + max output tokens.

    Polls /v1/models on the main app once at startup (and again after the
    cache misses for an unknown model). Keys are model ids like ``gpt-5.4``.
    """

    def __init__(self, app_api_url: str) -> None:
        self._app_api_url = app_api_url
        self._info: dict[str, dict[str, int | None]] = {}
        self._loaded = False
        self._lock = asyncio.Lock()

    async def _load(self) -> None:
        if not self._app_api_url:
            self._loaded = True
            return
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._app_api_url}/v1/models")
                resp.raise_for_status()
                payload = resp.json() or {}
                for entry in payload.get("data", []) or []:
                    if not isinstance(entry, dict):
                        continue
                    mid = entry.get("id") or entry.get("model")
                    if not mid:
                        continue
                    self._info[mid] = {
                        "context_window": (
                            entry.get("context_window")
                            or entry.get("contextWindow")
                            or entry.get("max_input_tokens")
                        ),
                        "max_output_tokens": (
                            entry.get("max_output_tokens")
                            or entry.get("maxOutputTokens")
                        ),
                    }
        except Exception as e:
            logger.debug("Model info preload failed: %s", e)
        self._loaded = True

    async def get(self, model: str) -> dict[str, int | None]:
        async with self._lock:
            if not self._loaded:
                await self._load()
            cached = self._info.get(model)
            if cached is not None:
                return cached
            # Try a refetch once — model list may have grown since startup.
            self._loaded = False
            await self._load()
            return self._info.get(model, {"context_window": None, "max_output_tokens": None})


# --- Bridge -----------------------------------------------------------------


class CodexBridge:
    """Reads chat.send commands from Redis, runs them through the Codex
    SDK, and publishes OpenClawEvent-formatted results back to Redis."""

    DEFAULT_REQUEST_TIMEOUT = 300

    # TASK-209: cache cleanup parameters
    _CLEANUP_INTERVAL = 6 * 3600  # every 6 hours
    _CACHE_MAX_AGE = 24 * 3600
    _CACHE_DIRS = (
        "cache",
        "log",
        "tmp",
        ".tmp",
        "shell_snapshots",
    )
    # auth/config files we never delete:
    _PRESERVE_FILENAMES = frozenset(
        {"auth.json", "config.toml", "installation_id", "version.json"}
    )

    def __init__(
        self,
        publisher: RedisPublisher,
        *,
        backend_name: str = "codex",
        app_api_url: str = "",
        default_model: str = "gpt-5.4",
        cwd: str = "/home/bridge/dev",
        codex_bin: str | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self._publisher = publisher
        self._backend_name = backend_name
        self._app_api_url = app_api_url
        self._default_model = default_model
        self._cwd = cwd
        self._request_timeout = request_timeout or self.DEFAULT_REQUEST_TIMEOUT
        self._command_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._redis = None
        self._session_queue = SessionQueue()
        self._transport = CodexTransport(codex_bin=codex_bin)
        self._model_info = _ModelInfoCache(app_api_url)

    # ----- Public properties (used by __main__ for log lines) -------------

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def cwd(self) -> str:
        return self._cwd

    # ----- Lifecycle ------------------------------------------------------

    async def start(self) -> None:
        self._command_task = asyncio.create_task(self._command_listener())
        self._cleanup_task = asyncio.create_task(self._periodic_cache_cleanup())
        logger.info(
            "CodexBridge started (backend=%s, model=%s, cwd=%s)",
            self._backend_name, self._default_model, self._cwd,
        )

    async def stop(self) -> None:
        for task in (self._command_task, self._cleanup_task):
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        await self._transport.shutdown()
        self._publisher.close()
        logger.info("CodexBridge stopped")

    async def run_forever(self) -> None:
        await self.start()
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    # ----- TASK-204: supervisor / self-heal -------------------------------

    async def _ensure_codex(self):
        """Return a live AsyncCodex; rebuilds on transport death."""
        return await self._transport.ensure_codex()

    async def _supervisor_teardown(self) -> None:
        """Tear down AsyncCodex so the next ensure rebuilds.

        Re-validates auth.json defensively — useful when a fresh
        ``codex login`` is the recovery path.
        """
        try:
            validate_auth_json()
        except RuntimeError as e:
            logger.warning("auth.json still invalid during teardown: %s", e)
        await self._transport.shutdown()

    # ----- TASK-206: session persistence helpers --------------------------

    async def _get_session(self, bot_id: str) -> tuple[str, str] | None:
        """Get (thread_id, model) from the bot's agent_backend_config."""
        if not self._app_api_url or not bot_id:
            return None
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
                            if ":" in sk:
                                # Legacy bug: routing keys like "snark:nick"
                                # were once stored as session ids. Reject.
                                logger.warning(
                                    "Ignoring invalid persisted session_key for %s: %s",
                                    bot_id, sk,
                                )
                                return None
                            return (sk, model)
                        return None
        except Exception as e:
            logger.warning("Failed to get session for %s: %s", bot_id, e)
        return None

    async def _set_session(self, bot_id: str, thread_id: str, model: str) -> None:
        if not self._app_api_url or not bot_id:
            logger.warning("No API URL or bot_id — session not persisted")
            return
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._app_api_url}/v1/bots")
                resp.raise_for_status()
                bc: dict = {}
                for bot in resp.json().get("data", []):
                    if bot.get("slug") == bot_id:
                        bc = dict(bot.get("agent_backend_config") or {})
                        break
                bc["session_key"] = thread_id
                bc["model"] = model
                await client.patch(
                    f"{self._app_api_url}/v1/bots/{bot_id}/profile",
                    json={"agent_backend_config": bc},
                )
            logger.info("Session persisted: %s -> %s", bot_id, thread_id)
        except Exception as e:
            logger.warning("Failed to persist session for %s: %s", bot_id, e)

    async def _clear_session(self, bot_id: str) -> bool:
        if not self._app_api_url or not bot_id:
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._app_api_url}/v1/bots")
                resp.raise_for_status()
                bc: dict = {}
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

    # ----- TASK-205: Redis command listener -------------------------------

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
            # id="$" so a brand-new codex-bridge group skips historical
            # RPCs (which were authored before the codex bridge existed
            # and target other backends). Once the group is established,
            # this id is no longer consulted — xreadgroup uses the
            # group's last delivered id, so restart-on-codex-RPC works.
            await async_redis.xgroup_create(
                COMMANDS_STREAM, CONSUMER_GROUP, id="$", mkstream=True,
            )
        except Exception:
            pass

        logger.info(
            "Command listener started on %s (group=%s)",
            COMMANDS_STREAM, CONSUMER_GROUP,
        )

        while True:
            try:
                results = await async_redis.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
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
                            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
                            continue
                        # Filter RPCs that explicitly target a different backend.
                        # Legacy callers send RPCs without `backend` — we still
                        # accept those (they may target any agent backend).
                        if action == "rpc.call" and backend and backend != self._backend_name:
                            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
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
                            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)

            except asyncio.CancelledError:
                await async_redis.aclose()
                raise
            except Exception:
                logger.exception("Command listener error")
                await asyncio.sleep(2)

    # ----- TASK-205+207+208: chat.send handler ----------------------------

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
            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
            return

        # /new resets the session (TASK-206)
        if message.lstrip().startswith("/new"):
            cleared = await self._clear_session(bot_slug or session_key)
            logger.info(
                "Session reset via /new: %s (had_session=%s)",
                bot_slug or session_key, cleared,
            )
            message = message.lstrip().removeprefix("/new").strip()
            if not message:
                self._publish_event(
                    request_id, session_key, 1,
                    kind=OpenClawEventKind.ASSISTANT_DONE,
                    text="Session reset. Ready for a new conversation.",
                    model=model,
                )
                self._publisher.publish_run_done(request_id)
                await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
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
            actual_model: str = model

            try:
                # Build prompt input — multimodal if attachments present (TASK-208)
                prompt_input = self._build_prompt_input(message, attachments)

                # MCP context injection (matches claude bridge)
                if system_prompt and bot_slug:
                    system_prompt += (
                        f"\n\n## MCP Tool Context\n"
                        f"Your bot_id is \"{bot_slug}\". When using llm-bawt-memory MCP tools:\n"
                        f"- Memory/message tools: always pass bot_id=\"{bot_slug}\"\n"
                        f"- Profile tool with entity_type=\"user\": use entity_id=\"nick\" (the user)\n"
                        f"- Profile tool with entity_type=\"bot\": use entity_id=\"{bot_slug}\" (yourself)"
                    )

                # Resolve resume vs fresh thread (TASK-206)
                existing = await self._get_session(bot_slug)
                resume_id: str | None = None
                if existing:
                    prev_sid, prev_model = existing
                    if prev_model == model:
                        resume_id = prev_sid
                    else:
                        logger.info(
                            "Model changed (%s -> %s), starting new thread for %s",
                            prev_model, model, bot_slug or session_key,
                        )
                        await self._clear_session(bot_slug or session_key)

                # Cooperative cancel event for chat.abort (TASK-205)
                cancel_event = (
                    self._session_queue.cancel_event(session_key) if session_key else None
                )
                if cancel_event is not None and cancel_event.is_set():
                    cancel_event.clear()

                fresh_session_retry = False
                while True:
                    try:
                        codex = await self._ensure_codex()
                    except RuntimeError as auth_err:
                        # auth.json missing/invalid at startup — surface to user
                        seq += 1
                        self._publish_event(
                            request_id, session_key, seq,
                            kind=OpenClawEventKind.ERROR,
                            text=f"Codex OAuth failed — {auth_err}",
                            model=model,
                        )
                        self._publisher.publish_run_done(request_id)
                        return

                    aborted = False
                    handle = None
                    session_persisted = bool(resume_id)
                    item_buffers: dict[str, dict[str, Any]] = {}

                    try:
                        # Open or resume thread
                        if resume_id:
                            thread = await codex.thread_resume(
                                resume_id,
                                model=model,
                                cwd=self._cwd,
                                approval_policy="never",
                                sandbox="danger-full-access",
                                developer_instructions=system_prompt,
                            )
                        else:
                            thread = await codex.thread_start(
                                model=model,
                                cwd=self._cwd,
                                approval_policy="never",
                                sandbox="danger-full-access",
                                developer_instructions=system_prompt,
                                base_instructions=None,
                                ephemeral=False,
                            )

                        handle = await thread.turn(prompt_input)

                        # Register handle so chat.abort can interrupt server-side
                        if session_key:
                            self._session_queue.set_active_stream(session_key, handle)

                        async for note in self._iter_notes(handle):
                            if cancel_event is not None and cancel_event.is_set():
                                logger.info(
                                    "chat.abort signalled, interrupting Codex turn: session=%s request=%s",
                                    session_key, request_id,
                                )
                                aborted = True
                                try:
                                    await handle.interrupt()
                                except Exception:
                                    logger.debug("interrupt() raised", exc_info=True)
                                break

                            seq, persisted_now = await self._handle_note(
                                note,
                                bot_slug=bot_slug,
                                session_key=session_key,
                                request_id=request_id,
                                seq=seq,
                                text_parts=text_parts,
                                model=model,
                                actual_model_ref=[actual_model],
                                resume_id=resume_id,
                                session_persisted=session_persisted,
                                item_buffers=item_buffers,
                            )
                            if persisted_now:
                                session_persisted = True

                        # Drain final result for usage. Bounded timeout — an
                        # interrupted turn should return quickly; a stuck one
                        # shouldn't wedge the bridge.
                        try:
                            turn_result = await asyncio.wait_for(
                                handle.run(), timeout=30.0
                            )
                        except (asyncio.TimeoutError, Exception) as run_err:
                            logger.debug("handle.run() raised: %s", run_err)
                            turn_result = None

                        # ASSISTANT_DONE if not already sent by turn/completed
                        if not aborted and not self._already_sent_done(seq, item_buffers):
                            token_usage = self._extract_token_usage(turn_result, actual_model_or_default=actual_model)
                            seq += 1
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=OpenClawEventKind.ASSISTANT_DONE,
                                text="".join(text_parts),
                                model=actual_model,
                                token_usage=await self._merge_model_info(token_usage, actual_model),
                            )
                        elif aborted:
                            seq += 1
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=OpenClawEventKind.ASSISTANT_DONE,
                                text="".join(text_parts),
                                model=actual_model,
                            )
                        break

                    except asyncio.CancelledError:
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
                            pass
                        try:
                            self._publisher.publish_run_done(request_id)
                        except Exception:
                            pass
                        # Try to interrupt Codex too
                        if handle is not None:
                            try:
                                await handle.interrupt()
                            except Exception:
                                pass
                        raise
                    except Exception as e:
                        # Auth failure → publish ERROR, tear down transport, no retry.
                        if _is_auth_failure(e):
                            logger.warning(
                                "Codex auth failure for %s: %s — tearing down transport",
                                request_id, e,
                            )
                            await self._supervisor_teardown()
                            seq += 1
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=OpenClawEventKind.ERROR,
                                text="Codex OAuth failed — re-run codex login on echo",
                                model=model,
                            )
                            self._publisher.publish_run_done(request_id)
                            return

                        # Process death → tear down so next request rebuilds.
                        if _is_process_dead(e):
                            logger.warning(
                                "Codex app-server died for %s: %s — tearing down",
                                request_id, e,
                            )
                            await self._supervisor_teardown()
                            seq += 1
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=OpenClawEventKind.ERROR,
                                text=(
                                    "Codex subprocess crashed; supervisor will restart "
                                    "on next request."
                                ),
                                model=model,
                            )
                            self._publisher.publish_run_done(request_id)
                            return

                        # Recoverable session error → retry once with fresh thread.
                        if (
                            not fresh_session_retry
                            and not text_parts
                            and _is_codex_session_error(e)
                        ):
                            fresh_session_retry = True
                            logger.warning(
                                "Recoverable session error for %s: %s — clearing session and retrying fresh",
                                request_id, e,
                            )
                            await self._clear_session(bot_slug or session_key)
                            resume_id = None
                            continue

                        raise
                    finally:
                        if session_key and handle is not None:
                            current = self._session_queue.get_active_stream(session_key)
                            if current is handle:
                                self._session_queue.pop_active_stream(session_key)

                self._publisher.publish_run_done(request_id)
                if aborted:
                    logger.info(
                        "Send aborted via chat.abort: request_id=%s session=%s",
                        request_id, session_key,
                    )
                else:
                    logger.info(
                        "Send completed: request_id=%s session=%s",
                        request_id, session_key,
                    )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("Send failed: request_id=%s", request_id)
                seq += 1
                self._publish_event(
                    request_id, session_key, seq,
                    kind=OpenClawEventKind.ERROR,
                    text=str(e),
                    model=model,
                )
                self._publisher.publish_run_done(request_id)
            finally:
                await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)

    # ----- Notification iteration with timeout ----------------------------

    async def _iter_notes(self, handle):
        """Async-iterate handle.stream() with a per-note timeout.

        TimeoutError surfaces so the retry-on-session-error path can clear
        a stuck thread and retry fresh (TASK-206 acceptance criteria).

        ``handle.stream()`` may be either a regular method returning an async
        iterator, or an ``async def`` coroutine that returns one. Probe both.
        """
        maybe_stream = handle.stream()
        if asyncio.iscoroutine(maybe_stream):
            stream = await maybe_stream
        else:
            stream = maybe_stream
        agen = stream.__aiter__() if hasattr(stream, "__aiter__") else stream
        while True:
            try:
                note = await asyncio.wait_for(
                    agen.__anext__(),
                    timeout=self._request_timeout,
                )
            except StopAsyncIteration:
                return
            except TimeoutError:
                raise TimeoutError(
                    f"No Codex notifications for {self._request_timeout}s — app-server may be hung"
                )
            yield note

    @staticmethod
    def _already_sent_done(seq: int, item_buffers: dict) -> bool:
        # turn/completed handler sets this sentinel.
        return bool(item_buffers.get("__done_emitted__"))

    # ----- TASK-208: prompt input building --------------------------------

    @staticmethod
    def _build_prompt_input(message: str, attachments: list[dict]):
        if not attachments:
            return message
        # Lazy import so __init__ doesn't pull the SDK before validate_auth_json
        # has had a chance to fail loudly.
        from codex_app_server import ImageInput, TextInput

        items: list = [TextInput(text=message)]
        for att in attachments:
            data = att.get("content")
            if not data:
                continue
            mime = att.get("mimeType") or "image/png"
            items.append(ImageInput(data=data, mime_type=mime))
        return items

    # ----- TASK-208: token usage extraction -------------------------------

    @staticmethod
    def _extract_token_usage(turn_result: Any, *, actual_model_or_default: str) -> dict | None:
        if turn_result is None:
            return None
        usage = getattr(turn_result, "usage", None)
        if usage is None and isinstance(turn_result, dict):
            usage = turn_result.get("usage")
        if usage is None:
            return None
        # Codex usage may be a dataclass or dict
        def _g(key: str, default=0):
            if hasattr(usage, key):
                return getattr(usage, key)
            if isinstance(usage, dict):
                return usage.get(key, default)
            return default
        try:
            return {
                "input_tokens": int(_g("input_tokens", 0) or 0),
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "output_tokens": int(_g("output_tokens", 0) or 0),
                "context_window": None,  # filled in by _merge_model_info
                "max_output_tokens": None,
                "total_cost_usd": None,
            }
        except Exception:
            return None

    async def _merge_model_info(self, usage: dict | None, model: str) -> dict | None:
        if not usage:
            return usage
        info = await self._model_info.get(model)
        usage["context_window"] = info.get("context_window")
        usage["max_output_tokens"] = info.get("max_output_tokens")
        return usage

    # ----- TASK-207: notification → OpenClawEvent mapping -----------------

    async def _handle_note(
        self,
        note: Any,
        *,
        bot_slug: str,
        session_key: str,
        request_id: str,
        seq: int,
        text_parts: list[str],
        model: str,
        actual_model_ref: list[str],
        resume_id: str | None,
        session_persisted: bool,
        item_buffers: dict[str, dict[str, Any]],
    ) -> tuple[int, bool]:
        """Translate a single Codex notification into OpenClawEvents.

        Returns (next_seq, did_persist_session).
        """
        method = self._note_method(note)
        payload = self._note_payload(note)

        # ---- thread/started: capture thread_id, persist on first turn ----
        if method == "thread/started":
            if not session_persisted and not resume_id and bot_slug:
                thread_id = self._dig(payload, "thread_id") or self._dig(payload, "thread", "id")
                if thread_id:
                    await self._set_session(bot_slug, str(thread_id), model)
                    return seq, True
            return seq, False

        # ---- turn/started: no-op ----
        if method == "turn/started":
            return seq, False

        # ---- turn/completed: ASSISTANT_DONE or ERROR ----
        if method == "turn/completed":
            status = (self._dig(payload, "status") or "").lower()
            usage = self._dig(payload, "usage")
            full_text = "".join(text_parts)
            if status == "failed":
                err_msg = self._dig(payload, "error", "message") or "Codex turn failed"
                seq += 1
                self._publish_event(
                    request_id, session_key, seq,
                    kind=OpenClawEventKind.ERROR,
                    text=str(err_msg),
                    model=actual_model_ref[0],
                )
                item_buffers["__done_emitted__"] = True
                return seq, False

            token_usage = None
            if usage is not None:
                fake_carrier = type("U", (), {"usage": usage})()
                token_usage = self._extract_token_usage(fake_carrier, actual_model_or_default=actual_model_ref[0])
                token_usage = await self._merge_model_info(token_usage, actual_model_ref[0])

            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=OpenClawEventKind.ASSISTANT_DONE,
                text=full_text,
                model=actual_model_ref[0],
                token_usage=token_usage,
            )
            item_buffers["__done_emitted__"] = True
            return seq, False

        # ---- agentMessage delta ----
        if method == "item/agentMessage/delta":
            delta = self._dig(payload, "delta") or self._dig(payload, "text") or ""
            if delta:
                text_parts.append(delta)
                seq += 1
                self._publish_event(
                    request_id, session_key, seq,
                    kind=OpenClawEventKind.ASSISTANT_DELTA,
                    text=delta,
                )
            return seq, False

        # ---- item/started: TOOL_START for command/file/web/mcp ----
        if method == "item/started":
            item = self._dig(payload, "item") or payload
            seq = self._emit_tool_start(
                item, request_id, session_key, seq, item_buffers,
            )
            return seq, False

        # ---- buffer command output (don't stream tool result chunks) ----
        if method == "item/commandExecution/outputDelta":
            item_id = self._dig(payload, "item_id") or self._dig(payload, "id") or ""
            chunk = self._dig(payload, "chunk") or self._dig(payload, "text") or ""
            if item_id and chunk:
                buf = item_buffers.setdefault(item_id, {"output": []})
                buf.setdefault("output", []).append(chunk)
            return seq, False

        # ---- item/completed: TOOL_END (or possibly final agentMessage) ----
        if method == "item/completed":
            item = self._dig(payload, "item") or payload
            seq = self._emit_tool_end(
                item, request_id, session_key, seq, item_buffers, text_parts,
            )
            return seq, False

        # ---- error notification ----
        if method == "error":
            err_msg = self._dig(payload, "message") or str(payload)
            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=OpenClawEventKind.ERROR,
                text=str(err_msg),
                model=actual_model_ref[0],
            )
            return seq, False

        # ---- suppressed: reasoning, plan, review, contextCompaction ----
        if (
            method.startswith("item/reasoning/")
            or method.startswith("item/plan")
            or method.startswith("item/planUpdate")
            or method == "item/enteredReviewMode"
            or method == "item/exitedReviewMode"
            or method == "item/contextCompaction"
        ):
            return seq, False

        logger.debug("Unhandled Codex notification: method=%s", method)
        return seq, False

    # ----- Item-shape helpers ---------------------------------------------

    def _emit_tool_start(
        self,
        item: Any,
        request_id: str,
        session_key: str,
        seq: int,
        item_buffers: dict,
    ) -> int:
        item_type = self._dig(item, "type") or self._dig(item, "kind") or ""
        item_id = self._dig(item, "id") or ""

        tool_name: str | None = None
        tool_args: dict = {}

        if item_type in ("commandExecution", "command_execution"):
            tool_name = "Bash"
            cmd = self._dig(item, "command") or self._dig(item, "cmd") or ""
            cwd = self._dig(item, "cwd") or self._dig(item, "working_dir") or ""
            tool_args = {"command": cmd, "description": cwd}
            item_buffers.setdefault(item_id, {"output": [], "tool_name": tool_name, "type": "command"})
        elif item_type in ("fileChange", "file_change"):
            # Each fileChange item may contain multiple changes; emit one
            # TOOL_START per change so the UI gets per-file cards.
            return self._emit_filechange_starts(item, request_id, session_key, seq, item_buffers)
        elif item_type in ("webSearch", "web_search"):
            action = self._dig(item, "action") or {}
            atype = self._dig(action, "type") or ""
            if atype == "search":
                tool_name = "WebSearch"
                tool_args = {"query": self._dig(action, "query") or ""}
            elif atype == "openPage":
                tool_name = "WebFetch"
                tool_args = {"url": self._dig(action, "url") or ""}
            elif atype == "findInPage":
                tool_name = "Grep"
                tool_args = {
                    "pattern": self._dig(action, "pattern") or "",
                    "path": self._dig(action, "url") or "",
                }
            else:
                tool_name = "WebSearch"
                tool_args = {"query": str(action)}
            item_buffers.setdefault(item_id, {"tool_name": tool_name, "type": "web"})
        elif item_type in ("mcpToolCall", "mcp_tool_call"):
            tool_name = self._dig(item, "tool") or self._dig(item, "name") or "mcp_tool"
            tool_args = self._dig(item, "arguments") or self._dig(item, "args") or {}
            if not isinstance(tool_args, dict):
                tool_args = {"raw": str(tool_args)}
            item_buffers.setdefault(item_id, {"tool_name": tool_name, "type": "mcp"})
        elif item_type in ("dynamicToolCall", "dynamic_tool_call"):
            tool_name = self._dig(item, "tool") or self._dig(item, "name") or "tool"
            tool_args = self._dig(item, "arguments") or {}
            if not isinstance(tool_args, dict):
                tool_args = {"raw": str(tool_args)}
            item_buffers.setdefault(item_id, {"tool_name": tool_name, "type": "dynamic"})
        elif item_type in ("imageView", "image_view"):
            tool_name = "Read"
            tool_args = {"file_path": self._dig(item, "path") or ""}
            item_buffers.setdefault(item_id, {"tool_name": tool_name, "type": "image"})
        elif item_type in ("agentMessage", "agent_message"):
            # Final agent message — accumulated via deltas; no tool card.
            return seq

        if tool_name is None:
            return seq

        seq += 1
        self._publish_event(
            request_id, session_key, seq,
            kind=OpenClawEventKind.TOOL_START,
            tool_name=tool_name,
            tool_arguments=tool_args,
        )
        return seq

    def _emit_filechange_starts(
        self,
        item: Any,
        request_id: str,
        session_key: str,
        seq: int,
        item_buffers: dict,
    ) -> int:
        item_id = self._dig(item, "id") or ""
        changes = self._dig(item, "changes") or []
        # Group hunks by file_path so multi-hunk modifies get a single MultiEdit.
        by_path: dict[str, list[dict]] = {}
        for change in changes or []:
            path = self._dig(change, "path") or self._dig(change, "file_path") or ""
            if not path:
                continue
            by_path.setdefault(path, []).append(change)

        emitted: list[dict] = []
        for path, hunks in by_path.items():
            kinds = [self._dig(h, "kind") or self._dig(h, "type") or "modify" for h in hunks]
            if "create" in kinds:
                tool_name = "Write"
                diff_text = self._dig(hunks[0], "diff") or ""
                content_lines = [
                    line[1:] for line in diff_text.splitlines() if line.startswith("+")
                ]
                args = {"file_path": path, "content": "\n".join(content_lines)}
            elif "delete" in kinds:
                tool_name = "DeleteFile"
                args = {"file_path": path}
            elif len(hunks) == 1:
                tool_name = "Edit"
                diff_text = self._dig(hunks[0], "diff") or ""
                old, new = _synthesize_old_new(diff_text)
                args = {"file_path": path, "old_string": old, "new_string": new}
            else:
                tool_name = "MultiEdit"
                edits: list[dict] = []
                for h in hunks:
                    diff_text = self._dig(h, "diff") or ""
                    old, new = _synthesize_old_new(diff_text)
                    edits.append({"old_string": old, "new_string": new})
                args = {"file_path": path, "edits": edits}

            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=OpenClawEventKind.TOOL_START,
                tool_name=tool_name,
                tool_arguments=args,
            )
            emitted.append({"tool_name": tool_name, "args": args, "path": path})

        if item_id:
            item_buffers.setdefault(item_id, {})["filechange_started"] = emitted
            item_buffers[item_id]["type"] = "file"
        return seq

    def _emit_tool_end(
        self,
        item: Any,
        request_id: str,
        session_key: str,
        seq: int,
        item_buffers: dict,
        text_parts: list[str],
    ) -> int:
        item_type = self._dig(item, "type") or self._dig(item, "kind") or ""
        item_id = self._dig(item, "id") or ""
        buf = item_buffers.get(item_id, {})

        if item_type in ("agentMessage", "agent_message"):
            phase = self._dig(item, "phase") or self._dig(item, "stage") or ""
            if phase == "final_answer":
                final_text = (
                    self._dig(item, "text")
                    or self._dig(item, "content")
                    or ""
                )
                if final_text and not text_parts:
                    text_parts.append(str(final_text))
            return seq

        if item_type in ("commandExecution", "command_execution"):
            output = "".join(buf.get("output", []))
            exit_code = self._dig(item, "exit_code") or self._dig(item, "exitCode") or 0
            duration_ms = self._dig(item, "duration_ms") or self._dig(item, "durationMs") or 0
            try:
                exit_code_int = int(exit_code) if exit_code is not None else 0
            except (ValueError, TypeError):
                exit_code_int = 0
            if exit_code_int:
                result = json.dumps({
                    "output": output,
                    "exitCode": exit_code_int,
                    "durationMs": duration_ms,
                })
            else:
                result = output
            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=OpenClawEventKind.TOOL_END,
                tool_name="Bash",
                tool_result=result[:4000],
            )
            return seq

        if item_type in ("fileChange", "file_change"):
            # Emit one TOOL_END per matching TOOL_START (path-keyed).
            started = buf.get("filechange_started") or []
            for entry in started:
                seq += 1
                self._publish_event(
                    request_id, session_key, seq,
                    kind=OpenClawEventKind.TOOL_END,
                    tool_name=entry.get("tool_name") or "Edit",
                    tool_result="ok",
                )
            return seq

        if item_type in ("webSearch", "web_search"):
            tool_name = buf.get("tool_name") or "WebSearch"
            results = self._dig(item, "results") or self._dig(item, "result") or ""
            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=OpenClawEventKind.TOOL_END,
                tool_name=tool_name,
                tool_result=str(results)[:4000],
            )
            return seq

        if item_type in ("mcpToolCall", "mcp_tool_call"):
            tool_name = buf.get("tool_name") or self._dig(item, "tool") or "mcp_tool"
            result = self._dig(item, "result") or self._dig(item, "error") or ""
            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=OpenClawEventKind.TOOL_END,
                tool_name=tool_name,
                tool_result=str(result)[:4000],
            )
            return seq

        if item_type in ("dynamicToolCall", "dynamic_tool_call"):
            tool_name = buf.get("tool_name") or self._dig(item, "tool") or "tool"
            result = self._dig(item, "result") or self._dig(item, "error") or ""
            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=OpenClawEventKind.TOOL_END,
                tool_name=tool_name,
                tool_result=str(result)[:4000],
            )
            return seq

        if item_type in ("imageView", "image_view"):
            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=OpenClawEventKind.TOOL_END,
                tool_name="Read",
                tool_result="[image]",
            )
            return seq

        return seq

    @staticmethod
    def _note_method(note: Any) -> str:
        if hasattr(note, "method"):
            return getattr(note, "method") or ""
        if isinstance(note, dict):
            return note.get("method", "") or ""
        return ""

    @staticmethod
    def _note_payload(note: Any) -> Any:
        if hasattr(note, "payload"):
            return getattr(note, "payload")
        if isinstance(note, dict):
            return note.get("payload") or note.get("params") or note
        return note

    @staticmethod
    def _dig(obj: Any, *keys: str) -> Any:
        cur: Any = obj
        for k in keys:
            if cur is None:
                return None
            if hasattr(cur, k):
                cur = getattr(cur, k)
                continue
            if isinstance(cur, dict):
                cur = cur.get(k)
                continue
            return None
        return cur

    # ----- TASK-205: RPC handler -----------------------------------------

    async def _bot_uses_codex(self, bot_id: str) -> bool:
        """Look up a bot's agent_backend; return True only if it's codex.

        Used to defensively skip legacy RPCs (no `backend` field on the
        message) that target bots owned by other bridges. Without this
        guard, the codex bridge would happily clear session_keys on
        claude-code / openclaw bots — a cross-backend interference bug.
        """
        if not self._app_api_url or not bot_id:
            # No way to verify — fall through to legacy behavior. The
            # operations the RPC triggers (clear_session, signal_cancel)
            # are no-ops on unknown bots/sessions anyway.
            return True
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._app_api_url}/v1/bots")
                resp.raise_for_status()
                for bot in resp.json().get("data", []):
                    if bot.get("slug") == bot_id:
                        return (bot.get("agent_backend") or "") == self._backend_name
        except Exception as e:
            logger.debug("agent_backend lookup failed for %s: %s", bot_id, e)
            # On lookup failure, default to True so we don't drop our own
            # RPCs because the API blipped.
            return True
        # Bot not found — definitely not ours.
        return False

    async def _handle_rpc(
        self, fields: dict, msg_id: str, async_redis,
    ) -> None:
        request_id = fields.get("request_id", "")
        method = fields.get("method", "")
        message_backend = (fields.get("backend") or "").strip()
        params_raw = fields.get("params", "{}")
        try:
            params = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
        except (json.JSONDecodeError, TypeError):
            params = {}

        # Defensive bot-ownership check: when the message has no `backend`
        # field (legacy callers), look up the bot and skip silently if it
        # isn't ours. The command-listener already filtered explicit
        # backend=other; this catches the no-hint case.
        if not message_backend:
            session_key = (params.get("sessionKey") or "") if isinstance(params, dict) else ""
            target = _bot_slug_from_session_key(session_key) or session_key
            if target and not await self._bot_uses_codex(target):
                await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
                return

        try:
            if method == "session.reset":
                session_key = params.get("sessionKey", "")
                target = _bot_slug_from_session_key(session_key) or session_key
                if target:
                    cleared = await self._clear_session(target)
                    logger.info(
                        "Session reset: %s (had_session=%s)", target, cleared,
                    )
                    self._publisher.publish_rpc_result(
                        request_id, {"ok": True, "reset": target, "had_session": cleared},
                    )
                else:
                    self._publisher.publish_rpc_result(
                        request_id, {"ok": False, "error": "missing sessionKey"},
                    )
            elif method == "chat.abort":
                session_key = params.get("sessionKey", "")
                self._session_queue.signal_cancel(session_key)
                handle = self._session_queue.pop_active_stream(session_key)
                handle_interrupted = False
                if handle is not None:
                    try:
                        await asyncio.wait_for(handle.interrupt(), timeout=10.0)
                        handle_interrupted = True
                    except asyncio.TimeoutError:
                        logger.warning(
                            "handle.interrupt() timed out for session %s", session_key,
                        )
                    except Exception:
                        logger.debug(
                            "handle.interrupt() raised on chat.abort for session %s",
                            session_key, exc_info=True,
                        )
                cancelled = self._session_queue.cancel_active(session_key)
                detail_parts: list[str] = []
                if cancelled:
                    detail_parts.append("task_cancelled")
                if handle_interrupted:
                    detail_parts.append("turn_interrupted")
                if not detail_parts:
                    detail_parts.append("no_active_task")
                logger.info(
                    "chat.abort: session=%s detail=%s",
                    session_key, ",".join(detail_parts),
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
                self._publisher.publish_rpc_result(
                    request_id, {"ok": False, "error": f"unknown method: {method}"},
                )
        except Exception as e:
            logger.warning("RPC %s failed: %s", method, e)
            self._publisher.publish_rpc_result(
                request_id, {"ok": False, "error": str(e)},
            )
        finally:
            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)

    # ----- TASK-209: cache cleanup --------------------------------------

    async def _periodic_cache_cleanup(self) -> None:
        codex_dir = Path.home() / ".codex"
        sessions_dir = codex_dir / "sessions"
        while True:
            try:
                await asyncio.sleep(self._CLEANUP_INTERVAL)
                now = time.time()
                total_removed = 0

                # Per-name top-level dirs to prune
                for dirname in self._CACHE_DIRS:
                    cache_path = codex_dir / dirname
                    if not cache_path.is_dir():
                        continue
                    for entry in cache_path.iterdir():
                        if entry.name in self._PRESERVE_FILENAMES:
                            continue
                        try:
                            age = now - entry.stat().st_mtime
                            if age > self._CACHE_MAX_AGE:
                                if entry.is_dir():
                                    shutil.rmtree(entry, ignore_errors=True)
                                else:
                                    entry.unlink()
                                total_removed += 1
                        except OSError:
                            pass

                # Rollout JSONLs under sessions/YYYY/MM/DD/*.jsonl
                if sessions_dir.is_dir():
                    for jsonl in sessions_dir.rglob("*.jsonl"):
                        try:
                            age = now - jsonl.stat().st_mtime
                            if age > self._CACHE_MAX_AGE:
                                jsonl.unlink()
                                total_removed += 1
                        except OSError:
                            pass
                    # Best-effort empty-dir cleanup
                    self._prune_empty_dirs(sessions_dir)

                if total_removed:
                    logger.info(
                        "Cache cleanup: removed %d stale entries", total_removed,
                    )

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Cache cleanup error")

    @staticmethod
    def _prune_empty_dirs(root: Path) -> None:
        # Walk bottom-up; rmdir empty leaves.
        for path in sorted(root.rglob("*"), reverse=True):
            try:
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
            except OSError:
                pass

    # ----- Event publishing ---------------------------------------------

    @staticmethod
    def _shorten_paths(s: str | None) -> str | None:
        if s is None:
            return None
        return s.replace("/home/bridge/", "~/").replace("/home/bridge", "~")

    @classmethod
    def _shorten_paths_in_dict(cls, d: dict | None) -> dict | None:
        if d is None:
            return None
        out: dict = {}
        for k, v in d.items():
            if isinstance(v, str):
                out[k] = cls._shorten_paths(v)
            elif isinstance(v, dict):
                out[k] = cls._shorten_paths_in_dict(v)
            elif isinstance(v, list):
                out[k] = [
                    cls._shorten_paths(item) if isinstance(item, str)
                    else cls._shorten_paths_in_dict(item) if isinstance(item, dict)
                    else item
                    for item in v
                ]
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
