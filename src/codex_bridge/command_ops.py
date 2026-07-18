from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from agent_bridge.events import AgentEvent, AgentEventKind, synthesize_event_id
from agent_bridge.publisher import COMMANDS_STREAM, RedisPublisher
from agent_bridge.session_queue import SessionQueue

from .transport import CodexTransport, validate_auth_json
from ._bridge_helpers import (
    CONSUMER_GROUP,
    CONSUMER_NAME,
    MCP_TOOL_CONTEXT_KEY,
    _MCP_TOOL_CONTEXT_FALLBACK,
    _ModelInfoCache,
    _bot_slug_from_session_key,
    _is_auth_failure,
    _is_codex_session_error,
)

logger = logging.getLogger("codex_bridge.bridge")


class CodexCommandMixin:
    """Codex Redis command listener + send/RPC handlers (TASK-555).

    Split out of ``CodexBridge`` (TASK-555). Composed back on via
    inheritance, so ``self.*`` state set in ``CodexBridge.__init__`` and
    methods on sibling mixins all resolve on the assembled instance.
    """

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
                            asyncio.create_task(
                                self._handle_send(fields, msg_id, async_redis)
                            )
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

    async def _handle_send(
        self, fields: dict, msg_id: str, async_redis,
    ) -> None:
        request_id = fields.get("request_id", "")
        session_key = fields.get("session_key", "")
        bot_slug = (fields.get("bot_id", "") or "").strip() or _bot_slug_from_session_key(session_key)
        message = fields.get("message", "")
        system_prompt = fields.get("system_prompt") or None
        model = fields.get("model") or self._default_model
        # Frontend-supplied user-message UUID; stamped on every emitted event
        # so the frontend can bucket tool activity under the originating user
        # message without falling back to turn_id heuristics.
        trigger_message_id = (fields.get("trigger_message_id") or "").strip() or None
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

        if trigger_message_id:
            self._trigger_message_ids[request_id] = trigger_message_id

        # /new resets the session (TASK-206)
        if message.lstrip().startswith("/new"):
            cleared = await self._clear_session(bot_slug or session_key)
            logger.info(
                "Session reset via /new: %s (had_session=%s)",
                bot_slug or session_key, cleared,
            )
            # Deterministic SESSION_RESET unified event for the frontend
            # (TASK-249) — clears the visible buffer without racing
            # turn_complete timing.
            self._publish_session_reset_unified(
                bot_slug or session_key, session_key, had_session=cleared,
            )
            message = message.lstrip().removeprefix("/new").strip()
            if not message:
                self._publish_event(
                    request_id, session_key, 1,
                    kind=AgentEventKind.ASSISTANT_DONE,
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

        async with self._session_queue.active(session_key):
            logger.info(
                "Handling send: request_id=%s session=%s model=%s system_prompt=%s msg=%.60s...",
                request_id, session_key, model,
                f"{len(system_prompt)} chars" if system_prompt else "none",
                message,
            )

            seq = 0
            text_parts: list[str] = []
            actual_model: str = model
            tmp_image_paths: list[str] = []

            try:
                # MCP context injection (matches claude bridge). Body from the
                # registry (TASK-490) with a byte-identical local fallback.
                if system_prompt and bot_slug:
                    mcp_ctx = await self._get_mcp_tool_context(bot_slug)
                    system_prompt += f"\n\n{mcp_ctx}"

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
                        codex = self._ensure_codex()
                    except RuntimeError as auth_err:
                        # auth.json missing/invalid at startup — surface to user
                        seq += 1
                        self._publish_event(
                            request_id, session_key, seq,
                            kind=AgentEventKind.ERROR,
                            text=f"Codex OAuth failed — {auth_err}",
                            model=model,
                        )
                        self._publisher.publish_run_done(request_id)
                        return

                    # Build prompt input fresh each retry attempt — temp files
                    # for images get rebuilt too (the inner loop may clear and
                    # rebuild after a recoverable session error).
                    self._cleanup_tmp_files(tmp_image_paths)
                    # TASK-288 observability: log the system_prompt value AS SENT
                    # (prepended to the user message by _build_prompt_input),
                    # paired with resume state — the only place the resume-gate
                    # decision is visible.
                    logger.info(
                        "Codex prompt: resume=%s system_prompt_sent=%s",
                        bool(resume_id),
                        f"{len(system_prompt)} chars" if system_prompt else "none",
                    )
                    prompt_input, tmp_image_paths = self._build_prompt_input(
                        message,
                        attachments,
                        # TASK-288: inject the system prompt on EVERY turn, resume
                        # included. Previously gated to fresh threads only, which
                        # meant the persona decayed to nothing on long-lived codex
                        # sessions (the steady state, since the thread is resumed
                        # every turn). The prompt is now byte-stable (temporal +
                        # response-style moved off it in core/base.py), so re-
                        # sending keeps persona alive. Codex prepends it to the
                        # user message rather than a cached system param, so this
                        # adds the (stable) persona to each turn's input.
                        system_prompt=system_prompt,
                    )

                    aborted = False
                    item_text: dict[str, str] = {}  # last seen text per agent_message item
                    item_buffers: dict[str, dict[str, Any]] = {}
                    final_usage: Any = None
                    session_persisted = bool(resume_id)

                    # Per-turn AbortController so chat.abort can interrupt the
                    # codex subprocess server-side via the SDK's signal plumbing.
                    from openai_codex_sdk import AbortController, AbortError

                    controller = AbortController()
                    if session_key:
                        self._session_queue.set_active_client(session_key, controller)

                    try:
                        # Open or resume thread (sync API)
                        thread_options = {
                            "model": model,
                            "working_directory": self._cwd,
                            "approval_policy": "never",
                            "sandbox_mode": "danger-full-access",
                            "skip_git_repo_check": True,
                        }
                        if resume_id:
                            thread = codex.resume_thread(resume_id, thread_options)
                        else:
                            thread = codex.start_thread(thread_options)

                        streamed = await thread.run_streamed(
                            prompt_input,
                            {"signal": controller.signal},
                        )

                        async for event in self._iter_events(streamed):
                            if cancel_event is not None and cancel_event.is_set():
                                logger.info(
                                    "chat.abort signalled, aborting Codex turn: session=%s request=%s",
                                    session_key, request_id,
                                )
                                aborted = True
                                try:
                                    controller.abort("chat.abort")
                                except Exception:
                                    logger.debug("controller.abort raised", exc_info=True)
                                break

                            (
                                seq,
                                persisted_now,
                                usage_now,
                            ) = await self._handle_event(
                                event,
                                bot_slug=bot_slug,
                                session_key=session_key,
                                request_id=request_id,
                                seq=seq,
                                text_parts=text_parts,
                                item_text=item_text,
                                model=model,
                                actual_model_ref=[actual_model],
                                resume_id=resume_id,
                                session_persisted=session_persisted,
                                item_buffers=item_buffers,
                                thread=thread,
                            )
                            if persisted_now:
                                session_persisted = True
                            if usage_now is not None:
                                final_usage = usage_now

                        # On a fresh thread, capture the id once the stream is
                        # done in case we never saw a typed thread.started event.
                        if (
                            not session_persisted
                            and not resume_id
                            and bot_slug
                            and getattr(thread, "id", None)
                        ):
                            await self._set_session(bot_slug, str(thread.id), model)
                            session_persisted = True

                        if not aborted and not self._already_sent_done(seq, item_buffers):
                            token_usage = self._extract_token_usage(
                                final_usage,
                                actual_model_or_default=actual_model,
                            )
                            token_usage = self._sanitize_context_usage(
                                await self._merge_model_info(
                                    token_usage, actual_model,
                                )
                            )
                            seq += 1
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=AgentEventKind.ASSISTANT_DONE,
                                text="".join(text_parts),
                                model=actual_model,
                                token_usage=token_usage,
                            )
                        elif aborted:
                            seq += 1
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=AgentEventKind.ASSISTANT_DONE,
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
                                kind=AgentEventKind.ASSISTANT_DONE,
                                text="".join(text_parts),
                                model=actual_model,
                            )
                        except Exception:
                            pass
                        try:
                            self._publisher.publish_run_done(request_id)
                        except Exception:
                            pass
                        try:
                            controller.abort("task_cancelled")
                        except Exception:
                            pass
                        raise

                    except AbortError:
                        # Triggered by controller.abort() — treat as a normal
                        # aborted turn so the user sees their partial response.
                        logger.info(
                            "Codex turn aborted: request_id=%s session=%s",
                            request_id, session_key,
                        )
                        aborted = True
                        seq += 1
                        self._publish_event(
                            request_id, session_key, seq,
                            kind=AgentEventKind.ASSISTANT_DONE,
                            text="".join(text_parts),
                            model=actual_model,
                        )
                        break

                    except Exception as e:
                        # Auth failure → publish ERROR, drop SDK handle, no retry.
                        if _is_auth_failure(e):
                            logger.warning(
                                "Codex auth failure for %s: %s — resetting SDK handle",
                                request_id, e,
                            )
                            self._supervisor_teardown()
                            seq += 1
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=AgentEventKind.ERROR,
                                text="Codex OAuth failed — re-run codex login on echo",
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
                        if session_key:
                            current = self._session_queue.get_active_client(session_key)
                            if current is controller:
                                self._session_queue.pop_active_client(session_key)

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
                    kind=AgentEventKind.ERROR,
                    text=str(e),
                    model=model,
                )
                self._publisher.publish_run_done(request_id)
            finally:
                self._cleanup_tmp_files(tmp_image_paths)
                # Drop the per-run trigger_message_id mapping so we don't leak.
                self._trigger_message_ids.pop(request_id, None)
                await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)

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
                    # SESSION_RESET unified event so any active frontend SSE
                    # consumer for this bot clears its visible buffer.
                    # TASK-249.
                    self._publish_session_reset_unified(
                        target, session_key or target, had_session=cleared,
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
                controller = self._session_queue.pop_active_client(session_key)
                turn_interrupted = False
                if controller is not None:
                    try:
                        controller.abort("chat.abort")
                        turn_interrupted = True
                    except Exception:
                        logger.debug(
                            "controller.abort raised on chat.abort for session %s",
                            session_key, exc_info=True,
                        )
                cancelled = self._session_queue.cancel_active(session_key)
                detail_parts: list[str] = []
                if cancelled:
                    detail_parts.append("task_cancelled")
                if turn_interrupted:
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
