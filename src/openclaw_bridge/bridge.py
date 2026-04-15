from __future__ import annotations
import asyncio
import uuid
import json
import logging
from datetime import datetime

from .events import OpenClawEvent, OpenClawEventKind
from .ingest import EventIngestPipeline
from .metrics import get_metrics
from .publisher import COMMANDS_STREAM, RedisPublisher
from .session_queue import SessionQueue
from .store import EventStore
from .ws_client import OpenClawWsClient

logger = logging.getLogger(__name__)


class SessionBridge:
    def __init__(
        self,
        ws_client: OpenClawWsClient,
        ingest: EventIngestPipeline,
        store: EventStore,
        publisher: RedisPublisher,
        *,
        session_to_bot: dict[str, str] | None = None,
    ) -> None:
        self._ws_client = ws_client
        self._ingest = ingest
        self._store = store
        self._publisher = publisher
        self._run_buffers: dict[str, list[str]] = {}  # run_id -> text deltas
        self._run_tool_calls: dict[str, list[dict]] = {}  # run_id -> tool calls
        self._session_to_bot: dict[str, str] = session_to_bot or {}
        self._command_task: asyncio.Task | None = None
        # Shared session queue — serializes sends per session and tracks
        # active tasks for abort support.
        self._session_queue = SessionQueue()
        # When the live run stream ends without assistant text, poll history
        # briefly to recover replies produced by upstream provider fallback.
        self._history_reply_timeout_s = 45.0
        self._history_reply_poll_interval_s = 1.0

    async def start(self) -> None:
        """Start the bridge: connect WS, begin consuming events, and listen for commands."""
        # Passive event feed: only publishes tool events to the unified SSE
        # stream so the frontend sees real-time activity without an active
        # chat request. Does NOT write history or manage run state.
        # self._ws_client.on_event(self._on_passive_event)  # disabled — use active path only
        await self._ws_client.connect()
        self._command_task = asyncio.create_task(self._command_listener())
        logger.info("SessionBridge started (sessions=%s)", list(self._ws_client.subscribed_sessions))

    async def stop(self) -> None:
        if self._command_task:
            self._command_task.cancel()
            try:
                await self._command_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._ws_client.disconnect()
        self._publisher.close()
        logger.info("SessionBridge stopped")

    async def run_forever(self) -> None:
        """Block forever, keeping the bridge alive. For standalone mode."""
        await self.start()
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def _command_listener(self) -> None:
        """Read send commands from Redis and dispatch them via WS."""
        import redis.asyncio as aioredis

        redis_url = self._publisher._redis.connection_pool.connection_kwargs.get("db", 0)
        # Reuse the same Redis URL by extracting from the sync publisher
        # We need an async client for blocking reads
        try:
            # Build URL from the sync publisher's connection
            conn_kwargs = self._publisher._redis.connection_pool.connection_kwargs
            host = conn_kwargs.get("host", "localhost")
            port = conn_kwargs.get("port", 6379)
            db = conn_kwargs.get("db", 0)
            async_redis = aioredis.Redis(host=host, port=port, db=db, decode_responses=True)
            await async_redis.ping()
        except Exception as e:
            logger.error("Command listener: cannot connect to Redis: %s", e)
            return

        # Create consumer group
        try:
            await async_redis.xgroup_create(
                COMMANDS_STREAM, "bridge", id="0", mkstream=True
            )
        except Exception:
            pass  # group already exists

        logger.info("Command listener started on %s", COMMANDS_STREAM)

        while True:
            try:
                results = await async_redis.xreadgroup(
                    "bridge", "worker-0",
                    {COMMANDS_STREAM: ">"},
                    count=1,
                    block=5000,
                )
                if not results:
                    continue

                for _stream, messages in results:
                    for msg_id, fields in messages:
                        action = fields.get("action", "")
                        session_key = fields.get("session_key", "")

                        # Skip commands for sessions this bridge doesn't own
                        if session_key and not self._resolve_bot_id(session_key):
                            await async_redis.xack(COMMANDS_STREAM, "bridge", msg_id)
                            continue

                        if action == "chat.send":
                            asyncio.create_task(
                                self._handle_send_command(fields, msg_id, async_redis)
                            )
                        elif action == "rpc.call":
                            asyncio.create_task(
                                self._handle_rpc_command(fields, msg_id, async_redis)
                            )
                        else:
                            logger.warning("Unknown command action: %s", action)
                            await async_redis.xack(COMMANDS_STREAM, "bridge", msg_id)

            except asyncio.CancelledError:
                await async_redis.aclose()
                raise
            except Exception:
                logger.exception("Command listener error")
                await asyncio.sleep(2)

    async def _handle_send_command(
        self, fields: dict, msg_id: str, async_redis
    ) -> None:
        """Handle a chat.send command: send via WS, stream events back via Redis."""
        request_id = fields.get("request_id", "")
        session_key = fields.get("session_key", "main")
        message = fields.get("message", "")
        attachments_raw = fields.get("attachments", "")
        attachments: list = []
        if attachments_raw:
            try:
                import json as _json
                attachments = _json.loads(attachments_raw)
            except Exception:
                logger.warning("Failed to parse attachments for request_id=%s", request_id)

        if not request_id or not message:
            logger.warning("Invalid send command: missing request_id or message")
            await async_redis.xack(COMMANDS_STREAM, "bridge", msg_id)
            return

        logger.info(
            "Handling send command: request_id=%s session=%s msg=%.60s...",
            request_id, session_key, message,
        )

        async with self._session_queue.lock(session_key):
            try:
                # Clear any previous cancel signal before starting a new send
                self._ws_client.clear_session_cancel(session_key)
                bot_id = self._resolve_bot_id(session_key)
                # Resolve user_id — for now use "default" as bridge doesn't track per-user
                unified_user_id = "default"
                saw_assistant_text = False

                async for raw_event in self._ws_client.send_and_stream(
                    session_key, message, attachments=attachments,
                ):
                    # Parse through ingest pipeline to get structured event
                    event = self._ingest.parse(raw_event, session_key)
                    if event is not None:
                        if event.kind == OpenClawEventKind.ASSISTANT_DELTA and event.text:
                            saw_assistant_text = True

                        # If ASSISTANT_DONE carries inline text but no deltas
                        # were streamed, emit a synthetic delta so the consumer
                        # sees the text.  Do NOT fall back to chat history —
                        # that risks replaying a stale response from a prior turn.
                        if (
                            event.kind == OpenClawEventKind.ASSISTANT_DONE
                            and not saw_assistant_text
                            and event.text
                        ):
                            saw_assistant_text = True
                            self._publisher.publish_run_event(
                                request_id,
                                OpenClawEvent(
                                    event_id=f"{event.event_id}:final-text",
                                    session_key=event.session_key,
                                    run_id=event.run_id,
                                    kind=OpenClawEventKind.ASSISTANT_DELTA,
                                    origin=event.origin,
                                    text=event.text,
                                    model=event.model,
                                    seq=event.seq,
                                    timestamp=event.timestamp,
                                    raw=event.raw,
                                ),
                            )

                        self._publisher.publish_run_event(request_id, event)

                        # Tool events are published to the unified stream by
                        # chat_streaming.py (with proper call_id and user_id).
                        # Do not duplicate here.

                if not saw_assistant_text:
                    history_text = await self._wait_for_history_reply(session_key)
                    if history_text:
                        saw_assistant_text = True
                        logger.info(
                            "Recovered assistant text from chat.history after empty stream: "
                            "request_id=%s session=%s chars=%d",
                            request_id, session_key, len(history_text),
                        )
                        self._publisher.publish_run_event(
                            request_id,
                            OpenClawEvent(
                                event_id=f"history_{request_id}",
                                session_key=session_key,
                                run_id=None,
                                kind=OpenClawEventKind.ASSISTANT_DELTA,
                                origin="bridge",
                                text=history_text,
                            ),
                        )

                self._publisher.publish_run_done(request_id)
                get_metrics().incr("openclaw.commands_completed")
                logger.info("Send command completed: request_id=%s", request_id)

            except Exception as e:
                logger.exception("Send command failed: request_id=%s", request_id)
                # Publish error event
                err_event = OpenClawEvent(
                    event_id=f"err_{request_id}",
                    session_key=session_key,
                    run_id=None,
                    kind=OpenClawEventKind.ERROR,
                    origin="bridge",
                    text=str(e),
                )
                self._publisher.publish_run_event(request_id, err_event)
                self._publisher.publish_run_done(request_id)
            finally:
                await async_redis.xack(COMMANDS_STREAM, "bridge", msg_id)

    async def _handle_rpc_command(
        self, fields: dict, msg_id: str, async_redis
    ) -> None:
        """Handle an rpc.call command: call WS RPC, publish result via Redis."""
        import json as _json

        request_id = fields.get("request_id", "")
        method = fields.get("method", "")
        params_raw = fields.get("params", "{}")

        try:
            params = _json.loads(params_raw) if isinstance(params_raw, str) else params_raw
        except (_json.JSONDecodeError, TypeError):
            params = {}

        if not request_id or not method:
            logger.warning("Invalid rpc.call command: missing request_id or method")
            await async_redis.xack(COMMANDS_STREAM, "bridge", msg_id)
            return

        logger.info("Handling RPC command: method=%s request_id=%s", method, request_id)

        try:
            # If this is a chat.abort, signal the active send_and_stream to stop
            # so the session lock is released and the next send can proceed.
            if method == "chat.abort":
                params_session = params.get("sessionKey", "")
                if params_session:
                    self._ws_client.cancel_session(params_session)

            res = await self._ws_client._request(method, params)
            payload = res.get("payload", {})
            self._publisher.publish_rpc_result(request_id, {"ok": True, "payload": payload})
        except Exception as e:
            logger.warning("RPC command failed: method=%s error=%s", method, e)
            self._publisher.publish_rpc_result(request_id, {"ok": False, "error": str(e)})
        finally:
            await async_redis.xack(COMMANDS_STREAM, "bridge", msg_id)

    async def _wait_for_history_reply(self, session_key: str) -> str | None:
        """Poll chat.history for a fresh assistant reply after empty streams.

        Some gateway/provider fallback paths end the original run without
        emitting assistant deltas, then continue the conversation internally
        and only persist the final reply to session history. In that case we
        recover the most recent non-empty assistant message that follows the
        latest non-empty user message in the session.
        """
        deadline = asyncio.get_event_loop().time() + self._history_reply_timeout_s
        while asyncio.get_event_loop().time() < deadline:
            try:
                messages = await self._ws_client.get_chat_history(session_key, limit=20)
            except Exception:
                logger.debug("chat.history fallback failed for session=%s", session_key, exc_info=True)
                return None

            reply = self._extract_latest_history_reply(messages)
            if reply:
                return reply

            await asyncio.sleep(self._history_reply_poll_interval_s)

        return None

    @classmethod
    def _extract_latest_history_reply(cls, messages: list[dict] | None) -> str | None:
        """Return the latest non-empty assistant reply after the latest user message."""
        if not isinstance(messages, list) or not messages:
            return None

        ordered = sorted(
            enumerate(messages),
            key=lambda item: (cls._message_sort_key(item[1], item[0]), item[0]),
        )
        normalized = [message for _, message in ordered]

        latest_user_idx: int | None = None
        for idx in range(len(normalized) - 1, -1, -1):
            msg = normalized[idx]
            if cls._message_role(msg) == "user" and cls._extract_message_text(msg):
                latest_user_idx = idx
                break

        if latest_user_idx is None:
            return None

        reply_text: str | None = None
        for msg in normalized[latest_user_idx + 1:]:
            if cls._message_role(msg) != "assistant":
                continue
            text = cls._extract_message_text(msg)
            if text:
                reply_text = text

        return reply_text

    @staticmethod
    def _message_role(message: dict) -> str:
        role = message.get("role")
        if isinstance(role, str):
            return role
        inner = message.get("message")
        if isinstance(inner, dict) and isinstance(inner.get("role"), str):
            return inner["role"]
        return ""

    @staticmethod
    def _message_sort_key(message: dict, fallback_index: int) -> tuple[int, float]:
        ts = message.get("timestamp")
        if ts is None and isinstance(message.get("message"), dict):
            ts = message["message"].get("timestamp")
        if isinstance(ts, (int, float)):
            return (0, float(ts))
        if isinstance(ts, str):
            try:
                return (0, datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
            except ValueError:
                pass
        return (1, float(fallback_index))

    @staticmethod
    def _extract_message_text(message: dict) -> str:
        if not isinstance(message, dict):
            return ""

        if isinstance(message.get("text"), str):
            return message["text"]

        inner = message.get("message")
        if isinstance(inner, dict):
            message = inner

        content = message.get("content")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue

            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            if isinstance(text, dict):
                value = text.get("value") or text.get("text")
                if isinstance(value, str) and value:
                    parts.append(value)
                    continue

            value = item.get("value")
            if isinstance(value, str) and value:
                parts.append(value)

        return "".join(parts)

    # Per-run tool call counters for generating stable iteration numbers
    _passive_tool_counters: dict[str, int] = {}
    # Stack of call_ids per run — tool calls can nest (e.g. Agent containing
    # Grep/Read), so a single slot loses the outer call_id.
    _passive_call_id_stacks: dict[str, list[str]] = {}

    async def _on_passive_event(self, raw: dict) -> None:
        """Lightweight passive handler: only publishes tool events to the
        unified SSE stream. No history writes, no Postgres, no run state."""
        import time as _time

        session_key = raw.get("session_key", "")
        if not session_key:
            sessions = self._ws_client.subscribed_sessions
            session_key = next(iter(sessions), "unknown")

        event = self._ingest.parse(raw, session_key)
        if event is None:
            return

        bot_id = self._resolve_bot_id(event.session_key)
        logger.info("PASSIVE: session=%s kind=%s bot=%s run=%s locked=%s",
                     event.session_key, event.kind.value if event.kind else "?",
                     bot_id, event.run_id,
                     self._session_queue.is_busy(event.session_key))
        if not bot_id:
            return

        # Default user_id — could be parsed from session key if needed
        user_id = "nick"
        run_id = event.run_id or "unknown"
        ts = event.timestamp.timestamp() if event.timestamp else _time.time()

        # Skip if this session has an active chat request streaming —
        # the active path (chat_streaming.py) already publishes tool events.
        if self._session_queue.is_busy(event.session_key):
            return

        if event.kind == OpenClawEventKind.TOOL_START:
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            self._passive_tool_counters.setdefault(run_id, 0)
            self._passive_tool_counters[run_id] += 1
            iteration = self._passive_tool_counters[run_id]
            self._passive_call_id_stacks.setdefault(run_id, []).append(call_id)

            self._publisher.publish_unified_event(bot_id, user_id, {
                "_type": "tool_event",
                "event": "tool_start",
                "turn_id": run_id,
                "bot_id": bot_id,
                "user_id": user_id,
                "tool_name": event.tool_name or "unknown",
                "arguments": event.tool_arguments or {},
                "call_id": call_id,
                "iteration": iteration,
                "ts": ts,
            })

        elif event.kind == OpenClawEventKind.TOOL_END:
            stack = self._passive_call_id_stacks.get(run_id, [])
            call_id = stack.pop() if stack else f"call_{uuid.uuid4().hex[:8]}"
            iteration = self._passive_tool_counters.get(run_id, 1)
            self._publisher.publish_unified_event(bot_id, user_id, {
                "_type": "tool_event",
                "event": "tool_end",
                "turn_id": run_id,
                "bot_id": bot_id,
                "user_id": user_id,
                "tool_name": event.tool_name or "unknown",
                "call_id": call_id,
                "result": str(event.tool_result or "")[:2000],
                "iteration": iteration,
                "ts": ts,
            })

        elif event.kind == OpenClawEventKind.RUN_COMPLETED:
            self._passive_tool_counters.pop(run_id, None)
            self._passive_call_id_stacks.pop(run_id, None)
            self._publisher.publish_unified_event(bot_id, user_id, {
                "_type": "turn_complete",
                "turn_id": run_id,
                "bot_id": bot_id,
                "user_id": user_id,
                "status": "completed",
                "ts": ts,
            })

    async def _on_raw_event(self, raw: dict) -> None:
        """Process a raw WS message: parse -> store -> update run state -> publish to Redis."""
        metrics = get_metrics()
        metrics.incr("openclaw.ws_messages_received")

        session_key = raw.get("session_key", "")
        if not session_key:
            sessions = self._ws_client.subscribed_sessions
            session_key = next(iter(sessions), "unknown")

        event = self._ingest.parse(raw, session_key)
        if event is None:
            metrics.incr("openclaw.events_dropped", reason="parse_failed")
            return

        metrics.incr("openclaw.events_parsed", kind=event.kind.value, session=session_key)

        # Store to Postgres (idempotent)
        stored = self._store.store(event)
        if not stored:
            return

        # Update session cursor
        if event.db_id:
            self._store.update_session_cursor(event.session_key, event.db_id)

        # Run state management
        if event.kind == OpenClawEventKind.RUN_STARTED and event.run_id:
            self._store.create_run(event.run_id, event.session_key, event.model, event.origin)
            self._run_buffers[event.run_id] = []
            self._run_tool_calls[event.run_id] = []

        elif event.kind == OpenClawEventKind.ASSISTANT_DELTA and event.run_id:
            if event.run_id in self._run_buffers:
                self._run_buffers[event.run_id].append(event.text or "")

        elif event.kind == OpenClawEventKind.TOOL_START and event.run_id:
            if event.run_id in self._run_tool_calls:
                self._run_tool_calls[event.run_id].append({
                    "name": event.tool_name,
                    "arguments": event.tool_arguments,
                })

        elif event.kind == OpenClawEventKind.TOOL_END and event.run_id:
            if event.run_id in self._run_tool_calls and self._run_tool_calls[event.run_id]:
                self._run_tool_calls[event.run_id][-1]["result"] = event.tool_result

        elif event.kind == OpenClawEventKind.ASSISTANT_DONE:
            # Publish history-persist command for the main app
            bot_id = self._resolve_bot_id(event.session_key)
            if bot_id and event.text:
                self._publisher.publish_history(bot_id, "assistant", event.text)

        elif event.kind == OpenClawEventKind.USER_MESSAGE:
            if event.text and self._ingest.should_drop_content(event.text):
                logger.debug("Dropping user message matching content filter: %.80s…", event.text)
            else:
                bot_id = self._resolve_bot_id(event.session_key)
                if bot_id and event.text:
                    self._publisher.publish_history(bot_id, "user", event.text)

        elif event.kind == OpenClawEventKind.RUN_COMPLETED and event.run_id:
            full_text = "".join(self._run_buffers.pop(event.run_id, []))
            if not full_text:
                full_text = self._store.assemble_run_text(event.run_id)
            tool_calls = self._run_tool_calls.pop(event.run_id, [])
            self._store.complete_run(event.run_id, full_text, tool_calls or None)
            metrics.incr("openclaw.runs_completed", session=event.session_key)
            metrics.gauge("openclaw.last_run_text_len", float(len(full_text)), session=event.session_key)
            metrics.gauge("openclaw.last_run_tool_calls", float(len(tool_calls)), session=event.session_key)
            logger.info(
                "OpenClaw run completed: run_id=%s text_len=%d tools=%d",
                event.run_id, len(full_text), len(tool_calls),
            )

        # Publish to Redis Stream for consumers
        self._publisher.publish_event(event)

    def _resolve_bot_id(self, session_key: str) -> str | None:
        if not self._session_to_bot:
            return None
        # Direct match first
        bot_id = self._session_to_bot.get(session_key)
        if bot_id:
            return bot_id
        # Sub-agent sessions: "agent:{bot}:subagent:{uuid}" -> resolve to parent bot
        parts = session_key.split(":")
        if len(parts) >= 3 and parts[0] == "agent" and "subagent" in parts:
            parent_key = f"agent:{parts[1]}:main"
            return self._session_to_bot.get(parent_key)
        return None

    @property
    def connected(self) -> bool:
        return self._ws_client.connected
