from __future__ import annotations
import asyncio
import json
import logging

from .events import OpenClawEventKind
from .ingest import EventIngestPipeline
from .metrics import get_metrics
from .publisher import COMMANDS_STREAM, RedisPublisher
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

    async def start(self) -> None:
        """Start the bridge: connect WS, begin consuming events, and listen for commands."""
        self._ws_client.on_event(self._on_raw_event)
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

        if not request_id or not message:
            logger.warning("Invalid send command: missing request_id or message")
            await async_redis.xack(COMMANDS_STREAM, "bridge", msg_id)
            return

        logger.info(
            "Handling send command: request_id=%s session=%s msg=%.60s...",
            request_id, session_key, message,
        )

        try:
            async for raw_event in self._ws_client.send_and_stream(
                session_key, message, timeout=600,
            ):
                # Parse through ingest pipeline to get structured event
                event = self._ingest.parse(raw_event, session_key)
                if event is not None:
                    self._publisher.publish_run_event(request_id, event)

            self._publisher.publish_run_done(request_id)
            get_metrics().incr("openclaw.commands_completed")
            logger.info("Send command completed: request_id=%s", request_id)

        except Exception as e:
            logger.exception("Send command failed: request_id=%s", request_id)
            # Publish error event
            from .events import OpenClawEvent, OpenClawEventKind
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
            res = await self._ws_client._request(method, params)
            payload = res.get("payload", {})
            self._publisher.publish_rpc_result(request_id, {"ok": True, "payload": payload})
        except Exception as e:
            logger.warning("RPC command failed: method=%s error=%s", method, e)
            self._publisher.publish_rpc_result(request_id, {"ok": False, "error": str(e)})
        finally:
            await async_redis.xack(COMMANDS_STREAM, "bridge", msg_id)

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
        return self._session_to_bot.get(session_key)

    @property
    def connected(self) -> bool:
        return self._ws_client.connected
