from __future__ import annotations
import asyncio
import logging

from .events import OpenClawEventKind
from .ingest import EventIngestPipeline
from .metrics import get_metrics
from .publisher import RedisPublisher
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

    async def start(self) -> None:
        """Start the bridge: connect WS and begin consuming events."""
        self._ws_client.on_event(self._on_raw_event)
        await self._ws_client.connect()
        logger.info("SessionBridge started (sessions=%s)", list(self._ws_client.subscribed_sessions))

    async def stop(self) -> None:
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
        bot_id = self._session_to_bot.get(session_key)
        if not bot_id:
            normalized = EventIngestPipeline._normalize_session_key(session_key)
            bot_id = self._session_to_bot.get(normalized)
        return bot_id

    @property
    def connected(self) -> bool:
        return self._ws_client.connected
