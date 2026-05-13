from __future__ import annotations
import asyncio
import logging
from typing import Any, Callable

from .agent_bridge_events import AgentEventKind
from .agent_bridge_fanout import FanoutHub
from .openclaw_ingest import EventIngestPipeline
from .agent_bridge_store import EventStore
from .openclaw_ws import OpenClawWsClient

logger = logging.getLogger(__name__)


class SessionBridge:
    def __init__(
        self,
        ws_client: OpenClawWsClient,
        ingest: EventIngestPipeline,
        store: EventStore,
        fanout: FanoutHub,
        *,
        history_sink: Callable[[str, str, str], None] | None = None,
        session_to_bot: dict[str, str] | None = None,
    ) -> None:
        self._ws_client = ws_client
        self._ingest = ingest
        self._store = store
        self._fanout = fanout
        self._run_buffers: dict[str, list[str]] = {}  # run_id -> text deltas
        self._run_tool_calls: dict[str, list[dict]] = {}  # run_id -> tool calls
        # Callback(bot_id, role, content) to persist messages to chat history
        self._history_sink = history_sink
        # Mapping of normalized session_key -> bot_id for history routing
        self._session_to_bot: dict[str, str] = session_to_bot or {}
        # Run IDs initiated via the API (already persisted by _stream_via_bridge)
        self._api_run_ids: set[str] = set()

    async def start(self) -> None:
        """Start the bridge: connect WS, subscribe, begin consuming events."""
        self._ws_client.on_event(self._on_raw_event)
        await self._ws_client.connect()
        logger.info("SessionBridge started")

    async def stop(self) -> None:
        """Graceful shutdown."""
        await self._ws_client.disconnect()
        logger.info("SessionBridge stopped")

    async def send_user_message(self, session_key: str, text: str) -> str:
        """Send a user message via WS and return the run_id."""
        return await self._ws_client.send_user_message(session_key, text)

    async def _on_raw_event(self, raw: dict) -> None:
        """Process a raw WS message: parse -> store -> update run state -> fanout."""
        session_key = raw.get("session_key", "")
        if not session_key:
            # Try to get from subscribed sessions
            sessions = self._ws_client.subscribed_sessions
            session_key = next(iter(sessions), "unknown")

        event = self._ingest.parse(raw, session_key)
        if event is None:
            return

        # Store (idempotent)
        stored = self._store.store(event)
        if not stored:
            logger.debug("Duplicate event dropped: %s", event.event_id)
            return

        # Update session cursor
        if event.db_id:
            self._store.update_session_cursor(event.session_key, event.db_id)

        # Run state management
        if event.kind == AgentEventKind.RUN_STARTED and event.run_id:
            self._store.create_run(event.run_id, event.session_key, event.model, event.origin)
            self._run_buffers[event.run_id] = []
            self._run_tool_calls[event.run_id] = []

            # For non-API runs, fetch the user message from gateway chat history
            if event.run_id not in self._api_run_ids and self._history_sink:
                asyncio.create_task(self._fetch_and_persist_user_message(event.session_key))

        elif event.kind == AgentEventKind.ASSISTANT_DELTA and event.run_id:
            if event.run_id in self._run_buffers:
                self._run_buffers[event.run_id].append(event.text or "")

        elif event.kind == AgentEventKind.TOOL_START and event.run_id:
            if event.run_id in self._run_tool_calls:
                self._run_tool_calls[event.run_id].append({
                    "name": event.tool_name,
                    "arguments": event.tool_arguments,
                })

        elif event.kind == AgentEventKind.TOOL_END and event.run_id:
            if event.run_id in self._run_tool_calls and self._run_tool_calls[event.run_id]:
                self._run_tool_calls[event.run_id][-1]["result"] = event.tool_result

        elif event.kind == AgentEventKind.ASSISTANT_DONE:
            # Persist complete assistant message to chat history
            # Skip if this run was initiated via the API (already persisted by _stream_via_bridge)
            is_api_run = event.run_id and event.run_id in self._api_run_ids
            if is_api_run:
                # Discard here (not at RUN_COMPLETED) because ASSISTANT_DONE
                # can arrive after RUN_COMPLETED
                self._api_run_ids.discard(event.run_id)
            else:
                bot_id = self._resolve_bot_id(event.session_key)
                if self._history_sink and event.text and bot_id:
                    try:
                        self._history_sink(bot_id, "assistant", event.text)
                    except Exception:
                        logger.exception("Failed to persist assistant message to chat history")

        elif event.kind == AgentEventKind.USER_MESSAGE:
            # Persist async user messages (cron triggers, external sends)
            bot_id = self._resolve_bot_id(event.session_key)
            if self._history_sink and event.text and bot_id:
                try:
                    self._history_sink(bot_id, "user", event.text)
                except Exception:
                    logger.exception("Failed to persist user message to chat history")

        elif event.kind == AgentEventKind.RUN_COMPLETED and event.run_id:
            full_text = "".join(self._run_buffers.pop(event.run_id, []))
            if not full_text:
                full_text = self._store.assemble_run_text(event.run_id)
            tool_calls = self._run_tool_calls.pop(event.run_id, [])
            self._store.complete_run(event.run_id, full_text, tool_calls or None)
            logger.info(
                "OpenClaw run completed: run_id=%s text_len=%d tools=%d",
                event.run_id, len(full_text), len(tool_calls),
            )

        # Fanout to UI subscribers
        self._fanout.broadcast(event)

    async def _fetch_and_persist_user_message(self, session_key: str) -> None:
        """Fetch the latest user message from gateway history and persist it."""
        try:
            messages = await self._ws_client.get_chat_history(session_key, limit=5)
            # Find the most recent user message
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content") or ""
                    if isinstance(content, list):
                        # content: [{type: "text", text: "..."}]
                        parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("text"):
                                parts.append(item["text"])
                        content = "".join(parts)
                    if content and self._ingest.should_drop_content(content):
                        logger.debug("Dropping polled user message matching content filter: %.80s…", content)
                        break
                    bot_id = self._resolve_bot_id(session_key)
                    if content and self._history_sink and bot_id:
                        self._history_sink(bot_id, "user", content)
                    break
        except Exception:
            logger.debug("Could not fetch user message from chat.history")

    def _resolve_bot_id(self, session_key: str) -> str | None:
        """Resolve session_key to bot_id via the session_to_bot mapping."""
        if not self._session_to_bot:
            return None
        bot_id = self._session_to_bot.get(session_key)
        if not bot_id:
            logger.debug("No bot mapped for session_key=%s", session_key)
        return bot_id

    @property
    def connected(self) -> bool:
        return self._ws_client.connected
