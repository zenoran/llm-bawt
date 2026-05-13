from __future__ import annotations
import asyncio
import logging
from typing import AsyncIterator

from .agent_bridge_events import AgentEvent
from .agent_bridge_store import EventStore

logger = logging.getLogger(__name__)


class FanoutHub:
    def __init__(self, store: EventStore) -> None:
        self._store = store
        self._subscribers: dict[str, list[asyncio.Queue[AgentEvent | None]]] = {}

    async def subscribe(
        self, session_key: str, *, since_event_id: int | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Subscribe to live events. If since_event_id provided, replays gap first."""
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue(maxsize=1000)

        if session_key not in self._subscribers:
            self._subscribers[session_key] = []
        self._subscribers[session_key].append(queue)

        try:
            # Replay gap if requested
            if since_event_id is not None:
                gap_events = self._store.get_events(session_key, since_id=since_event_id)
                for event in gap_events:
                    yield event

            # Stream live events
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            if session_key in self._subscribers:
                try:
                    self._subscribers[session_key].remove(queue)
                except ValueError:
                    pass
                if not self._subscribers[session_key]:
                    del self._subscribers[session_key]

    def broadcast(self, event: AgentEvent) -> None:
        """Push to all subscribers of this session."""
        queues = self._subscribers.get(event.session_key, [])
        for queue in queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "FanoutHub subscriber queue full for session %s, dropping event",
                    event.session_key,
                )

    def close_session(self, session_key: str) -> None:
        """Signal all subscribers for a session to stop."""
        queues = self._subscribers.get(session_key, [])
        for queue in queues:
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    @property
    def subscriber_count(self) -> int:
        return sum(len(q) for q in self._subscribers.values())
