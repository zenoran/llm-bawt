"""Per-session message serialization and active-task tracking.

Shared between OpenClaw and Claude Code bridges to ensure consistent
behavior: only one send runs at a time per session, and abort support
works uniformly.

Inspired by the Claude Code CLI's ``messageQueueManager`` which queues
messages arriving mid-turn and processes them sequentially after the
current turn completes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


class SessionQueue:
    """Serialize sends per session and track active tasks for abort.

    Both the OpenClaw gateway and Claude Code SDK process one turn at a
    time per session.  Sending concurrent messages to the same session
    causes state corruption (run-queue overwrites, stale history fallback,
    tangled SDK session state).

    Usage in a bridge command listener::

        queue = SessionQueue()

        # In _command_listener, when dispatching a chat.send:
        task = asyncio.create_task(self._handle_send(fields, ...))
        queue.set_active_task(session_key, task)

        # Inside _handle_send, wrap the actual work:
        async with queue.lock(session_key):
            # ... do the send ...

        # For abort / chat.abort RPC:
        queue.cancel_active(session_key)
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._active_tasks: dict[str, asyncio.Task] = {}
        # Cooperative cancel signals — handlers check this between SDK
        # messages so an abort takes effect without waiting for the next
        # `await` point to fire CancelledError.
        self._cancel_events: dict[str, asyncio.Event] = {}
        # The active SDK message stream (an async generator) per session.
        # Only used by the Claude Code bridge today; the OpenClaw bridge's
        # cooperative cancel happens via `WSClient.cancel_session` instead.
        # Tracking it here lets `chat.abort` call `aclose()` on the
        # generator, which propagates GeneratorExit through the SDK's
        # `try/finally` and forces the underlying `claude` CLI subprocess
        # to be SIGTERM/SIGKILL'd.  Without this, `task.cancel()` only
        # raises `CancelledError` at the next `await` point, which lets
        # the SDK keep running mid-tool-call indefinitely.
        self._active_streams: dict[str, AsyncIterator[Any]] = {}

    # -- Lock management --------------------------------------------------

    def lock(self, session_key: str) -> asyncio.Lock:
        """Get or create the asyncio.Lock for a session.

        Returns the lock object for use with ``async with``.
        """
        existing = self._locks.get(session_key)
        if existing is None:
            existing = asyncio.Lock()
            self._locks[session_key] = existing
        return existing

    def is_busy(self, session_key: str) -> bool:
        """Return True if a session has an active send (lock is held)."""
        lk = self._locks.get(session_key)
        return lk is not None and lk.locked()

    # -- Active task tracking (for abort) ---------------------------------

    def set_active_task(self, session_key: str, task: asyncio.Task) -> None:
        """Register *task* as the active send for *session_key*.

        When the task completes (success or failure) it is automatically
        removed from the registry.
        """
        self._active_tasks[session_key] = task
        task.add_done_callback(
            lambda _t, sk=session_key: self._active_tasks.pop(sk, None)
        )

    def cancel_active(self, session_key: str) -> bool:
        """Cancel the active task for *session_key*.

        Returns True if a running task was found and cancelled.
        """
        task = self._active_tasks.get(session_key)
        if task and not task.done():
            task.cancel()
            logger.info("Cancelled active task for session: %s", session_key)
            return True
        return False

    def has_active_task(self, session_key: str) -> bool:
        """Return True if *session_key* has a running (non-done) task."""
        task = self._active_tasks.get(session_key)
        return task is not None and not task.done()

    # -- Cooperative cancel events ---------------------------------------

    def cancel_event(self, session_key: str) -> asyncio.Event:
        """Get or create the cooperative cancel event for *session_key*.

        Handlers should check ``event.is_set()`` between SDK messages and
        bail out cleanly when set. The event survives across handler
        invocations until explicitly cleared with ``clear_cancel_event``.
        """
        existing = self._cancel_events.get(session_key)
        if existing is None:
            existing = asyncio.Event()
            self._cancel_events[session_key] = existing
        return existing

    def signal_cancel(self, session_key: str) -> bool:
        """Set the cooperative cancel event for *session_key*.

        Returns True if a fresh signal was raised (event went from clear
        to set), False if the event was already set or no event existed.
        Creates the event if it doesn't yet exist so a chat.abort that
        races a slow chat.send still arms the flag for the next iteration.
        """
        event = self.cancel_event(session_key)
        already_set = event.is_set()
        event.set()
        if not already_set:
            logger.info("Cooperative cancel signalled: %s", session_key)
        return not already_set

    def clear_cancel_event(self, session_key: str) -> None:
        """Drop the cancel event for *session_key*.

        Call this from the handler's `finally` block so the next send
        starts with a fresh, un-set event.
        """
        self._cancel_events.pop(session_key, None)

    # -- Active SDK stream tracking (for forced subprocess kill) ----------

    def set_active_stream(
        self, session_key: str, stream: AsyncIterator[Any]
    ) -> None:
        """Register the active SDK message stream for *session_key*.

        The stream is expected to be an async generator (e.g. the one
        returned by `claude_agent_sdk.query()`). On abort, the bridge
        will call `aclose()` on this generator to force the underlying
        subprocess to terminate even mid-tool-call.
        """
        self._active_streams[session_key] = stream

    def pop_active_stream(self, session_key: str) -> AsyncIterator[Any] | None:
        """Remove and return the active SDK stream for *session_key*."""
        return self._active_streams.pop(session_key, None)

    def get_active_stream(
        self, session_key: str
    ) -> AsyncIterator[Any] | None:
        """Return the active SDK stream for *session_key* without removing it."""
        return self._active_streams.get(session_key)
