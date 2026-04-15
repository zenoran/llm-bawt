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
