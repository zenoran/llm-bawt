"""Shared bridge lifecycle for per-turn changed-file manifests (TASK-661)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .changed_files import ChangedFileTracker
from .events import AgentEvent, AgentEventKind, synthesize_event_id

logger = logging.getLogger(__name__)


class ChangedFileLifecycleMixin:
    """Arm, finalize, and publish changed-file manifests for any bridge.

    Concrete bridges provide ``self._publisher`` and may provide
    ``self._backend_name`` plus ``self._trigger_message_ids``. Keeping the git
    lifecycle here makes provider coverage additive without copy-pasting the
    failure-tolerant capture contract into every command handler.
    """

    @staticmethod
    async def _arm_changed_file_tracker() -> ChangedFileTracker | None:
        if ChangedFileTracker.disabled():
            return None
        try:
            tracker = ChangedFileTracker()
            await tracker.start()
            return tracker
        except Exception:
            logger.debug("changed-file tracker failed to arm", exc_info=True)
            return None

    async def _publish_changed_files(
        self,
        tracker: ChangedFileTracker | None,
        *,
        request_id: str,
        session_key: str,
        seq: int,
        trigger_message_id: str | None = None,
    ) -> int:
        """Finalize and publish a manifest, returning the next event sequence.

        Idempotence lives in :meth:`ChangedFileTracker.finalize`, so provider
        handlers can safely call this from more than one terminal path.
        """
        if tracker is None:
            return seq
        try:
            manifest = await tracker.finalize()
        except Exception:
            logger.debug("changed-file finalize failed", exc_info=True)
            return seq
        if not manifest:
            return seq

        seq += 1
        active_triggers: Any = getattr(self, "_trigger_message_ids", None)
        if trigger_message_id is None and isinstance(active_triggers, dict):
            trigger_message_id = active_triggers.get(request_id)
        backend_name = str(getattr(self, "_backend_name", "agent-bridge"))
        event = AgentEvent(
            event_id=synthesize_event_id(
                session_key,
                AgentEventKind.CHANGED_FILES.value,
                {"files": manifest.get("files"), "seq": seq},
                seq,
            ),
            session_key=session_key,
            run_id=request_id,
            kind=AgentEventKind.CHANGED_FILES,
            origin="system",
            seq=seq,
            timestamp=datetime.now(timezone.utc),
            raw=manifest,
            provider=backend_name,
            trigger_message_id=trigger_message_id,
        )
        try:
            self._publisher.publish_run_event(request_id, event)
        except Exception:
            logger.debug("changed-file publish failed", exc_info=True)
            return seq

        overlapped = manifest.get("overlapping_repos") or []
        logger.info(
            "Changed files: %d file(s)%s%s request_id=%s provider=%s",
            len(manifest.get("files") or []),
            " [truncated]" if manifest.get("truncated") else "",
            f" [overlapping: {', '.join(overlapped)}]" if overlapped else "",
            request_id,
            backend_name,
        )
        return seq
