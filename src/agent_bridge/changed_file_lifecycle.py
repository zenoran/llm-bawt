"""Shared bridge hooks for tool-call-based changed-file attribution."""

from __future__ import annotations

import logging

from .changed_files import ToolChangedFileCapture
from .events import AgentEvent, AgentEventKind

logger = logging.getLogger(__name__)


class ChangedFileLifecycleMixin:
    """Attach per-tool file capture to standardized bridge events.

    Concrete bridges keep publishing TOOL_START/TOOL_END normally.  This mixin
    snapshots only the declared ``file_path`` and emits a FILE_CHANGED event as
    soon as that specific tool completes.  No workspace-wide turn snapshots,
    overlap registry, or finalize phase are involved.
    """

    def _changed_file_capture(self) -> ToolChangedFileCapture:
        capture = getattr(self, "_tool_changed_file_capture", None)
        if capture is None:
            capture = ToolChangedFileCapture(cwd=getattr(self, "_cwd", "/app"))
            self._tool_changed_file_capture = capture
        return capture

    def _capture_changed_file_event(
        self,
        event: AgentEvent,
        *,
        request_id: str | None = None,
    ) -> AgentEvent | None:
        try:
            capture = self._changed_file_capture()
            capture_request_id = request_id or event.run_id or ""
            if event.kind == AgentEventKind.TOOL_START:
                capture.start(
                    request_id=capture_request_id,
                    tool_use_id=event.tool_use_id,
                    tool_name=event.tool_name,
                    arguments=event.tool_arguments,
                )
                return None
            if event.kind != AgentEventKind.TOOL_END:
                return None
            finished = capture.finish(
                request_id=capture_request_id,
                tool_use_id=event.tool_use_id,
                tool_name=event.tool_name,
            )
            if finished is None:
                return None
            tool_name, arguments, file_data = finished
            return AgentEvent(
                event_id=f"{event.event_id}:file-changed",
                session_key=event.session_key,
                run_id=event.run_id,
                kind=AgentEventKind.FILE_CHANGED,
                origin="system",
                tool_name=tool_name,
                tool_arguments=arguments,
                tool_use_id=event.tool_use_id,
                parent_tool_use_id=event.parent_tool_use_id,
                seq=event.seq,
                timestamp=event.timestamp,
                raw={"file": file_data},
                provider=event.provider,
                trigger_message_id=event.trigger_message_id,
            )
        except Exception:
            logger.debug("changed-file tool attribution failed", exc_info=True)
            return None

    def _publish_run_event_with_changed_file(
        self,
        request_id: str,
        event: AgentEvent,
    ) -> None:
        """Capture before TOOL_START; publish FILE_CHANGED after TOOL_END."""
        if event.kind == AgentEventKind.TOOL_START:
            self._capture_changed_file_event(event, request_id=request_id)
            self._publisher.publish_run_event(request_id, event)
            return
        self._publisher.publish_run_event(request_id, event)
        file_event = self._capture_changed_file_event(event, request_id=request_id)
        if file_event is not None:
            self._publisher.publish_run_event(request_id, file_event)

    def _discard_changed_file_request(self, request_id: str) -> None:
        try:
            self._changed_file_capture().discard_request(request_id)
        except Exception:
            logger.debug("changed-file request cleanup failed", exc_info=True)
