"""Persist and shape tool-attributed changed-file events."""

from __future__ import annotations

import logging
from typing import Any

from agent_bridge.changed_files import is_file_modifying_tool
from .changed_files_store import ChangedFileInput, ChangedFilesStore, decode_file_changes

logger = logging.getLogger(__name__)


class ToolChangedFilesCoordinator:
    """Upsert one completed file-tool event and return its canonical summary."""

    def __init__(self, engine: Any) -> None:
        self.store = ChangedFilesStore(engine)

    def persist(self, event: dict[str, Any]) -> dict[str, Any] | None:
        turn_id = event.get("turn_id")
        if not turn_id:
            return None
        raw_file = event.get("file")
        files = decode_file_changes([raw_file]) if isinstance(raw_file, dict) else []
        if not files:
            tool_name = event.get("tool_name")
            arguments = event.get("arguments")
            if not is_file_modifying_tool(tool_name) or not isinstance(arguments, dict):
                return None
            raw_path = arguments.get("file_path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                return None
            normalized = raw_path.replace("\\", "/")
            files = [ChangedFileInput(
                repo_key="workspace",
                repo_label=None,
                path=normalized,
                change_kind="modified",
            )]
        source_id = event.get("tool_use_id") or event.get("call_id")
        if source_id:
            files[0].source_tool_call_ids = [str(source_id)]
        saved = self.store.save_turn_files(
            turn_id=str(turn_id),
            bot_id=event.get("bot_id"),
            user_id=event.get("user_id"),
            trigger_message_id=event.get("trigger_message_id"),
            files=files,
        )
        if not saved:
            return None
        summary = self.store.summary_for_turn(str(turn_id))
        changed = next(
            (
                item for item in summary.get("files", [])
                if item.get("repo_key") == files[0].repo_key
                and item.get("path") == files[0].path
            ),
            None,
        )
        if changed is None:
            logger.warning("persisted changed file missing from summary turn=%s", turn_id)
            return None
        return {"file": changed, "summary": summary}
