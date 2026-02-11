"""Detail panel for showing selected item details."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static


class DetailPanel(Vertical):
    """Panel for displaying details of selected item."""

    DEFAULT_CSS = """
    DetailPanel {
        border-left: solid $panel;
        width: 36;
        min-width: 28;
        background: $surface;
        padding: 1;
    }

    DetailPanel .detail-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    DetailPanel .detail-body {
        height: 1fr;
        overflow-y: auto;
    }

    DetailPanel .empty-state {
        color: $text-muted;
        text-style: italic;
    }

    DetailPanel .button-row {
        height: auto;
        align: right middle;
        margin-top: 1;
    }

    DetailPanel Button {
        margin-left: 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._data: dict[str, Any] | None = None
        self._data_type = "memory"

    def compose(self):
        yield Static("Details", classes="detail-title")
        yield Static("Select a row to inspect details.", classes="detail-body empty-state", id="detail_body")
        with Horizontal(classes="button-row"):
            yield Button("Edit", id="edit")
            yield Button("Delete", id="delete", variant="error")

    def set_data(self, data: dict[str, Any] | None, data_type: str = "memory"):
        """Set selected data and update rendering."""
        self._data = data
        self._data_type = data_type

        title = self.query_one(".detail-title", Static)
        body = self.query_one("#detail_body", Static)

        if data is None:
            title.update("Details")
            body.update("Select a row to inspect details.")
            body.set_class(True, "empty-state")
            return

        title_map = {
            "memory": "Memory",
            "message": "Message",
            "profile": "Profile",
            "summary": "Summary",
            "generic": "Details",
        }
        title.update(title_map.get(data_type, "Details"))
        body.set_class(False, "empty-state")
        body.update(self._render_text(data, data_type))

    def _render_text(self, data: dict[str, Any], data_type: str) -> str:
        """Render selected row details as readable text."""
        if data_type == "memory":
            memory_id = str(data.get("id", "unknown"))
            content = str(data.get("content", ""))
            tags = data.get("tags", [])
            if isinstance(tags, str):
                tags_text = tags
            else:
                tags_text = ", ".join(str(tag) for tag in tags[:12])
            importance = data.get("importance", "?")
            relevance = data.get("relevance") or data.get("similarity")
            created = self._format_time(data.get("created_at") or data.get("timestamp"))
            lines = [
                f"id: {memory_id}",
                f"importance: {importance}",
                f"relevance: {relevance if relevance is not None else '-'}",
                f"created: {created}",
                f"tags: {tags_text or '-'}",
                "",
                "content:",
                content,
            ]
            return "\n".join(lines)

        if data_type == "message":
            lines = [
                f"id: {data.get('id', 'unknown')}",
                f"role: {data.get('role', '?')}",
                f"timestamp: {self._format_time(data.get('timestamp'))}",
                f"session: {str(data.get('session_id', '-'))}",
                "",
                "content:",
                str(data.get("content", "")),
            ]
            return "\n".join(lines)

        if data_type == "profile":
            attrs = data.get("attributes")
            attr_count = len(attrs) if isinstance(attrs, list) else 0
            lines = [
                f"entity_type: {data.get('entity_type', '?')}",
                f"entity_id: {data.get('entity_id', '?')}",
                f"display_name: {data.get('display_name', '-')}",
                f"attributes: {attr_count}",
                "",
                f"summary: {data.get('summary', '-')}",
            ]
            return "\n".join(lines)

        return json.dumps(data, indent=2, default=str)

    def _format_time(self, ts: float | str | datetime | None) -> str:
        if ts is None:
            return "-"
        try:
            if isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts)
            elif isinstance(ts, str):
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            else:
                dt = ts
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(ts)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Forward actions to the app."""
        self.post_message(self.Action(event.button.id, self._data, self._data_type))

    class Action(Message):
        """Detail panel action message."""

        def __init__(self, action: str, data: dict[str, Any] | None, data_type: str):
            self.action = action
            self.data = data
            self.data_type = data_type
            super().__init__()
