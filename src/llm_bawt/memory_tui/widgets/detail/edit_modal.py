"""Edit modals for cell and row editing."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, TextArea, Static, Label


class EditModal(ModalScreen[dict[str, Any] | None]):
    """Base edit modal."""
    
    DEFAULT_CSS = """
    EditModal {
        align: center middle;
        background: $background;
    }
    
    EditModal > Vertical {
        border: solid $primary;
        padding: 1 2;
        width: 80;
        height: auto;
        max-height: 40;
    }
    
    EditModal .modal-title {
        color: $accent;
        text-style: bold;
        text-align: center;
        border-bottom: solid $surface;
        padding-bottom: 1;
        margin-bottom: 1;
    }
    
    EditModal .field-label {
        color: $text-muted;
        text-style: bold;
        margin-top: 1;
    }
    
    EditModal .field-value {
        margin-bottom: 1;
    }
    
    EditModal Input {
        width: 100%;
        margin-bottom: 1;
        border: none;
        background: $surface;
    }
    
    EditModal Input:focus {
        border-left: solid $primary;
    }
    
    EditModal TextArea {
        width: 100%;
        height: 10;
        margin-bottom: 1;
        border: none;
        background: $surface;
    }
    
    EditModal .button-row {
        align: right middle;
        margin-top: 1;
    }
    
    EditModal Button {
        margin-left: 1;
    }
    """
    
    def __init__(self, title: str = "Edit", **kwargs):
        super().__init__(**kwargs)
        self._title = title


class CellEditModal(EditModal):
    """Modal for editing a single cell."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        column: str,
        value: Any,
        row: dict[str, Any],
        **kwargs
    ):
        super().__init__(title=f"Edit {column}", **kwargs)
        self.column = column
        self.value = value
        self.row = row
    
    def compose(self) -> ComposeResult:
        """Compose the modal."""
        with Vertical():
            yield Static(self._title, classes="modal-title")
            
            yield Label("Column:", classes="field-label")
            yield Static(self.column, classes="field-value")
            
            yield Label("Value:", classes="field-label")
            
            # Use TextArea for long text, Input for short
            value_str = str(self.value) if self.value is not None else ""
            if len(value_str) > 50 or "\n" in value_str:
                yield TextArea(
                    text=value_str,
                    id="value_input",
                    language=None,
                )
            else:
                yield Input(
                    value=value_str,
                    id="value_input",
                )
            
            with Horizontal(classes="button-row"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Save", variant="primary", id="save")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "save":
            value_widget = self.query_one("#value_input")
            if isinstance(value_widget, TextArea):
                new_value = value_widget.text
            else:
                new_value = value_widget.value
            
            self.dismiss({
                "column": self.column,
                "value": new_value,
                "row": self.row,
            })

    def action_cancel(self) -> None:
        self.dismiss(None)


class RowEditModal(EditModal):
    """Modal for editing an entire row."""

    DEFAULT_CSS = """
    RowEditModal > Vertical {
        width: 80;
        max-height: 80%;
        overflow-y: auto;
    }

    RowEditModal .field-label {
        color: $text-muted;
        text-style: bold;
        margin-top: 1;
    }

    RowEditModal TextArea {
        height: 6;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        row: dict[str, Any],
        editable_columns: list[str] | None = None,
        **kwargs
    ):
        super().__init__(title="Edit Row", **kwargs)
        self.row = row
        # Only offer known useful columns for editing
        skip = {"id", "entity_id", "created_at", "updated_at", "embedding", "timestamp"}
        self.editable_columns = editable_columns or [
            c for c in row.keys() if c not in skip
        ]
        self._inputs: dict[str, Input | TextArea] = {}

    def compose(self) -> ComposeResult:
        """Compose the modal."""
        with Vertical():
            yield Static(self._title, classes="modal-title")

            # PK display
            pk = self.row.get("id") or self.row.get("entity_id")
            if pk:
                yield Label(f"ID: {pk}", classes="field-label")

            # Editable fields
            for col in self.editable_columns:
                value = self.row.get(col)
                value_str = str(value) if value is not None else ""

                # Determine input type
                if isinstance(value, (list, dict)):
                    import json
                    value_str = json.dumps(value, indent=2)
                elif isinstance(value, float):
                    value_str = f"{value}"

                yield Label(col, classes="field-label")

                # Use TextArea for long/complex values
                if len(value_str) > 80 or "\n" in value_str or isinstance(value, (list, dict)):
                    widget = TextArea(
                        text=value_str,
                        id=f"field_{col}",
                        language="json" if isinstance(value, (list, dict)) else None,
                        show_line_numbers=False,
                    )
                else:
                    widget = Input(
                        value=value_str,
                        id=f"field_{col}",
                    )

                self._inputs[col] = widget
                yield widget

            with Horizontal(classes="button-row"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Save", variant="primary", id="save")

    def action_cancel(self) -> None:
        """Dismiss on Escape."""
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "save":
            updates = {}
            for col, widget in self._inputs.items():
                if isinstance(widget, TextArea):
                    value = widget.text
                    # Try to parse JSON
                    try:
                        import json
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                else:
                    value = widget.value
                    # Try to convert numbers
                    try:
                        if "." in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
                
                updates[col] = value
            
            self.dismiss({
                "updates": updates,
                "row": self.row,
            })


class ConfirmModal(ModalScreen[bool]):
    """Confirmation modal."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    ConfirmModal {
        align: center middle;
        background: $background;
    }
    
    ConfirmModal > Vertical {
        border: solid $primary;
        padding: 1 2;
        width: 60;
        height: auto;
    }
    
    ConfirmModal .modal-title {
        color: $accent;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    
    ConfirmModal .modal-message {
        text-align: center;
        margin-bottom: 1;
    }
    
    ConfirmModal .button-row {
        align: center middle;
        margin-top: 1;
    }
    
    ConfirmModal Button {
        margin: 0 1;
    }
    """
    
    def __init__(self, title: str, message: str, **kwargs):
        super().__init__(**kwargs)
        self._title = title
        self._message = message
    
    def compose(self) -> ComposeResult:
        """Compose the modal."""
        with Vertical():
            yield Static(self._title, classes="modal-title")
            yield Static(self._message, classes="modal-message")
            with Horizontal(classes="button-row"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Confirm", variant="error", id="confirm")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        self.dismiss(event.button.id == "confirm")

    def action_cancel(self) -> None:
        self.dismiss(False)


class PromptModal(ModalScreen[str | None]):
    """Simple input prompt modal."""

    DEFAULT_CSS = """
    PromptModal {
        align: center middle;
        background: $background 70%;
    }

    PromptModal > Vertical {
        border: round $primary;
        background: $surface;
        width: 72;
        height: auto;
        padding: 1 2;
    }

    PromptModal .modal-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    PromptModal .modal-help {
        color: $text-muted;
        margin-bottom: 1;
    }

    PromptModal Input {
        width: 100%;
        margin-bottom: 1;
    }

    PromptModal .button-row {
        align: right middle;
    }
    """

    def __init__(self, title: str, placeholder: str = "", value: str = "", **kwargs):
        super().__init__(**kwargs)
        self._title = title
        self._placeholder = placeholder
        self._value = value

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._title, classes="modal-title")
            yield Static("Press Enter to submit, Esc to cancel", classes="modal-help")
            yield Input(value=self._value, placeholder=self._placeholder, id="prompt_input")
            with Horizontal(classes="button-row"):
                yield Button("Cancel", id="cancel")
                yield Button("Search", id="submit", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#prompt_input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            value = self.query_one("#prompt_input", Input).value.strip()
            self.dismiss(value or None)
        else:
            self.dismiss(None)
