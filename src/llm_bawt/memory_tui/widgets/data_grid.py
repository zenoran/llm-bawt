"""Data grid widget with editing capabilities."""

from __future__ import annotations

from typing import Any

from textual.widgets import DataTable
from textual.reactive import reactive
from textual.message import Message


class DataGrid(DataTable):
    """Enhanced DataTable with editing and event handling."""
    
    DEFAULT_CSS = """
    DataGrid {
        border: none;
        background: transparent;
        height: 1fr;
    }
    
    DataGrid:focus {
        border-left: solid $primary;
    }
    
    DataGrid > .datatable--header {
        text-style: bold;
        color: $accent;
    }
    
    DataGrid > .datatable--cursor {
        color: $primary;
        text-style: bold;
    }
    """
    
    BINDINGS = [
        ("enter", "edit_cell", "Edit"),
        ("e", "edit_row", "Edit Row"),
        ("d", "delete_row", "Delete"),
        ("c", "copy_cell", "Copy Cell"),
        ("C", "copy_row", "Copy Row"),
        ("r", "refresh", "Refresh"),
    ]
    
    # Reactive state
    column_names = reactive(list)
    rows_data = reactive(list)
    primary_key = reactive(None)
    editable = reactive(True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._column_keys = []
        self._row_ids = []
        self.column_names = []
        self.rows_data = []
    
    def clear_data(self):
        """Clear all data."""
        self.clear(columns=True)
        self._column_keys = []
        self._row_ids = []
        self.column_names = []
        self.rows_data = []
    
    def set_columns(self, columns: list[str]):
        """Set column headers."""
        if not columns:
            columns = ["No Data"]
        self.clear(columns=True)
        self._column_keys = []
        self.column_names = columns
        
        for col in columns:
            key = self.add_column(col, key=col)
            self._column_keys.append(key)
    
    def set_rows(self, rows: list[dict[str, Any]]):
        """Set row data."""
        self.rows_data = rows
        self.clear()
        self._row_ids = []
        
        if not rows:
            return
        
        for i, row_data in enumerate(rows):
            values = []
            for col in self.column_names:
                val = row_data.get(col, "")
                # Format values
                if val is None:
                    val = ""
                elif isinstance(val, (list, dict)):
                    import json
                    val = json.dumps(val)[:50]
                elif isinstance(val, float):
                    val = f"{val:.3f}"
                values.append(str(val)[:100])  # Truncate long strings
            
            row_key = self.add_row(*values)
            self._row_ids.append(row_data.get(self.primary_key, i))
    
    def get_selected_row_data(self) -> dict[str, Any] | None:
        """Get data for currently selected row."""
        cursor = self.cursor_coordinate
        if cursor is None or cursor.row >= len(self.rows_data):
            return None
        return self.rows_data[cursor.row]
    
    def get_selected_cell_value(self) -> str:
        """Get value of currently selected cell."""
        cursor = self.cursor_coordinate
        if cursor is None:
            return ""
        
        row_data = self.get_selected_row_data()
        if row_data is None or cursor.column >= len(self.column_names):
            return ""
        
        col_name = self.column_names[cursor.column]
        return str(row_data.get(col_name, ""))
    
    def action_edit_cell(self):
        """Emit edit cell event."""
        cursor = self.cursor_coordinate
        if cursor is None:
            return
        
        row_data = self.get_selected_row_data()
        if row_data is None:
            return
        
        col_name = self.column_names[cursor.column] if cursor.column < len(self.column_names) else ""
        self.post_message(self.EditCell(row_data, col_name, row_data.get(col_name, "")))
    
    def action_edit_row(self):
        """Emit edit row event."""
        row_data = self.get_selected_row_data()
        if row_data:
            self.post_message(self.EditRow(row_data))
    
    def action_delete_row(self):
        """Emit delete row event."""
        row_data = self.get_selected_row_data()
        if row_data:
            self.post_message(self.DeleteRow(row_data))
    
    def action_copy_cell(self):
        """Copy cell to clipboard."""
        value = self.get_selected_cell_value()
        try:
            import pyperclip
            pyperclip.copy(value)
            self.notify(f"Copied: {value[:50]}..." if len(value) > 50 else f"Copied: {value}")
        except ImportError:
            self.notify("pyperclip not installed", severity="warning")
        except Exception as e:
            self.notify(f"Copy failed: {e}", severity="error")
    
    def action_copy_row(self):
        """Copy row as JSON to clipboard."""
        import json
        
        row_data = self.get_selected_row_data()
        if row_data:
            json_str = json.dumps(row_data, indent=2, default=str)
            try:
                import pyperclip
                pyperclip.copy(json_str)
                self.notify(f"Copied row ({len(json_str)} chars)")
            except ImportError:
                self.notify("pyperclip not installed", severity="warning")
            except Exception as e:
                self.notify(f"Copy failed: {e}", severity="error")
    
    def action_refresh(self):
        """Emit refresh event."""
        self.post_message(self.Refresh())
    
    # Message classes
    class EditCell(Message):
        """User wants to edit a cell."""
        def __init__(self, row: dict, column: str, value: Any):
            self.row = row
            self.column = column
            self.value = value
            super().__init__()
    
    class EditRow(Message):
        """User wants to edit a row."""
        def __init__(self, row: dict):
            self.row = row
            super().__init__()
    
    class DeleteRow(Message):
        """User wants to delete a row."""
        def __init__(self, row: dict):
            self.row = row
            super().__init__()
    
    class Refresh(Message):
        """User wants to refresh data."""
        pass
    
    class RowSelected(Message):
        """User selected a row."""
        def __init__(self, row: dict, index: int):
            self.row = row
            self.index = index
            super().__init__()
    
    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        """Handle cell highlight (keyboard and mouse)."""
        row = self.get_selected_row_data()
        if row:
            self.post_message(self.RowSelected(row, event.coordinate.row))

    def on_data_table_cell_selected(self, event: DataTable.CellSelected):
        """Handle cell selection (double click)."""
        self.action_edit_cell()
