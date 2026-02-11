"""SQL editor and display widgets."""

from __future__ import annotations

from textual.widgets import TextArea, Static
from textual.reactive import reactive
from textual.containers import Horizontal
from textual.message import Message
from rich.syntax import Syntax
from rich.text import Text


class SQLDisplay(Static):
    """Read-only SQL display for API mode."""
    
    DEFAULT_CSS = """
    SQLDisplay {
        height: auto;
        max-height: 4;
        border: none;
        background: transparent;
        padding: 0 1;
    }
    """
    
    sql = reactive("")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._syntax_highlight = True
    
    def watch_sql(self, sql: str):
        """React to SQL changes."""
        self.update(self._render_sql())
    
    def _render_sql(self) -> Syntax | Text:
        """Render SQL with syntax highlighting."""
        if not self.sql:
            return Text("No query", style="dim")
        
        if self._syntax_highlight:
            return Syntax(
                self.sql,
                "sql",
                theme="monokai",
                line_numbers=False,
                word_wrap=True,
            )
        return Text(self.sql)
    
    def set_sql(self, sql: str):
        """Set the displayed SQL."""
        self.sql = sql
    
    def render(self):
        """Render the widget."""
        return self._render_sql()


class SQLEditor(TextArea):
    """Editable SQL editor for DB mode."""
    
    DEFAULT_CSS = """
    SQLEditor {
        height: auto;
        max-height: 6;
        border: none;
        background: transparent;
        padding: 0 1;
    }
    
    SQLEditor:focus {
        border-left: solid $primary;
    }
    """
    
    BINDINGS = [
        ("f5", "execute", "Execute"),
        ("ctrl+enter", "execute", "Execute"),
    ]
    
    def __init__(self, **kwargs):
        super().__init__(
            language="sql",
            theme="monokai",
            show_line_numbers=False,
            **kwargs
        )
    
    def action_execute(self):
        """Emit execute event."""
        self.post_message(self.Execute(self.text))
    
    class Execute(Message):
        """Message sent when user wants to execute SQL."""
        
        def __init__(self, sql: str):
            self.sql = sql
            super().__init__()
    
    def get_sql(self) -> str:
        """Get current SQL text."""
        return self.text
    
    def set_sql(self, sql: str):
        """Set SQL text."""
        self.text = sql
        self.move_cursor((0, 0))


class SQLContainer(Horizontal):
    """Container for SQL display/editor with mode indicator."""
    
    DEFAULT_CSS = """
    SQLContainer {
        height: auto;
        max-height: 5;
        border-bottom: solid $surface;
        background: transparent;
    }
    
    SQLContainer > .sql-mode-indicator {
        width: 3;
        content-align: center middle;
        color: $text-muted;
    }
    
    SQLContainer > .sql-content {
        width: 1fr;
    }
    """
    
    def __init__(self, editable: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.editable = editable
        self._editor: SQLEditor | SQLDisplay | None = None
    
    def compose(self):
        """Compose the widget."""
        from textual.widgets import Static
        
        # Mode indicator
        if self.editable:
            yield Static(">", classes="sql-mode-indicator mode-editable")
        else:
            yield Static("|", classes="sql-mode-indicator mode-locked")
        
        # SQL content
        if self.editable:
            self._editor = SQLEditor(classes="sql-content")
        else:
            self._editor = SQLDisplay(classes="sql-content")
        
        yield self._editor
    
    def set_sql(self, sql: str):
        """Set SQL text."""
        if self._editor:
            self._editor.set_sql(sql)
    
    def get_sql(self) -> str:
        """Get SQL text."""
        if isinstance(self._editor, SQLEditor):
            return self._editor.get_sql()
        return ""
    
    def set_editable(self, editable: bool):
        """Switch between editable and read-only mode."""
        if self.editable == editable:
            return
        
        self.editable = editable
        current_sql = self.get_sql()

        indicator = self.query_one(".sql-mode-indicator", Static)
        indicator.update(">" if editable else "|")
        indicator.set_class(editable, "mode-editable")
        indicator.set_class(not editable, "mode-locked")

        if self._editor:
            self._editor.remove()

        self._editor = SQLEditor(classes="sql-content") if editable else SQLDisplay(classes="sql-content")
        self._editor.set_sql(current_sql)
        self.mount(self._editor)
