"""IPython console panel widget."""

from __future__ import annotations

import sys
from io import StringIO
from typing import Any
from contextlib import redirect_stdout, redirect_stderr

from textual.widgets import Input, Static, Log
from textual.containers import Vertical
from textual.reactive import reactive
from textual.message import Message


class ConsolePanel(Vertical):
    """IPython console panel for advanced scripting."""
    
    DEFAULT_CSS = """
    ConsolePanel {
        border-top: solid $surface;
        height: 12;
        dock: bottom;
        background: transparent;
    }
    
    ConsolePanel.collapsed {
        display: none;
    }
    
    ConsolePanel > .console-output {
        height: 1fr;
        background: transparent;
        border: none;
        padding: 0 1;
    }
    
    ConsolePanel > .console-input {
        height: 1;
        border-top: solid $surface;
        border-bottom: none;
        border-left: none;
        border-right: none;
        background: transparent;
    }
    
    ConsolePanel > .console-input:focus {
        border-left: solid $primary;
    }
    
    ConsolePanel > .console-prompt {
        height: 1;
        color: $text-muted;
        content-align: left middle;
        padding: 0 1;
    }
    """
    
    BINDINGS = [
        ("up", "history_prev", "Previous"),
        ("down", "history_next", "Next"),
        ("ctrl+l", "clear", "Clear"),
    ]
    
    # Reactive state
    mode = reactive("command")  # command, python, sql
    visible = reactive(True)
    history = reactive(list)
    history_index = reactive(-1)
    
    def __init__(
        self,
        api_client: Any | None = None,
        db_client: Any | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_client = api_client
        self.db_client = db_client
        self._ipython = None
        self._user_ns = {}
        self._setup_namespace()
    
    def _setup_namespace(self):
        """Setup the IPython namespace with useful objects."""
        self._user_ns = {
            "api": self.api_client,
            "db": self.db_client,
            "config": None,  # Will be set lazily
            "json": __import__("json"),
            "pd": None,  # Will try to import pandas
            "print": self._console_print,
            "refresh": self._trigger_refresh,
            "clipboard": self._copy_to_clipboard,
            "selected_row": None,
            "current_sql": "",
            "current_data": [],
            "to_dataframe": self._to_dataframe,
        }
        
        # Try to import pandas
        try:
            import pandas as pd
            self._user_ns["pd"] = pd
        except ImportError:
            pass
    
    def _console_print(self, *args, **kwargs):
        """Custom print that outputs to console log."""
        text = " ".join(str(a) for a in args)
        self._add_output(text)
    
    def _trigger_refresh(self):
        """Trigger data refresh."""
        self.post_message(self.RefreshData())
    
    def _copy_to_clipboard(self, text: str):
        """Copy text to clipboard."""
        try:
            import pyperclip
            pyperclip.copy(str(text))
            self._add_output("Copied to clipboard!")
        except Exception as e:
            self._add_output(f"Error copying: {e}")
    
    def _to_dataframe(self, data: list[dict]) -> Any:
        """Convert data to pandas DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame(data)
        except ImportError:
            self._add_output("pandas not available")
            return data
    
    def _add_output(self, text: str, style: str = ""):
        """Add output to console log."""
        log = self.query_one(".console-output", Log)
        if style:
            log.write(f"[{style}]{text}")
        else:
            log.write(text)
    
    def compose(self):
        """Compose the console panel."""
        yield Static(">>> Console (Tab to focus, Ctrl+L clear)", classes="console-prompt")
        yield Log(classes="console-output", highlight=True)
        yield Input(
            placeholder="Type a command...",
            classes="console-input",
        )
    
    def on_mount(self):
        """Initialize IPython on mount."""
        self._init_ipython()
        self._add_output("Ready. Pre-loaded: api, db, to_dataframe(), clipboard(), refresh()")
        if self._user_ns.get("pd"):
            self._add_output("pandas available as 'pd'")
    
    def _init_ipython(self):
        """Initialize IPython embed."""
        try:
            from IPython.terminal.embed import InteractiveShellEmbed
            
            self._ipython = InteractiveShellEmbed(
                user_ns=self._user_ns,
                banner1="",
                exit_msg="",
            )
            self._ipython.colors = "Linux"
        except ImportError:
            self._add_output("IPython not available, using basic Python exec")
            self._ipython = None
    
    def execute(self, code: str):
        """Execute code in console."""
        if not code.strip():
            return
        
        # Add to history
        if not self.history or self.history[-1] != code:
            self.history.append(code)
        self.history_index = len(self.history)
        
        # Echo input
        prompt = ">>>" if self.mode == "command" else f"{self.mode}>>>"
        self._add_output(f"{prompt} {code}", style="cyan")
        
        try:
            if self.mode == "python" and self._ipython:
                # Use IPython
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                captured = StringIO()
                sys.stdout = captured
                sys.stderr = captured
                
                try:
                    result = self._ipython.run_cell(code)
                    output = captured.getvalue()
                    if output:
                        self._add_output(output)
                    if result.result is not None:
                        self._add_output(repr(result.result))
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    
            elif self.mode == "sql":
                # Execute SQL on DB
                if self.db_client and self.db_client.is_connected():
                    result = self.db_client.execute_query(code)
                    if result["success"]:
                        self._add_output(f"Rows: {result['rowcount']}")
                        if result["rows"]:
                            import json
                            self._add_output(json.dumps(result["rows"][:5], indent=2, default=str))
                    else:
                        self._add_output(f"Error: {result['error']}", style="red")
                else:
                    self._add_output("DB not connected", style="red")
                    
            else:
                # Command mode - parse as command
                self._execute_command(code)
                
        except Exception as e:
            self._add_output(f"Error: {e}", style="red")
    
    def _execute_command(self, cmd: str):
        """Execute a command in command mode."""
        parts = cmd.strip().split()
        if not parts:
            return
        
        command = parts[0].lower()
        args = parts[1:]
        
        if command in ("help", "?"):
            self._show_help()
        elif command == "clear":
            self.query_one(".console-output", Log).clear()
        elif command == "mode":
            if args:
                self.set_mode(args[0])
            else:
                self._add_output(f"Current mode: {self.mode}")
        elif command == "refresh":
            self._trigger_refresh()
        elif command == "bot":
            if args:
                self.post_message(self.SetBot(args[0]))
            else:
                self._add_output("Usage: bot <bot_id>")
        elif command == "search":
            if args:
                query = " ".join(args)
                self.post_message(self.SearchMemories(query))
            else:
                self._add_output("Usage: search <query>")
        else:
            self._add_output(f"Unknown command: {command}")
    
    def _show_help(self):
        """Show help text."""
        help_text = """
Commands:
  help              Show this help
  mode [python|sql|command]  Switch console mode
  clear             Clear console output
  refresh           Refresh current data view
  bot <id>          Switch bot context
  search <query>    Search memories

Variables available:
  api               API client for service calls
  db                Database client (DB mode)
  selected_row      Currently selected row data
  current_sql       Current SQL query
  clipboard(text)   Copy text to clipboard
  refresh()         Refresh data view
  to_dataframe(data) Convert to pandas DataFrame
        """
        self._add_output(help_text)
    
    def set_mode(self, mode: str):
        """Set console mode."""
        mode = mode.lower()
        if mode in ("command", "python", "sql"):
            self.mode = mode
            prompt = self.query_one(".console-prompt", Static)
            prompt_text = {
                "command": ">>> Console (Tab to focus, Ctrl+L clear)",
                "python": ">>> Python mode",
                "sql": ">>> SQL mode",
            }.get(mode, ">>>")
            prompt.update(prompt_text)
            self._add_output(f"Switched to {mode} mode")
            
            # Update input placeholder to reflect mode
            input_widget = self.query_one(".console-input", Input)
            placeholders = {
                "command": "Type a command...",
                "python": "Python expression...",
                "sql": "SQL query...",
            }
            input_widget.placeholder = placeholders.get(mode, "Type a command...")
    
    def toggle(self):
        """Toggle console visibility."""
        self.visible = not self.visible
        if self.visible:
            self.remove_class("collapsed")
        else:
            self.add_class("collapsed")
    
    def on_input_submitted(self, event: Input.Submitted):
        """Handle console input submission."""
        code = event.value.strip()
        if code:
            self.execute(code)
        event.input.value = ""
    
    def action_history_prev(self):
        """Show previous history item."""
        if self.history and self.history_index > 0:
            self.history_index -= 1
            input_widget = self.query_one(".console-input", Input)
            input_widget.value = self.history[self.history_index]
    
    def action_history_next(self):
        """Show next history item."""
        if self.history and self.history_index < len(self.history) - 1:
            self.history_index += 1
            input_widget = self.query_one(".console-input", Input)
            input_widget.value = self.history[self.history_index]
    
    def action_clear(self):
        """Clear console output."""
        self.query_one(".console-output", Log).clear()
    
    def update_namespace(self, **kwargs):
        """Update the console namespace with new variables."""
        self._user_ns.update(kwargs)
        if self._ipython:
            self._ipython.user_ns.update(kwargs)
    
    # Message classes
    class RefreshData(Message):
        """Request data refresh."""
        pass
    
    class SetBot(Message):
        """Request bot switch."""
        def __init__(self, bot_id: str):
            self.bot_id = bot_id
            super().__init__()
    
    class SearchMemories(Message):
        """Request memory search."""
        def __init__(self, query: str):
            self.query = query
            super().__init__()
