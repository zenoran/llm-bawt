"""Memory TUI main application."""

from __future__ import annotations

import re
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

from .api_client import MemoryAPIClient
from .db_client import MemoryDBClient
from .themes import NEURAL_THEME
from .widgets import APINavTree, APINavigate, ConsolePanel, DBNavTree, DBNavigate, DataGrid
from .widgets.detail import CellEditModal, ConfirmModal, DetailPanel, PromptModal, RowEditModal
from .widgets.sql_editor import SQLContainer, SQLEditor


class MemoryTUIApp(App):
    """Main Memory TUI Application."""

    CSS = """
    Screen {
        background: $background;
        color: $foreground;
    }

    .main-container {
        width: 100%;
        height: 1fr;
    }

    .sidebar {
        width: 34;
        min-width: 28;
        border-right: solid $panel;
        background: $surface;
    }

    .sidebar-title {
        height: 3;
        content-align: center middle;
        text-style: bold;
        color: $accent;
        border-bottom: solid $panel;
    }

    #api_nav,
    #db_nav {
        height: 1fr;
    }

    .content-area {
        width: 1fr;
        height: 1fr;
    }

    .data-section {
        height: 1fr;
    }

    #sql_container {
        border-bottom: solid $panel;
    }

    #data_grid {
        width: 1fr;
        background: $surface;
    }

    #detail_panel {
        width: 38;
    }

    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        ("f2", "toggle_mode", "API/DB Mode"),
        ("f5", "refresh", "Refresh"),
        ("/", "quick_search", "Search"),
        ("f10", "toggle_theme", "Theme"),
        ("f12", "toggle_console", "Console"),
        ("`", "toggle_console", "Console"),
        ("q", "quit", "Quit"),
        ("question", "help", "Help"),
    ]

    mode = reactive("api")
    current_bot = reactive("nova")
    current_view = reactive("memory_browse")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_client = MemoryAPIClient()
        self.db_client = MemoryDBClient()
        self._data: list[dict[str, Any]] = []
        self._current_sql = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(classes="main-container"):
            with Vertical(classes="sidebar"):
                yield Static("Memory Explorer", classes="sidebar-title")
                yield APINavTree(id="api_nav")
                db_nav = DBNavTree(id="db_nav")
                db_nav.add_class("hidden")
                yield db_nav

            with Vertical(classes="content-area"):
                yield SQLContainer(editable=False, id="sql_container")
                with Horizontal(classes="data-section"):
                    yield DataGrid(id="data_grid")
                    yield DetailPanel(id="detail_panel")

        yield ConsolePanel(api_client=self.api_client, db_client=self.db_client, id="console_panel")
        yield Footer()

    async def on_mount(self):
        """Initialize app and initial data."""
        self.register_theme(NEURAL_THEME)
        self.theme = "neural"

        self._set_header_subtitle()

        grid = self.query_one("#data_grid", DataGrid)
        grid.set_columns(["loading"])
        grid.set_rows([{"loading": "Loading data..."}])

        try:
            await self.api_client.health_check()
            self.notify("Connected to llm-service")
        except Exception as exc:
            self.notify(f"Service unavailable: {exc}", severity="error")

        if self.db_client.connect():
            self.notify("Database connected")
        else:
            self.notify("Database not available, API mode only", severity="warning")

        await self._populate_bots_nav()
        await self._load_api_data()
        self._sync_console_namespace()

    async def _populate_bots_nav(self) -> None:
        """Load bots into API nav tree."""
        bots: list[dict] = []
        # Try API first
        try:
            result = await self.api_client.list_bots()
            bots = result.get("bots", [])
        except Exception:
            pass

        # Fallback: load from local bots.yaml
        if not bots:
            try:
                from .bots import BotManager
                manager = BotManager()
                bots = [
                    {"id": b.slug, "name": b.name}
                    for b in manager.list_bots()
                ]
            except Exception:
                pass

        if bots:
            self.query_one("#api_nav", APINavTree).update_bots(bots)

    def watch_mode(self, mode: str):
        """Handle API/DB mode switch."""
        api_nav = self.query_one("#api_nav", APINavTree)
        db_nav = self.query_one("#db_nav", DBNavTree)
        sql_container = self.query_one("#sql_container", SQLContainer)

        if mode == "api":
            api_nav.remove_class("hidden")
            db_nav.add_class("hidden")
            sql_container.set_editable(False)
        else:
            api_nav.add_class("hidden")
            db_nav.remove_class("hidden")
            sql_container.set_editable(True)
            if self.db_client.is_connected():
                db_nav.update_tables(self.db_client.get_tables(), self.current_bot)

        self._set_header_subtitle()

    def _set_header_subtitle(self) -> None:
        header = self.query_one(Header)
        if self.mode == "api":
            header.sub_title = f"API mode | bot={self.current_bot}"
        else:
            header.sub_title = f"DB mode | bot={self.current_bot}"

    def action_toggle_mode(self):
        self.mode = "db" if self.mode == "api" else "api"

    async def _load_api_data(self):
        """Load data from API based on current view."""
        view = self.current_view
        try:
            if view in {"memory_browse", "memory_importance", "memory_tags"}:
                result = await self.api_client.list_memories(self.current_bot, limit=200)
                memories = result.get("results", [])

                if view == "memory_importance":
                    memories = sorted(memories, key=lambda row: row.get("importance", 0), reverse=True)
                    self._set_data(
                        memories,
                        f"SELECT * FROM {self.current_bot}_memories ORDER BY importance DESC LIMIT 200",
                    )
                elif view == "memory_tags":
                    tag_rows: dict[str, int] = {}
                    for memory in memories:
                        tags = memory.get("tags", [])
                        if isinstance(tags, str):
                            tags = [tags]
                        for tag in tags:
                            tag_rows[str(tag)] = tag_rows.get(str(tag), 0) + 1
                    rows = [
                        {"tag": tag, "count": count}
                        for tag, count in sorted(tag_rows.items(), key=lambda item: item[1], reverse=True)
                    ]
                    self._set_data(rows, "SELECT tag, COUNT(*) FROM memories GROUP BY tag ORDER BY count DESC")
                else:
                    self._set_data(
                        memories,
                        f"SELECT * FROM {self.current_bot}_memories ORDER BY importance DESC LIMIT 200",
                    )

            elif view == "messages_recent":
                result = await self.api_client.get_messages(self.current_bot, limit=200)
                self._set_data(
                    result.get("messages", []),
                    f"SELECT * FROM {self.current_bot}_messages ORDER BY timestamp DESC LIMIT 200",
                )

            elif view == "messages_forgotten":
                result = await self.api_client.preview_forgotten_messages(self.current_bot)
                rows = result.get("messages", []) or result.get("results", [])
                self._set_data(rows, f"SELECT * FROM {self.current_bot}_forgotten_messages ORDER BY timestamp DESC")

            elif view == "messages_summaries":
                result = await self.api_client.list_summaries(self.current_bot)
                rows = result.get("summaries", [])
                self._set_data(rows, f"SELECT * FROM {self.current_bot}_summaries ORDER BY created_at DESC")

            elif view == "profiles_users":
                result = await self.api_client.list_profiles("user")
                self._set_data(result.get("profiles", []), "SELECT * FROM user_profiles")

            elif view == "profiles_bots":
                result = await self.api_client.list_profiles("bot")
                self._set_data(result.get("profiles", []), "SELECT * FROM bot_profiles")

            elif view == "stats":
                stats = await self.api_client.get_memory_stats(self.current_bot)
                self._set_data(self._stats_to_rows(stats), f"SELECT stats FROM {self.current_bot}")

            elif view == "cleanup":
                status = await self.api_client.get_status()
                rows = [{"key": key, "value": value} for key, value in status.items()]
                self._set_data(rows, "SELECT service_status")

            else:
                self._set_data([], "")

        except Exception as exc:
            self.notify(f"Error loading data: {exc}", severity="error")
            self._set_data([], "")

    def _set_data(self, rows: list[dict[str, Any]], sql: str) -> None:
        """Set table data and update dependent widgets."""
        self._data = rows
        self._update_sql_display(sql)
        self._update_grid()

        detail = self.query_one("#detail_panel", DetailPanel)
        detail_type = self._infer_detail_type()
        detail.set_data(rows[0] if rows else None, detail_type)

        self._sync_console_namespace()

    def _update_grid(self, columns: list[str] | None = None):
        grid = self.query_one("#data_grid", DataGrid)
        if not self._data:
            grid.clear_data()
            grid.set_columns(["status"])
            grid.set_rows([{"status": "No results"}])
            return

        if columns is None:
            columns = list(self._data[0].keys())

        grid.set_columns(columns)
        grid.set_rows(self._data)
        grid.refresh()

    def _update_sql_display(self, sql: str):
        self._current_sql = sql
        self.query_one("#sql_container", SQLContainer).set_sql(sql)

    def _sync_console_namespace(self) -> None:
        console = self.query_one(ConsolePanel)
        console.update_namespace(
            selected_row=self._data[0] if self._data else None,
            current_sql=self._current_sql,
            current_data=self._data,
        )

    def _infer_detail_type(self) -> str:
        if self.current_view.startswith("messages"):
            return "message"
        if self.current_view.startswith("profiles"):
            return "profile"
        if self.current_view == "messages_summaries":
            return "summary"
        return "memory"

    async def _prompt(self, title: str, placeholder: str = "") -> str | None:
        result = await self.push_screen_wait(PromptModal(title=title, placeholder=placeholder))
        if not result:
            return None
        return result

    async def _execute_db_query(self, sql: str):
        if not self.db_client.is_connected():
            self.notify("Database not connected", severity="error")
            return

        result = self.db_client.execute_query(sql)
        if result["success"]:
            self._data = result["rows"]
            self._update_grid(columns=result.get("columns"))
            self.notify(f"Query returned {result['rowcount']} rows")
            self._update_sql_display(sql)
        else:
            self.notify(f"Query error: {result['error']}", severity="error")

    @work
    async def on_apinavigate(self, event: APINavigate):
        """Handle API navigation."""
        data = event.data
        nav_type = data.get("type")

        if nav_type == "select_bot":
            self.current_bot = data.get("bot_id", self.current_bot)
            self._set_header_subtitle()
            await self._load_api_data()
            return

        if nav_type == "memory_search":
            query = await self._prompt("Search memories", "Type search text")
            if query:
                await self._run_memory_search(query)
            return

        if nav_type == "messages_search":
            query = await self._prompt("Search messages", "Type search text")
            if query:
                await self._run_message_search(query)
            return

        if nav_type == "profiles_attrs":
            entity_id = await self._prompt("Open profile by entity", "user id or bot id")
            if entity_id:
                await self._run_profile_lookup(entity_id)
            return

        if nav_type == "memory_consolidate":
            await self._run_consolidate()
            return

        if nav_type == "memory_embeddings":
            await self._run_regenerate_embeddings()
            return

        self.current_view = nav_type
        await self._load_api_data()

    async def on_dbnavigate(self, event: DBNavigate):
        """Handle DB navigation."""
        nav_type = event.data.get("type")

        if nav_type == "table":
            table = event.data.get("table")
            sql = f"SELECT * FROM {table} LIMIT 100"
            await self._execute_db_query(sql)
            return

        if nav_type == "new_query":
            self._update_sql_display("SELECT * FROM ")
            return

        if nav_type == "db_info":
            rows = [
                {"key": "connected", "value": self.db_client.is_connected()},
                {"key": "bot", "value": self.current_bot},
                {"key": "table_count", "value": len(self.db_client.get_tables()) if self.db_client.is_connected() else 0},
            ]
            self._set_data(rows, "SELECT db_info")
            return

        self.notify(f"Not implemented yet: {nav_type}", severity="warning")

    async def on_sql_editor_execute(self, event: SQLEditor.Execute):
        """Run SQL from the DB mode editor."""
        await self._execute_db_query(event.sql)

    async def _run_memory_search(self, query: str) -> None:
        try:
            result = await self.api_client.search_memories(query, bot_id=self.current_bot, limit=100)
            self.current_view = "memory_search"
            safe_query = query.replace("'", "''")
            self._set_data(
                result.get("results", []),
                f"SELECT * FROM {self.current_bot}_memories WHERE content ILIKE '%{safe_query}%' LIMIT 100",
            )
        except Exception as exc:
            self.notify(f"Search failed: {exc}", severity="error")

    async def _run_message_search(self, query: str) -> None:
        try:
            result = await self.api_client.search_messages(query, bot_id=self.current_bot, limit=100)
            self.current_view = "messages_search"
            safe_query = query.replace("'", "''")
            self._set_data(
                result.get("messages", []) or result.get("results", []),
                f"SELECT * FROM {self.current_bot}_messages WHERE content ILIKE '%{safe_query}%' LIMIT 100",
            )
        except Exception as exc:
            self.notify(f"Message search failed: {exc}", severity="error")

    async def _run_profile_lookup(self, entity_id: str) -> None:
        try:
            profile = await self.api_client.get_profile(entity_id)
            rows = []
            attrs = profile.get("attributes", [])
            if attrs:
                for item in attrs:
                    rows.append(
                        {
                            "id": item.get("id", ""),
                            "category": item.get("category", ""),
                            "key": item.get("key", ""),
                            "value": item.get("value", ""),
                            "confidence": item.get("confidence", ""),
                        }
                    )
            else:
                rows = [profile]
            self.current_view = "profiles_attrs"
            self._set_data(rows, f"SELECT * FROM profile_attributes WHERE entity_id='{entity_id}'")
        except Exception as exc:
            self.notify(f"Profile lookup failed: {exc}", severity="error")

    def on_data_grid_row_selected(self, event: DataGrid.RowSelected):
        detail = self.query_one("#detail_panel", DetailPanel)
        detail.set_data(event.row, self._infer_detail_type())

        console = self.query_one(ConsolePanel)
        console.update_namespace(selected_row=event.row)

    @work
    async def on_data_grid_edit_cell(self, event: DataGrid.EditCell):
        result = await self.push_screen_wait(CellEditModal(column=event.column, value=event.value, row=event.row))
        if result:
            if self.mode == "api":
                await self._handle_api_update(result)
            else:
                await self._handle_db_update(result)

    @work
    async def on_data_grid_edit_row(self, event: DataGrid.EditRow):
        await self._edit_row(event.row)

    @work
    async def on_data_grid_delete_row(self, event: DataGrid.DeleteRow):
        await self._delete_row(event.row)

    @work
    async def on_detail_panel_action(self, event: DetailPanel.Action):
        """Wire detail panel actions to edit/delete/copy operations."""
        row = event.data
        if not row:
            return

        if event.action == "edit":
            await self._edit_row(row)
        elif event.action == "delete":
            await self._delete_row(row)

    async def _edit_row(self, row: dict[str, Any]) -> None:
        result = await self.push_screen_wait(RowEditModal(row=row))
        if result:
            if self.mode == "api":
                await self._handle_api_update(result)
            else:
                await self._handle_db_update(result)

    async def _delete_row(self, row: dict[str, Any]) -> None:
        row_id = row.get("id") or row.get("entity_id") or "selected item"
        confirmed = await self.push_screen_wait(
            ConfirmModal(title="Confirm Delete", message=f"Delete {str(row_id)[:48]}?")
        )
        if not confirmed:
            return

        if self.mode == "api":
            await self._handle_api_delete(row)
        else:
            await self._handle_db_delete(row)

    async def _handle_api_update(self, result: dict):
        self.notify("API update is limited, refreshing data")
        await self._load_api_data()

    async def _handle_db_update(self, result: dict):
        row = result["row"]
        updates = result.get("updates", {})

        table = self._extract_table_from_sql(self._current_sql)
        if not table:
            self.notify("Cannot determine table", severity="error")
            return

        pk_col = self.db_client.get_table_primary_key(table)
        pk_val = row.get(pk_col) if pk_col else row.get("id")

        if pk_col and pk_val is not None:
            db_result = self.db_client.update_row(table, pk_col, pk_val, updates)
            if db_result["success"]:
                self.notify(f"Updated {db_result['rowcount']} row(s)")
                await self._execute_db_query(self._current_sql)
            else:
                self.notify(f"Update failed: {db_result['error']}", severity="error")

    async def _handle_api_delete(self, row: dict):
        try:
            if self.current_view.startswith("messages"):
                msg_id = row.get("id")
                if msg_id:
                    await self.api_client.forget_messages(message_id=msg_id, bot_id=self.current_bot)
                    self.notify("Message forgotten")
            else:
                mem_id = row.get("id")
                if mem_id:
                    await self.api_client.delete_memory(mem_id, bot_id=self.current_bot)
                    self.notify("Memory deleted")

            await self._load_api_data()
        except Exception as exc:
            self.notify(f"Delete failed: {exc}", severity="error")

    async def _handle_db_delete(self, row: dict):
        table = self._extract_table_from_sql(self._current_sql)
        if not table:
            self.notify("Cannot determine table", severity="error")
            return

        pk_col = self.db_client.get_table_primary_key(table)
        pk_val = row.get(pk_col) if pk_col else row.get("id")

        if pk_col and pk_val is not None:
            result = self.db_client.delete_row(table, pk_col, pk_val)
            if result["success"]:
                self.notify(f"Deleted {result['rowcount']} row(s)")
                await self._execute_db_query(self._current_sql)
            else:
                self.notify(f"Delete failed: {result['error']}", severity="error")

    def _extract_table_from_sql(self, sql: str) -> str | None:
        match = re.search(r"FROM\s+(\w+)", sql, re.IGNORECASE)
        return match.group(1) if match else None

    async def _run_consolidate(self):
        confirmed = await self.push_screen_wait(
            ConfirmModal(
                title="Consolidate memories",
                message="Merge redundant memories for current bot?",
            )
        )
        if not confirmed:
            return

        try:
            result = await self.api_client.consolidate_memories(self.current_bot)
            merged = result.get("clusters_merged", 0)
            self.notify(f"Consolidated {merged} clusters")
            await self._load_api_data()
        except Exception as exc:
            self.notify(f"Consolidation failed: {exc}", severity="error")

    async def _run_regenerate_embeddings(self):
        confirmed = await self.push_screen_wait(
            ConfirmModal(
                title="Regenerate embeddings",
                message="Regenerate all memory embeddings for current bot?",
            )
        )
        if not confirmed:
            return

        try:
            result = await self.api_client.regenerate_embeddings(self.current_bot)
            self.notify(f"Updated {result.get('updated', 0)} embeddings")
            await self._load_api_data()
        except Exception as exc:
            self.notify(f"Regeneration failed: {exc}", severity="error")

    async def on_console_panel_refresh_data(self, event: ConsolePanel.RefreshData):
        if self.mode == "api":
            await self._load_api_data()
        else:
            sql = self.query_one("#sql_container", SQLContainer).get_sql()
            if sql:
                await self._execute_db_query(sql)

    async def on_console_panel_set_bot(self, event: ConsolePanel.SetBot):
        self.current_bot = event.bot_id
        self._set_header_subtitle()
        await self._load_api_data()

    async def on_console_panel_search_memories(self, event: ConsolePanel.SearchMemories):
        await self._run_memory_search(event.query)

    async def action_refresh(self):
        if self.mode == "api":
            await self._load_api_data()
        else:
            sql = self.query_one("#sql_container", SQLContainer).get_sql()
            if sql:
                await self._execute_db_query(sql)

    @work
    async def action_quick_search(self):
        if self.mode != "api":
            self.notify("Quick search is API mode only", severity="warning")
            return

        if self.current_view.startswith("messages"):
            query = await self._prompt("Search messages", "Type search text")
            if query:
                await self._run_message_search(query)
        else:
            query = await self._prompt("Search memories", "Type search text")
            if query:
                await self._run_memory_search(query)

    def action_toggle_console(self):
        self.query_one(ConsolePanel).toggle()

    def action_toggle_theme(self):
        themes = ["neural", "textual-dark"]
        idx = themes.index(self.theme) if self.theme in themes else 0
        next_theme = themes[(idx + 1) % len(themes)]
        self.theme = next_theme
        self.notify(f"Theme: {next_theme}")

    def action_help(self):
        self.notify(
            "Tab/Shift+Tab switch panels | Enter select/edit | / search | F2 mode | F5 refresh | F12 console | q quit",
            title="Help",
            timeout=8,
        )

    def _stats_to_rows(self, stats: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        memories = stats.get("memories", {})
        messages = stats.get("messages", {})

        rows.append({"metric": "total_memories", "value": memories.get("total", 0)})
        rows.append({"metric": "total_messages", "value": messages.get("total", 0)})
        rows.append({"metric": "forgotten_messages", "value": messages.get("forgotten", 0)})

        for tag, count in list(memories.get("tag_counts", {}).items())[:20]:
            rows.append({"metric": f"tag:{tag}", "value": count})

        return rows

    async def on_unmount(self):
        await self.api_client.close()


def main():
    """Entry point."""
    app = MemoryTUIApp()
    app.run()


if __name__ == "__main__":
    main()
