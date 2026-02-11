"""Navigation trees for API and DB modes."""

from __future__ import annotations

from typing import Any

from textual.widgets import Tree
from textual.message import Message


class APINavTree(Tree):
    """Structured navigation tree for API mode."""
    
    ALLOW_SELECT = True
    
    BINDINGS = [
        ("enter", "select_cursor", "Select"),
    ]
    
    DEFAULT_CSS = """
    APINavTree {
        width: 1fr;
        border: none;
        background: transparent;
        padding: 0 1;
    }
    
    APINavTree:focus {
        border-left: solid $primary;
    }
    
    APINavTree > .tree--cursor {
        color: $primary;
        text-style: bold;
    }
    
    APINavTree > .tree--highlight {
        color: $accent;
        text-style: bold;
    }
    
    APINavTree > .tree--guides,
    APINavTree > .tree--guide {
        color: $surface;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__("API Views", **kwargs)
        self._setup_tree()
    
    def _setup_tree(self):
        """Setup the tree structure."""
        # Expand root
        self.root.expand()
        
        # Memories section
        memories = self.root.add("Memories", expand=True)
        memories.add_leaf("Search", data={"type": "memory_search"})
        memories.add_leaf("Browse All", data={"type": "memory_browse"})
        memories.add_leaf("By Tags", data={"type": "memory_tags"})
        memories.add_leaf("By Importance", data={"type": "memory_importance"})
        memories.add_leaf("Consolidate", data={"type": "memory_consolidate"})
        memories.add_leaf("Regenerate Embeddings", data={"type": "memory_embeddings"})
        
        # Messages section
        messages = self.root.add("Messages", expand=True)
        messages.add_leaf("Recent", data={"type": "messages_recent"})
        messages.add_leaf("Search", data={"type": "messages_search"})
        messages.add_leaf("Forgotten", data={"type": "messages_forgotten"})
        messages.add_leaf("Summaries", data={"type": "messages_summaries"})
        
        # Profiles section
        profiles = self.root.add("Profiles", expand=True)
        profiles.add_leaf("Users", data={"type": "profiles_users"})
        profiles.add_leaf("Bots", data={"type": "profiles_bots"})
        profiles.add_leaf("By Entity", data={"type": "profiles_attrs"})
        
        # Maintenance section
        maint = self.root.add("Maintenance", expand=True)
        maint.add_leaf("Statistics", data={"type": "stats"})
        maint.add_leaf("Cleanup Tasks", data={"type": "cleanup"})
        
        # Bots section (will be populated dynamically)
        self._bots_node = self.root.add("Bots", expand=False)
    
    def update_bots(self, bots: list[dict[str, Any]]):
        """Update the bots list in tree."""
        self._bots_node.remove_children()
        for bot in bots:
            bot_id = bot.get("id", "unknown")
            name = bot.get("name", bot_id)
            self._bots_node.add_leaf(name, data={"type": "select_bot", "bot_id": bot_id})
    
    def on_key(self, event) -> None:
        """Handle key events - vim-style navigation."""
        if event.key in ("j", "down"):
            event.stop()
            self.action_cursor_down()
        elif event.key in ("k", "up"):
            event.stop()
            self.action_cursor_up()
    
    def action_select_cursor(self):
        """Handle selection (Enter key)."""
        node = self.cursor_node
        if node and node.data:
            self.post_message(APINavigate(node.data))

    def on_tree_node_selected(self, event: Tree.NodeSelected[dict[str, Any]]) -> None:
        """Handle mouse/node selection."""
        if event.node.data:
            self.post_message(APINavigate(event.node.data))


class APINavigate(Message):
    """Navigation message for API tree."""
    
    def __init__(self, data: dict[str, Any]):
        self.data = data
        super().__init__()


class DBNavTree(Tree):
    """Table browser tree for DB mode."""
    
    ALLOW_SELECT = True
    
    DEFAULT_CSS = """
    DBNavTree {
        width: 1fr;
        border: none;
        background: transparent;
        padding: 0 1;
    }
    
    DBNavTree:focus {
        border-left: solid $primary;
    }
    
    DBNavTree > .tree--cursor {
        color: $primary;
        text-style: bold;
    }
    
    DBNavTree > .tree--highlight {
        color: $accent;
        text-style: bold;
    }
    
    DBNavTree > .tree--guides,
    DBNavTree > .tree--guide {
        color: $surface;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__("Database", **kwargs)
        self._tables_node = None
        self._queries_node = None
        self._setup_tree()
    
    def _setup_tree(self):
        """Setup the tree structure."""
        # Expand root
        self.root.expand()
        
        # Tables section (populated dynamically)
        self._tables_node = self.root.add("Tables", expand=True)
        
        # Custom queries
        self._queries_node = self.root.add("Queries", expand=True)
        self._queries_node.add_leaf("New Query...", data={"type": "new_query"})
        self._queries_node.add_leaf("Recent Queries", data={"type": "recent_queries"})
        self._queries_node.add_leaf("Saved Queries", data={"type": "saved_queries"})
        
        # System
        system = self.root.add("System", expand=False)
        system.add_leaf("Connection Info", data={"type": "db_info"})
    
    def update_tables(self, tables: list[dict[str, Any]], bot_id: str | None = None):
        """Update tables list from database."""
        self._tables_node.remove_children()
        
        # Group by category
        categories: dict[str, list[dict]] = {}
        for table in tables:
            cat = table.get("category", "other")
            categories.setdefault(cat, []).append(table)
        
        # Add by category
        category_icons = {
            "memories": "memories",
            "messages": "messages",
            "forgotten": "forgotten",
            "summaries": "summaries",
            "profiles": "profiles",
            "attributes": "attributes",
            "other": "other",
        }
        
        for cat in ["memories", "messages", "forgotten", "summaries", "profiles", "attributes", "other"]:
            if cat not in categories:
                continue
            
            icon = category_icons.get(cat, "other")
            cat_node = self._tables_node.add(icon.title(), expand=(cat in ["memories", "messages"]))
            
            for table in categories[cat]:
                name = table["name"]
                # For bot-specific tables, show cleaner name
                display_name = name
                if bot_id and name.startswith(f"{bot_id}_"):
                    display_name = name[len(bot_id)+1:]
                
                cat_node.add_leaf(display_name, data={
                    "type": "table",
                    "table": name,
                    "category": cat,
                })
    
    def on_key(self, event) -> None:
        """Handle key events - vim-style navigation."""
        if event.key in ("j", "down"):
            event.stop()
            self.action_cursor_down()
        elif event.key in ("k", "up"):
            event.stop()
            self.action_cursor_up()
    
    def action_select_cursor(self):
        """Handle selection (Enter key)."""
        node = self.cursor_node
        if node and node.data:
            self.post_message(DBNavigate(node.data))

    def on_tree_node_selected(self, event: Tree.NodeSelected[dict[str, Any]]) -> None:
        """Handle mouse/node selection."""
        if event.node.data:
            self.post_message(DBNavigate(event.node.data))


class DBNavigate(Message):
    """Navigation message for DB tree."""
    
    def __init__(self, data: dict[str, Any]):
        self.data = data
        super().__init__()
