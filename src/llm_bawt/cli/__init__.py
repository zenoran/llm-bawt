"""CLI package for llm-bawt.

This package provides the command-line interface, split into logical modules:
- app: Main CLI application, argument parsing, and command handlers
- config_setup: Interactive .env configuration walkthrough (llm --setup)
- bot_editor: Bot YAML editing
- commands/: Subcommand implementations (status, profile, models)
"""

# Re-export main entry points from app module
from .app import main, run_app, show_status, show_bots, show_users
from .app import show_user_profile, run_user_profile_setup, ensure_user_profile

__all__ = [
    "main",
    "run_app",
    "show_status",
    "show_bots",
    "show_users",
    "show_user_profile",
    "run_user_profile_setup",
    "ensure_user_profile",
]
