"""CLI package for llmbothub.

This package provides the command-line interface, split into logical modules:
- app: Main CLI application and command handlers
- parser: Argument parsing
- commands/: Subcommand implementations (status, profile, models)
"""

# Re-export main entry points from app module
from .app import main, run_app, show_status, show_bots, show_users
from .app import show_user_profile, run_user_profile_setup, ensure_user_profile
from .app import parse_arguments

__all__ = [
    "main",
    "run_app",
    "parse_arguments",
    "show_status",
    "show_bots",
    "show_users",
    "show_user_profile",
    "run_user_profile_setup",
    "ensure_user_profile",
]
