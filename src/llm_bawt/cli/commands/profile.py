"""Profile command - user profile management."""

# Re-export from the app module
from ..app import show_user_profile, run_user_profile_setup, ensure_user_profile

__all__ = ["show_user_profile", "run_user_profile_setup", "ensure_user_profile"]
