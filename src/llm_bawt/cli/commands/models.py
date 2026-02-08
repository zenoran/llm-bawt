"""Models command - model listing."""

# Re-export from the main cli module for now  
# This will be migrated to standalone implementation in future
# Note: list_models comes from model_manager, not cli
from ...model_manager import list_models as show_models

__all__ = ["show_models"]
