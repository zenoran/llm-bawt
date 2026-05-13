"""Re-export from the shared agent_bridge package."""
from agent_bridge.store import EventStore, create_agent_event_tables

__all__ = ["EventStore", "create_agent_event_tables"]
