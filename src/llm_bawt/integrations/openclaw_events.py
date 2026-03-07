"""Re-export from standalone openclaw_bridge package."""
from openclaw_bridge.events import OpenClawEvent, OpenClawEventKind, synthesize_event_id

__all__ = ["OpenClawEvent", "OpenClawEventKind", "synthesize_event_id"]
