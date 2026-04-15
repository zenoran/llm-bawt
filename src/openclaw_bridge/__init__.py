"""OpenClaw WebSocket bridge — standalone consumer service.

Can run as a separate process/container: python -m openclaw_bridge
Also importable from the main llm-bawt app for shared types.
"""

from .events import OpenClawEvent, OpenClawEventKind
from .session_queue import SessionQueue

__all__ = ["OpenClawEvent", "OpenClawEventKind", "SessionQueue"]
