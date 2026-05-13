"""Agent bridge — generic transport shared by all bridge backends.

This package holds the wire-format envelope, event store, Redis
publisher/subscriber, and per-session queue that the OpenClaw, Codex,
and Claude Code bridges all use. Backend-specific code (WebSocket
clients, gateway HTTP, SDK adapters) lives in its own package
(``openclaw_bridge``, ``codex_bridge``, ``claude_code_bridge``).
"""

from .events import AgentEvent, AgentEventKind, synthesize_event_id
from .session_queue import SessionQueue

__all__ = ["AgentEvent", "AgentEventKind", "SessionQueue", "synthesize_event_id"]
