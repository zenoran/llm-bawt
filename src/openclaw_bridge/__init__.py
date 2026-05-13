"""OpenClaw WebSocket bridge — standalone consumer service.

Can run as a separate process/container: python -m openclaw_bridge
Connects to the OpenClaw gateway over WebSocket, normalizes its event
protocol, and publishes ``AgentEvent``s to Redis via the shared
``agent_bridge`` transport package.
"""

from agent_bridge import AgentEvent, AgentEventKind, SessionQueue

__all__ = ["AgentEvent", "AgentEventKind", "SessionQueue"]
