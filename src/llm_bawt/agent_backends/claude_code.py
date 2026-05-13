"""Claude Code agent backend.

Uses the shared agent-bridge Redis command/event protocol, but routes
to the claude-code-bridge (Agent SDK) instead of the openclaw-bridge
(WebSocket gateway).

Each user gets their own Claude conversation thread — the session key
is scoped by user_id (e.g. ``claude-code:main:nick``).
"""

from __future__ import annotations

from .agent_bridge import AgentBridgeBackend


class ClaudeCodeBackend(AgentBridgeBackend):
    """Agent backend for Claude Code via the Agent SDK bridge.

    Inherits the full Redis command/event protocol from AgentBridgeBackend.
    Session keys are user-scoped so each user gets their own conversation.

    Configuration keys (``agent_backend_config`` in bot profile):
        session_key: Base session key (default: "claude-code:main")
        timeout_seconds: Max wait for response (default: 120)
        model: Claude model alias (default, opus[1m], haiku, sonnet[1m])
    """

    name = "claude-code"

    def _resolve_session_key(self, config: dict) -> str:
        # Route by bot + user so each user gets an independent Claude session
        # for a given bot. The session_key in agent_backend_config is an SDK
        # session UUID written by the bridge and should not be used directly
        # as a routing key.
        bot_id = str(config.get("bot_id") or "main").strip() or "main"
        user_id = str(config.get("user_id") or "default").strip() or "default"
        return f"{bot_id}:{user_id}"
