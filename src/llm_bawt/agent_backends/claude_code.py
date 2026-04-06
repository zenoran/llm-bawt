"""Claude Code agent backend.

Uses the same Redis command/event protocol as the OpenClaw backend,
but routes to the claude-code-bridge (Agent SDK) instead of the
openclaw-bridge (WebSocket gateway).

Each user gets their own Claude conversation thread — the session key
is scoped by user_id (e.g. ``claude-code:main:nick``).
"""

from __future__ import annotations

from .openclaw import OpenClawBackend


class ClaudeCodeBackend(OpenClawBackend):
    """Agent backend for Claude Code via the Agent SDK bridge.

    Inherits the full Redis command/event protocol from OpenClawBackend.
    Session keys are user-scoped so each user gets their own conversation.

    Configuration keys (``agent_backend_config`` in bot profile):
        session_key: Base session key (default: "claude-code:main")
        timeout_seconds: Max wait for response (default: 120)
        model: Claude model alias (default, opus[1m], haiku, sonnet[1m])
    """

    name = "claude-code"

    def _resolve_session_key(self, config: dict) -> str:
        # Always use bot_id — the session_key in agent_backend_config
        # is the SDK session UUID (written by the bridge), not a routing key
        return config.get("bot_id", "main")
