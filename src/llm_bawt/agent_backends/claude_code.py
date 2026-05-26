"""Claude Code agent backend.

Uses the shared agent-bridge Redis command/event protocol, but routes
to the claude-code-bridge (Agent SDK) instead of the openclaw-bridge
(WebSocket gateway).

Each user gets their own Claude conversation thread — the session key
is scoped by user_id (e.g. ``claude-code:main:nick``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .agent_bridge import AgentBridgeBackend


class ClaudeCodeBackend(AgentBridgeBackend):
    """Agent backend for Claude Code via the Agent SDK bridge.

    Inherits the full Redis command/event protocol from AgentBridgeBackend.
    Session keys are user-scoped so each user gets their own conversation.

    Configuration keys (``agent_backend_config`` in bot profile):
        session_key: Base session key (default: "claude-code:main")
        timeout_seconds: Max wait for response (default: 120)
        model: Claude model SDK ID — REQUIRED, no fallback.
    """

    name = "claude-code"

    def stream_raw(
        self,
        prompt: str,
        config: dict,
        attachments: list | None = None,
        trigger_message_id: str | None = None,
    ) -> Iterator[str | dict[str, Any]]:
        # Hard-require an explicit model. No silent fallback to Sonnet (or
        # any other model) — if the bot's agent_backend_config is missing
        # ``model``, fail loudly with a message that names the bot so the
        # operator can fix the config.
        model = str(config.get("model") or "").strip()
        if not model:
            bot_id = str(config.get("bot_id") or "?").strip() or "?"
            raise ValueError(
                f"Claude Code backend: bot={bot_id} has no 'model' in "
                f"agent_backend_config. Set it on the bot's profile — the "
                f"bridge will not fall back to a default."
            )
        return super().stream_raw(
            prompt,
            config,
            attachments=attachments,
            trigger_message_id=trigger_message_id,
        )

    def _resolve_session_key(self, config: dict) -> str:
        # Route by bot + user so each user gets an independent Claude session
        # for a given bot. The session_key in agent_backend_config is an SDK
        # session UUID written by the bridge and should not be used directly
        # as a routing key.
        bot_id = str(config.get("bot_id") or "main").strip() or "main"
        user_id = str(config.get("user_id") or "default").strip() or "default"
        return f"{bot_id}:{user_id}"
