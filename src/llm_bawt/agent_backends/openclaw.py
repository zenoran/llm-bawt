"""OpenClaw agent backend.

Routes chat through the OpenClaw bridge's persistent WebSocket connection:

    Main App → Redis command → openclaw-bridge → WS chat.send → Gateway
    Gateway  → WS events     → openclaw-bridge → Redis run stream → Main App

Inherits the full Redis command/event protocol from
:class:`AgentBridgeBackend`. Only differences from the shared base:

- ``name = "openclaw"`` (registry key; also stamped as event ``provider``)
- ``_resolve_session_key`` falls back to ``OPENCLAW_SESSION_KEY`` env var
"""

from __future__ import annotations

import os

from .agent_bridge import AgentBridgeBackend


class OpenClawBackend(AgentBridgeBackend):
    """Agent backend for OpenClaw via the openclaw-bridge."""

    name = "openclaw"

    def _resolve_session_key(self, config: dict) -> str:
        explicit = str(config.get("session_key") or "").strip()
        if explicit:
            return explicit
        return os.getenv("OPENCLAW_SESSION_KEY", "main")
