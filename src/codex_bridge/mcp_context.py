"""Request-local trusted MCP context for Codex subprocesses."""

from __future__ import annotations

from collections.abc import Mapping

from agent_bridge.mcp_call_context import (
    MCP_REQUEST_CONTEXT_ENV,
    MCP_TASK_TURN_CONTEXT_ENV,
    mint_mcp_request_context,
)


def codex_mcp_environment(
    base_environment: Mapping[str, str],
    *,
    capability: str,
    agent_request_id: str,
    session_key: str,
    backend: str,
) -> dict[str, str]:
    """Return a fresh environment carrying correlation for exactly one turn."""
    environment = dict(base_environment)
    environment[MCP_TASK_TURN_CONTEXT_ENV] = str(capability or "").strip()
    environment[MCP_REQUEST_CONTEXT_ENV] = mint_mcp_request_context(
        capability=capability,
        agent_request_id=agent_request_id,
        session_key=session_key,
        backend=backend,
    )
    return environment


__all__ = ["codex_mcp_environment"]
