"""Codex bridge — translates OpenAI Codex SDK events to OpenClaw event protocol.

Reads chat.send commands from the shared Redis ``agent:commands`` stream,
runs them through the OpenAI Codex SDK (which spawns the codex Rust binary),
and publishes AgentEvent-compatible events back to Redis run streams.

The main app sees identical events regardless of whether the openclaw-bridge,
the claude-code-bridge, or this codex-bridge handled the request.
"""
