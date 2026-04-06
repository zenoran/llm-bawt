"""Claude Code bridge — translates Agent SDK events to OpenClaw event protocol.

Reads chat.send commands from the shared Redis ``openclaw:commands`` stream,
runs them through the Claude Agent SDK (which spawns the Claude Code binary),
and publishes OpenClawEvent-compatible events back to Redis run streams.

The main app sees identical events regardless of whether the openclaw-bridge
or this claude-code-bridge handled the request.
"""
