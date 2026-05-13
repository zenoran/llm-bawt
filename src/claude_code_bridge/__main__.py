"""Standalone Claude Code bridge service.

Usage:
    python -m claude_code_bridge
"""

import asyncio
import logging
import os
import signal
import sys

from agent_bridge.publisher import RedisPublisher

from .bridge import ClaudeCodeBridge


async def _health_server(bridge: ClaudeCodeBridge, publisher: RedisPublisher, port: int) -> None:
    """Minimal TCP health check."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readline()
            redis_ok = publisher.connected
            status = "200 OK" if redis_ok else "503 Service Unavailable"
            body = f'{{"redis": {str(redis_ok).lower()}}}'
            resp = (
                f"HTTP/1.1 {status}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"\r\n"
                f"{body}"
            )
            writer.write(resp.encode())
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "0.0.0.0", port)
    logger = logging.getLogger("claude_code_bridge.health")
    logger.info("Health check listening on :%d", port)
    async with server:
        await server.serve_forever()


def main() -> None:
    log_level = os.getenv("CLAUDE_CODE_BRIDGE_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy httpx request logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger("claude_code_bridge")

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    publisher = RedisPublisher(redis_url, default_provider="claude-code")
    if not publisher.connected:
        logger.error("Cannot connect to Redis at %s", redis_url)
        sys.exit(1)

    # Check for OAuth token (env var or credentials file)
    from .bridge import _get_fresh_oauth_token
    if not _get_fresh_oauth_token():
        logger.error(
            "No OAuth token available. Either set CLAUDE_CODE_OAUTH_TOKEN "
            "or mount ~/.claude/.credentials.json into the container."
        )
        sys.exit(1)

    add_dirs_raw = os.getenv("CLAUDE_CODE_ADD_DIRS", "")
    add_dirs = [d.strip() for d in add_dirs_raw.split(",") if d.strip()]

    bridge = ClaudeCodeBridge(
        publisher=publisher,
        backend_name=os.getenv("CLAUDE_CODE_BACKEND_NAME", "claude-code"),
        app_api_url=os.getenv("LLM_BAWT_API_URL", ""),
        default_model=os.getenv("CLAUDE_CODE_MODEL", "claude-sonnet-4-20250514"),
        cwd=os.getenv("CLAUDE_CODE_CWD", "/app"),
        permission_mode=os.getenv("CLAUDE_CODE_PERMISSION_MODE", "bypassPermissions"),
        add_dirs=add_dirs,
        request_timeout=float(os.getenv("CLAUDE_CODE_REQUEST_TIMEOUT", "300")),
    )

    health_port = int(os.getenv("CLAUDE_CODE_BRIDGE_HEALTH_PORT", "8681"))

    logger.info(
        "Starting Claude Code bridge (backend=%s, model=%s)",
        os.getenv("CLAUDE_CODE_BACKEND_NAME", "claude-code"),
        os.getenv("CLAUDE_CODE_MODEL", "claude-sonnet-4-20250514"),
    )

    async def _run() -> None:
        shutdown_event = asyncio.Event()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_event.set)

        health_task = asyncio.create_task(
            _health_server(bridge, publisher, health_port)
        )
        bridge_task = asyncio.create_task(bridge.run_forever())

        await shutdown_event.wait()
        logger.info("Shutting down...")

        bridge_task.cancel()
        health_task.cancel()
        await bridge.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
