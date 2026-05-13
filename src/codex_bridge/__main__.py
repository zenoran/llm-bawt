"""Standalone Codex bridge service.

Usage:
    python -m codex_bridge

Environment:
    REDIS_URL                       — Redis connection (default redis://localhost:6379/0)
    LLM_BAWT_API_URL                — main app base URL for /v1/bots calls
    CODEX_AUTH_PATH                 — auth.json path (default /home/bridge/.codex/auth.json)
    CODEX_HOME                      — exported as CODEX_HOME for the SDK (default /home/bridge/.codex)
    CODEX_MODEL                     — default model when chat.send omits it
    CODEX_BACKEND_NAME              — backend filter on commands stream (default 'codex')
    CODEX_BRIDGE_HEALTH_PORT        — TCP port for /health (default 8682)
    CODEX_BRIDGE_LOG_LEVEL          — log level (default INFO)
    CODEX_BRIDGE_REQUEST_TIMEOUT    — per-call SDK timeout, seconds (default 300)
    CODEX_BRIDGE_CWD                — codex thread cwd (default /home/bridge/dev)
    CODEX_BIN                       — explicit codex binary path (default bundled)
    CODEX_LOCAL_PLUGINS_ENABLED     — stage repo-managed local plugins into ~/.agents + ~/plugins
    CODEX_LOCAL_PLUGINS_SRC         — source mapping root (default /home/bridge/dev/agent-skills/codex)
    CODEX_DEV_ROOT                  — dev root used to resolve repo-managed skills (default /home/bridge/dev)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from agent_bridge.publisher import RedisPublisher

from .bridge import CodexBridge
from .exec_patch import install as install_exec_patch
from .local_plugins import install_repo_local_plugins
from .parser_patch import install as install_parser_patch
from .transport import auth_path, scrub_api_key_env, validate_auth_json


async def _health_server(publisher: RedisPublisher, port: int) -> None:
    """Minimal TCP health check.

    Returns 200 when Redis is connected, 503 otherwise. Mirrors the shape of
    claude_code_bridge's health server so docker compose healthchecks across
    bridges look identical.
    """

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
    logger = logging.getLogger("codex_bridge.health")
    logger.info("Health check listening on :%d", port)
    async with server:
        await server.serve_forever()


def _redact(s: str | None, *, keep: int = 6) -> str:
    if not s:
        return "<unset>"
    if len(s) <= keep:
        return "*" * len(s)
    return s[:keep] + "*" * (len(s) - keep)


def main() -> None:
    log_level = os.getenv("CODEX_BRIDGE_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger("codex_bridge")

    # Patch the SDK's strict-literal parser so a single mismatched item
    # status doesn't kill the whole turn (see parser_patch.py).
    install_parser_patch()
    install_exec_patch()

    # --- TASK-204: scrub API key env vars before constructing the SDK ---
    scrubbed = scrub_api_key_env()
    if scrubbed:
        logger.warning(
            "Removed %s from environment — Codex bridge is OAuth-only.",
            ", ".join(scrubbed),
        )

    # --- TASK-204: validate auth.json before any SDK call ---
    auth_file = auth_path()
    try:
        auth_data = validate_auth_json(auth_file)
    except RuntimeError as e:
        logger.error("Auth bootstrap failed: %s", e)
        sys.exit(1)

    tokens = auth_data.get("tokens") or {}
    logger.info(
        "Codex OAuth ok: path=%s auth_mode=%s account_id=%s last_refresh=%s",
        auth_file,
        (tokens.get("auth_mode") or auth_data.get("auth_mode") or "chatgpt"),
        _redact(tokens.get("account_id") or auth_data.get("account_id")),
        auth_data.get("last_refresh") or tokens.get("last_refresh") or "<unknown>",
    )

    # Make sure CODEX_HOME points at the dir containing auth.json so the SDK
    # picks up the same files we just validated.
    codex_home = os.getenv("CODEX_HOME") or str(auth_file.parent)
    os.environ["CODEX_HOME"] = codex_home
    install_repo_local_plugins(
        logger=logger,
        codex_home=Path(codex_home),
    )

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    publisher = RedisPublisher(
        redis_url,
        default_provider=os.getenv("CODEX_BACKEND_NAME", "codex"),
    )
    if not publisher.connected:
        logger.error("Cannot connect to Redis at %s", redis_url)
        sys.exit(1)

    bridge = CodexBridge(
        publisher=publisher,
        backend_name=os.getenv("CODEX_BACKEND_NAME", "codex"),
        app_api_url=os.getenv("LLM_BAWT_API_URL", ""),
        default_model=os.getenv("CODEX_MODEL", "gpt-5.4"),
        cwd=os.getenv("CODEX_BRIDGE_CWD", "/home/bridge/dev"),
        codex_bin=os.getenv("CODEX_BIN") or None,
        request_timeout=float(os.getenv("CODEX_BRIDGE_REQUEST_TIMEOUT", "900")),
    )

    health_port = int(os.getenv("CODEX_BRIDGE_HEALTH_PORT", "8682"))

    logger.info(
        "Starting Codex bridge (backend=%s, model=%s, cwd=%s, codex_home=%s)",
        bridge.backend_name,
        bridge.default_model,
        bridge.cwd,
        codex_home,
    )

    async def _run() -> None:
        shutdown_event = asyncio.Event()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_event.set)

        health_task = asyncio.create_task(_health_server(publisher, health_port))
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
