"""Standalone local-model bridge service.

Usage:
    python -m local_model_bridge

Environment:
    REDIS_URL                          — Redis connection (default redis://localhost:6379/0)
    LLM_BAWT_API_URL                   — main app base URL for /v1/models calls
    LOCAL_MODEL_BACKEND_NAME           — backend filter on commands stream (default 'local')
    LOCAL_MODEL_BRIDGE_HEALTH_PORT     — TCP port for /health (default 8683)
    LOCAL_MODEL_BRIDGE_EMBED_PORT      — TCP port for the /embed API (default 8684)
    LOCAL_MODEL_BRIDGE_LOG_LEVEL       — log level (default INFO)
    LOCAL_MODEL_BRIDGE_REQUEST_TIMEOUT — per-call generation timeout, seconds (default 1800)
    LOCAL_MODEL_EMBED_MODEL            — sentence-transformers model (default all-MiniLM-L6-v2)
    LOCAL_MODEL_EMBED_DEVICE           — embed model device (default "cpu")
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from agent_bridge.publisher import RedisPublisher

from .bridge import LocalModelBridge


async def _health_server(publisher: RedisPublisher, port: int) -> None:
    """Minimal TCP health check.

    Returns 200 when Redis is connected, 503 otherwise. Mirrors the shape of
    the other bridges' health servers so docker compose healthchecks across
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
    logger = logging.getLogger("local_model_bridge.health")
    logger.info("Health check listening on :%d", port)
    async with server:
        await server.serve_forever()


def main() -> None:
    log_level = os.getenv("LOCAL_MODEL_BRIDGE_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger("local_model_bridge")

    backend_name = os.getenv("LOCAL_MODEL_BACKEND_NAME", "local")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    publisher = RedisPublisher(redis_url, default_provider=backend_name)
    if not publisher.connected:
        logger.error("Cannot connect to Redis at %s", redis_url)
        sys.exit(1)

    bridge = LocalModelBridge(
        publisher=publisher,
        backend_name=backend_name,
        app_api_url=os.getenv("LLM_BAWT_API_URL", ""),
        request_timeout=float(os.getenv("LOCAL_MODEL_BRIDGE_REQUEST_TIMEOUT", "1800")),
    )

    health_port = int(os.getenv("LOCAL_MODEL_BRIDGE_HEALTH_PORT", "8683"))
    embed_port = int(os.getenv("LOCAL_MODEL_BRIDGE_EMBED_PORT", "8684"))

    logger.info(
        "Starting local-model bridge (backend=%s, app_api_url=%s, health_port=%d, embed_port=%d)",
        bridge.backend_name,
        os.getenv("LLM_BAWT_API_URL", ""),
        health_port,
        embed_port,
    )

    async def _run() -> None:
        shutdown_event = asyncio.Event()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_event.set)

        # Embed API (TASK-277): MiniLM lives here now; the app POSTs to /embed.
        # Imported lazily so a sentence-transformers/fastapi import error can't
        # stop the chat bridge from starting.
        from .embed_server import serve_embed

        health_task = asyncio.create_task(_health_server(publisher, health_port))
        bridge_task = asyncio.create_task(bridge.run_forever())
        embed_task = asyncio.create_task(serve_embed(embed_port))

        await shutdown_event.wait()
        logger.info("Shutting down...")

        bridge_task.cancel()
        health_task.cancel()
        embed_task.cancel()
        await bridge.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
