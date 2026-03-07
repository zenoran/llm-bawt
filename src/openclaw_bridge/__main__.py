"""Standalone OpenClaw bridge service.

Usage:
    python -m openclaw_bridge
"""

import asyncio
import logging
import signal
import sys

from .bridge import SessionBridge
from .config import BridgeConfig
from .ingest import EventIngestPipeline, IngestFilterConfig
from .metrics import get_metrics
from .publisher import RedisPublisher
from .store import EventStore, create_openclaw_tables
from .ws_client import OpenClawWsClient, OpenClawWsConfig


async def _health_server(bridge: SessionBridge, publisher: RedisPublisher, port: int) -> None:
    """Minimal TCP health check on the given port.

    Returns HTTP 200 if WS + Redis are connected, 503 otherwise.
    Docker HEALTHCHECK can use: curl -f http://localhost:<port>/health
    """
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readline()  # consume request line
            ws_ok = bridge.connected
            redis_ok = publisher.connected
            ok = ws_ok and redis_ok
            status = "200 OK" if ok else "503 Service Unavailable"
            body = f'{{"ws": {str(ws_ok).lower()}, "redis": {str(redis_ok).lower()}}}'
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
    logger = logging.getLogger("openclaw_bridge.health")
    logger.info("Health check listening on :%d", port)
    async with server:
        await server.serve_forever()


def main() -> None:
    config = BridgeConfig.from_env()

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger("openclaw_bridge")

    if not config.ws_url:
        logger.error("OPENCLAW_WS_URL not set, cannot start bridge")
        sys.exit(1)

    # PostgreSQL
    from sqlalchemy import create_engine
    engine = create_engine(config.postgres_url)
    create_openclaw_tables(engine)

    # Redis
    publisher = RedisPublisher(config.redis_url)
    if not publisher.connected:
        logger.error("Cannot connect to Redis at %s", config.redis_url)
        sys.exit(1)

    # WS client
    ws_config = OpenClawWsConfig(
        url=config.ws_url,
        token=config.ws_token,
        session_keys=config.session_keys,
        reconnect_max_delay=config.reconnect_max_delay,
    )
    ws_client = OpenClawWsClient(ws_config)

    # Ingest filters
    ingest_filter = IngestFilterConfig.from_env(
        drop_patterns_csv=config.ingest_drop_patterns,
        drop_events_csv=config.ingest_drop_events,
        drop_msg_types_csv=config.ingest_drop_msg_types,
    )

    # Fetch session->bot mapping from main app API
    session_to_bot = config.fetch_session_to_bot()

    # Assemble bridge
    bridge = SessionBridge(
        ws_client=ws_client,
        ingest=EventIngestPipeline(filter_config=ingest_filter),
        store=EventStore(engine),
        publisher=publisher,
        session_to_bot=session_to_bot,
    )

    # Metrics
    metrics = get_metrics()
    metrics.set_tags(service="openclaw-bridge")

    logger.info(
        "Starting OpenClaw bridge (sessions=%s, bot_map=%s)",
        config.session_keys,
        session_to_bot,
    )

    loop = asyncio.new_event_loop()

    def _shutdown(sig: signal.Signals) -> None:
        logger.info("Received %s, shutting down...", sig.name)
        loop.create_task(bridge.stop())
        loop.call_later(2, loop.stop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown, sig)

    async def _run() -> None:
        # Start health check server
        health_task = asyncio.create_task(
            _health_server(bridge, publisher, config.health_port)
        )
        try:
            await bridge.run_forever()
        finally:
            health_task.cancel()

    try:
        loop.run_until_complete(_run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(bridge.stop())
        metrics.close()
        loop.close()


if __name__ == "__main__":
    main()
