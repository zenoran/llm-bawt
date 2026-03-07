"""Redis Streams subscriber — consumer side counterpart to RedisPublisher.

Used by the main app to read events from Redis Streams instead of
the in-memory FanoutHub.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

import redis.asyncio as aioredis

from .events import OpenClawEvent
from .publisher import COMMANDS_STREAM, EVENTS_STREAM_PREFIX, HISTORY_STREAM, RUN_STREAM_PREFIX

logger = logging.getLogger(__name__)


class RedisSubscriber:
    def __init__(self, redis_url: str) -> None:
        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._connected = False

    async def connect(self) -> None:
        try:
            await self._redis.ping()
            self._connected = True
            logger.info("Redis subscriber connected")
        except Exception as e:
            logger.error("Redis subscriber connect failed: %s", e)

    async def subscribe(
        self,
        session_key: str,
        *,
        run_id: str | None = None,
        timeout_s: float = 300,
    ) -> AsyncIterator[OpenClawEvent]:
        """Stream events for a session from Redis Streams.

        If run_id is provided, yields events until RUN_COMPLETED for that run,
        then stops. Otherwise streams indefinitely until cancelled.
        """
        stream_key = f"{EVENTS_STREAM_PREFIX}{session_key}"
        last_id = "$"  # only new events
        deadline = asyncio.get_event_loop().time() + timeout_s

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return

            # XREAD with block (in ms), max 1s blocks to allow timeout checks
            block_ms = min(int(remaining * 1000), 1000)
            try:
                results = await self._redis.xread(
                    {stream_key: last_id},
                    count=100,
                    block=block_ms,
                )
            except Exception:
                logger.exception("Redis XREAD error on %s", stream_key)
                await asyncio.sleep(1)
                continue

            if not results:
                continue

            for _stream_name, messages in results:
                for msg_id, fields in messages:
                    last_id = msg_id
                    payload_str = fields.get("payload", "{}")
                    try:
                        data = json.loads(payload_str)
                        event = OpenClawEvent.from_dict(data)
                    except Exception:
                        logger.debug("Failed to parse event from Redis: %s", payload_str[:200])
                        continue

                    yield event

                    # If tracking a specific run, stop on completion
                    if run_id and event.run_id == run_id:
                        from .events import OpenClawEventKind
                        if event.kind in (OpenClawEventKind.RUN_COMPLETED, OpenClawEventKind.ERROR):
                            return

    async def send_command(
        self,
        session_key: str,
        message: str,
        request_id: str,
    ) -> None:
        """Publish a chat.send command to the bridge's command stream."""
        await self._redis.xadd(
            COMMANDS_STREAM,
            {
                "action": "chat.send",
                "session_key": session_key,
                "message": message,
                "request_id": request_id,
            },
            maxlen=1000,
            approximate=True,
        )
        logger.debug("Sent command: request_id=%s session=%s", request_id, session_key)

    async def subscribe_run(
        self,
        request_id: str,
        *,
        timeout_s: float = 600,
    ) -> AsyncIterator[OpenClawEvent]:
        """Subscribe to the per-run response stream published by the bridge.

        Yields events until the bridge signals 'done' or timeout.
        """
        stream_key = f"{RUN_STREAM_PREFIX}{request_id}"
        last_id = "0"  # read from beginning (stream is created fresh per run)
        deadline = asyncio.get_event_loop().time() + timeout_s

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning("subscribe_run timeout for request_id=%s", request_id)
                return

            block_ms = min(int(remaining * 1000), 2000)
            try:
                results = await self._redis.xread(
                    {stream_key: last_id},
                    count=100,
                    block=block_ms,
                )
            except Exception:
                logger.exception("Redis XREAD error on %s", stream_key)
                await asyncio.sleep(1)
                continue

            if not results:
                continue

            for _stream_name, messages in results:
                for msg_id, fields in messages:
                    last_id = msg_id

                    # Check for done sentinel
                    if fields.get("done"):
                        return

                    payload_str = fields.get("payload", "{}")
                    try:
                        data = json.loads(payload_str)
                        event = OpenClawEvent.from_dict(data)
                    except Exception:
                        logger.debug("Failed to parse run event: %s", payload_str[:200])
                        continue

                    yield event

    async def drain_history(
        self,
        callback,
        *,
        consumer_group: str = "llm-bawt",
        consumer_name: str = "main",
    ) -> None:
        """Continuously drain the openclaw:history stream and call callback(bot_id, role, content).

        Uses consumer groups so multiple main-app instances can share the load.
        """
        # Ensure consumer group exists
        try:
            await self._redis.xgroup_create(
                HISTORY_STREAM, consumer_group, id="0", mkstream=True
            )
        except Exception:
            pass  # group already exists

        while True:
            try:
                results = await self._redis.xreadgroup(
                    consumer_group,
                    consumer_name,
                    {HISTORY_STREAM: ">"},
                    count=10,
                    block=5000,
                )
                if not results:
                    continue

                for _stream, messages in results:
                    for msg_id, fields in messages:
                        bot_id = fields.get("bot_id", "")
                        role = fields.get("role", "")
                        content = fields.get("content", "")
                        if bot_id and role and content:
                            try:
                                callback(bot_id, role, content)
                            except Exception:
                                logger.exception("History callback failed for bot=%s", bot_id)
                        # ACK the message
                        await self._redis.xack(HISTORY_STREAM, consumer_group, msg_id)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("History drain error")
                await asyncio.sleep(2)

    async def close(self) -> None:
        try:
            await self._redis.aclose()
        except Exception:
            pass

    @property
    def connected(self) -> bool:
        return self._connected
