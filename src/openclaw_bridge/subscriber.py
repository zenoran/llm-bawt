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
from .publisher import (
    COMMANDS_STREAM,
    EVENTS_STREAM_PREFIX,
    HISTORY_STREAM,
    RUN_STREAM_PREFIX,
    UNIFIED_EVENTS_PREFIX,
    UNIFIED_STREAM_MAXLEN,
)

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
        attachments: list | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        backend: str | None = None,
        bot_id: str | None = None,
    ) -> None:
        """Publish a chat.send command to the bridge's command stream."""
        fields: dict = {
            "action": "chat.send",
            "session_key": session_key,
            "message": message,
            "request_id": request_id,
        }
        if attachments:
            fields["attachments"] = json.dumps(attachments, ensure_ascii=False)
        if system_prompt:
            fields["system_prompt"] = system_prompt
        if model:
            fields["model"] = model
        if backend:
            fields["backend"] = backend
        if bot_id:
            fields["bot_id"] = bot_id
        await self._redis.xadd(
            COMMANDS_STREAM,
            fields,
            maxlen=1000,
            approximate=True,
        )
        logger.debug(
            "Sent command: request_id=%s session=%s attachments=%d",
            request_id, session_key, len(attachments or []),
        )

    async def send_rpc(
        self,
        method: str,
        params: dict,
        request_id: str,
        *,
        timeout_s: float = 15,
    ) -> dict:
        """Send an RPC command to the bridge and wait for the result."""
        import json as _json

        await self._redis.xadd(
            COMMANDS_STREAM,
            {
                "action": "rpc.call",
                "method": method,
                "params": _json.dumps(params, ensure_ascii=False),
                "request_id": request_id,
            },
            maxlen=1000,
            approximate=True,
        )

        # Wait for result on openclaw:rpc:{request_id}
        stream_key = f"openclaw:rpc:{request_id}"
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            remaining_ms = int((deadline - asyncio.get_event_loop().time()) * 1000)
            if remaining_ms <= 0:
                break
            results = await self._redis.xread(
                {stream_key: "0-0"}, count=1, block=min(remaining_ms, 2000),
            )
            if results:
                for _stream, messages in results:
                    for _msg_id, fields in messages:
                        payload_raw = fields.get("payload", "{}")
                        return _json.loads(payload_raw)
        raise TimeoutError(f"RPC {method} timed out after {timeout_s}s")

    async def subscribe_run(
        self,
        request_id: str,
        *,
        timeout_s: float = 600,
    ) -> AsyncIterator[OpenClawEvent]:
        """Subscribe to the per-run response stream published by the bridge.

        Yields events until the bridge signals 'done' or inactivity timeout.
        The timeout resets on each received event, so long-running tasks
        (e.g. backups) that continue producing events won't time out.
        """
        stream_key = f"{RUN_STREAM_PREFIX}{request_id}"
        last_id = "0"  # read from beginning (stream is created fresh per run)
        loop_time = asyncio.get_event_loop().time
        deadline = loop_time() + timeout_s
        logger.info(
            "subscribe_run started: request_id=%s timeout_s=%.0f stream=%s",
            request_id, timeout_s, stream_key,
        )

        while True:
            remaining = deadline - loop_time()
            if remaining <= 0:
                logger.warning("subscribe_run inactivity timeout for request_id=%s", request_id)
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
                    # Reset inactivity deadline on each event
                    deadline = loop_time() + timeout_s

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

    # ---- Unified event stream (consumer groups) ----

    async def ensure_consumer_group(
        self,
        stream_key: str,
        group_name: str,
    ) -> None:
        """Idempotent XGROUP CREATE — creates the group starting from latest."""
        try:
            await self._redis.xgroup_create(
                stream_key, group_name, id="$", mkstream=True,
            )
        except Exception:
            pass  # group already exists

    async def subscribe_group(
        self,
        bot_id: str | list[str],
        user_id: str,
        consumer_id: str,
        *,
        timeout_s: float = 86400,
    ) -> AsyncIterator[dict | None]:
        """Subscribe to unified event stream(s) via consumer groups.

        ``bot_id`` may be a single bot or a list of bots.  When multiple bots
        are given, events from ALL of them are interleaved and yielded.

        Two-phase read per stream:
        1. Replay pending (unacked) messages from previous connection
        2. Read new messages as they arrive

        Yields raw event dicts (not OpenClawEvent — unified stream carries
        tool events, OpenClaw events, and application events).
        """
        bot_ids = [bot_id] if isinstance(bot_id, str) else list(bot_id)
        group_name = f"ui:{consumer_id}"
        consumer_name = "reader"

        # Build {stream_key: bot_id} mapping
        stream_keys: dict[str, str] = {}
        for bid in bot_ids:
            sk = f"{UNIFIED_EVENTS_PREFIX}{bid}:{user_id}"
            stream_keys[sk] = bid

        # Ensure consumer groups exist for all streams
        for sk in stream_keys:
            await self.ensure_consumer_group(sk, group_name)

        # Phase 1: replay pending (unacked) messages from all streams
        try:
            results = await self._redis.xreadgroup(
                group_name, consumer_name,
                {sk: "0" for sk in stream_keys},
                count=500,
            )
            if results:
                # Interleave messages from all streams by Redis message ID
                # (timestamp-based) so events aren't serialized per-stream.
                all_msgs: list[tuple[str, str, dict]] = []
                for stream_name, messages in results:
                    for msg_id, fields in messages:
                        all_msgs.append((msg_id, stream_name, fields))
                all_msgs.sort(key=lambda x: x[0])

                for msg_id, stream_name, fields in all_msgs:
                    payload_str = fields.get("payload", "{}")
                    try:
                        data = json.loads(payload_str)
                    except Exception:
                        data = {"raw": payload_str}
                    data["_replayed"] = True
                    yield data
                    await self._redis.xack(stream_name, group_name, msg_id)
        except Exception:
            logger.exception("Error replaying pending messages for %s", list(stream_keys))

        # Phase 2: live events from all streams
        deadline = asyncio.get_event_loop().time() + timeout_s
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return

            block_ms = min(int(remaining * 1000), 2000)
            try:
                results = await self._redis.xreadgroup(
                    group_name, consumer_name,
                    {sk: ">" for sk in stream_keys},
                    count=100,
                    block=block_ms,
                )
            except Exception:
                logger.exception("Redis XREADGROUP error on %s", list(stream_keys))
                await asyncio.sleep(1)
                continue

            if not results:
                yield None  # keepalive tick — lets caller send SSE ping
                continue

            # Interleave messages from all streams by Redis message ID
            # so concurrent bot events are delivered in chronological order
            # instead of all-of-stream-A then all-of-stream-B.
            all_msgs: list[tuple[str, str, dict]] = []
            for stream_name, messages in results:
                for msg_id, fields in messages:
                    all_msgs.append((msg_id, stream_name, fields))
            all_msgs.sort(key=lambda x: x[0])

            for msg_id, stream_name, fields in all_msgs:
                deadline = asyncio.get_event_loop().time() + timeout_s
                payload_str = fields.get("payload", "{}")
                try:
                    data = json.loads(payload_str)
                except Exception:
                    data = {"raw": payload_str}
                yield data
                await self._redis.xack(stream_name, group_name, msg_id)

    async def publish_tool_event(
        self,
        bot_id: str,
        user_id: str,
        event: dict,
    ) -> str | None:
        """Publish a tool_start/tool_end event to the unified stream."""
        stream_key = f"{UNIFIED_EVENTS_PREFIX}{bot_id}:{user_id}"
        try:
            fields = {"payload": json.dumps(event, ensure_ascii=False, default=str)}
            stream_id = await self._redis.xadd(
                stream_key,
                fields,
                maxlen=UNIFIED_STREAM_MAXLEN,
                approximate=True,
            )
            return stream_id
        except Exception:
            logger.exception("Failed to publish tool event to %s", stream_key)
            return None

    async def cleanup_stale_groups(
        self,
        stream_key: str,
        *,
        max_idle_ms: int = 1_800_000,
    ) -> int:
        """Destroy consumer groups that have been idle too long.

        Only targets ui:* groups (not system groups like llm-bawt).
        Returns the number of groups destroyed.
        """
        destroyed = 0
        try:
            groups = await self._redis.xinfo_groups(stream_key)
        except Exception:
            return 0

        for group in groups:
            name = group.get("name", "")
            if not name.startswith("ui:"):
                continue
            # Check last delivered ID and pending count
            # If no pending and idle > threshold, destroy
            consumers = group.get("consumers", 0)
            pending = group.get("pending", 0)
            if pending > 0:
                continue  # has unacked messages, keep alive
            # Check consumer idle time
            try:
                consumer_info = await self._redis.xinfo_consumers(stream_key, name)
                all_idle = all(
                    (c.get("idle", 0) > max_idle_ms) for c in consumer_info
                ) if consumer_info else (consumers == 0)
            except Exception:
                all_idle = consumers == 0

            if all_idle:
                try:
                    await self._redis.xgroup_destroy(stream_key, name)
                    destroyed += 1
                    logger.debug("Destroyed stale consumer group %s on %s", name, stream_key)
                except Exception:
                    logger.debug("Failed to destroy group %s on %s", name, stream_key)

        return destroyed

    async def drain_tool_events(
        self,
        callback,
        *,
        consumer_group: str = "persist:tools",
        consumer_name: str = "writer",
    ) -> None:
        """Continuously drain tool events from ALL unified streams.

        Uses a dedicated consumer group so persistence is independent
        of UI consumers. Calls ``callback(event_data)`` for each tool event.
        """
        while True:
            try:
                # Discover active unified streams
                streams = await self.list_unified_streams()
                if not streams:
                    await asyncio.sleep(5)
                    continue

                # Ensure consumer group on each stream
                for stream_key in streams:
                    await self.ensure_consumer_group(stream_key, consumer_group)

                # Build read dict
                read_streams = {sk: ">" for sk in streams}

                results = await self._redis.xreadgroup(
                    consumer_group,
                    consumer_name,
                    read_streams,
                    count=50,
                    block=5000,
                )
                if not results:
                    continue

                for stream_key, messages in results:
                    for msg_id, fields in messages:
                        payload_str = fields.get("payload", "{}")
                        try:
                            data = json.loads(payload_str)
                        except Exception:
                            await self._redis.xack(stream_key, consumer_group, msg_id)
                            continue

                        # Only process tool events
                        if data.get("_type") == "tool_event":
                            try:
                                callback(data)
                            except Exception:
                                logger.exception("Tool event persistence callback failed")
                        await self._redis.xack(stream_key, consumer_group, msg_id)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Tool event drain error")
                await asyncio.sleep(2)

    async def list_unified_streams(self) -> list[str]:
        """List all active unified event stream keys."""
        try:
            keys = []
            async for key in self._redis.scan_iter(match=f"{UNIFIED_EVENTS_PREFIX}*", count=100):
                keys.append(key)
            return keys
        except Exception:
            logger.exception("Failed to list unified streams")
            return []

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
