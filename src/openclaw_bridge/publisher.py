"""Redis Streams publisher — replaces the in-memory FanoutHub.

Events are published to a Redis Stream keyed by session:
    openclaw:events:{session_key}

History-persist commands are published to:
    openclaw:history

Consumers (main app) read via XREAD or consumer groups.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis

from .events import OpenClawEvent
from .metrics import get_metrics

logger = logging.getLogger(__name__)

# Stream names
EVENTS_STREAM_PREFIX = "openclaw:events:"
HISTORY_STREAM = "openclaw:history"

# Keep last 10k events per session stream (auto-trimmed)
STREAM_MAXLEN = 10_000


class RedisPublisher:
    def __init__(self, redis_url: str) -> None:
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._connected = False
        try:
            self._redis.ping()
            self._connected = True
            logger.info("Redis publisher connected: %s", redis_url)
        except redis.ConnectionError as e:
            logger.error("Redis publisher connect failed: %s", e)

    def publish_event(self, event: OpenClawEvent) -> str | None:
        """Publish an event to the session's Redis Stream. Returns the stream ID."""
        if not self._connected:
            return None
        stream_key = f"{EVENTS_STREAM_PREFIX}{event.session_key}"
        try:
            data = event.to_dict()
            # Redis Streams fields must be flat str->str
            fields = {"payload": json.dumps(data, ensure_ascii=False, default=str)}
            stream_id = self._redis.xadd(
                stream_key,
                fields,
                maxlen=STREAM_MAXLEN,
                approximate=True,
            )
            get_metrics().incr("openclaw.redis_publish", stream="events")
            return stream_id
        except Exception:
            logger.exception("Failed to publish event to Redis stream %s", stream_key)
            get_metrics().incr("openclaw.redis_publish_errors", stream="events")
            return None

    def publish_history(self, bot_id: str, role: str, content: str) -> str | None:
        """Publish a history-persist command for the main app to consume."""
        if not self._connected:
            return None
        try:
            fields = {
                "bot_id": bot_id,
                "role": role,
                "content": content,
            }
            stream_id = self._redis.xadd(
                HISTORY_STREAM,
                fields,
                maxlen=1000,
                approximate=True,
            )
            get_metrics().incr("openclaw.redis_publish", stream="history")
            return stream_id
        except Exception:
            logger.exception("Failed to publish history command")
            get_metrics().incr("openclaw.redis_publish_errors", stream="history")
            return None

    def close(self) -> None:
        try:
            self._redis.close()
        except Exception:
            pass

    @property
    def connected(self) -> bool:
        return self._connected
