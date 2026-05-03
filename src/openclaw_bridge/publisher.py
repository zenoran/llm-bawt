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

logger = logging.getLogger(__name__)

# Stream names
EVENTS_STREAM_PREFIX = "openclaw:events:"
HISTORY_STREAM = "openclaw:history"
COMMANDS_STREAM = "openclaw:commands"
RUN_STREAM_PREFIX = "openclaw:run:"
# Unified event stream for all bots (native + OpenClaw): events:{bot_id}:{user_id}
UNIFIED_EVENTS_PREFIX = "events:"

# Keep last 10k events per session stream (auto-trimmed)
STREAM_MAXLEN = 10_000
# Unified event streams are smaller — tool events + lifecycle only
UNIFIED_STREAM_MAXLEN = 5_000
# Run response streams are shorter-lived
RUN_STREAM_MAXLEN = 5_000


class RedisPublisher:
    def __init__(self, redis_url: str, *, default_provider: str | None = None) -> None:
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._connected = False
        # Bridges instantiate one publisher and pass their backend name here
        # ("claude-code", "codex", "openclaw"). Stamps every event at the
        # publish boundary so downstream consumers can dispatch on provider
        # without each call site remembering to set it.
        self._default_provider = default_provider
        try:
            self._redis.ping()
            self._connected = True
            logger.info("Redis publisher connected: %s", redis_url)
        except redis.ConnectionError as e:
            logger.error("Redis publisher connect failed: %s", e)

    def _stamp_provider(self, event: OpenClawEvent) -> OpenClawEvent:
        if event.provider is None and self._default_provider is not None:
            event.provider = self._default_provider
        return event

    def publish_event(self, event: OpenClawEvent) -> str | None:
        """Publish an event to the session's Redis Stream. Returns the stream ID."""
        if not self._connected:
            return None
        stream_key = f"{EVENTS_STREAM_PREFIX}{event.session_key}"
        self._stamp_provider(event)
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
            return stream_id
        except Exception:
            logger.exception("Failed to publish event to Redis stream %s", stream_key)
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
            return stream_id
        except Exception:
            logger.exception("Failed to publish history command")
            return None

    def publish_run_event(self, request_id: str, event: OpenClawEvent) -> str | None:
        """Publish an event to a per-run response stream for the requesting client."""
        if not self._connected:
            return None
        stream_key = f"{RUN_STREAM_PREFIX}{request_id}"
        self._stamp_provider(event)
        try:
            data = event.to_dict()
            fields = {"payload": json.dumps(data, ensure_ascii=False, default=str)}
            stream_id = self._redis.xadd(
                stream_key,
                fields,
                maxlen=RUN_STREAM_MAXLEN,
                approximate=True,
            )
            return stream_id
        except Exception:
            logger.exception("Failed to publish run event to %s", stream_key)
            return None

    def publish_rpc_result(self, request_id: str, result: dict) -> None:
        """Publish an RPC call result to a short-lived response stream."""
        if not self._connected:
            return
        stream_key = f"openclaw:rpc:{request_id}"
        try:
            fields = {"payload": json.dumps(result, ensure_ascii=False, default=str)}
            self._redis.xadd(stream_key, fields, maxlen=10)
            self._redis.expire(stream_key, 60)
        except Exception:
            logger.exception("Failed to publish RPC result to %s", stream_key)

    def publish_run_done(self, request_id: str) -> None:
        """Signal that a run is complete by writing a sentinel entry."""
        if not self._connected:
            return
        stream_key = f"{RUN_STREAM_PREFIX}{request_id}"
        try:
            self._redis.xadd(stream_key, {"done": "1"}, maxlen=RUN_STREAM_MAXLEN, approximate=True)
            # Expire the run stream after 5 minutes (cleanup)
            self._redis.expire(stream_key, 300)
        except Exception:
            logger.exception("Failed to publish run done to %s", stream_key)

    def publish_unified_event(
        self,
        bot_id: str,
        user_id: str,
        event_data: dict[str, Any],
    ) -> str | None:
        """Publish an event to the unified event stream for a bot+user pair.

        Used for tool execution events, OpenClaw agent events, and any
        application-level events that the UI consumes via consumer groups.
        """
        if not self._connected:
            return None
        stream_key = f"{UNIFIED_EVENTS_PREFIX}{bot_id}:{user_id}"
        try:
            fields = {"payload": json.dumps(event_data, ensure_ascii=False, default=str)}
            stream_id = self._redis.xadd(
                stream_key,
                fields,
                maxlen=UNIFIED_STREAM_MAXLEN,
                approximate=True,
            )
            return stream_id
        except Exception:
            logger.exception("Failed to publish unified event to %s", stream_key)
            return None

    def close(self) -> None:
        try:
            self._redis.close()
        except Exception:
            pass

    @property
    def connected(self) -> bool:
        return self._connected
