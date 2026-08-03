"""Live provider-neutral usage checkpoint and event construction."""
from __future__ import annotations

import time
from typing import Any


class TurnUsageCoordinator:
    """Own latest-snapshot persistence and public live usage payloads."""

    def __init__(self, turn_log_store: Any) -> None:
        self._turn_log_store = turn_log_store

    def capture(
        self,
        *,
        turn_id: str,
        trigger_message_id: str | None,
        bot_id: str,
        user_id: str,
        model: str,
        token_usage: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Checkpoint a partial snapshot and return its unified event envelope."""
        if not isinstance(token_usage, dict) or not token_usage:
            return None
        snapshot = {**token_usage, "usage_status": "partial"}
        self._turn_log_store.update_turn(turn_id=turn_id, token_usage=snapshot)
        return {
            "_type": "turn_usage",
            "turn_id": turn_id,
            "trigger_message_id": trigger_message_id,
            "bot_id": bot_id,
            "user_id": user_id,
            "model": model,
            "token_usage": snapshot,
            "ts": time.time(),
        }

    @staticmethod
    def http_chunk(event: dict[str, Any]) -> dict[str, Any]:
        """Return the metadata-only OpenAI-compatible initiating-stream chunk."""
        return {
            "object": "chat.completion.usage",
            "turn_id": event["turn_id"],
            "trigger_message_id": event.get("trigger_message_id"),
            "model": event.get("model"),
            "token_usage": event["token_usage"],
        }
