"""Shared validation for dispatcher-owned inter-bot correlation fields."""

from __future__ import annotations

from typing import Any


_CORRELATION_FIELDS = (
    "inter_bot_turn_id",
    "inter_bot_bridge_request_id",
    "inter_bot_claim_token",
    "inter_bot_timeout_seconds",
    "inter_bot_session_policy",
    "inter_bot_seed_session_id",
)


def validate_inter_bot_claim(service: Any, request: Any) -> None:
    """Reject spoofed or stale durable-delivery authority on every chat path."""
    delivery_id = getattr(request, "inter_bot_delivery_id", None)
    if delivery_id:
        dispatcher = getattr(service, "_inter_bot_dispatcher", None)
        if dispatcher is None or not dispatcher.store.validate_claim(
            delivery_id=delivery_id,
            claim_token=getattr(request, "inter_bot_claim_token", None) or "",
            target_bot_id=(getattr(request, "bot_id", None) or service._default_bot),
            turn_id=getattr(request, "inter_bot_turn_id", None) or "",
            user_message_id=getattr(request, "user_message_id", None) or "",
            bridge_request_id=(
                getattr(request, "inter_bot_bridge_request_id", None) or ""
            ),
        ):
            raise ValueError("invalid or stale inter-bot delivery claim")
        return

    if any(getattr(request, field, None) for field in _CORRELATION_FIELDS):
        raise ValueError(
            "inter-bot correlation fields require a valid delivery claim"
        )
