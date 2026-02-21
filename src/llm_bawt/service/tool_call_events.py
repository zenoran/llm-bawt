"""Helpers for correlating tool-call turn logs to history message IDs."""

from __future__ import annotations

from typing import Any


def collect_message_identifiers(message: dict[str, Any]) -> list[str]:
    """Collect known message identifier keys from one message payload."""
    ids: list[str] = []
    for key in ("id", "db_id", "message_id"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            ids.append(value.strip())
    return ids


def extract_trigger_message(request_payload: Any) -> tuple[str, str, float | None] | None:
    """Extract the trigger message for a turn (last user message with an ID)."""
    if not isinstance(request_payload, dict):
        return None
    messages = request_payload.get("messages")
    if not isinstance(messages, list):
        return None

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "") != "user":
            continue
        identifiers = collect_message_identifiers(message)
        if not identifiers:
            continue
        ts_value = message.get("timestamp")
        timestamp = float(ts_value) if isinstance(ts_value, (int, float)) else None
        return identifiers[0], "user", timestamp
    return None


def parse_message_filters(message_id: str | None, message_ids: list[str] | None) -> set[str]:
    """Normalize message-id filters from single and repeated/CSV query params."""
    out: set[str] = set()
    if message_id and message_id.strip():
        out.add(message_id.strip())
    for value in message_ids or []:
        if not value:
            continue
        for chunk in str(value).split(","):
            chunk = chunk.strip()
            if chunk:
                out.add(chunk)
    return out


def message_id_matches(candidate: str, targets: set[str]) -> bool:
    """Match full IDs and short-prefix IDs (common in CLI usage)."""
    if not targets:
        return True
    if candidate in targets:
        return True
    for target in targets:
        if len(target) < 36 and candidate.startswith(target):
            return True
    return False
