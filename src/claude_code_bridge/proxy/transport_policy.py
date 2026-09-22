"""Catalog-driven ChatGPT Responses transport policy.

The model catalog owns the selected transport.  This module is the shared
normalization boundary used by app-side catalog validation, bridge command
parsing, request metadata, and the ChatGPT adapter.
"""

from __future__ import annotations

RESPONSES_TRANSPORT_SSE = "sse"
RESPONSES_TRANSPORT_LITE_WS = "lite_ws"
DEFAULT_RESPONSES_TRANSPORT = RESPONSES_TRANSPORT_SSE
VALID_RESPONSES_TRANSPORTS = frozenset(
    {RESPONSES_TRANSPORT_SSE, RESPONSES_TRANSPORT_LITE_WS}
)


def normalize_responses_transport(value: object) -> str | None:
    """Return one canonical transport value, or ``None`` when invalid/absent."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in VALID_RESPONSES_TRANSPORTS else None
