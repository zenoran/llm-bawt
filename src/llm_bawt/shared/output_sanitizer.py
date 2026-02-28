"""Utilities for stripping model/tool protocol leakage from user-facing text."""

from __future__ import annotations

import re


_ROUTING_FRAGMENT_RE = re.compile(
    r"(?:^|\s)to=(?:functions|multi_tool_use)\.[^\n\r]*",
    re.IGNORECASE,
)

_TOOL_USES_JSON_RE = re.compile(
    r'\{\s*"tool_uses"\s*:\s*\[.*?\]\s*\}',
    re.DOTALL | re.IGNORECASE,
)


def strip_tool_protocol_leakage(text: str) -> str:
    """Remove leaked tool-routing protocol fragments from model text.

    This strips artifacts such as:
    - ``to=functions.some_tool commentary ...``
    - ``to=multi_tool_use.parallel ...``
    - Raw JSON payloads like ``{"tool_uses": [...]}``
    """
    if not text:
        return ""

    cleaned = _TOOL_USES_JSON_RE.sub("", text)
    cleaned = _ROUTING_FRAGMENT_RE.sub(" ", cleaned)

    # Remove control characters often left behind after malformed protocol output.
    cleaned = re.sub(r"[\x00-\x08\x0e-\x1f\x7f-\x9f]", "", cleaned)

    # Normalize whitespace while preserving line breaks.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()
