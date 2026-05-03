"""Lenient parsing for ``openai_codex_sdk`` thread events.

The SDK's ``parse_thread_item`` strictly validates each item against its
Pydantic model. The codex CLI sometimes emits values that don't match the
SDK's literal types (e.g. ``FileChangeItem.status`` arriving as
``"in_progress"`` mid-edit, against the SDK's ``Literal["completed",
"failed"]``). A single such mismatch raises ``ValidationError`` from
inside the streaming async generator, exhausting it and aborting the
whole turn.

We patch ``parse_thread_item`` once at startup to fall back to
``UnknownThreadItem`` (``extra="allow"``, no literal constraints) when
strict validation fails, so the stream survives. ``_handle_event``
already reads item attributes via ``getattr``, so an UnknownThreadItem
behaves like the typed item for our rendering purposes.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

logger = logging.getLogger(__name__)


def install() -> None:
    """Replace ``openai_codex_sdk.parsing.parse_thread_item`` in-place."""
    from openai_codex_sdk import parsing
    from openai_codex_sdk.errors import EventParseError
    from openai_codex_sdk.types import UnknownThreadItem

    if getattr(parsing, "_codex_bridge_lenient", False):
        return  # already patched

    original = parsing.parse_thread_item

    def lenient_parse_thread_item(data: Any):
        try:
            return original(data)
        except ValidationError as e:
            item_type = data.get("type") if isinstance(data, dict) else None
            logger.debug(
                "Lenient parse: %s validation failed (%s); using UnknownThreadItem",
                item_type, e,
            )
            return UnknownThreadItem.model_validate(data)
        except EventParseError:
            raise

    parsing.parse_thread_item = lenient_parse_thread_item
    parsing._codex_bridge_lenient = True
    logger.info("Installed lenient parse_thread_item patch")
