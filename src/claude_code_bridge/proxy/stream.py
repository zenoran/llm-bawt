"""Responses API stream events → Anthropic Messages API SSE events.

Anthropic event sequence:

    message_start
      content_block_start (index=0, text)
        content_block_delta (text_delta)*
      content_block_stop  (index=0)
      content_block_start (index=N, tool_use)
        content_block_delta (input_json_delta)*
      content_block_stop  (index=N)
    message_delta (stop_reason, usage)
    message_stop

Responses API events we consume (from the openai-python AsyncStream):

    response.created
    response.output_text.delta            (text streaming)
    response.output_item.added            (function_call appears)
    response.function_call_arguments.delta
    response.function_call_arguments.done
    response.output_item.done             (function_call closes)
    response.completed                    (terminator, has usage)
    response.error
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


def _sse(event: str, data: dict) -> bytes:
    """Format one SSE frame. Anthropic streams use both ``event:`` and
    ``data:`` lines so curious clients can switch on either."""
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode()


_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "stop_sequence",
}


def _anthropic_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


async def responses_to_anthropic_sse(
    upstream_stream: AsyncIterator[Any],
    anthropic_model: str,
) -> AsyncIterator[bytes]:
    """Translate a Responses API event stream into Anthropic SSE bytes."""

    message_id = _anthropic_message_id()

    # Emit message_start up front so the SDK has the envelope before any
    # content arrives. usage gets refined by message_delta at the end.
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": anthropic_model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        },
    )

    # Content-block index allocation. Index 0 is reserved for the text block
    # (opened lazily on first text delta); tool_use blocks claim subsequent
    # indices in the order Responses emits ``output_item.added``.
    text_index: int | None = None
    text_open = False
    tool_blocks: dict[str, dict] = {}  # item_id → {"index", "open"}
    next_index = 0

    stop_reason: str | None = None
    output_tokens = 0
    input_tokens = 0
    cache_read_input_tokens = 0

    try:
        async for event in upstream_stream:
            etype = getattr(event, "type", "") or ""

            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if not delta:
                    continue
                if text_index is None:
                    text_index = next_index
                    next_index += 1
                if not text_open:
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": text_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                    text_open = True
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": text_index,
                        "delta": {"type": "text_delta", "text": delta},
                    },
                )

            elif etype == "response.output_item.added":
                item = getattr(event, "item", None)
                if not item:
                    continue
                if getattr(item, "type", "") != "function_call":
                    continue
                item_id = getattr(item, "id", "") or ""
                call_id = getattr(item, "call_id", "") or item_id or f"call_{uuid.uuid4().hex[:16]}"
                name = getattr(item, "name", "") or ""
                # Close the text block (if open) before a tool_use block,
                # mirroring Anthropic's strict ordering.
                if text_open and text_index is not None:
                    yield _sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": text_index},
                    )
                    text_open = False
                index = next_index
                next_index += 1
                tool_blocks[item_id] = {
                    "index": index,
                    "open": True,
                    "call_id": call_id,
                    "name": name,
                }
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": call_id,
                            "name": name,
                            "input": {},
                        },
                    },
                )

            elif etype == "response.function_call_arguments.delta":
                item_id = getattr(event, "item_id", "") or ""
                partial = getattr(event, "delta", "") or ""
                block = tool_blocks.get(item_id)
                if block is None or not partial:
                    continue
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block["index"],
                        "delta": {"type": "input_json_delta", "partial_json": partial},
                    },
                )

            elif etype == "response.output_item.done":
                item = getattr(event, "item", None)
                if not item or getattr(item, "type", "") != "function_call":
                    continue
                item_id = getattr(item, "id", "") or ""
                block = tool_blocks.get(item_id)
                if block is None or not block["open"]:
                    continue
                yield _sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": block["index"]},
                )
                block["open"] = False
                stop_reason = stop_reason or "tool_use"

            elif etype == "response.completed":
                resp = getattr(event, "response", None)
                if resp is not None:
                    usage = getattr(resp, "usage", None)
                    if usage is not None:
                        output_tokens = getattr(usage, "output_tokens", 0) or output_tokens
                        input_tokens = getattr(usage, "input_tokens", 0) or input_tokens
                        input_details = getattr(usage, "input_tokens_details", None)
                        if input_details is not None:
                            cache_read_input_tokens = (
                                getattr(input_details, "cached_tokens", 0)
                                or cache_read_input_tokens
                            )
                    # Tool-use stop_reason wins over any text-completion code
                    # — Anthropic semantics: if the turn ended because the
                    # model wants to call a tool, that's the only stop_reason
                    # the client cares about.
                    if stop_reason != "tool_use":
                        raw_stop = (
                            getattr(resp, "status", "")
                            or getattr(resp, "stop_reason", "")
                        )
                        if raw_stop in _STOP_REASON_MAP:
                            stop_reason = _STOP_REASON_MAP[raw_stop]
                # Loop will exit naturally — finalize below.

            elif etype == "response.error":
                err = getattr(event, "error", None)
                msg = getattr(err, "message", "") or "upstream error"
                code = getattr(err, "code", "") or "upstream_error"
                # Emit an Anthropic-shaped error frame and stop. The SDK
                # surfaces these to the bridge as a failed turn.
                yield _sse(
                    "error",
                    {"type": "error", "error": {"type": code, "message": msg}},
                )
                return

        # ── finalize ─────────────────────────────────────────────────────
        # Close any still-open content blocks. text_open being true at this
        # point means a streaming response ended without an explicit close
        # event — Responses API does this when the entire response is text.
        if text_open and text_index is not None:
            yield _sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": text_index},
            )
            text_open = False
        for block in tool_blocks.values():
            if block["open"]:
                yield _sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": block["index"]},
                )
                block["open"] = False

        # If nothing landed in stop_reason, infer from whether any tool blocks
        # were emitted. ``end_turn`` is the catch-all when neither tool nor
        # length limit fired.
        if stop_reason is None:
            stop_reason = "tool_use" if tool_blocks else "end_turn"

        uncached_input_tokens = max((input_tokens or 0) - (cache_read_input_tokens or 0), 0)
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {
                    "input_tokens": uncached_input_tokens,
                    "output_tokens": output_tokens or 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": cache_read_input_tokens or 0,
                },
            },
        )
        yield _sse(
            "message_stop",
            {"type": "message_stop"},
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception("Stream translation failed")
        yield _sse(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "proxy_stream_error",
                    "message": f"Proxy stream translation failed: {exc}",
                },
            },
        )
