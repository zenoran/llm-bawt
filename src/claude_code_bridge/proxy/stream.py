"""Responses API stream events → Anthropic Messages API SSE events.

This translator is **feature-complete** against the OpenAI Responses streaming
event set (`openai.types.responses.ResponseStreamEvent`, 53 event types as of
openai-python 2.x) and emits a fully spec-compliant Anthropic Messages SSE
stream so the Claude Agent SDK consumes a non-Anthropic provider exactly as if
it were talking to api.anthropic.com.

Anthropic event sequence we produce:

    message_start
      content_block_start (index, thinking)      ← reasoning summary / CoT
        content_block_delta (thinking_delta)*
        content_block_delta (signature_delta)     ← reasoning encrypted_content
      content_block_stop  (index)
      content_block_start (index, text)
        content_block_delta (text_delta)*
      content_block_stop  (index)
      content_block_start (index, tool_use)
        content_block_delta (input_json_delta)*
      content_block_stop  (index)
    message_delta (stop_reason, usage)
    message_stop

plus periodic ``ping`` frames (Anthropic emits these too) so a slow upstream
never reads as a dead connection.

Why reasoning matters: high-effort reasoning models (gpt-5.x via the codex
backend, GLM, …) stream a long burst of ``response.reasoning*`` events BEFORE
any visible text. The previous translator dropped them, so the whole reasoning
window was silent SSE — the Claude SDK and the UI saw a frozen turn until the
first text token. Surfacing reasoning as Anthropic ``thinking`` blocks (a) keeps
the stream alive and (b) shows the model's thinking, matching native Anthropic.

Responses events consumed (grouped):

    lifecycle:   response.created / .in_progress / .queued
                 response.completed / .incomplete / .failed / .error / error
    reasoning:   response.output_item.added(type=reasoning)
                 response.reasoning_text.delta / .done
                 response.reasoning_summary_text.delta / .done
                 response.reasoning_summary_part.added / .done
                 response.output_item.done(type=reasoning)   → encrypted_content
    text:        response.content_part.added / .done
                 response.output_text.delta / .done
                 response.output_text.annotation.added
                 response.refusal.delta / .done
    tools:       response.output_item.added(type=function_call|custom_tool_call)
                 response.function_call_arguments.delta / .done
                 response.custom_tool_call_input.delta / .done
                 response.output_item.done(type=function_call|custom_tool_call)
    server-side: web_search_call / code_interpreter_call / mcp_call /
                 image_generation_call / file_search_call (kept alive, see note)
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


def _ping() -> bytes:
    return _sse("ping", {"type": "ping"})


# Responses ``status`` / ``incomplete reason`` → Anthropic stop_reason.
_STOP_REASON_MAP = {
    "stop": "end_turn",
    "completed": "end_turn",
    "length": "max_tokens",
    "max_output_tokens": "max_tokens",
    "max_tokens": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
    "refusal": "refusal",
}


def _anthropic_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _usage_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from an object or dict (Responses usage is either)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_usage(resp: Any) -> tuple[int, int, int, int]:
    """Pull (input, output, cache_read, cache_create) from a Response object.

    Responses API reports cached prompt tokens under
    ``usage.input_tokens_details.cached_tokens``; ``input_tokens`` is the FULL
    prompt (cached + uncached), unlike Anthropic where ``input_tokens`` is the
    uncached remainder. The caller normalises to Anthropic's split.

    Also accepts dict-shaped usage (some SDKs / xAI edge paths) and the
    chat-completions alias ``prompt_tokens_details.cached_tokens``.
    """
    usage = _usage_get(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return 0, 0, 0, 0
    input_tokens = int(
        _usage_get(usage, "input_tokens", None)
        or _usage_get(usage, "prompt_tokens", 0)
        or 0
    )
    output_tokens = int(
        _usage_get(usage, "output_tokens", None)
        or _usage_get(usage, "completion_tokens", 0)
        or 0
    )
    cache_read = 0
    details = (
        _usage_get(usage, "input_tokens_details", None)
        or _usage_get(usage, "prompt_tokens_details", None)
    )
    if details is not None:
        cache_read = int(_usage_get(details, "cached_tokens", 0) or 0)
    return input_tokens, output_tokens, cache_read, 0


async def responses_to_anthropic_sse(
    upstream_stream: AsyncIterator[Any],
    anthropic_model: str,
) -> AsyncIterator[bytes]:
    """Translate a Responses API event stream into Anthropic SSE bytes."""

    message_id = _anthropic_message_id()

    # Emit message_start up front so the SDK has the envelope before any
    # content arrives. Our Responses-backed providers only know real usage at
    # response.completed, so we start with zeroes and refine at message_delta.
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

    # Anthropic permits at most one *open* content block at a time and requires
    # strict open→delta→stop ordering. Responses emits output items
    # sequentially (reasoning, then message text, then function calls), so we
    # model a single ``open_block`` and a per-item index map for routing arg
    # deltas to the right tool block.
    next_index = 0
    open_block: dict | None = None          # {"index", "kind", "item_id", "call_id"?}
    blocks_by_item: dict[str, dict] = {}    # item_id → block dict (tools)

    saw_tool_use = False
    saw_refusal = False
    explicit_stop: str | None = None
    input_tokens = output_tokens = cache_read = cache_create = 0

    def _open_start_frame(kind: str, idx: int, **extra: Any) -> bytes:
        if kind == "thinking":
            block = {"type": "thinking", "thinking": "", "signature": ""}
        elif kind == "tool":
            block = {
                "type": "tool_use",
                "id": extra.get("call_id", ""),
                "name": extra.get("name", ""),
                "input": {},
            }
        else:  # text
            block = {"type": "text", "text": ""}
        return _sse(
            "content_block_start",
            {"type": "content_block_start", "index": idx, "content_block": block},
        )

    def _stop_frame(idx: int) -> bytes:
        return _sse("content_block_stop", {"type": "content_block_stop", "index": idx})

    try:
        async for event in upstream_stream:
            etype = getattr(event, "type", "") or ""

            # ── lifecycle: pre-content keepalive ────────────────────────────
            if etype in ("response.created", "response.in_progress", "response.queued"):
                yield _ping()
                continue

            # ── output item opened ──────────────────────────────────────────
            if etype == "response.output_item.added":
                item = getattr(event, "item", None)
                itype = getattr(item, "type", "") if item else ""
                if itype == "reasoning":
                    if open_block is not None:
                        yield _stop_frame(open_block["index"])
                        open_block = None
                    idx = next_index
                    next_index += 1
                    open_block = {"index": idx, "kind": "thinking",
                                  "item_id": getattr(item, "id", "") or ""}
                    yield _open_start_frame("thinking", idx)
                elif itype in ("function_call", "custom_tool_call"):
                    if open_block is not None:
                        yield _stop_frame(open_block["index"])
                        open_block = None
                    item_id = getattr(item, "id", "") or ""
                    call_id = (
                        getattr(item, "call_id", "")
                        or item_id
                        or f"call_{uuid.uuid4().hex[:16]}"
                    )
                    name = getattr(item, "name", "") or ""
                    idx = next_index
                    next_index += 1
                    block = {"index": idx, "kind": "tool",
                             "item_id": item_id, "call_id": call_id}
                    open_block = block
                    blocks_by_item[item_id] = block
                    saw_tool_use = True
                    yield _open_start_frame("tool", idx, call_id=call_id, name=name)
                elif itype == "message":
                    # Text container — the actual text block opens lazily on the
                    # first content_part.added / output_text.delta.
                    pass
                else:
                    # Server-side tool calls (web_search, code_interpreter, mcp,
                    # image_generation, file_search). The Claude Agent SDK drives
                    # this proxy with function tools only, so these are not
                    # reachable in practice — but keep the stream alive and log
                    # if a provider ever emits one so we notice.
                    logger.debug("Unhandled output item type=%r — keepalive", itype)
                    yield _ping()
                continue

            # ── reasoning deltas ────────────────────────────────────────────
            if etype in ("response.reasoning_text.delta",
                         "response.reasoning_summary_text.delta"):
                delta = getattr(event, "delta", "") or ""
                if not delta:
                    continue
                if open_block is None or open_block["kind"] != "thinking":
                    if open_block is not None:
                        yield _stop_frame(open_block["index"])
                    idx = next_index
                    next_index += 1
                    open_block = {"index": idx, "kind": "thinking", "item_id": ""}
                    yield _open_start_frame("thinking", idx)
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": open_block["index"],
                        "delta": {"type": "thinking_delta", "thinking": delta},
                    },
                )
                continue

            if etype == "response.reasoning_summary_part.added":
                # A new summary section. Separate sections with a blank line so
                # multi-part reasoning summaries stay readable.
                summary_index = getattr(event, "summary_index", 0) or 0
                if (summary_index > 0 and open_block is not None
                        and open_block["kind"] == "thinking"):
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": open_block["index"],
                            "delta": {"type": "thinking_delta", "thinking": "\n\n"},
                        },
                    )
                continue

            # ── text / refusal deltas ───────────────────────────────────────
            if etype in ("response.output_text.delta", "response.refusal.delta"):
                delta = getattr(event, "delta", "") or ""
                if not delta:
                    continue
                if open_block is None or open_block["kind"] != "text":
                    if open_block is not None:
                        yield _stop_frame(open_block["index"])
                    idx = next_index
                    next_index += 1
                    open_block = {"index": idx, "kind": "text", "item_id": ""}
                    yield _open_start_frame("text", idx)
                if etype == "response.refusal.delta":
                    saw_refusal = True
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": open_block["index"],
                        "delta": {"type": "text_delta", "text": delta},
                    },
                )
                continue

            if etype == "response.content_part.added":
                # Open the text block eagerly when a text/refusal part starts so
                # an empty part still produces a well-formed block pair.
                part = getattr(event, "part", None)
                ptype = getattr(part, "type", "") if part else ""
                if ptype in ("output_text", "text", "refusal"):
                    if open_block is None or open_block["kind"] != "text":
                        if open_block is not None:
                            yield _stop_frame(open_block["index"])
                        idx = next_index
                        next_index += 1
                        open_block = {"index": idx, "kind": "text", "item_id": ""}
                        yield _open_start_frame("text", idx)
                continue

            # ── function / custom tool argument deltas ──────────────────────
            if etype in ("response.function_call_arguments.delta",
                         "response.custom_tool_call_input.delta"):
                item_id = getattr(event, "item_id", "") or ""
                partial = getattr(event, "delta", "") or ""
                block = blocks_by_item.get(item_id)
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
                continue

            # ── output item closed ──────────────────────────────────────────
            if etype == "response.output_item.done":
                item = getattr(event, "item", None)
                itype = getattr(item, "type", "") if item else ""
                if itype == "reasoning":
                    # Surface the opaque reasoning blob as Anthropic's thinking
                    # signature so the block is well-formed. encrypted_content is
                    # only present when the provider returns it; fall back to a
                    # deterministic non-empty sentinel (the SDK requires a
                    # non-empty signature, and our return-trip drops it anyway —
                    # store:false stateless reasoning).
                    if open_block is not None and open_block["kind"] == "thinking":
                        sig = (
                            getattr(item, "encrypted_content", None)
                            or f"reasoning:{getattr(item, 'id', '') or message_id}"
                        )
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": open_block["index"],
                                "delta": {"type": "signature_delta", "signature": sig},
                            },
                        )
                        yield _stop_frame(open_block["index"])
                        open_block = None
                elif itype in ("function_call", "custom_tool_call"):
                    item_id = getattr(item, "id", "") or ""
                    block = blocks_by_item.get(item_id) or open_block
                    if block is not None:
                        yield _stop_frame(block["index"])
                        if open_block is block:
                            open_block = None
                    saw_tool_use = True
                elif itype == "message":
                    if open_block is not None and open_block["kind"] == "text":
                        yield _stop_frame(open_block["index"])
                        open_block = None
                continue

            # ── terminal: success ───────────────────────────────────────────
            if etype == "response.completed":
                resp = getattr(event, "response", None)
                if resp is not None:
                    input_tokens, output_tokens, cache_read, cache_create = _extract_usage(resp)
                    logger.info(
                        "Responses usage: input=%d cached=%d uncached=%d output=%d cache_hit=%.1f%%",
                        input_tokens, cache_read,
                        max(input_tokens - cache_read, 0), output_tokens,
                        (100.0 * cache_read / input_tokens) if input_tokens else 0.0,
                    )
                    raw_stop = (
                        getattr(resp, "status", "")
                        or getattr(resp, "stop_reason", "")
                    )
                    if raw_stop in _STOP_REASON_MAP:
                        explicit_stop = _STOP_REASON_MAP[raw_stop]
                continue

            # ── terminal: incomplete (e.g. hit max_output_tokens) ───────────
            if etype == "response.incomplete":
                resp = getattr(event, "response", None)
                if resp is not None:
                    input_tokens, output_tokens, cache_read, cache_create = _extract_usage(resp)
                    details = getattr(resp, "incomplete_details", None)
                    reason = getattr(details, "reason", "") if details else ""
                    explicit_stop = _STOP_REASON_MAP.get(reason, "max_tokens")
                continue

            # ── terminal: failure / error ───────────────────────────────────
            if etype in ("response.failed", "response.error", "error"):
                err = getattr(event, "error", None)
                if err is None:
                    resp = getattr(event, "response", None)
                    err = getattr(resp, "error", None) if resp is not None else None
                msg = getattr(err, "message", "") or "upstream error"
                code = getattr(err, "code", "") or "upstream_error"
                # Close any open block before the error so the SDK's parser
                # doesn't choke on a dangling block.
                if open_block is not None:
                    yield _stop_frame(open_block["index"])
                    open_block = None
                yield _sse(
                    "error",
                    {"type": "error", "error": {"type": code, "message": msg}},
                )
                return

            # ── events with no Anthropic analog: close-out / annotations /
            #    *.done markers / server-tool progress / audio. Safe to skip,
            #    but emit a keepalive so a quiet provider never stalls. ───────
            if etype in (
                "response.output_text.done",
                "response.refusal.done",
                "response.content_part.done",
                "response.output_text.annotation.added",
                "response.reasoning_text.done",
                "response.reasoning_summary_text.done",
                "response.reasoning_summary_part.done",
                "response.function_call_arguments.done",
                "response.custom_tool_call_input.done",
            ):
                continue

            # Anything else (server-tool progress, audio, image partials, …):
            # keep the connection warm and move on.
            logger.debug("Passthrough keepalive for event type=%r", etype)
            yield _ping()

        # ── finalize ────────────────────────────────────────────────────────
        if open_block is not None:
            yield _stop_frame(open_block["index"])
            open_block = None

        # Precedence: a tool call ends the turn for the agent loop; an explicit
        # non-default terminal (max_tokens, content-filter refusal) outranks a
        # generic completion; a refusal content block outranks a plain
        # end_turn (the upstream still reports status=completed on refusals).
        if saw_tool_use:
            stop_reason = "tool_use"
        elif explicit_stop is not None and explicit_stop != "end_turn":
            stop_reason = explicit_stop
        elif saw_refusal:
            stop_reason = "refusal"
        elif explicit_stop is not None:
            stop_reason = explicit_stop
        else:
            stop_reason = "end_turn"

        uncached_input_tokens = max((input_tokens or 0) - (cache_read or 0), 0)
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {
                    "input_tokens": uncached_input_tokens,
                    "output_tokens": output_tokens or 0,
                    "cache_creation_input_tokens": cache_create or 0,
                    "cache_read_input_tokens": cache_read or 0,
                },
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})

    except Exception as exc:  # noqa: BLE001
        logger.exception("Stream translation failed")
        if open_block is not None:
            try:
                yield _stop_frame(open_block["index"])
            except Exception:  # noqa: BLE001
                pass
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
