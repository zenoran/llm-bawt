"""FastAPI routes for the Anthropic-compatible proxy.

Only one substantive endpoint right now: ``POST /v1/messages``. The
Anthropic Messages API spec also defines ``/v1/messages/count_tokens`` and
some metadata endpoints; the Claude Agent SDK doesn't currently call them
in the workflows we care about, so they're not implemented yet.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .adapters import lookup
from .errors import anthropic_error, auth_failed_error
from .heartbeat import DEFAULT_INTERVAL, with_heartbeat

# Seconds of upstream silence before the proxy injects a keepalive ping.
# Override with CLAUDE_CODE_BRIDGE_PROXY_PING_INTERVAL (0 disables).
try:
    _PING_INTERVAL = float(
        os.getenv("CLAUDE_CODE_BRIDGE_PROXY_PING_INTERVAL", DEFAULT_INTERVAL)
    )
except ValueError:
    _PING_INTERVAL = DEFAULT_INTERVAL

logger = logging.getLogger(__name__)

router = APIRouter()


def _split_model(model_field: str) -> tuple[str, str]:
    """Split ``"<provider>/<upstream>"`` → ``(provider, upstream)``.

    Forward-slash is the separator because LiteLLM/OpenRouter already use it
    and the Claude SDK passes the model field through unchanged.
    """
    if "/" not in model_field:
        raise anthropic_error(
            400,
            f"Proxy expects model='<provider>/<upstream>' (e.g. "
            f"'openai_chatgpt/gpt-5.4'); got {model_field!r}.",
            error_type="invalid_request_error",
        )
    provider, _, upstream = model_field.partition("/")
    if not provider or not upstream:
        raise anthropic_error(
            400,
            f"Invalid namespaced model {model_field!r}.",
            error_type="invalid_request_error",
        )
    return provider, upstream


def _error_status(error_type: str | None) -> int:
    return {
        "invalid_request_error": 400,
        "authentication_error": 401,
        "permission_error": 403,
        "not_found_error": 404,
        "rate_limit_error": 429,
    }.get((error_type or "").strip(), 500)


async def _proxy_iter(
    adapter,
    body: dict[str, Any],
    upstream_model: str,
    *,
    provider: str,
) -> AsyncIterator[bytes]:
    """Run one proxy request, surfacing adapter failures as Anthropic SSE."""
    started = time.perf_counter()
    try:
        # Heartbeat-wrap every adapter so neither the OpenAI translate path
        # nor the Z.AI passthrough can go silent long enough to read as a
        # dead connection (covers the reasoning-window stall).
        async for chunk in with_heartbeat(
            adapter.call(body, upstream_model), interval=_PING_INTERVAL
        ):
            yield chunk
    except RuntimeError as e:
        # Adapter failures (auth errors, upstream HTTP errors, etc.)
        # surface as RuntimeError. Classify them so the SDK handles
        # retries appropriately instead of treating all as auth errors.
        msg = str(e)
        if "401" in msg or "403" in msg or "auth" in msg.lower():
            etype = "authentication_error"
        elif "429" in msg or "rate" in msg.lower():
            etype = "rate_limit_error"
        elif "overloaded" in msg.lower() or "503" in msg or "529" in msg:
            etype = "overloaded_error"
        else:
            etype = "api_error"
        logger.warning("Adapter error (type=%s): %s", etype, e)
        err_payload = {
            "type": "error",
            "error": {
                "type": etype,
                "message": msg,
            },
        }
        yield (
            b"event: error\n"
            b"data: " + json.dumps(err_payload).encode() + b"\n\n"
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Unexpected adapter failure")
        err_payload = {
            "type": "error",
            "error": {"type": "api_error", "message": f"Proxy error: {e}"},
        }
        yield (
            b"event: error\n"
            b"data: " + json.dumps(err_payload).encode() + b"\n\n"
        )
    finally:
        logger.info(
            "Proxy /v1/messages completed provider=%s upstream_model=%s elapsed_ms=%.1f",
            provider,
            upstream_model,
            (time.perf_counter() - started) * 1000,
        )


async def _anthropic_sse_to_message(
    source: AsyncIterator[bytes],
    *,
    fallback_model: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Buffer an Anthropic SSE stream into one Messages API response object.

    Claude Code occasionally issues ``stream=false`` requests (confirmed on the
    z.ai / GLM path during the first user turn). Our adapters are stream-native,
    so for the non-streaming route we coerce the upstream call to streaming,
    then rebuild the final ``Message`` JSON body from the emitted SSE frames.

    Returns ``(message_payload, error_payload)`` where ``error_payload`` is the
    Anthropic ``{"type": "error", ...}`` object if the stream surfaced an
    error frame.
    """
    buf = ""
    message: dict[str, Any] | None = None
    blocks: dict[int, dict[str, Any]] = {}
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: dict[str, Any] | None = None
    error_payload: dict[str, Any] | None = None

    def _finalize_tool_inputs() -> None:
        for block in blocks.values():
            raw = block.pop("__input_json_buffer", None)
            if raw is None:
                continue
            if raw == "":
                block["input"] = {}
                continue
            try:
                block["input"] = json.loads(raw)
            except json.JSONDecodeError:
                block["input"] = raw

    async for chunk in source:
        if not chunk:
            continue
        buf += chunk.decode("utf-8", "ignore")
        while "\n\n" in buf:
            block_text, buf = buf.split("\n\n", 1)
            if not block_text.strip():
                continue

            data_lines: list[str] = []
            event_name = ""
            for line in block_text.splitlines():
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if not data_lines:
                continue

            try:
                payload = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                logger.debug("Proxy non-stream buffer skipped malformed SSE block")
                continue

            ptype = payload.get("type") or event_name
            if ptype == "ping":
                continue
            if ptype == "error":
                error_payload = payload
                break
            if ptype == "message_start":
                started = payload.get("message") or {}
                message = dict(started)
                if isinstance(started.get("usage"), dict):
                    usage = started["usage"]
                continue
            if ptype == "content_block_start":
                idx = int(payload.get("index", 0) or 0)
                content_block = payload.get("content_block") or {}
                blocks[idx] = dict(content_block)
                if blocks[idx].get("type") == "tool_use":
                    blocks[idx]["__input_json_buffer"] = ""
                continue
            if ptype == "content_block_delta":
                idx = int(payload.get("index", 0) or 0)
                block = blocks.setdefault(idx, {"type": "text", "text": ""})
                delta = payload.get("delta") or {}
                dtype = delta.get("type")
                if dtype == "text_delta":
                    block["text"] = f"{block.get('text', '')}{delta.get('text', '')}"
                elif dtype == "thinking_delta":
                    block["thinking"] = f"{block.get('thinking', '')}{delta.get('thinking', '')}"
                elif dtype == "signature_delta":
                    block["signature"] = f"{block.get('signature', '')}{delta.get('signature', '')}"
                elif dtype == "input_json_delta":
                    block["__input_json_buffer"] = (
                        f"{block.get('__input_json_buffer', '')}{delta.get('partial_json', '')}"
                    )
                continue
            if ptype == "message_delta":
                delta = payload.get("delta") or {}
                stop_reason = delta.get("stop_reason")
                stop_sequence = delta.get("stop_sequence")
                if isinstance(payload.get("usage"), dict):
                    usage = payload["usage"]
                continue
            if ptype == "message_stop":
                _finalize_tool_inputs()
                continue

        if error_payload is not None:
            break

    _finalize_tool_inputs()
    if message is None:
        message = {
            "id": f"msg_proxy_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": fallback_model,
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence,
            "usage": usage
            or {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
    message["content"] = [blocks[i] for i in sorted(blocks)]
    message["model"] = message.get("model") or fallback_model
    message["stop_reason"] = (
        stop_reason if stop_reason is not None else message.get("stop_reason")
    )
    message["stop_sequence"] = (
        stop_sequence if stop_sequence is not None else message.get("stop_sequence")
    )
    if usage is not None:
        message["usage"] = usage
    return message, error_payload


@router.post("/v1/messages", response_model=None)
async def messages(request: Request) -> JSONResponse | StreamingResponse:
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        raise anthropic_error(400, f"Body is not valid JSON: {e}",
                              error_type="invalid_request_error")

    if not isinstance(body, dict):
        raise anthropic_error(400, "Body must be a JSON object.",
                              error_type="invalid_request_error")

    model_field = (body.get("model") or "").strip()
    if not model_field:
        raise anthropic_error(400, "Missing required field 'model'.",
                              error_type="invalid_request_error")

    provider, upstream_model = _split_model(model_field)
    adapter = lookup(provider)
    if adapter is None:
        raise anthropic_error(
            400,
            f"Unknown proxy provider {provider!r}. Registered: "
            f"{sorted((lookup.__globals__.get('REGISTRY') or {}).keys())}",
            error_type="invalid_request_error",
        )

    # The SDK CLI sends stream=true on the first init call but can omit the
    # field (stream=None) or set stream=false on internal calls (title gen,
    # non-streaming tool-loop iterations).  In both cases the CLI expects a
    # JSON Messages response, not SSE.  Our adapters are stream-native, so
    # the non-streaming path buffers upstream SSE via _anthropic_sse_to_message
    # and returns the assembled JSON body.
    stream_requested = bool(body.get("stream"))

    logger.info(
        "Proxy /v1/messages provider=%s upstream_model=%s messages=%d tools=%d",
        provider,
        upstream_model,
        len(body.get("messages") or []),
        len(body.get("tools") or []),
    )

    # Some Claude Agent SDK flows issue a non-streaming Anthropic Messages call
    # on the first turn. Our adapters are stream-native, so emulate
    # non-streaming by forcing the upstream to stream and buffering the SSE back
    # into one Messages API JSON response.
    body_for_adapter = dict(body)
    body_for_adapter["stream"] = True

    if not stream_requested:
        message, error_payload = await _anthropic_sse_to_message(
            _proxy_iter(adapter, body_for_adapter, upstream_model, provider=provider),
            fallback_model=body.get("model") or upstream_model,
        )
        if error_payload is not None:
            err = error_payload.get("error") if isinstance(error_payload, dict) else {}
            return JSONResponse(error_payload, status_code=_error_status(err.get("type")))
        return JSONResponse(message)

    return StreamingResponse(
        _proxy_iter(adapter, body_for_adapter, upstream_model, provider=provider),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable any proxy buffering en route
        },
    )


# Auth-related router exports kept here for future expansion.
__all__ = ["router", "auth_failed_error"]
