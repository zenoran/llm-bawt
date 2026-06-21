"""FastAPI routes for the Anthropic-compatible proxy.

Only one substantive endpoint right now: ``POST /v1/messages``. The
Anthropic Messages API spec also defines ``/v1/messages/count_tokens`` and
some metadata endpoints; the Claude Agent SDK doesn't currently call them
in the workflows we care about, so they're not implemented yet.
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .adapters import lookup
from .errors import anthropic_error, auth_failed_error

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

    stream_requested = bool(body.get("stream"))
    if not stream_requested:
        # Anthropic spec allows non-streaming, but the Claude Agent SDK only
        # ever sets stream=true. Surface a clear 400 instead of silently
        # downgrading; if we ever need non-streaming we'll buffer the
        # stream and emit a final Message object.
        raise anthropic_error(
            400,
            "Proxy currently only supports streaming requests "
            "(set stream=true).",
            error_type="invalid_request_error",
        )

    logger.info(
        "Proxy /v1/messages provider=%s upstream_model=%s messages=%d tools=%d",
        provider,
        upstream_model,
        len(body.get("messages") or []),
        len(body.get("tools") or []),
    )

    async def _iter():
        started = time.perf_counter()
        try:
            async for chunk in adapter.call(body, upstream_model):
                yield chunk
        except RuntimeError as e:
            # OAuth / file errors from the adapter surface as RuntimeError;
            # turn them into a proper Anthropic error frame inside the
            # stream so the SDK reports the failure rather than hanging.
            logger.warning("Adapter error: %s", e)
            err_payload = {
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": str(e),
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

    return StreamingResponse(
        _iter(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable any proxy buffering en route
        },
    )


# Auth-related router exports kept here for future expansion.
__all__ = ["router", "auth_failed_error"]
