"""Z.AI (Zhipu GLM) adapter — Anthropic Messages API passthrough.

Unlike the OpenAI ChatGPT adapter, Z.AI exposes a **native Anthropic
Messages API** surface (the same one their Claude Code integration targets):

    https://api.z.ai/api/anthropic/v1/messages

So there's nothing to translate — the Claude Agent SDK already speaks this
wire format. This adapter overrides ``call`` to stream the inbound Anthropic
body straight through to Z.AI and relay the upstream SSE bytes back
unchanged. The only rewrite is the ``model`` field: the proxy hands us the
bare upstream model (``glm-4.6``) after stripping the ``zai/`` provider
prefix, and we substitute it for the namespaced value the SDK sent.

Auth: a Z.AI API key (the value you'd normally set as ``ANTHROPIC_AUTH_TOKEN``
for Claude Code). Read from ``ZAI_API_KEY`` (``Z_AI_API_KEY`` accepted as an
alias). Z.AI's Anthropic endpoint authenticates with an
``x-api-key`` header, same as api.anthropic.com.
"""

from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator, ClassVar

import httpx

from .base import ProviderAdapter

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.z.ai/api/anthropic"
BASE_URL_ENV = "ZAI_BASE_URL"
API_KEY_ENVS = ("ZAI_API_KEY", "Z_AI_API_KEY")

# Anthropic API version the SDK negotiates against. Z.AI mirrors the public
# Anthropic header contract; this matches what the Claude CLI sends.
_ANTHROPIC_VERSION = "2023-06-01"

# Generous read timeout: agent turns with many tool calls stream for a long
# time. No total timeout — rely on the connect/read granular limits.
_TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=15.0)


class ZaiAdapter(ProviderAdapter):
    """Pure Anthropic→Anthropic passthrough to Z.AI's GLM models."""

    name: ClassVar[str] = "zai"

    def _api_key(self) -> str:
        for env in API_KEY_ENVS:
            key = os.getenv(env)
            if key:
                return key
        raise RuntimeError(
            "Z.AI API key required: set ZAI_API_KEY (or Z_AI_API_KEY) on the "
            "claude-code-bridge container."
        )

    async def authorize(self) -> tuple[str, str]:
        # Not used by the overridden ``call`` below, but the ABC requires it
        # and it keeps the adapter usable by the default Responses-API path
        # should Z.AI ever expose one. base_url is the Anthropic root.
        return self._api_key(), self._base_url()

    @staticmethod
    def _base_url() -> str:
        return (os.getenv(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")

    async def call(
        self,
        anthropic_body: dict,
        upstream_model: str,
    ) -> AsyncIterator[bytes]:
        """Stream the Anthropic request straight to Z.AI, relay SSE back."""
        api_key = self._api_key()
        url = f"{self._base_url()}/v1/messages"

        # The SDK sent model="zai/glm-4.6"; Z.AI wants the bare upstream name.
        body = dict(anthropic_body)
        body["model"] = upstream_model
        body["stream"] = True  # proxy only supports streaming (routes.py)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
            "accept": "text/event-stream",
        }

        logger.debug(
            "Z.AI passthrough → %s model=%s tools=%d messages=%d",
            url,
            upstream_model,
            len(body.get("tools") or []),
            len(body.get("messages") or []),
        )

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode("utf-8", "replace")
                    raise RuntimeError(
                        f"Z.AI upstream {resp.status_code}: {detail[:500]}"
                    )
                # Best-effort usage tap: relay raw bytes untouched (the SDK
                # consumes them), and on the side accumulate a decoded copy
                # to extract the `message_start` usage block so we can see
                # whether z.ai's automatic context cache is actually hitting.
                buf = ""
                usage_logged = False
                async for chunk in resp.aiter_raw():
                    if not chunk:
                        continue
                    yield chunk
                    if usage_logged:
                        continue
                    try:
                        buf += chunk.decode("utf-8", "ignore")
                        buf, usage_logged = self._tap_usage(
                            buf, upstream_model, usage_logged
                        )
                    except Exception:  # noqa: BLE001 — logging must never break the stream
                        usage_logged = True

    @staticmethod
    def _tap_usage(
        buf: str, upstream_model: str, already: bool
    ) -> tuple[str, bool]:
        """Scan buffered SSE for the message_start usage block and log it.

        Returns (remaining_buffer, logged). Anthropic streams usage in the
        first ``message_start`` event:
            usage = {input_tokens, cache_creation_input_tokens,
                     cache_read_input_tokens, output_tokens}
        where ``input_tokens`` is the UNCACHED prompt portion. Cache hit % is
        cache_read / (input + cache_read + cache_creation). If z.ai doesn't
        populate the cache fields over its Anthropic endpoint they read 0,
        which is itself the answer.
        """
        if already:
            return buf, True
        # Process complete SSE event blocks; keep any partial tail.
        while "\n\n" in buf:
            block, buf = buf.split("\n\n", 1)
            if "message_start" not in block:
                continue
            for line in block.splitlines():
                if not line.startswith("data:"):
                    continue
                try:
                    evt = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                usage = (evt.get("message") or {}).get("usage") or {}
                uncached = int(usage.get("input_tokens") or 0)
                cache_read = int(usage.get("cache_read_input_tokens") or 0)
                cache_create = int(usage.get("cache_creation_input_tokens") or 0)
                total_in = uncached + cache_read + cache_create
                hit_pct = (cache_read / total_in * 100) if total_in else 0.0
                logger.info(
                    "Z.AI usage model=%s input=%d cached=%d uncached=%d "
                    "cache_create=%d cache_hit=%.1f%%",
                    upstream_model, total_in, cache_read, uncached,
                    cache_create, hit_pct,
                )
                return buf, True
        # Cap buffer growth if message_start never shows a usage line.
        if len(buf) > 65536:
            return buf[-4096:], True
        return buf, False
