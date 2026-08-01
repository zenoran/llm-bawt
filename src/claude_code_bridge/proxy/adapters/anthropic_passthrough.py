"""Shared base for providers exposing a native **Anthropic Messages API**.

Some upstreams (Z.AI, Moonshot/Kimi) ship an Anthropic-compatible
``/v1/messages`` surface — the same one their Claude Code integration
targets. For those there is nothing to translate: the Claude Agent SDK
already speaks the wire format, so the adapter streams the inbound body
straight through and relays the upstream SSE bytes back.

That passthrough is identical across providers except for three things:

1. **Where** it posts (base URL + env override).
2. **How** it authenticates (``x-api-key`` vs ``Authorization: Bearer``).
3. **What** it calls itself in logs.

Everything else — the namespaced-model rewrite, the streaming relay, and the
prompt-cache usage tap — is common, so it lives here once. A new
Anthropic-native provider is a subclass that sets a few class vars; it should
not need to reimplement ``call``.

Contrast with :class:`~.base.ProviderAdapter`'s default ``call``, which targets
the **OpenAI Responses API** and is the right base for providers that speak
that instead (e.g. xAI).
"""

from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator, ClassVar

import httpx

from .base import ProviderAdapter
from ..request_context import ProxyRequestContext

logger = logging.getLogger(__name__)

# Anthropic API version the SDK negotiates against. Compatible endpoints
# mirror the public Anthropic header contract; this matches what the Claude
# CLI sends.
_ANTHROPIC_VERSION = "2023-06-01"

# Generous read timeout: agent turns with many tool calls stream for a long
# time. No total timeout — rely on the connect/read granular limits.
_TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=15.0)


class AnthropicPassthroughAdapter(ProviderAdapter):
    """Anthropic→Anthropic passthrough. Subclass and set the class vars."""

    #: Upstream Anthropic API root (no trailing ``/v1/messages``).
    DEFAULT_BASE_URL: ClassVar[str] = ""
    #: Env var that overrides :attr:`DEFAULT_BASE_URL`.
    BASE_URL_ENV: ClassVar[str] = ""
    #: Env vars searched in order for the API key.
    API_KEY_ENVS: ClassVar[tuple[str, ...]] = ()
    #: Human label used in log lines and the missing-key error.
    LABEL: ClassVar[str] = ""
    #: Path appended to the base URL for the messages endpoint.
    MESSAGES_PATH: ClassVar[str] = "/v1/messages"

    # -- credentials -----------------------------------------------------

    def _api_key(self) -> str:
        for env in self.API_KEY_ENVS:
            key = os.getenv(env)
            if key:
                return key
        primary, *aliases = self.API_KEY_ENVS
        names = primary
        if aliases:
            names += f" (or {', '.join(aliases)})"
        raise RuntimeError(
            f"{self.LABEL} API key required: set {names} on the "
            "claude-code-bridge container."
        )

    @classmethod
    def _base_url(cls) -> str:
        return (os.getenv(cls.BASE_URL_ENV) or cls.DEFAULT_BASE_URL).rstrip("/")

    def _auth_headers(self, api_key: str) -> dict[str, str]:
        """Provider auth headers.

        Default is ``x-api-key``, matching api.anthropic.com (and what the SDK
        sends for ``ANTHROPIC_API_KEY``). Providers that authenticate the
        ``ANTHROPIC_AUTH_TOKEN`` way override this to send a Bearer header.
        """
        return {"x-api-key": api_key}

    async def authorize(self) -> tuple[str, str]:
        # Not used by ``call`` below (it builds its own headers), but the ABC
        # requires it and it keeps the adapter usable by the default
        # Responses-API path should the provider ever expose one.
        return self._api_key(), self._base_url()

    # -- the shared passthrough -----------------------------------------

    async def call(
        self,
        anthropic_body: dict,
        upstream_model: str,
        context: ProxyRequestContext | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream the Anthropic request straight upstream, relay SSE back."""
        api_key = self._api_key()
        url = f"{self._base_url()}{self.MESSAGES_PATH}"

        # The SDK sent model="<provider>/<upstream>"; the upstream wants the
        # bare name.
        body = dict(anthropic_body)
        original_model = body.get("model", upstream_model)
        body["model"] = upstream_model
        body["stream"] = True  # proxy only supports streaming (routes.py)

        headers = {
            **self._auth_headers(api_key),
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
            "accept": "text/event-stream",
        }

        logger.debug(
            "%s passthrough → %s model=%s tools=%d messages=%d",
            self.LABEL,
            url,
            upstream_model,
            len(body.get("tools") or []),
            len(body.get("messages") or []),
        )

        client = await self.http_client()
        async with client.stream(
            "POST", url, json=body, headers=headers, timeout=_TIMEOUT
        ) as resp:
            if resp.status_code >= 400:
                detail = (await resp.aread()).decode("utf-8", "replace")
                raise RuntimeError(
                    f"{self.LABEL} upstream {resp.status_code}: {detail[:500]}"
                )
            # Best-effort usage tap: relay raw bytes untouched (the SDK
            # consumes them), and on the side accumulate a decoded copy
            # to extract the `message_start` usage block so we can see
            # whether the upstream's context cache is actually hitting.
            buf = ""
            usage_logged = False
            # The upstream echoes the bare model name ("glm-5.2") in
            # message_start. The SDK CLI validates that the response
            # model matches what it sent ("zai/glm-5.2"). Rewrite the
            # model field in the raw SSE so the CLI doesn't reject the
            # response as malformed.
            bare_model_bytes = f'"model": "{upstream_model}"'.encode()
            full_model_bytes = f'"model": "{original_model}"'.encode()
            async for chunk in resp.aiter_raw():
                if not chunk:
                    continue
                if bare_model_bytes in chunk:
                    chunk = chunk.replace(bare_model_bytes, full_model_bytes)
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

    @classmethod
    def _tap_usage(
        cls, buf: str, upstream_model: str, already: bool
    ) -> tuple[str, bool]:
        """Scan buffered SSE for the message_start usage block and log it.

        Returns (remaining_buffer, logged). Anthropic streams usage in the
        first ``message_start`` event:
            usage = {input_tokens, cache_creation_input_tokens,
                     cache_read_input_tokens, output_tokens}
        where ``input_tokens`` is the UNCACHED prompt portion. Cache hit % is
        cache_read / (input + cache_read + cache_creation). If the upstream
        doesn't populate the cache fields over its Anthropic endpoint they
        read 0, which is itself the answer.
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
                    "%s usage model=%s input=%d cached=%d uncached=%d "
                    "cache_create=%d cache_hit=%.1f%%",
                    cls.LABEL, upstream_model, total_in, cache_read, uncached,
                    cache_create, hit_pct,
                )
                return buf, True
        # Cap buffer growth if message_start never shows a usage line.
        if len(buf) > 65536:
            return buf[-4096:], True
        return buf, False
