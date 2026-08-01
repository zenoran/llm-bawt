"""ProviderAdapter ABC.

Adapters are stateful (cached credentials, refresh timers) so the registry
stores live instances, not classes. The default ``call`` implementation
targets the OpenAI Responses API and is shared by every Responses-API
provider — adapters only override when their upstream shape differs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, ClassVar

import httpx

from ..request_context import ProxyRequestContext

logger = logging.getLogger(__name__)


class ProviderAdapter(ABC):
    """Abstract base for upstream-provider adapters.

    Subclasses MUST set ``name`` and implement ``authorize``.

    The default ``call`` translates Anthropic Messages → Responses API →
    Anthropic SSE using the shared translate/stream helpers. Override
    ``call`` only when an upstream doesn't speak Responses API (e.g. a
    legacy Chat Completions–only provider).
    """

    name: ClassVar[str]

    def __init__(self) -> None:
        self._lifecycle_lock = asyncio.Lock()
        self._http_client: httpx.AsyncClient | None = None
        self._responses_client = None

    async def start(self) -> None:
        """Create the adapter-owned connection pool once per proxy lifecycle."""
        if self._http_client is not None:
            return
        async with self._lifecycle_lock:
            if self._http_client is not None:
                return
            from openai import AsyncOpenAI

            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=15.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
            # Request-local ``with_options`` wrappers below share this exact
            # underlying pool while carrying fresh credentials and headers.
            self._responses_client = AsyncOpenAI(
                api_key="lifecycle-managed",
                base_url="http://127.0.0.1",
                http_client=self._http_client,
            )

    async def close(self) -> None:
        """Close the lifecycle-owned pool exactly once."""
        async with self._lifecycle_lock:
            client = self._responses_client
            self._responses_client = None
            self._http_client = None
            if client is not None:
                await client.close()

    async def http_client(self) -> httpx.AsyncClient:
        """Return the reusable raw HTTP client for non-Responses adapters."""
        await self.start()
        assert self._http_client is not None
        return self._http_client

    def account_hash(self) -> str:
        """Non-secret account scope for concurrency telemetry."""
        return "default"

    @abstractmethod
    async def authorize(self) -> tuple[str, str]:
        """Return ``(bearer_token, base_url)``.

        Refresh credentials in-place when needed. ``base_url`` is the API
        root (e.g. ``https://api.openai.com/v1``), passed straight to the
        ``openai`` client.
        """

    def extra_headers(
        self,
        responses_body: dict,
        context: ProxyRequestContext | None = None,
    ) -> dict[str, str]:
        """Per-request headers merged into the upstream HTTP client.

        Default: none. Override for providers that need extra auth/routing
        headers (e.g. the ChatGPT backend's ``chatgpt-account-id``).
        """
        return {}

    def prepare_request(
        self,
        responses_body: dict,
        context: ProxyRequestContext | None = None,
    ) -> dict:
        """Last-chance hook to adapt the translated Responses body to an
        upstream's quirks (strip unsupported params, force defaults, etc.).

        Default: identity. Override only when an upstream rejects standard
        Responses fields. Mutates and returns the same dict for convenience.
        """
        return responses_body

    async def call(
        self,
        anthropic_body: dict,
        upstream_model: str,
        context: ProxyRequestContext | None = None,
    ) -> AsyncIterator[bytes]:
        """Default Responses API call. Yields Anthropic-shaped SSE bytes."""
        # Local imports keep the module import-graph trivial (the proxy
        # subpackage imports adapters, which imports openai, which is heavy).
        from .. import stream as stream_mod
        from .. import translate

        call_started = time.perf_counter()
        bearer, base_url = await self.authorize()
        await self.start()
        assert self._responses_client is not None
        responses_body = translate.anthropic_to_responses(
            anthropic_body, upstream_model
        )
        responses_body = self.prepare_request(responses_body, context)
        headers = self.extra_headers(responses_body, context) or None
        # ``with_options`` creates a cheap request wrapper around the SAME
        # lifecycle-owned httpx pool. Credentials and default headers are
        # immutable on that wrapper, so concurrent requests cannot cross-talk.
        client = self._responses_client.with_options(
            api_key=bearer,
            base_url=base_url,
            set_default_headers=headers,
        )
        if context is not None:
            context.local_setup_ms = (
                time.perf_counter() - call_started
            ) * 1000
        # Prompt-cache divergence diagnostic (no-op unless PROXY_CACHE_DIAG is
        # set). Runs on the exact post-prepare body.
        from .. import cache_diag

        cache_diag.record(responses_body)
        logger.debug(
            "Proxy → Responses API model=%s tools=%d input_items=%d",
            responses_body.get("model"),
            len(responses_body.get("tools", []) or []),
            len(responses_body.get("input", []) or []),
        )
        upstream_started = time.perf_counter()
        # ``stream=True`` returns after response headers arrive. This is the
        # upstream TTFB boundary; it intentionally excludes local setup/auth.
        upstream_stream = await client.responses.create(**responses_body)
        if context is not None:
            context.upstream_ttfb_ms = (
                time.perf_counter() - upstream_started
            ) * 1000
        # Best-effort: stash codex plan-usage from the response headers for
        # /v1/usage. Synchronous header peek + fire-and-forget Redis write;
        # never blocks or breaks inference.
        from .. import usage_capture

        usage_capture.schedule_capture(upstream_stream)
        async for chunk in stream_mod.responses_to_anthropic_sse(
            upstream_stream,
            anthropic_model=anthropic_body.get("model", upstream_model),
            tool_schemas=anthropic_body.get("tools"),
            on_usage=context.record_usage if context is not None else None,
        ):
            yield chunk
