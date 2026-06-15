"""ProviderAdapter ABC.

Adapters are stateful (cached credentials, refresh timers) so the registry
stores live instances, not classes. The default ``call`` implementation
targets the OpenAI Responses API and is shared by every Responses-API
provider — adapters only override when their upstream shape differs.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, ClassVar

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

    @abstractmethod
    async def authorize(self) -> tuple[str, str]:
        """Return ``(bearer_token, base_url)``.

        Refresh credentials in-place when needed. ``base_url`` is the API
        root (e.g. ``https://api.openai.com/v1``), passed straight to the
        ``openai`` client.
        """

    def extra_headers(self) -> dict[str, str]:
        """Per-request headers merged into the upstream HTTP client.

        Default: none. Override for providers that need extra auth/routing
        headers (e.g. the ChatGPT backend's ``chatgpt-account-id``).
        """
        return {}

    def prepare_request(self, responses_body: dict) -> dict:
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
    ) -> AsyncIterator[bytes]:
        """Default Responses API call. Yields Anthropic-shaped SSE bytes."""
        # Local imports keep the module import-graph trivial (the proxy
        # subpackage imports adapters, which imports openai, which is heavy).
        from openai import AsyncOpenAI

        from .. import stream as stream_mod
        from .. import translate

        bearer, base_url = await self.authorize()
        headers = self.extra_headers() or None
        client = AsyncOpenAI(api_key=bearer, base_url=base_url, default_headers=headers)
        try:
            responses_body = translate.anthropic_to_responses(
                anthropic_body, upstream_model
            )
            responses_body = self.prepare_request(responses_body)
            logger.debug(
                "Proxy → Responses API model=%s tools=%d input_items=%d",
                responses_body.get("model"),
                len(responses_body.get("tools", []) or []),
                len(responses_body.get("input", []) or []),
            )
            # ``stream=True`` returns an AsyncStream of typed events.
            upstream_stream = await client.responses.create(**responses_body)
            async for chunk in stream_mod.responses_to_anthropic_sse(
                upstream_stream,
                anthropic_model=anthropic_body.get("model", upstream_model),
            ):
                yield chunk
        finally:
            await client.close()
