"""xAI (Grok) adapter — OpenAI Responses API, API-key auth.

Unlike Z.AI (native Anthropic passthrough) and the ChatGPT codex backend
(a quirky non-standard Responses surface), xAI exposes a **standard
OpenAI-compatible Responses API**:

    https://api.x.ai/v1/responses      Authorization: Bearer $XAI_API_KEY

Verified live against grok-4.5: it accepts the exact body our
``translate.anthropic_to_responses`` emits (``store:false``, ``stream:true``,
``max_output_tokens``, ``temperature``, function ``tools``, and
``reasoning:{effort, summary}``) and streams the same Responses events
(``response.reasoning_summary_text.delta``, ``function_call`` output items)
that ``stream.py`` already converts back to Anthropic SSE.

So this adapter needs **no** ``call``/``prepare_request`` override — the
default Responses path in ``ProviderAdapter`` handles the whole round-trip.
All it supplies is credentials: ``authorize()`` returns
``(bearer, base_url)`` and the shared base class does the rest.

Auth: a standard xAI API key. Read from ``XAI_API_KEY``
(``LLM_BAWT_XAI_API_KEY`` accepted as an alias so a single key in ``.env``
serves both the app and this bridge). Models are namespaced ``xai/<model>``
(e.g. ``xai/grok-4.5``, ``xai/grok-4.3``); the proxy strips the ``xai/``
prefix before handing the bare upstream model to ``call``.
"""

from __future__ import annotations

import logging
import os
from typing import ClassVar

from .base import ProviderAdapter

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.x.ai/v1"
BASE_URL_ENV = "XAI_BASE_URL"
API_KEY_ENVS = ("XAI_API_KEY", "LLM_BAWT_XAI_API_KEY", "GROK_API_KEY")


class XaiAdapter(ProviderAdapter):
    """Grok via xAI's standard OpenAI Responses API (default call path)."""

    name: ClassVar[str] = "xai"

    def _api_key(self) -> str:
        for env in API_KEY_ENVS:
            key = os.getenv(env)
            if key:
                return key
        raise RuntimeError(
            "xAI API key required: set XAI_API_KEY (or LLM_BAWT_XAI_API_KEY) "
            "on the claude-code-bridge container."
        )

    @staticmethod
    def _base_url() -> str:
        return (os.getenv(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")

    async def authorize(self) -> tuple[str, str]:
        """Return ``(bearer, base_url)`` for the shared Responses-API path.

        xAI keys are static — no refresh, no OAuth bundle — so this just
        reads the key and hands back the API root the ``openai`` client
        appends ``/responses`` to.
        """
        return self._api_key(), self._base_url()
