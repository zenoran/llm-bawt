"""Grok (xAI) API client.

Uses the Responses API (/v1/responses) via the OpenAI SDK with xAI's base URL.
API documentation: https://docs.x.ai/api
"""

from __future__ import annotations

import os
import logging

from .responses_client import ResponsesClient
from ..utils.config import Config

logger = logging.getLogger(__name__)


class GrokClient(ResponsesClient):
    """Client for Grok (xAI) API via the Responses API.

    The xAI API is OpenAI-compatible, so we extend ResponsesClient
    with the xAI base URL and API key handling.
    """

    SUPPORTS_STREAMING = True
    XAI_BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        model: str,
        config: Config,
        api_key: str | None = None,
        model_definition: dict | None = None,
    ):
        self._provided_api_key = api_key
        effective_key = api_key or self._get_api_key()

        if not effective_key:
            raise ValueError(
                "xAI API key not found. Set XAI_API_KEY environment variable "
                "or LLM_BAWT_XAI_API_KEY in your config."
            )

        super().__init__(
            model=model,
            config=config,
            base_url=self.XAI_BASE_URL,
            api_key=effective_key,
            model_definition=model_definition,
        )

    def _get_api_key(self) -> str | None:
        return os.getenv("XAI_API_KEY") or os.getenv("LLM_BAWT_XAI_API_KEY")

    def get_styling(self) -> tuple[str | None, str]:
        return None, "bright_magenta"
