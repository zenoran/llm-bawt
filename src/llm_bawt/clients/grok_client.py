"""Grok (xAI) API client.

Uses the OpenAI-compatible API provided by xAI.
API documentation: https://docs.x.ai/api
"""

from __future__ import annotations

import os
from typing import List, Iterator, Any

from .openai_client import OpenAIClient
from ..utils.config import Config
from ..models.message import Message
import logging

logger = logging.getLogger(__name__)


class GrokClient(OpenAIClient):
    """Client for Grok (xAI) API.
    
    The xAI API is OpenAI-compatible, so we extend OpenAIClient
    with the xAI base URL and API key handling.
    """
    SUPPORTS_STREAMING = True

    # xAI API base URL
    XAI_BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        model: str,
        config: Config,
        api_key: str | None = None,
        model_definition: dict | None = None,
    ):
        """Initialize Grok client.
        
        Args:
            model: Model name/ID (e.g., "grok-2", "grok-2-vision")
            config: Application config
            api_key: Optional API key (defaults to XAI_API_KEY env var)
            model_definition: Optional model definition dict from models.yaml
        """
        self._provided_api_key = api_key
        effective_key = api_key or self._get_api_key()
        
        if not effective_key:
            raise ValueError(
                "xAI API key not found. Set XAI_API_KEY environment variable "
                "or LLM_BAWT_XAI_API_KEY in your config."
            )
        
        # Initialize parent with xAI base URL
        super().__init__(
            model=model,
            config=config,
            base_url=self.XAI_BASE_URL,
            api_key=effective_key,
            model_definition=model_definition,
        )

    def _get_api_key(self) -> str | None:
        """Get API key from environment."""
        return os.getenv("XAI_API_KEY") or os.getenv("LLM_BAWT_XAI_API_KEY")

    def supports_native_tools(self) -> bool:
        """Grok supports native tool calling."""
        return True

    def get_styling(self) -> tuple[str | None, str]:
        """Return Grok-specific styling."""
        return None, "bright_magenta"
