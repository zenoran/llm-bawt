"""Core LLMBawt class - simple OpenAI-compatible chat client with memory.

This module provides the main entry point for CLI usage. For the modular
pipeline components, see the other modules in this package.
"""

import logging

from ..clients import LLMClient
from ..clients.openai_client import OpenAIClient
from .base import BaseLLMBawt
from ..utils.config import Config

logger = logging.getLogger(__name__)


class LLMBawt(BaseLLMBawt):
    """Simple LLM client with optional memory augmentation.
    
    Uses OpenAI-compatible API only. For local model support (GGUF),
    use ServiceLLMBawt via the background service.
    
    Memory features work when database is available, otherwise falls back
    to filesystem-only mode.
    """
    
    def __init__(
        self,
        resolved_model_alias: str,
        config: Config,
        local_mode: bool = False,
        bot_id: str = "nova",
        user_id: str = "",  # Required - must be passed explicitly
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initialize LLMBawt.
        
        Args:
            resolved_model_alias: Model alias from models.yaml
            config: Application configuration
            local_mode: Skip database features
            bot_id: Bot personality to use
            user_id: User profile ID (required)
            verbose: Enable verbose logging (--verbose)
            debug: Enable debug logging (--debug)
        """
        # Store verbose/debug before calling super().__init__
        # because the base class uses them during initialization
        super().__init__(
            resolved_model_alias=resolved_model_alias,
            config=config,
            local_mode=local_mode,
            bot_id=bot_id,
            user_id=user_id,
            verbose=verbose,
            debug=debug,
        )
    
    def _initialize_client(self) -> LLMClient:
        """Initialize OpenAI-compatible client.
        
        Only supports OpenAI-compatible APIs. For local GGUF models,
        use the background service which has ServiceLLMBawt.
        """
        model_type = self.model_definition.get("type")
        model_id = self.model_definition.get("model_id")
        
        if model_type != "openai":
            raise ValueError(
                f"Only 'openai' model type is supported in CLI mode. Got '{model_type}'. "
                "Use --service flag for local models, or an OpenAI-compatible server."
            )
        
        if not model_id:
            raise ValueError(
                f"Missing 'model_id' in definition for '{self.resolved_model_alias}'"
            )
        
        base_url = self.model_definition.get("base_url")
        api_key = self.model_definition.get("api_key")
        
        if self.verbose:
            logger.info(f"Initializing OpenAI client for model: {model_id}")
        
        return OpenAIClient(
            model_id,
            config=self.config,
            base_url=base_url,
            api_key=api_key,
            model_definition=self.model_definition,
        )
