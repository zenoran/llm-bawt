"""Core LLMBawt class - simple OpenAI-compatible chat client with memory.

This module provides the main entry point for CLI usage. For the modular
pipeline components, see the other modules in this package.
"""

import logging

from ..clients import LLMClient, GrokClient, AgentBackendClient
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
        
        Supports OpenAI-compatible APIs and Grok (xAI). For local GGUF models,
        use the background service which has ServiceLLMBawt.
        """
        model_type = self.model_definition.get("type")
        model_id = self.model_definition.get("model_id")
        
        if model_type == "openclaw":
            backend_config = {
                "transport": "gateway_api",
                "gateway_url": self.model_definition.get("gateway_url"),
                "token": self.model_definition.get("token"),
                "token_env": self.model_definition.get("token_env"),
                "agent_id": self.model_definition.get("agent_id", "main"),
                "session_key": self.model_definition.get("session_key"),
                "message_channel": self.model_definition.get("message_channel"),
                "account_id": self.model_definition.get("account_id"),
                "model": self.model_definition.get("model_id"),
                "timeout_seconds": self.model_definition.get("timeout_seconds", 120),
                "tool_history_limit": self.model_definition.get("tool_history_limit", 8),
            }
            return AgentBackendClient(
                backend_name="openclaw",
                config=self.config,
                bot_config=backend_config,
                model_definition=self.model_definition,
            )

        if model_type not in ("openai", "grok"):
            raise ValueError(
                f"Only 'openai', 'grok', and 'openclaw' model types are supported in CLI mode. "
                f"Got '{model_type}'. Use --service flag for local models."
            )
        
        if not model_id:
            raise ValueError(
                f"Missing 'model_id' in definition for '{self.resolved_model_alias}'"
            )
        
        # Handle Grok (xAI) models
        if model_type == "grok":
            api_key = self.model_definition.get("api_key") or self.config.XAI_API_KEY
            
            if self.verbose:
                logger.info(f"Initializing Grok client for model: {model_id}")
            
            return GrokClient(
                model_id,
                config=self.config,
                api_key=api_key,
                model_definition=self.model_definition,
            )
        
        # Handle OpenAI-compatible APIs
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
