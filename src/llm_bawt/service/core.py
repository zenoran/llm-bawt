"""Service-side LLMBawt class - supports loading local models (GGUF, etc.)

This is separate from the CLI's core.py which only supports OpenAI API calls.
The service runs on the server and can load models directly.
"""

import logging
import time
from typing import TYPE_CHECKING

from rich.console import Console

from ..clients import LLMClient, GrokClient
from ..clients.openai_client import OpenAIClient
from ..core.base import BaseLLMBawt
from ..utils.config import Config, is_llama_cpp_available
from ..utils.history import Message
from ..tools import query_with_tools
from .logging import get_service_logger

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
slog = get_service_logger(__name__)
console = Console()


class ServiceLLMBawt(BaseLLMBawt):
    """Server-side LLM class that can load local models directly.
    
    Supports:
    - OpenAI-compatible API
    - GGUF models via llama-cpp-python
    - Memory augmentation when database is available
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
        existing_client: "LLMClient | None" = None,
    ):
        """Initialize ServiceLLMBawt.
        
        Args:
            resolved_model_alias: Model alias from models.yaml
            config: Application configuration
            local_mode: Skip database features
            bot_id: Bot personality to use
            user_id: User profile ID (required)
            verbose: Enable verbose logging (--verbose)
            debug: Enable debug logging (--debug)
            existing_client: Reuse this LLM client instead of loading model again
        """
        super().__init__(
            resolved_model_alias=resolved_model_alias,
            config=config,
            local_mode=local_mode,
            bot_id=bot_id,
            user_id=user_id,
            verbose=verbose,
            debug=debug,
            existing_client=existing_client,
        )
        self._last_history_load_at: float = time.time()

    def _refresh_history_if_stale(self) -> None:
        """Refresh history from DB only when cache is stale."""
        ttl = max(
            0.0,
            float(
                self._resolve_setting(
                    "history_reload_ttl_seconds",
                    getattr(self.config, "HISTORY_RELOAD_TTL_SECONDS", 0.0),
                )
            ),
        )
        now = time.time()
        is_empty = not bool(getattr(self.history_manager, "messages", []))
        is_stale = ttl == 0.0 or (now - self._last_history_load_at) >= ttl
        if is_empty or is_stale:
            self.load_history()
            self._last_history_load_at = now

    def invalidate_history_cache(self) -> None:
        """Force a history refresh on the next request."""
        self._last_history_load_at = 0.0
    
    def _initialize_client(self) -> LLMClient:
        """Initialize client based on model type - supports local models."""
        model_type = self.model_definition.get("type")
        model_id = self.model_definition.get("model_id")
        
        # Log model loading
        slog.model_loading(self.resolved_model_alias, model_type)
        start_time = time.perf_counter()
        
        if model_type == "openai":
            if not model_id:
                raise ValueError(
                    f"Missing 'model_id' in definition for '{self.resolved_model_alias}'"
                )
            base_url = self.model_definition.get("base_url")
            api_key = self.model_definition.get("api_key")
            client = OpenAIClient(
                model_id,
                config=self.config,
                base_url=base_url,
                api_key=api_key,
                model_definition=self.model_definition,
            )
            load_time_ms = (time.perf_counter() - start_time) * 1000
            slog.model_loaded(self.resolved_model_alias, model_type, load_time_ms)
            return client
        
        elif model_type == "grok":
            if not model_id:
                raise ValueError(
                    f"Missing 'model_id' in definition for '{self.resolved_model_alias}'"
                )
            api_key = self.model_definition.get("api_key") or self.config.XAI_API_KEY
            client = GrokClient(
                model_id,
                config=self.config,
                api_key=api_key,
                model_definition=self.model_definition,
            )
            load_time_ms = (time.perf_counter() - start_time) * 1000
            slog.model_loaded(self.resolved_model_alias, model_type, load_time_ms)
            return client
        
        elif model_type == "gguf":
            if not is_llama_cpp_available():
                raise ImportError(
                    "llama-cpp-python is required for GGUF models. "
                    "Install with: pip install llama-cpp-python"
                )
            from ..clients.llama_cpp_client import LlamaCppClient
            from ..gguf_handler import get_or_download_gguf_model
            
            repo_id = self.model_definition.get("repo_id")
            filename = self.model_definition.get("filename")
            if not repo_id or not filename:
                raise ValueError(
                    f"Missing 'repo_id' or 'filename' in GGUF definition for "
                    f"'{self.resolved_model_alias}'"
                )
            
            # Download model if needed
            model_path = get_or_download_gguf_model(repo_id, filename, self.config)
            if not model_path:
                raise FileNotFoundError(
                    f"Could not download GGUF model: {repo_id}/{filename}"
                )
            
            # Check if this GGUF should be run through vLLM instead of llama-cpp
            backend = self.model_definition.get("backend", "llama-cpp")
            if backend == "vllm":
                from ..utils.config import is_vllm_available
                if not is_vllm_available():
                    raise ImportError(
                        "vllm is required for backend: vllm. "
                        "Install with: pip install vllm"
                    )
                from ..clients.vllm_client import VLLMClient
                
                client = VLLMClient(
                    str(model_path),  # Pass local GGUF path to vLLM
                    config=self.config,
                    model_definition=self.model_definition,
                )
                load_time_ms = (time.perf_counter() - start_time) * 1000
                slog.model_loaded(self.resolved_model_alias, "gguf-vllm", load_time_ms)
                return client
            
            # Get optional chat_format from model definition (for models like MythoMax)
            chat_format = self.model_definition.get("chat_format")
            client = LlamaCppClient(model_path, config=self.config, chat_format=chat_format, model_definition=self.model_definition)
            load_time_ms = (time.perf_counter() - start_time) * 1000
            slog.model_loaded(self.resolved_model_alias, model_type, load_time_ms)
            return client
        
        elif model_type == "vllm":
            from ..utils.config import is_vllm_available
            if not is_vllm_available():
                raise ImportError(
                    "vllm is required for vLLM models. "
                    "Install with: pip install vllm"
                )
            from ..clients.vllm_client import VLLMClient
            
            # model_id can be HuggingFace model ID or GGUF path (for backend: vllm)
            model_id = self.model_definition.get("model_id", self.resolved_model_alias)
            client = VLLMClient(
                model_id,
                config=self.config,
                model_definition=self.model_definition,
            )
            load_time_ms = (time.perf_counter() - start_time) * 1000
            slog.model_loaded(self.resolved_model_alias, model_type, load_time_ms)
            return client
        
        else:
            raise ValueError(
                f"Unsupported model type: '{model_type}'. "
                f"Supported types: openai, grok, gguf, vllm"
            )
    
    # =========================================================================
    # Service API methods - used by api.py for request handling
    # =========================================================================
    
    def prepare_messages_for_query(self, prompt: str) -> list[Message]:
        """Prepare messages for query including history and memory context.

        Called by the service API before sending to the LLM.

        Args:
            prompt: User's prompt

        Returns:
            List of messages ready for LLM query
        """
        # Refresh from DB only when stale to reduce per-turn read pressure.
        # Set HISTORY_RELOAD_TTL_SECONDS=0 to force legacy "reload every request".
        self._refresh_history_if_stale()

        # Add user message to history first
        self.history_manager.add_message("user", prompt)

        # Build context with system prompt, memory, and history
        return self._build_context_messages(prompt)
    
    def execute_llm_query(
        self,
        messages: list[Message],
        plaintext_output: bool = False,
        stream: bool = False,
    ) -> tuple[str, str, list[dict]]:
        """Execute the LLM query and return response.
        
        Handles tool calling loop if bot has tools enabled.
        Called by the service API to get the response.
        
        Args:
            messages: Prepared messages from prepare_messages_for_query
            plaintext_output: Skip rich formatting
            stream: Enable streaming output
            
        Returns:
            Tuple of (response, tool_context, tool_call_details).
            tool_context is empty if no tools used.
            tool_call_details is a list of per-call dicts for debug logging.
        """
        # Use tool loop if bot has tools enabled and at least one tool backend is available
        if self.bot.uses_tools and (self.memory or self.home_client):
            tool_definitions = self._get_tool_definitions()
            if not tool_definitions:
                response = self.client.query(
                    messages,
                    plaintext_output=plaintext_output,
                    stream=stream,
                )
                return response, "", []
            return query_with_tools(
                messages=messages,
                client=self.client,
                memory_client=self.memory,
                profile_manager=self.profile_manager,
                search_client=self.search_client,
                home_client=self.home_client,
                model_lifecycle=self.model_lifecycle,
                config=self.config,
                user_id=self.user_id,
                bot_id=self.bot_id,
                stream=stream,
                tool_format=self.tool_format,
                tools=tool_definitions,
                adapter=self.adapter,
                history_manager=self.history_manager,
            )
        
        response = self.client.query(
            messages,
            plaintext_output=plaintext_output,
            stream=stream,
        )
        return response, "", []
    
    def finalize_response(
        self,
        response: str,
        tool_context: str = "",
    ):
        """Finalize the response by saving to history.
        
        Called by the service API after receiving the response.
        
        Args:
            response: The assistant's response
            tool_context: Optional tool results to save to history
        """
        # Save tool context first so it appears before the response in history
        if tool_context:
            self.history_manager.add_message(
                "system", f"[Tool Results]\n{tool_context}"
            )
        
        if response:
            self.history_manager.add_message("assistant", response)
    
    def refine_prompt(self, prompt: str, history: list | None = None) -> str:
        """Refine the user's prompt using context.
        
        Overrides base to add service-specific logging.
        """
        slog.debug(f"Refining prompt: {prompt[:50]}...")
        return super().refine_prompt(prompt, history)
