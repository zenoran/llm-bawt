"""Service-side LLMBawt class - supports loading local models (GGUF, etc.)

This is separate from the CLI's core.py which only supports OpenAI API calls.
The service runs on the server and can load models directly.
"""

import logging
import time
import threading
import uuid
from typing import TYPE_CHECKING

from rich.console import Console

from ..clients import LLMClient
from ..clients.openai_client import OpenAIClient
from ..core.base import BaseLLMBawt
from ..utils.config import Config, is_llama_cpp_available
from ..utils.history import Message
from ..tools import query_with_tools
from ..bots import get_system_prompt
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
            
            # Get optional chat_format from model definition (for models like MythoMax)
            chat_format = self.model_definition.get("chat_format")
            client = LlamaCppClient(model_path, config=self.config, chat_format=chat_format, model_definition=self.model_definition)
            load_time_ms = (time.perf_counter() - start_time) * 1000
            slog.model_loaded(self.resolved_model_alias, model_type, load_time_ms)
            return client
        
        else:
            raise ValueError(
                f"Unsupported model type: '{model_type}'. "
                f"Supported types: openai, gguf"
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
        # Reload history from database to ensure we have the latest messages.
        # The ServiceLLMBawt instance is cached across requests, so in-memory
        # history may be stale (missing messages from CLI, other sessions, or
        # messages that were only in DB but not loaded at init time).
        self.load_history()

        # Add user message to history first
        self.history_manager.add_message("user", prompt)

        # Build context with system prompt, memory, and history
        return self._build_context_messages(prompt)
    
    def execute_llm_query(
        self,
        messages: list[Message],
        plaintext_output: bool = False,
        stream: bool = False,
    ) -> tuple[str, str]:
        """Execute the LLM query and return response.
        
        Handles tool calling loop if bot has tools enabled.
        Called by the service API to get the response.
        
        Args:
            messages: Prepared messages from prepare_messages_for_query
            plaintext_output: Skip rich formatting
            stream: Enable streaming output
            
        Returns:
            Tuple of (response, tool_context). tool_context is empty if no tools used.
        """
        # Use tool loop if bot has tools enabled
        if self.bot.uses_tools and self.memory:
            tool_definitions = self._get_tool_definitions()
            return query_with_tools(
                messages=messages,
                client=self.client,
                memory_client=self.memory,
                profile_manager=self.profile_manager,
                search_client=self.search_client,
                model_lifecycle=self.model_lifecycle,
                config=self.config,
                user_id=self.user_id,
                bot_id=self.bot_id,
                stream=stream,
                tool_format=self.tool_format,
                tools=tool_definitions,
                adapter=self.adapter,
            )
        
        response = self.client.query(
            messages,
            plaintext_output=plaintext_output,
            stream=stream,
        )
        return response, ""
    
    def finalize_response(
        self,
        user_prompt: str,
        response: str,
        tool_context: str = "",
    ):
        """Finalize the response by saving to history and triggering extraction.
        
        Called by the service API after receiving the response.
        
        Args:
            user_prompt: The original user prompt
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
            
            # Trigger background memory extraction for all memory-enabled bots
            if self.memory:
                self._trigger_memory_extraction(user_prompt, response)
    
    def _trigger_memory_extraction(self, user_prompt: str, assistant_response: str):
        """Trigger background memory extraction with service-specific logging."""
        if not self.memory:
            slog.debug("Skipping extraction - no memory client")
            return
        
        slog.debug(f"Triggering extraction for {self.bot_id}/{self.user_id}")
        
        def extract():
            try:
                from ..service import ServiceClient
                from ..service.tasks import create_extraction_task
                
                client = ServiceClient()
                if client.is_available():
                    task = create_extraction_task(
                        user_message=user_prompt,
                        assistant_message=assistant_response,
                        bot_id=self.bot_id,
                        user_id=self.user_id,
                        message_ids=[str(uuid.uuid4()), str(uuid.uuid4())],
                        model=self.resolved_model_alias,
                    )
                    client.submit_task(task)
                    slog.debug(f"Extraction task submitted: {task.task_id[:8]}")
                else:
                    slog.warning("Extraction skipped - service unavailable")
            except Exception as e:
                logger.exception(f"Memory extraction failed: {e}")
        
        thread = threading.Thread(target=extract, daemon=True)
        thread.start()
        thread.join(timeout=0.5)
    
    def refine_prompt(self, prompt: str, history: list | None = None) -> str:
        """Refine the user's prompt using context.
        
        Overrides base to add service-specific logging.
        """
        slog.debug(f"Refining prompt: {prompt[:50]}...")
        return super().refine_prompt(prompt, history)
