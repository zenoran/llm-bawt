"""vLLM client for high-throughput inference with HuggingFace models."""

import time
import logging
import os
import contextlib
import json
from typing import List, Dict, Any, Iterator, Optional

from rich.markdown import Markdown
from rich.rule import Rule
from rich.json import JSON

from ..clients.base import LLMClient
from ..utils.config import Config
from ..models.message import Message

try:
    from vllm import LLM, SamplingParams
    _vllm_available = True
except ImportError:
    LLM = None
    SamplingParams = None
    _vllm_available = False

logger = logging.getLogger(__name__)


class VLLMClient(LLMClient):
    """Client for running HuggingFace models using vLLM for high-throughput inference.
    
    vLLM is optimized for serving LLMs with features like:
    - PagedAttention for efficient KV cache management
    - Continuous batching for high throughput
    - Quantization support (AWQ, GPTQ, FP8)
    - Fast model loading with tensor parallelism
    
    Note: Model loading takes 30-150s due to CUDA graph compilation.
    """

    SUPPORTS_STREAMING = True

    def __init__(
        self,
        model_id: str,
        config: Config,
        model_definition: dict | None = None
    ):
        """Initialize vLLM client with lazy model loading.
        
        Args:
            model_id: HuggingFace model ID or local path to model
            config: Global configuration object
            model_definition: Model-specific configuration from models.yaml
        """
        if not _vllm_available:
            raise ImportError(
                "vLLM not found. Install with: pip install vllm\n"
                "For CUDA 12.1+: pip install vllm\n"
                "See https://docs.vllm.ai/en/latest/getting_started/installation.html"
            )
        
        super().__init__(model_id, config, model_definition=model_definition)
        self.model_id = model_id
        self.llm_engine: LLM | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the vLLM model with optimized settings.
        
        This takes 30-150 seconds due to CUDA graph compilation.
        Suppresses vLLM's verbose initialization output unless VERBOSE is enabled.
        """
        # Guard against double-loading
        if self.llm_engine is not None:
            logger.warning(f"Model {self.model_id} already loaded, skipping reload")
            return
        
        if self.config.VERBOSE:
            self.console.print(
                f"[bold bright_cyan]Loading vLLM model:[/bold bright_cyan] {self.model_id}"
            )
            self.console.print(
                "[dim]⚠ Model loading takes 30-150s (CUDA graph compilation)[/dim]"
            )

        # Extract vLLM-specific config from model definition
        quantization = self.model_definition.get("quantization")  # awq|gptq|fp8|null
        dtype = self.model_definition.get("dtype", "auto")  # float16|bfloat16|auto
        max_model_len = self.model_definition.get("max_model_len")
        gpu_memory_utilization = self.model_definition.get("gpu_memory_utilization", 0.85)
        enforce_eager = self.model_definition.get("enforce_eager", False)
        enable_prefix_caching = self.model_definition.get("enable_prefix_caching", True)
        
        # Build vLLM engine parameters
        engine_params = {
            "model": self.model_id,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "enable_prefix_caching": enable_prefix_caching,
            "trust_remote_code": True,  # Always True for community models
        }
        
        # Add optional parameters if specified
        # Note: Don't pass quantization parameter - let vLLM auto-detect from model config.
        # Only override if explicitly needed (e.g., forcing a different quantization method).
        # Passing quantization when it's already in the model's config.json causes validation errors.
        if quantization and self.config.VERBOSE:
            logger.info(f"Model specifies quantization: {quantization} (vLLM will auto-detect from config)")
        if max_model_len:
            engine_params["max_model_len"] = max_model_len

        if self.config.VERBOSE:
            log_params = {k: v for k, v in engine_params.items() if k != "model"}
            logger.debug(f"vLLM engine parameters: {log_params}")
            if quantization:
                self.console.print(f"[dim]Model quantization: {quantization} (auto-detected)[/dim]")
            self.console.print(f"[dim]Dtype: {dtype}[/dim]")
            self.console.print(f"[dim]GPU Memory: {gpu_memory_utilization*100:.0f}%[/dim]")
            if max_model_len:
                self.console.print(f"[dim]Max Model Length: {max_model_len}[/dim]")

        start_time = time.time()
        
        try:
            # Always show vLLM initialization output - it includes useful progress info
            # and prevents confusion during the 30-150s CUDA graph compilation.
            # vLLM's output is informative and shows the model is actually loading.
            if LLM is None:
                raise ImportError("LLM class not available from vllm")
            
            if not self.config.VERBOSE:
                # Non-verbose: at least show we're starting
                self.console.print(
                    "[yellow]⏳ Compiling CUDA graphs (30-150s)...[/yellow]"
                )
            
            self.llm_engine = LLM(**engine_params)
            
            load_time = time.time() - start_time
            
            if self.config.VERBOSE:
                self.console.print(
                    f"[green]✓ vLLM model loaded in {load_time:.1f}s[/green]"
                )
                # Show actual max model length if available
                if hasattr(self.llm_engine.llm_engine, 'model_config'):
                    actual_max_len = getattr(
                        self.llm_engine.llm_engine.model_config,
                        'max_model_len',
                        None
                    )
                    if actual_max_len:
                        self.console.print(f"[dim]Context window: {actual_max_len} tokens[/dim]")
            else:
                # Minimal output for non-verbose mode
                self.console.print(
                    f"[green]✓ Model loaded[/green] [dim]({load_time:.1f}s)[/dim]"
                )

        except Exception as e:
            self.console.print(
                f"[bold red]Error loading vLLM model {self.model_id}:[/bold red] {e}"
            )
            self.console.print(
                "[yellow]Troubleshooting tips:[/yellow]\n"
                "  1. Ensure CUDA is available and compatible\n"
                "  2. Check GPU has sufficient VRAM for model\n"
                "  3. Try reducing gpu_memory_utilization\n"
                "  4. For quantized models, verify quantization format matches model"
            )
            logger.exception("vLLM model loading failed")
            raise

    def _prepare_messages_for_vllm(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert Message objects to vLLM chat format.
        
        Args:
            messages: List of Message objects
            
        Returns:
            List of dicts with 'role' and 'content' keys
        """
        return [msg.to_api_format() for msg in messages]

    def _create_sampling_params(
        self,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | str | None = None,
    ) -> "SamplingParams":
        """Create vLLM SamplingParams from config and overrides.
        
        Args:
            max_tokens: Override for max output tokens
            temperature: Override for temperature
            top_p: Override for top_p
            stop: Stop sequences (string or list)
            
        Returns:
            SamplingParams object for vLLM generation
        """
        if SamplingParams is None:
            raise ImportError("SamplingParams not available from vllm")
        
        params = {
            "max_tokens": max_tokens or self.effective_max_tokens,
            "temperature": temperature if temperature is not None else self.config.TEMPERATURE,
            "top_p": top_p if top_p is not None else self.config.TOP_P,
        }
        
        # Handle stop sequences
        if stop:
            if isinstance(stop, str):
                params["stop"] = [stop]
            else:
                params["stop"] = stop
        
        return SamplingParams(**params)

    def query(
        self,
        messages: List[Message],
        plaintext_output: bool = False,
        stream: bool = True,
        stop: list[str] | str | None = None,
        **kwargs: Any
    ) -> str:
        """Query the vLLM model using chat template.
        
        Args:
            messages: List of Message objects
            plaintext_output: If True, return raw text without formatting
            stream: If True, stream output token-by-token (currently no-op for vLLM)
            stop: Stop sequences to terminate generation
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Generated response as string
        """
        if not self.llm_engine:
            error_msg = "Error: vLLM model not properly initialized."
            self.console.print(f"[bold red]{error_msg}[/bold red]")
            return error_msg

        # Convert messages to vLLM format
        vllm_messages = self._prepare_messages_for_vllm(messages)
        
        # Create sampling parameters
        sampling_params = self._create_sampling_params(stop=stop)
        
        if self.config.VERBOSE:
            self.console.print(Rule("Querying vLLM Model", style="bright_cyan"))
            self.console.print(
                f"[dim]Params:[/dim] [italic]max_tokens={sampling_params.max_tokens}, "
                f"temp={sampling_params.temperature}, top_p={sampling_params.top_p}[/italic]"
            )
            if sampling_params.stop:
                self.console.print(f"[dim]Stop sequences:[/dim] {sampling_params.stop}")
            
            self.console.print(Rule("Request Messages", style="dim bright_cyan"))
            try:
                payload_str = json.dumps(vllm_messages, indent=2)
                self.console.print(JSON(payload_str))
            except TypeError as e:
                logger.error(f"Could not serialize messages for display: {e}")
                import pprint
                self.console.print(pprint.pformat(vllm_messages))
            
            self.console.print(Rule(style="bright_cyan"))

        response_text = ""
        panel_title, panel_border_style = self.get_styling()
        
        try:
            start_time = time.time()
            
            # vLLM's chat method applies the chat template automatically
            # Returns a list of RequestOutput objects
            outputs = self.llm_engine.chat(
                messages=vllm_messages,
                sampling_params=sampling_params,
                use_tqdm=False,  # Disable progress bar
            )
            
            end_time = time.time()
            
            # Extract the generated text
            if outputs and len(outputs) > 0:
                output = outputs[0]
                response_text = output.outputs[0].text.strip()
                
                if self.config.VERBOSE:
                    # Calculate tokens/sec if possible
                    num_tokens = len(output.outputs[0].token_ids)
                    prompt_tokens = len(output.prompt_token_ids)
                    elapsed = end_time - start_time
                    
                    if elapsed > 0:
                        tokens_per_sec = num_tokens / elapsed
                        self.console.print(
                            f"[dim]Generated {num_tokens} tokens "
                            f"(prompt: {prompt_tokens}) in {elapsed:.2f}s "
                            f"({tokens_per_sec:.2f} tokens/sec)[/dim]"
                        )
                    else:
                        self.console.print(
                            f"[dim]Generated {num_tokens} tokens (prompt: {prompt_tokens})[/dim]"
                        )
                
                # Format output unless plaintext requested
                if not plaintext_output:
                    # Split on double newline for panel formatting
                    parts = response_text.split("\n\n", 1)
                    first_part = parts[0]
                    second_part = parts[1] if len(parts) > 1 else None
                    self._print_assistant_message(
                        first_part,
                        second_part=second_part,
                        panel_title=panel_title,
                        panel_border_style=panel_border_style
                    )
            else:
                self.console.print(
                    "[bold red]Error: No response generated by vLLM model.[/bold red]"
                )
                response_text = "Error: Failed to get response from vLLM model."
        
        except Exception as e:
            self.console.print(f"[bold red]Error during vLLM generation:[/bold red] {e}")
            logger.exception("Error during vLLM generation")
            response_text = f"Error: An exception occurred during generation: {e}"
        
        finally:
            if self.config.VERBOSE:
                self.console.print(Rule(style="bright_cyan"))
        
        return response_text

    def stream_raw(
        self,
        messages: List[Message],
        stop: list[str] | str | None = None,
        **kwargs: Any
    ) -> Iterator[str]:
        """Stream raw text chunks from vLLM (Phase 1: single chunk).
        
        Phase 1 Implementation: Returns entire response as single chunk.
        This is used by the API service for SSE streaming.
        
        Phase 2 TODO: Implement true streaming using AsyncLLMEngine:
        - Replace LLM with AsyncLLMEngine for async/await support
        - Use engine.generate() with async iteration
        - Yield tokens as they're generated
        - Requires async refactor of service layer
        
        Args:
            messages: List of Message objects
            stop: Stop sequences to terminate generation
            **kwargs: Additional arguments (ignored)
            
        Yields:
            Text chunks (currently entire response in one chunk)
        """
        if not self.llm_engine:
            raise RuntimeError("vLLM model not properly initialized.")
        
        # Phase 1: Generate complete response and yield as single chunk
        # This matches the fallback behavior in base.py but with vLLM generation
        vllm_messages = self._prepare_messages_for_vllm(messages)
        sampling_params = self._create_sampling_params(stop=stop)
        
        logger.debug("stream_raw: Generating response (Phase 1: single chunk)")
        
        try:
            outputs = self.llm_engine.chat(
                messages=vllm_messages,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            
            if outputs and len(outputs) > 0:
                response_text = outputs[0].outputs[0].text
                yield response_text
            else:
                yield "Error: No response from vLLM model"
        
        except Exception as e:
            logger.exception(f"Error in vLLM stream_raw: {e}")
            yield f"Error: {e}"

    def supports_native_tools(self) -> bool:
        """Return False - in-process vLLM doesn't produce OpenAI tool_call objects.
        
        vLLM's synchronous LLM class doesn't natively support tool calling format.
        For tool support, would need:
        1. OpenAI-compatible vLLM server (vllm serve) OR
        2. Custom chat template + grammar constraints
        
        Current implementation uses text-based tool calling via adapters.
        """
        return False

    def get_styling(self) -> tuple[str | None, str]:
        """Return vLLM-specific styling for output formatting.
        
        Returns:
            Tuple of (panel_title, border_style)
            - panel_title: None to use bot name from base class
            - border_style: "bright_cyan" for vLLM branding
        """
        return None, "bright_cyan"

    def unload(self) -> None:
        """Unload the vLLM model and free GPU memory.
        
        Deletes the engine, runs garbage collection, and clears CUDA cache.
        Called by ModelLifecycleManager when switching models.
        """
        if self.llm_engine is None:
            return
        
        logger.info(f"Unloading vLLM model: {self.model_id}")
        
        try:
            # Delete engine instance
            del self.llm_engine
            self.llm_engine = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    if self.config.VERBOSE:
                        self.console.print("[green]✓ CUDA cache cleared[/green]")
            except ImportError:
                logger.warning("PyTorch not available, cannot clear CUDA cache")
            
            if self.config.VERBOSE:
                self.console.print("[green]✓ vLLM model unloaded and memory freed[/green]")
        
        except Exception as e:
            logger.warning(f"Error during vLLM model unload: {e}")
            # Don't re-raise - unload is best-effort cleanup
