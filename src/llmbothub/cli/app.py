import argparse
import subprocess
import traceback
import sys  # Import sys for exit codes
from rich.console import Console
from rich.prompt import Prompt
from rich.columns import Columns
from llmbothub.utils.config import Config, has_database_credentials
from llmbothub.utils.config import set_config_value
from llmbothub.utils.input_handler import MultilineInputHandler
from llmbothub.core import LLMBotHub
from llmbothub.model_manager import list_models, update_models_interactive, delete_model, ModelManager, is_service_mode_enabled
from llmbothub.gguf_handler import handle_add_gguf
from llmbothub.bots import BotManager
from llmbothub.profiles import ProfileManager, EntityType, AttributeCategory
from llmbothub.utils.streaming import render_streaming_response, render_complete_response
from llmbothub.shared.logging import LogConfig
import logging
from pathlib import Path
from rich.table import Table
from rich.panel import Panel
from typing import Iterator

console = Console()

# Cache for service client
_service_client = None

def get_service_client(config: Config | None = None):
    """Get or create the service client singleton."""
    global _service_client
    if _service_client is None:
        try:
            from llmbothub.service import ServiceClient
            # Build service URL from config if not provided
            if config is None:
                config = Config()
            service_url = getattr(config, 'SERVICE_URL', None)
            if not service_url and hasattr(config, 'SERVICE_HOST') and hasattr(config, 'SERVICE_PORT'):
                service_url = f"http://{config.SERVICE_HOST}:{config.SERVICE_PORT}"
            _service_client = ServiceClient(http_url=service_url)
        except ImportError:
            _service_client = False  # Mark as unavailable
    return _service_client if _service_client else None


def query_via_service(
    prompt: str,
    model: str | None,
    bot_id: str | None,
    user_id: str | None,
    plaintext_output: bool,
    stream: bool,
    config: Config | None = None,
) -> bool:
    """
    Query via the background service if available.

    Returns True if query was handled by service, False if service unavailable.
    """
    from rich.markdown import Markdown
    from rich.align import Align

    if config is None:
        config = Config()
    client = get_service_client(config)
    if not client or not client.is_available():
        return False

    messages = [{"role": "user", "content": prompt}]

    # Get bot name for display
    bot_manager = BotManager(Config())
    bot = bot_manager.get_bot(bot_id) if bot_id else bot_manager.get_default_bot()
    bot_name = bot.name if bot else (bot_id or "Assistant")

    # Determine panel style based on bot
    panel_styles = {
        "mira": ("magenta", "magenta"),
        "nova": ("cyan", "cyan"),
        "spark": ("yellow", "yellow"),
    }
    title_style, border_style = panel_styles.get(bot_id or "", ("green", "green"))

    def format_panel_title(actual_model: str | None) -> str:
        """Format panel title with actual model used."""
        # Use \[ to escape opening bracket so Rich doesn't interpret [model] as markup
        # Use dim style for model and service parts to emphasize bot name
        model_part = f" [dim]\\[{actual_model}][/dim]" if actual_model else ""
        return f"[bold {title_style}]{bot_name}[/bold {title_style}]{model_part} [dim](service)[/dim]"

    try:
        if stream:
            # Stream response using shared streaming utilities
            raw_iterator = client.chat_completion(
                messages=messages,
                model=model,
                bot_id=bot_id,
                user_id=user_id,
                stream=True,
            )

            # If service returned None (failed), fall back to local
            if raw_iterator is None:
                return False

            # Extract model from first metadata chunk, then yield content
            actual_model = model  # Default to requested model
            got_metadata = False  # Track if service responded with model info

            def content_iterator() -> Iterator[str]:
                nonlocal actual_model, got_metadata
                for item in raw_iterator:
                    if isinstance(item, dict) and "warnings" in item:
                        # Service warnings (e.g. model fallback)
                        for warning in item["warnings"]:
                            console.print(f"[yellow]{warning}[/yellow]")
                        if item.get("model"):
                            actual_model = item["model"]
                            got_metadata = True
                    elif isinstance(item, dict) and "model" in item:
                        # Metadata chunk with actual model - service is responding
                        actual_model = item["model"]
                        got_metadata = True
                    elif isinstance(item, str):
                        yield item

            # We need to peek at the first item to get the model before rendering
            content_iter = content_iterator()
            first_content = None
            try:
                first_content = next(content_iter)
            except StopIteration:
                if got_metadata:
                    # Service responded (we got the model) but produced no content.
                    # This typically means a server-side error (e.g. tool execution failed).
                    # Show error instead of silently falling back to a different model.
                    error_detail = ""
                    if client.last_error:
                        error_detail = f" {client.last_error}"
                    console.print(f"[bold red]Service returned empty response.[/bold red]{error_detail}")
                    console.print("[dim]Check service logs for details.[/dim]")
                    return True  # Handled (don't fall back to different model)
                # Service didn't respond at all - allow fallback
                return False

            # Now we have the actual_model set, create panel title
            panel_title = format_panel_title(actual_model)

            # Create a new iterator that includes the first content
            def full_content_iterator() -> Iterator[str]:
                if first_content:
                    yield first_content
                yield from content_iter

            result = render_streaming_response(
                stream_iterator=full_content_iterator(),
                console=console,
                panel_title=panel_title,
                panel_border_style=border_style,
                plaintext_output=plaintext_output,
            )
            # If streaming returned no content (service error), fall back to local
            if not result:
                return False
            return True
        else:
            # Non-streaming: get full response to extract actual model
            response = client.chat_completion(
                messages=messages,
                model=model,
                bot_id=bot_id,
                user_id=user_id,
                stream=False,
            )
            if response and "choices" in response:
                content = response["choices"][0].get("message", {}).get("content")
                actual_model = response.get("model", model)  # Use actual model from response
                panel_title = format_panel_title(actual_model)

                if content:
                    if plaintext_output:
                        print(content)
                    else:
                        # Use shared render function for split display
                        render_complete_response(
                            response=content,
                            console=console,
                            panel_title=panel_title,
                            panel_border_style=border_style,
                        )
                    return True
    except Exception as e:
        # Store error in service client for later display
        service_client = get_service_client(config)
        if service_client:
            service_client.last_error = str(e)
        logging.warning(f"Service query failed: {e}")
        if config.VERBOSE:
            import traceback
            traceback.print_exc()

    return False


def show_status(config: Config, args: argparse.Namespace | None = None):
    """Display overall system status including dependencies, bots, memory, and configuration."""
    import importlib.util
    import os
    import shutil
    from llmbothub.utils.config import is_huggingface_available, is_llama_cpp_available, has_database_credentials
    
    console.print(Panel.fit("[bold magenta]llmbothub System Status[/bold magenta]", border_style="grey39"))
    console.print()

    def make_table() -> Table:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        return table

    def format_active_setting(label: str, env_key: str, env_value: object, config_value: object, redact: bool = False) -> str:
        if env_value is not None and env_value != "":
            display = "<set>" if redact else str(env_value)
            return f"{label}: {display} [dim]({env_key})[/dim]"
        if config_value is None or config_value == "":
            return f"{label}: [dim]<not set>[/dim]"
        display = "<set>" if redact else str(config_value)
        return f"{label}: {display} [dim](config)[/dim]"

    def format_env_tag(env_key: str) -> str:
        return f"[dim]({env_key})[/dim]"

    def format_compact_setting(label: str, env_key: str, env_value: object, config_value: object, redact: bool = False) -> str:
        if env_value is not None and env_value != "":
            display = "<set>" if redact else str(env_value)
            return f"{label}={display} {format_env_tag(env_key)}"
        if config_value is None or config_value == "":
            return f"{label}=<not set>"
        display = "<set>" if redact else str(config_value)
        return f"{label}={display} [dim](config)[/dim]"

    # Check if USE_SERVICE is enabled - uses centralized function from model_manager
    # Keep env/config vars for display purposes
    use_service_env = os.getenv("LLMBOTHUB_USE_SERVICE", "").lower() in ("true", "1", "yes")
    use_service_config = getattr(config, "USE_SERVICE", False)
    use_service = is_service_mode_enabled(config)  # Centralized check
    service_url_env = os.getenv("LLMBOTHUB_SERVICE_URL")
    service_url_config = getattr(config, "SERVICE_URL", None)
    service_url = service_url_env or service_url_config

    service_client = None
    service_status = None
    service_available = False
    service_error = None
    try:
        service_client = get_service_client(config)
        if service_client and service_client.is_available(force_check=True):
            service_available = True
            try:
                service_status = service_client.get_status(silent=True)
            except Exception as e:
                service_error = str(e)
    except Exception as e:
        service_error = str(e)

    sections: list[tuple[str, Table]] = []

    # --- Execution Section ---
    exec_table = make_table()
    if use_service:
        if use_service_env:
            mode_suffix = format_env_tag("LLMBOTHUB_USE_SERVICE")
        else:
            mode_suffix = "[dim](config: USE_SERVICE)[/dim]"
        exec_table.add_row("Mode", f"[bold cyan]Service Mode[/bold cyan] {mode_suffix}")
        if service_url:
            exec_table.add_row(
                "Service URL",
                f"[cyan]{service_url}[/cyan] {format_env_tag('LLMBOTHUB_SERVICE_URL')}"
                if service_url_env
                else f"[cyan]{service_url}[/cyan] [dim](config: SERVICE_HOST/PORT)[/dim]",
            )
        elif hasattr(config, "SERVICE_HOST") and hasattr(config, "SERVICE_PORT"):
            default_url = f"http://{config.SERVICE_HOST}:{config.SERVICE_PORT}"
            exec_table.add_row(
                "Service URL",
                f"[cyan]{default_url}[/cyan] [dim](default, config: SERVICE_HOST/PORT)[/dim]",
            )
        if service_available and service_status and getattr(service_status, "available", False):
            exec_table.add_row("API Usage", "[green]✓ Connected via service API[/green]")
        elif service_available:
            exec_table.add_row("API Usage", "[yellow]⚠ Service reachable but unhealthy[/yellow]")
        else:
            exec_table.add_row("API Usage", "[red]✗ Service not reachable[/red]")
    else:
        mode_suffix = "[dim](config: USE_SERVICE)[/dim]"
        exec_table.add_row("Mode", f"[bold green]Direct Execution[/bold green] [dim](local)[/dim] {mode_suffix}")
        exec_table.add_row("Info", "[dim]LLM queries run directly in this process[/dim]")
        exec_table.add_row("Hint", "[dim]Set LLMBOTHUB_USE_SERVICE=True to use service mode[/dim]")
        if service_available:
            exec_table.add_row("Notice", "[yellow]Service detected but not enabled[/yellow]")
            exec_table.add_row(" ", f"[dim]Service available at: {service_client.http_url}[/dim]")
    sections.append(("Execution", exec_table))

    # --- Current Session Section ---
    if args:
        session_table = make_table()

        # Determine effective bot
        bot_manager = BotManager(config)
        if args.bot:
            target_bot = bot_manager.get_bot(args.bot)
            if target_bot:
                bot_display = f"[bold cyan]{target_bot.name}[/bold cyan] ({args.bot}) [dim]--bot {args.bot}[/dim]"
            else:
                bot_display = f"[red]Unknown: {args.bot}[/red]"
        elif getattr(args, 'local', False):
            target_bot = bot_manager.get_default_bot(local_mode=True)
            bot_display = f"[bold yellow]{target_bot.name}[/bold yellow] ({target_bot.slug}) [dim]--local default[/dim]"
        else:
            target_bot = bot_manager.get_default_bot()
            bot_display = f"[bold cyan]{target_bot.name}[/bold cyan] ({target_bot.slug}) [dim]default[/dim]"
        session_table.add_row("Bot", bot_display)

        # Determine effective model: explicit > bot default > config default
        explicit_model = getattr(args, 'model', None)
        selection = bot_manager.select_model(
            explicit_model,
            bot_slug=target_bot.slug,
            local_mode=getattr(args, 'local', False),
        )
        model_alias = selection.alias
        if selection.source == "explicit":
            model_source = "[dim]-m flag[/dim]"
        elif selection.source == "bot_default":
            model_source = "[dim]bot default[/dim]"
        elif selection.source == "config_default":
            model_source = "[dim]config default[/dim]"
        else:
            model_source = ""

        if model_alias:
            # Check if model exists in defined models
            defined_models = config.defined_models.get("models", {})
            if model_alias in defined_models:
                model_def = defined_models.get(model_alias, {})
                model_type = model_def.get("type", "unknown")
                model_display = f"[bold green]{model_alias}[/bold green] [dim]({model_type})[/dim] {model_source}"
            else:
                # Check for partial matches
                matches = [a for a in defined_models.keys() if model_alias.lower() in a.lower()]
                if matches:
                    model_display = f"[yellow]{model_alias}[/yellow] [dim](partial match: {matches[0]})[/dim] {model_source}"
                else:
                    model_display = f"[red]{model_alias}[/red] [dim](not found)[/dim] {model_source}"
        else:
            model_display = "[dim]not set[/dim]"
        session_table.add_row("Model", model_display)

        # Determine effective user
        user_id = getattr(args, 'user', None) or config.DEFAULT_USER
        if getattr(args, 'local', False):
            user_display = "[dim]N/A (--local mode)[/dim]"
        else:
            try:
                from llmbothub.profiles import ProfileManager, EntityType

                manager = ProfileManager(config)
                profile, _ = manager.get_or_create_profile(EntityType.USER, user_id)
                if profile and profile.display_name:
                    user_display = f"[bold cyan]{profile.display_name}[/bold cyan] ({user_id})"
                else:
                    user_display = f"{user_id} [dim](no profile)[/dim]"
            except Exception:
                user_display = f"{user_id}"
        session_table.add_row("User", user_display)

        # Show mode
        mode_parts = []
        if getattr(args, 'local', False):
            mode_parts.append("[yellow]local[/yellow]")
        if getattr(args, 'plain', False):
            mode_parts.append("plain")
        if getattr(args, 'no_stream', False):
            mode_parts.append("no-stream")
        mode_display = ", ".join(mode_parts) if mode_parts else "[dim]default[/dim]"
        session_table.add_row("Flags", mode_display)

        sections.append(("Current Session", session_table))

    # --- Service Section ---
    if use_service or service_available or service_url:
        service_table = make_table()
        if service_available and service_status and getattr(service_status, "available", False):
            uptime_str = ""
            if service_status.uptime_seconds:
                hours, remainder = divmod(int(service_status.uptime_seconds), 3600)
                minutes, seconds = divmod(remainder, 60)
                if hours > 0:
                    uptime_str = f"{hours}h {minutes}m"
                elif minutes > 0:
                    uptime_str = f"{minutes}m {seconds}s"
                else:
                    uptime_str = f"{seconds}s"

            service_state = "[green]✓ Connected[/green]"
            if uptime_str:
                service_state += f" [dim](uptime: {uptime_str})[/dim]"
            service_table.add_row("Status", service_state)
            service_table.add_row("Version", f"{service_status.version or 'unknown'}")
            service_table.add_row("Tasks", f"{service_status.tasks_processed} processed / {service_status.tasks_pending} pending")
            if service_status.models_loaded:
                service_table.add_row("Models Loaded", ", ".join(service_status.models_loaded))
        elif service_available:
            service_table.add_row("Status", "[yellow]⚠ Service unhealthy[/yellow]")
        else:
            status_msg = "[red]✗ Cannot connect to service[/red]" if use_service else "[dim]○ Service not running[/dim]"
            service_table.add_row("Status", status_msg)
            if service_error:
                service_table.add_row("Error", f"[dim]{service_error[:80]}[/dim]")

        # Check MCP Memory Server (if using service mode)
        if use_service and service_url:
            try:
                import httpx
                # Extract host from service URL and check MCP port (usually 8001)
                mcp_status = "[dim]○ Not checked[/dim]"
                if "localhost" in service_url or "127.0.0.1" in service_url:
                    mcp_url = service_url.replace("8642", "8001")  # Assume MCP on 8001
                    try:
                        response = httpx.get(f"{mcp_url}/health", timeout=2.0)
                        if response.status_code == 200:
                            mcp_status = f"[green]✓ Running[/green] [dim]({mcp_url})[/dim]"
                        else:
                            mcp_status = f"[yellow]⚠ Unhealthy[/yellow] [dim](status {response.status_code})[/dim]"
                    except Exception:
                        mcp_status = f"[red]✗ Not reachable[/red] [dim]({mcp_url})[/dim]"
                service_table.add_row("MCP Server", mcp_status)
            except Exception:
                pass

        sections.append(("Service", service_table))

    # --- Local Dependencies Section ---
    deps_table = make_table()
    
    # Check CUDA availability
    cuda_status = "[dim]○ Not available[/dim]"
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        # Try to get CUDA version
        try:
            import subprocess
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Parse version from output like "Cuda compilation tools, release 12.2, V12.2.140"
                import re
                match = re.search(r"release ([\d.]+)", result.stdout)
                version = match.group(1) if match else "unknown"
                cuda_status = f"[green]✓ Available[/green] (CUDA {version})"
            else:
                cuda_status = f"[yellow]⚠ nvcc found but failed[/yellow]"
        except Exception:
            cuda_status = f"[green]✓ Available[/green] (nvcc at {nvcc_path})"
    deps_table.add_row("CUDA", cuda_status)
    
    # Check huggingface-hub (for GGUF downloads)
    hf_hub_available = importlib.util.find_spec("huggingface_hub") is not None
    hf_hub_status = "[green]✓ Installed[/green]" if hf_hub_available else "[red]✗ Not installed[/red] (pip install huggingface-hub)"
    deps_table.add_row("huggingface-hub", hf_hub_status)
    
    # Check llama-cpp-python (for GGUF inference)
    llama_cpp_available = is_llama_cpp_available()
    llama_cpp_status = "[green]✓ Installed[/green]" if llama_cpp_available else "[red]✗ Not installed[/red] (pip install llama-cpp-python)"
    # If llama-cpp is installed, check if it has CUDA support
    if llama_cpp_available:
        try:
            # Suppress the ggml_cuda_init logging during import (native C code writes to fd)
            import sys
            import os as os_module
            # Save original file descriptors
            devnull = os_module.open(os_module.devnull, os_module.O_WRONLY)
            old_stdout_fd = os_module.dup(1)
            old_stderr_fd = os_module.dup(2)
            try:
                os_module.dup2(devnull, 1)
                os_module.dup2(devnull, 2)
                from llama_cpp import llama_supports_gpu_offload
                has_gpu = llama_supports_gpu_offload()
            finally:
                os_module.dup2(old_stdout_fd, 1)
                os_module.dup2(old_stderr_fd, 2)
                os_module.close(devnull)
                os_module.close(old_stdout_fd)
                os_module.close(old_stderr_fd)
            if has_gpu:
                llama_cpp_status = "[green]✓ Installed[/green] (GPU support)"
            else:
                llama_cpp_status = "[green]✓ Installed[/green] [dim](CPU only)[/dim]"
        except (ImportError, AttributeError, OSError):
            llama_cpp_status = "[green]✓ Installed[/green]"
    deps_table.add_row("llama-cpp-python", llama_cpp_status)
    
    # Check HuggingFace transformers (for HF models)
    hf_available = is_huggingface_available()
    hf_status = "[green]✓ Installed[/green]" if hf_available else "[dim]○ Not installed[/dim] (pip install transformers torch)"
    deps_table.add_row("transformers + torch", hf_status)
    
    # Check ollama connectivity
    ollama_status = "[dim]○ Not checked[/dim]"
    if config.OLLAMA_URL:
        try:
            import httpx
            response = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=2.0)
            if response.status_code == 200:
                model_count = len(response.json().get("models", []))
                ollama_status = f"[green]✓ Connected[/green] ({model_count} models)"
            else:
                ollama_status = f"[yellow]⚠ Server responded with {response.status_code}[/yellow]"
        except Exception:
            ollama_status = f"[red]✗ Not reachable[/red] ({config.OLLAMA_URL})"
    providers_table = make_table()
    ollama_url_env = os.getenv("LLMBOTHUB_OLLAMA_URL")
    ollama_url_value = ollama_url_env or config.OLLAMA_URL
    if ollama_url_env:
        ollama_tag = format_env_tag("LLMBOTHUB_OLLAMA_URL")
    else:
        ollama_tag = "[dim](config: OLLAMA_URL)[/dim]"
    providers_table.add_row("Ollama", f"{ollama_status} {ollama_tag} [dim]({ollama_url_value})[/dim]")
    
    # Check OpenAI API key (from environment variable)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_status = "[green]✓ API key set[/green]" if openai_api_key else "[dim]○ No API key[/dim] (export OPENAI_API_KEY=...)"
    openai_tag = format_env_tag("OPENAI_API_KEY")
    openai_key_state = "<set>" if openai_api_key else "<not set>"
    providers_table.add_row("OpenAI", f"{openai_status} [dim](key: {openai_key_state} {openai_tag})[/dim]")

    sections.append(("Local Dependencies", deps_table))
    sections.append(("Providers", providers_table))

    # --- Bots Section ---
    bot_manager = BotManager(config)
    default_bot = bot_manager.get_default_bot()
    local_bot = bot_manager.get_default_bot(local_mode=True)
    
    bots_table = make_table()
    bots_table.add_row("Default", f"[bold cyan]{default_bot.name}[/bold cyan] - {default_bot.description}")
    bots_table.add_row("Local (--local)", f"[bold yellow]{local_bot.name}[/bold yellow] - {local_bot.description}")
    bots_table.add_row("Available", ", ".join(b.slug for b in bot_manager.list_bots()))
    sections.append(("Bots", bots_table))

    # --- Models Section ---
    defined_models = config.defined_models.get("models", {})
    models_table = make_table()
    if config.DEFAULT_MODEL_ALIAS:
        if config.DEFAULT_MODEL_ALIAS in defined_models:
            model_def = defined_models[config.DEFAULT_MODEL_ALIAS]
            model_type = model_def.get("type", "unknown")
            default_model_env = os.getenv("LLMBOTHUB_DEFAULT_MODEL_ALIAS")
            models_table.add_row(
                "Default Model",
                f"{config.DEFAULT_MODEL_ALIAS} [dim]({model_type})[/dim]",
            )
            if default_model_env:
                model_source = format_env_tag("LLMBOTHUB_DEFAULT_MODEL_ALIAS")
            else:
                model_source = "[dim](config: DEFAULT_MODEL_ALIAS)[/dim]"
            models_table.add_row("Default Model Source", f"{model_source}")

            # Check model-specific requirements
            model_check = "[green]✓ Valid[/green]"
            if model_type == "gguf":
                if not is_llama_cpp_available():
                    model_check = "[red]✗ llama-cpp-python not installed[/red]"
                elif not hf_hub_available:
                    model_check = "[yellow]⚠ huggingface-hub not installed (needed for downloads)[/yellow]"
                else:
                    # Check if model file exists
                    source = model_def.get("source", "")
                    if "/" in source:
                        parts = source.split("/")
                        if len(parts) >= 3:
                            repo_id = "/".join(parts[:2])
                            filename = "/".join(parts[2:])
                            cache_path = os.path.join(config.MODEL_CACHE_DIR, repo_id, filename)
                            if os.path.exists(cache_path):
                                model_check = f"[green]✓ Cached[/green] [dim]({os.path.getsize(cache_path) / (1024**3):.1f}GB)[/dim]"
                            else:
                                model_check = "[yellow]⚠ Not cached (will download on first use)[/yellow]"
            elif model_type == "ollama":
                if ollama_status.startswith("[green]"):
                    model_check = "[green]✓ Ollama available[/green]"
                else:
                    model_check = "[yellow]⚠ Ollama server not reachable[/yellow]"
            elif model_type == "openai":
                if openai_api_key:
                    model_check = "[green]✓ API key set[/green]"
                else:
                    model_check = "[red]✗ No OPENAI_API_KEY[/red]"
            elif model_type == "huggingface":
                if not hf_available:
                    model_check = "[red]✗ transformers/torch not installed[/red]"

            models_table.add_row("Default Model Check", f"{model_check}")
        else:
            models_table.add_row("Default Model", f"[red]✗ '{config.DEFAULT_MODEL_ALIAS}' not found in models.yaml[/red]")
    else:
        models_table.add_row("Default Model", "[yellow]⚠ Not configured[/yellow]")

    # Models config file exists and is valid
    models_config_path = config.MODELS_CONFIG_PATH
    models_config_env = os.getenv("LLMBOTHUB_MODELS_CONFIG_PATH")
    if os.path.exists(models_config_path):
        model_count = len(defined_models)
        if model_count > 0:
            models_table.add_row("Models Config", f"[green]✓ {model_count} models defined[/green]")
        else:
            models_table.add_row("Models Config", "[yellow]⚠ File exists but no models defined[/yellow]")
    else:
        models_table.add_row("Models Config", f"[red]✗ File not found: {models_config_path}[/red]")
    if models_config_env:
        config_source = format_env_tag("LLMBOTHUB_MODELS_CONFIG_PATH")
    else:
        config_source = "[dim](config: MODELS_CONFIG_PATH)[/dim]"
    models_table.add_row("Models Config Source", f"{config_source} [dim]({models_config_path})[/dim]")

    sections.append(("Models", models_table))

    # --- Memory Section ---
    db_status = "[yellow]Not Configured[/yellow]"
    long_term_count = 0
    messages_count = 0
    db_stats = None

    if not has_database_credentials(config):
        db_status = "[yellow]Not Configured[/yellow] [dim](set LLMBOTHUB_POSTGRES_PASSWORD)[/dim]"
    else:
        try:
            from llmbothub.memory.postgresql import PostgreSQLMemoryBackend
            backend = PostgreSQLMemoryBackend(config, bot_id=default_bot.slug)
            db_stats = backend.stats()
            long_term_count = db_stats.get('memories', {}).get('total_count', 0)
            messages_count = db_stats.get('messages', {}).get('total_count', 0)
            db_status = f"[green]Connected[/green] ({config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DATABASE})"
        except Exception as e:
            db_status = f"[red]Error: {e}[/red]"

    memory_table = make_table()
    memory_table.add_row("PostgreSQL", db_status)
    memory_table.add_row(
        "Postgres Config",
        ", ".join(
            [
                format_compact_setting(
                    "host",
                    "LLMBOTHUB_POSTGRES_HOST",
                    os.getenv("LLMBOTHUB_POSTGRES_HOST"),
                    config.POSTGRES_HOST,
                ),
                format_compact_setting(
                    "port",
                    "LLMBOTHUB_POSTGRES_PORT",
                    os.getenv("LLMBOTHUB_POSTGRES_PORT"),
                    config.POSTGRES_PORT,
                ),
                format_compact_setting(
                    "db",
                    "LLMBOTHUB_POSTGRES_DATABASE",
                    os.getenv("LLMBOTHUB_POSTGRES_DATABASE"),
                    config.POSTGRES_DATABASE,
                ),
                format_compact_setting(
                    "user",
                    "LLMBOTHUB_POSTGRES_USER",
                    os.getenv("LLMBOTHUB_POSTGRES_USER"),
                    config.POSTGRES_USER,
                ),
                format_compact_setting(
                    "pass",
                    "LLMBOTHUB_POSTGRES_PASSWORD",
                    os.getenv("LLMBOTHUB_POSTGRES_PASSWORD"),
                    None,
                    redact=True,
                ),
            ]
        ),
    )
    messages_display = f"[green]{messages_count}[/green]" if messages_count else "[dim]0[/dim]"
    memories_display = f"[green]{long_term_count}[/green]" if long_term_count else "[dim]0[/dim]"
    memory_table.add_row("Counts", f"messages={messages_display}, memories={memories_display}")

    if db_status.startswith("[green]"):
        try:
            from pgvector.psycopg2 import register_vector
            memory_table.add_row("Backend", "[green]✓ PostgreSQL + pgvector[/green]")
        except ImportError:
            memory_table.add_row("Backend", "[yellow]⚠ pgvector Python package not installed[/yellow]")
    elif not has_database_credentials(config):
        memory_table.add_row("Backend", "[dim]○ Not configured (set LLMBOTHUB_POSTGRES_PASSWORD)[/dim]")
    elif db_status.startswith("[red]"):
        memory_table.add_row("Backend", "[red]✗ PostgreSQL connection failed[/red]")
    else:
        memory_table.add_row("Backend", "[dim]○ Not configured (--local mode only)[/dim]")

    # History file location is writable
    history_dir = os.path.dirname(config.HISTORY_FILE)
    if os.path.exists(history_dir):
        if os.access(history_dir, os.W_OK):
            memory_table.add_row("History Storage", "[green]✓ Writable[/green]")
        else:
            memory_table.add_row("History Storage", f"[red]✗ Not writable: {history_dir}[/red]")
    else:
        # Check if we can create it
        try:
            os.makedirs(history_dir, exist_ok=True)
            memory_table.add_row("History Storage", "[green]✓ Created[/green]")
        except Exception as e:
            memory_table.add_row("History Storage", f"[red]✗ Cannot create: {e}[/red]")

    sections.append(("Memory", memory_table))

    idx = 0
    while idx < len(sections):
        title, table = sections[idx]
        if title == "Current Session" and idx + 1 < len(sections) and sections[idx + 1][0] == "Service":
            session_panel = Panel.fit(table, title=f"[bold]{title}[/bold]", border_style="grey39")
            service_title, service_table = sections[idx + 1]
            service_panel = Panel.fit(service_table, title=f"[bold]{service_title}[/bold]", border_style="grey39")
            console.print(Columns([session_panel, service_panel], equal=True, expand=True))
            console.print()
            idx += 2
            continue
        console.print(Panel.fit(table, title=f"[bold]{title}[/bold]", border_style="grey39"))
        console.print()
        idx += 1


def show_bots(config: Config):
    """Display available bots."""
    bot_manager = BotManager(config)
    bots = bot_manager.list_bots()
    default_bot = bot_manager.get_default_bot()
    
    console.print(Panel.fit("[bold cyan]Available Bots[/bold cyan]", border_style="cyan"))
    console.print()
    
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Slug", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Default Model")
    table.add_column("Memory", justify="center")
    
    for bot in bots:
        is_default = " ⭐" if bot.slug == default_bot.slug else ""
        memory_icon = "[green]✓[/green]" if bot.requires_memory else "[dim]✗[/dim]"
        selection = bot_manager.select_model(None, bot_slug=bot.slug)
        default_model = selection.alias or f"[dim]{config.DEFAULT_MODEL_ALIAS or 'global'}[/dim]"
        table.add_row(
            f"{bot.slug}{is_default}",
            bot.name,
            bot.description,
            default_model,
            memory_icon
        )
    
    console.print(table)
    console.print()
    console.print(f"[dim]⭐ = default bot | ✓ = requires database | Use -b/--bot <slug> to select[/dim]")
    console.print()


def show_job_status(config: Config):
    """Display scheduled background jobs and recent run history."""
    if not has_database_credentials(config):
        console.print("[yellow]Job scheduler requires database connection.[/yellow]")
        console.print("[dim]Set LLMBOTHUB_POSTGRES_PASSWORD in ~/.config/llmbothub/.env[/dim]")
        return

    try:
        from sqlmodel import Session, create_engine, select
        from urllib.parse import quote_plus
        from llmbothub.service.scheduler import ScheduledJob, JobRun, JobStatus, create_scheduler_tables
        from datetime import datetime

        encoded_password = quote_plus(config.POSTGRES_PASSWORD)
        postgres_url = (
            f"postgresql+psycopg2://{config.POSTGRES_USER}:{encoded_password}"
            f"@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DATABASE}"
        )
        engine = create_engine(postgres_url)
        
        # Ensure tables exist
        create_scheduler_tables(engine)

        with Session(engine) as session:
            # Get scheduled jobs
            jobs = session.exec(select(ScheduledJob).order_by(ScheduledJob.job_type)).all()
            
            # Get recent runs (last 10)
            runs = session.exec(
                select(JobRun)
                .order_by(JobRun.started_at.desc())
                .limit(10)
            ).all()

        # Display scheduled jobs
        console.print(Panel.fit("[bold cyan]Scheduled Jobs[/bold cyan]", border_style="cyan"))
        console.print()

        if not jobs:
            console.print("[dim]No scheduled jobs found. Start the service to initialize default jobs.[/dim]")
        else:
            job_table = Table(show_header=True, box=None, padding=(0, 2))
            job_table.add_column("Job Type", style="cyan")
            job_table.add_column("Bot", style="bold")
            job_table.add_column("Enabled", justify="center")
            job_table.add_column("Interval")
            job_table.add_column("Last Run")
            job_table.add_column("Next Run")

            now = datetime.utcnow()  # Use UTC to match database timestamps
            for job in jobs:
                enabled_icon = "[green]✓[/green]" if job.enabled else "[red]✗[/red]"
                interval_str = f"{job.interval_minutes}m"
                
                if job.last_run_at:
                    # Handle both naive and aware datetimes - convert to naive UTC
                    last_run = job.last_run_at.replace(tzinfo=None) if job.last_run_at.tzinfo else job.last_run_at
                    ago = now - last_run
                    if ago.total_seconds() < 3600:
                        last_run_str = f"{int(ago.total_seconds() // 60)}m ago"
                    elif ago.total_seconds() < 86400:
                        last_run_str = f"{int(ago.total_seconds() // 3600)}h ago"
                    else:
                        last_run_str = f"{int(ago.total_seconds() // 86400)}d ago"
                else:
                    last_run_str = "[dim]never[/dim]"

                if job.next_run_at:
                    # Handle both naive and aware datetimes
                    next_run = job.next_run_at.replace(tzinfo=None) if job.next_run_at.tzinfo else job.next_run_at
                    until = next_run - now
                    if until.total_seconds() < 0:
                        next_run_str = "[yellow]overdue[/yellow]"
                    elif until.total_seconds() < 3600:
                        next_run_str = f"in {int(until.total_seconds() // 60)}m"
                    else:
                        next_run_str = f"in {int(until.total_seconds() // 3600)}h"
                else:
                    next_run_str = "[dim]pending[/dim]"

                job_table.add_row(
                    job.job_type.value,
                    job.bot_id,
                    enabled_icon,
                    interval_str,
                    last_run_str,
                    next_run_str,
                )

            console.print(job_table)
        
        console.print()

        # Display recent runs
        console.print(Panel.fit("[bold cyan]Recent Job Runs[/bold cyan]", border_style="cyan"))
        console.print()

        if not runs:
            console.print("[dim]No job runs recorded yet.[/dim]")
        else:
            run_table = Table(show_header=True, box=None, padding=(0, 2))
            run_table.add_column("Started", style="dim")
            run_table.add_column("Bot", style="bold")
            run_table.add_column("Status", justify="center")
            run_table.add_column("Duration")
            run_table.add_column("Error")

            for run in runs:
                if run.status == JobStatus.SUCCESS:
                    status_str = "[green]✓ success[/green]"
                elif run.status == JobStatus.FAILED:
                    status_str = "[red]✗ failed[/red]"
                elif run.status == JobStatus.RUNNING:
                    status_str = "[yellow]⟳ running[/yellow]"
                elif run.status == JobStatus.SKIPPED:
                    status_str = "[dim]○ skipped[/dim]"
                else:
                    status_str = "[dim]○ pending[/dim]"

                duration_str = f"{run.duration_ms}ms" if run.duration_ms else "[dim]-[/dim]"
                error_str = run.error_message[:40] + "..." if run.error_message and len(run.error_message) > 40 else (run.error_message or "[dim]-[/dim]")
                started_str = run.started_at.strftime("%Y-%m-%d %H:%M") if run.started_at else "[dim]-[/dim]"

                run_table.add_row(
                    started_str,
                    run.bot_id,
                    status_str,
                    duration_str,
                    error_str,
                )

            console.print(run_table)

        console.print()
        console.print(f"[dim]Scheduler enabled: {config.SCHEDULER_ENABLED} | Check interval: {config.SCHEDULER_CHECK_INTERVAL_SECONDS}s[/dim]")
        console.print()

    except Exception as e:
        console.print(f"[red]Could not load job status: {e}[/red]")
        import traceback
        traceback.print_exc()


def trigger_job(config: Config, job_type: str):
    """Trigger a scheduled job to run immediately."""
    if not has_database_credentials(config):
        console.print("[yellow]Job scheduler requires database connection.[/yellow]")
        console.print("[dim]Set LLMBOTHUB_POSTGRES_PASSWORD in ~/.config/llmbothub/.env[/dim]")
        return

    valid_types = ["profile_maintenance", "memory_consolidation", "memory_decay"]
    if job_type not in valid_types:
        console.print(f"[red]Unknown job type: {job_type}[/red]")
        console.print(f"[dim]Valid types: {', '.join(valid_types)}[/dim]")
        return

    try:
        from sqlmodel import Session, create_engine, select
        from urllib.parse import quote_plus
        from llmbothub.service.scheduler import ScheduledJob, JobType, create_scheduler_tables
        from datetime import datetime, timedelta

        encoded_password = quote_plus(config.POSTGRES_PASSWORD)
        postgres_url = (
            f"postgresql+psycopg2://{config.POSTGRES_USER}:{encoded_password}"
            f"@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DATABASE}"
        )
        engine = create_engine(postgres_url)
        create_scheduler_tables(engine)

        with Session(engine) as session:
            # Find the job
            job_type_enum = JobType(job_type)
            job = session.exec(
                select(ScheduledJob).where(ScheduledJob.job_type == job_type_enum)
            ).first()

            if not job:
                console.print(f"[yellow]No scheduled job found for type: {job_type}[/yellow]")
                console.print("[dim]Start the service first to initialize default jobs.[/dim]")
                return

            # Set next_run_at to past to trigger immediate run
            job.next_run_at = datetime.utcnow() - timedelta(minutes=1)
            session.add(job)
            session.commit()

            console.print(f"[green]✓ Triggered job: {job_type}[/green]")
            console.print(f"[dim]The scheduler will pick it up within {config.SCHEDULER_CHECK_INTERVAL_SECONDS} seconds.[/dim]")
            console.print(f"[dim]Run 'llm --job-status' to check the result.[/dim]")

    except Exception as e:
        console.print(f"[red]Could not trigger job: {e}[/red]")
        import traceback
        traceback.print_exc()


def show_user_profile(config: Config, user_id: str):
    """Display user profile."""
    if not has_database_credentials(config):
        console.print("[yellow]User profiles require database connection.[/yellow]")
        console.print("[dim]Set LLMBOTHUB_POSTGRES_PASSWORD in ~/.config/llmbothub/.env[/dim]")
        return
    
    try:
        from llmbothub.profiles import ProfileManager, EntityType
        
        manager = ProfileManager(config)
        profile, _ = manager.get_or_create_profile(EntityType.USER, user_id)
        attributes = manager.get_all_attributes(EntityType.USER, user_id)
    except Exception as e:
        console.print(f"[red]Could not load user profile: {e}[/red]")
        return
    
    console.print(Panel.fit(f"[bold cyan]User Profile: {user_id}[/bold cyan]", border_style="cyan"))
    console.print()
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    
    table.add_row("[bold]Identity[/bold]", "")
    table.add_row("  Display Name", profile.display_name or "[dim]not set[/dim]")
    table.add_row("  Description", profile.description or "[dim]not set[/dim]")
    
    # Group attributes by category
    by_category = {}
    for attr in attributes:
        if attr.category not in by_category:
            by_category[attr.category] = []
        by_category[attr.category].append(attr)
    
    # Display attributes by category
    category_names = {"preference": "Preferences", "fact": "Facts", "interest": "Interests", "communication": "Communication", "context": "Context"}
    for category, attrs in sorted(by_category.items()):
        table.add_row("", "")
        table.add_row(f"[bold]{category_names.get(category, category.title())}[/bold]", "")
        for attr in sorted(attrs, key=lambda a: a.key):
            value_str = str(attr.value) if not isinstance(attr.value, str) or len(str(attr.value)) < 60 else str(attr.value)[:57] + "..."
            conf_str = f" [dim]({attr.confidence:.0%})[/dim]" if attr.confidence < 1.0 else ""
            table.add_row(f"  {attr.key}", f"{value_str}{conf_str}")
    
    console.print(table)
    console.print()
    console.print("[dim]Add attributes: llm --user-profile-set category.key=value[/dim]")
    console.print()


def show_users(config: Config):
    """Display all user profiles."""
    if not has_database_credentials(config):
        console.print("[yellow]User profiles require database connection.[/yellow]")
        console.print("[dim]Set LLMBOTHUB_POSTGRES_PASSWORD in ~/.config/llmbothub/.env[/dim]")
        return
    
    try:
        from llmbothub.profiles import ProfileManager, EntityType
        
        manager = ProfileManager(config)
        profiles = manager.list_all_profiles(EntityType.USER)
    except Exception as e:
        console.print(f"[red]Could not load users: {e}[/red]")
        return
    
    if not profiles:
        console.print("[yellow]No user profiles found.[/yellow]")
        console.print("[dim]Create one with: llm --user-profile-setup[/dim]")
        return
    
    console.print(Panel.fit("[bold cyan]User Profiles[/bold cyan]", border_style="cyan"))
    console.print()
    
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("User ID", style="cyan")
    table.add_column("Display Name")
    table.add_column("Description")
    
    for profile in profiles:
        table.add_row(
            profile.entity_id,
            profile.display_name or "[dim]-[/dim]",
            (profile.description[:50] + "..." if profile.description and len(profile.description) > 50 else profile.description) or "[dim]-[/dim]"
        )
    
    console.print(table)
    console.print()
    console.print(f"[dim]Use --user <id> to select a user profile[/dim]")
    console.print()


def run_user_profile_setup(config: Config, user_id: str) -> bool:
    """Run interactive user profile setup wizard."""
    if not has_database_credentials(config):
        console.print("[yellow]User profiles require database connection.[/yellow]")
        console.print("[dim]Set LLMBOTHUB_POSTGRES_PASSWORD in ~/.config/llmbothub/.env[/dim]")
        return False
    
    console.print(Panel.fit("[bold cyan]User Profile Setup[/bold cyan]", border_style="cyan"))
    console.print()
    console.print(f"[dim]Setting up profile for user: {user_id}[/dim]")
    console.print(f"[dim]Press Enter to skip any field.[/dim]")
    console.print()
    
    try:
        from llmbothub.profiles import ProfileManager, EntityType, AttributeCategory
        
        manager = ProfileManager(config)
        profile, _ = manager.get_or_create_profile(EntityType.USER, user_id)
        
        # Get existing attributes
        existing_attrs = {attr.key: attr.value for attr in manager.get_all_attributes(EntityType.USER, user_id)}
        
        # Prompt for basic info
        name = Prompt.ask(
            "What's your name?",
            default=profile.display_name or existing_attrs.get("name", "")
        )
        
        occupation = Prompt.ask(
            "What do you do? (occupation)",
            default=existing_attrs.get("occupation", "")
        )
        
        console.print()
        
        # Save to new profile system
        if name:
            manager.update_profile(EntityType.USER, user_id, display_name=name)
            manager.set_attribute(EntityType.USER, user_id, AttributeCategory.FACT, "name", name, source="explicit")
        
        if occupation:
            manager.set_attribute(EntityType.USER, user_id, AttributeCategory.FACT, "occupation", occupation, source="explicit")
        
        console.print(f"[green]✓ User profile saved for '{user_id}'[/green]")
        console.print()
        return True
            
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Setup cancelled.[/yellow]")
        return False
    except Exception as e:
        console.print(f"[red]Error during setup: {e}[/red]")
        return False


def ensure_user_profile(config: Config, user_id: str) -> bool:
    """Ensure user profile exists, prompting for setup if needed.
    
    Returns True if profile exists or setup succeeded, False if setup was cancelled.
    """
    if not has_database_credentials(config):
        logging.getLogger(__name__).debug("Database credentials not configured - skipping user profile")
        return True  # Not an error, just no profile
    
    try:
        from llmbothub.profiles import ProfileManager, EntityType
        
        manager = ProfileManager(config)
        profile, is_new = manager.get_or_create_profile(EntityType.USER, user_id)
        
        # If profile has no name, run setup wizard
        if not profile.display_name:
            console.print()
            console.print(f"[yellow]Welcome! Let's set up your user profile.[/yellow]")
            console.print()
            return run_user_profile_setup(config, user_id)
        
        return True
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not ensure user profile: {e}")
        return True  # Don't block on profile errors


def ensure_default_user(config: Config, args: argparse.Namespace) -> None:
    """Prompt for and persist DEFAULT_USER if missing and no --user is provided."""
    if getattr(args, "user", None):
        return
    default_user = getattr(config, "DEFAULT_USER", None)
    if default_user and str(default_user).strip():
        return
    if not sys.stdin.isatty():
        console.print("[bold red]DEFAULT_USER is required to continue.[/bold red]")
        console.print("[dim]Set LLMBOTHUB_DEFAULT_USER or pass --user.[/dim]")
        sys.exit(1)
    console.print("[yellow]No default user configured.[/yellow]")
    user_id = Prompt.ask("Enter default user id", default="default").strip()
    if not user_id:
        console.print("[red]No user id provided. Aborting.[/red]")
        sys.exit(1)
    if not set_config_value("DEFAULT_USER", user_id, config):
        console.print("[red]Failed to save DEFAULT_USER to config.[/red]")
        sys.exit(1)
    config.DEFAULT_USER = user_id
    console.print(f"[green]Default user set to '{user_id}'.[/green]")


def parse_arguments(config_obj: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query LLM models from the command line using model aliases defined in models.yaml")
    parser.add_argument("-m","--model",type=str,default=None,help=f"Model alias defined in {config_obj.MODELS_CONFIG_PATH}. Supports partial matching. (Default: bot's default or {config_obj.DEFAULT_MODEL_ALIAS or 'None'})")
    parser.add_argument("--list-models",action="store_true",help="List available model aliases defined in the configuration file and exit.")
    parser.add_argument("--add-gguf",type=str,metavar="REPO_ID",help="(Deprecated: use --add-model gguf) Add a GGUF model from a Hugging Face repo ID.")
    parser.add_argument("--add-model",type=str,choices=['ollama', 'openai', 'gguf'],metavar="TYPE",help="Add models: 'ollama' (refresh from server), 'openai' (query API), 'gguf' (add from HuggingFace repo)")
    parser.add_argument("--delete-model",type=str,metavar="ALIAS",help="Delete the specified model alias from the configuration file after confirmation.")
    parser.add_argument("--config-set", nargs=2, metavar=("KEY", "VALUE"), help="Set a configuration value (e.g., DEFAULT_MODEL_ALIAS) in the .env file.")
    parser.add_argument("--config-list", action="store_true", help="List the current effective configuration settings.")
    parser.add_argument("question", nargs="*", help="Your question for the LLM model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-dh", "--delete-history", action="store_true", help="Clear chat history")
    parser.add_argument("-ph", "--print-history", nargs="?", const=-1, type=int, default=None,help="Print chat history (optional: number of recent pairs)")
    parser.add_argument("-c", "--command", help="Execute command and add output to question")
    parser.add_argument("--plain", action="store_true", help="Use plain text output")
    parser.add_argument("--no-stream", action="store_true", default=False, help="Disable streaming output")
    parser.add_argument("--local", action="store_true", help="Use local filesystem for history instead of database")
    parser.add_argument("--service", action="store_true", help="Route queries through the background service (if running)")
    parser.add_argument("-b", "--bot", type=str, default=None, help="Bot to use (nova, spark, mira). Use --list-bots to see all.")
    parser.add_argument("--list-bots", action="store_true", help="List available bots and exit")
    parser.add_argument("--status", action="store_true", help="Show memory system status and configuration")
    parser.add_argument("--job-status", action="store_true", help="Show scheduled background jobs and recent run history")
    parser.add_argument("--run-job", type=str, metavar="TYPE", help="Trigger a scheduled job immediately (profile_maintenance, memory_consolidation, memory_decay)")
    parser.add_argument("--user", type=str, default=None, help="User profile to use (creates if not exists). Defaults to config.DEFAULT_USER")
    parser.add_argument("--list-users", action="store_true", help="List all user profiles")
    parser.add_argument("--user-profile", action="store_true", help="Show current user profile")
    parser.add_argument("--user-profile-set", metavar="FIELD=VALUE", help="Set a user profile field (e.g., name=\"Nick\")")
    parser.add_argument("--user-profile-setup", action="store_true", help="Run user profile setup wizard")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging output.")
    return parser.parse_args()

def main():
    try:
        # --- Minimal Pre-parsing for flags affecting config loading/display --- 
        # We need to parse --verbose, --plain early, but also handle config paths if specified.
        # Using a separate parser for this minimal set.
        prelim_parser = argparse.ArgumentParser(add_help=False)
        prelim_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
        prelim_parser.add_argument("--plain", action="store_true", help="Use plain text output")
        prelim_args, remaining_argv = prelim_parser.parse_known_args()

        # --- Instantiate Config --- 
        config_obj = Config()

    except Exception as e:
        # Catch potential Pydantic validation errors or file issues during Config init
        console.print(f"[bold red]Error initializing configuration:[/bold red] {e}")
        sys.exit(1)
    
    # Parse full arguments now that config is loaded
    args = parse_arguments(config_obj)
    config_obj.VERBOSE = args.verbose
    config_obj.DEBUG = args.debug
    config_obj.PLAIN_OUTPUT = args.plain
    config_obj.NO_STREAM = args.no_stream
    config_obj.INTERACTIVE_MODE = not args.question and not args.command  # Set interactive mode flag

    # Configure logging via centralized LogConfig
    LogConfig.configure(verbose=args.verbose, debug=args.debug)

    # Handle setting config values, guard missing attribute in stubbed args
    if getattr(args, 'config_set', None):
        key, value = args.config_set
        success = set_config_value(key, value, config_obj)
        if success:
            console.print(f"[green]Configuration '{key}' set to '{value}' in {config_obj.model_config.get('env_file', 'unknown')}.[/green]")
            sys.exit(0)
        else:
            console.print(f"[bold red]Failed to set configuration '{key}'.[/bold red]")
            sys.exit(1)
    # Handle listing config values
    elif getattr(args, 'config_list', False):
        env_file_path = config_obj.model_config.get('env_file', 'unknown')
        console.print(f"[bold magenta]Current Configuration Settings[/bold magenta]")
        console.print(f"[dim]Config file: {env_file_path}[/dim]\n")
        
        # Define which settings are important and should be set by the user
        # Format: field_name -> (is_secret, is_required, description)
        important_settings = {
            'OPENAI_API_KEY': (True, False, "Required for OpenAI/compatible APIs"),
            'DEFAULT_MODEL_ALIAS': (False, False, "Default model to use"),
            'DEFAULT_BOT': (False, False, "Default bot personality"),
            'DEFAULT_USER': (False, False, "Default user profile"),
            'POSTGRES_PASSWORD': (True, True, "Required for memory features"),
            'POSTGRES_HOST': (False, False, "PostgreSQL server hostname"),
            'OLLAMA_URL': (False, False, "Ollama server URL"),
        }
        
        # Check for missing important settings
        missing_settings = []
        for field_name, (is_secret, is_required, desc) in important_settings.items():
            value = getattr(config_obj, field_name, None)
            if not value or (isinstance(value, str) and not value.strip()):
                if is_required:
                    missing_settings.append((field_name, desc))
        
        if missing_settings:
            console.print("[yellow]⚠ Missing recommended settings:[/yellow]")
            for field_name, desc in missing_settings:
                env_var = f"LLMBOTHUB_{field_name}"
                console.print(f"  [red]✗[/red] {env_var} - {desc}")
            console.print()
        
        # Show all settings grouped by category
        exclude_fields = {'defined_models', 'available_ollama_models', 'ollama_checked', 'model_config', 'SYSTEM_MESSAGE'}
        secret_fields = {'OPENAI_API_KEY', 'POSTGRES_PASSWORD', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY'}
        
        console.print("[bold]All Settings:[/bold]")
        for field_name, field_info in sorted(config_obj.model_fields.items()):
            if field_name not in exclude_fields:
                current_value = getattr(config_obj, field_name)
                if isinstance(current_value, Path):
                    current_value_str = str(current_value)
                elif field_name in secret_fields and current_value:
                    # Mask secrets
                    current_value_str = "[green]****[/green] (set)"
                elif field_name in secret_fields:
                    current_value_str = "[dim]not set[/dim]"
                elif current_value is None or (isinstance(current_value, str) and not current_value.strip()):
                    current_value_str = "[dim]not set[/dim]"
                else:
                    current_value_str = repr(current_value)

                console.print(f"  [cyan]{field_name}[/cyan]: {current_value_str}")
        
        console.print(f"\n  [dim]SYSTEM_MESSAGE: (set by bot config in bots.yaml)[/dim]")
        console.print(f"\n[dim]To set a value: llm --config-set KEY value[/dim]")
        console.print(f"[dim]Or edit: {env_file_path}[/dim]")
        sys.exit(0)

    elif getattr(args, 'status', False):
        show_status(config_obj, args)
        sys.exit(0)

    elif getattr(args, 'job_status', False):
        show_job_status(config_obj)
        sys.exit(0)

    elif getattr(args, 'run_job', None):
        trigger_job(config_obj, args.run_job)
        sys.exit(0)

    elif getattr(args, 'list_bots', False):
        show_bots(config_obj)
        sys.exit(0)

    elif getattr(args, 'list_users', False):
        show_users(config_obj)
        sys.exit(0)

    elif getattr(args, 'user_profile', False):
        ensure_default_user(config_obj, args)
        user_id = args.user if args.user else config_obj.DEFAULT_USER
        show_user_profile(config_obj, user_id)
        sys.exit(0)
    
    elif getattr(args, 'user_profile_setup', False):
        ensure_default_user(config_obj, args)
        user_id = args.user if args.user else config_obj.DEFAULT_USER
        success = run_user_profile_setup(config_obj, user_id)
        sys.exit(0 if success else 1)
    
    elif getattr(args, 'user_profile_set', None):
        ensure_default_user(config_obj, args)
        if not has_database_credentials(config_obj):
            console.print("[yellow]User profiles require database connection.[/yellow]")
            console.print("[dim]Set LLMBOTHUB_POSTGRES_PASSWORD in ~/.config/llmbothub/.env[/dim]")
            sys.exit(1)
        try:
            from llmbothub.profiles import ProfileManager, EntityType, AttributeCategory
            
            field, value = args.user_profile_set.split("=", 1)
            field = field.strip()
            value = value.strip().strip('"').strip("'")
            user_id = args.user if args.user else config_obj.DEFAULT_USER
            
            manager = ProfileManager(config_obj)
            
            # Parse field: category.key or just key
            if "." in field:
                category_str, key = field.split(".", 1)
                category_map = {"preference": AttributeCategory.PREFERENCE, "fact": AttributeCategory.FACT, "interest": AttributeCategory.INTEREST, "communication": AttributeCategory.COMMUNICATION, "context": AttributeCategory.CONTEXT}
                category = category_map.get(category_str.lower())
                if not category:
                    console.print(f"[red]Invalid category: {category_str}[/red]")
                    console.print("[dim]Valid categories: preference, fact, interest, communication, context[/dim]")
                    sys.exit(1)
            else:
                # Default to fact category
                key = field
                category = AttributeCategory.FACT
            
            manager.set_attribute(EntityType.USER, user_id, category, key, value, source="explicit")
            console.print(f"[green]Set {category}.{key} = {value}[/green]")
        except ValueError:
            console.print("[red]Use format: --user-profile-set category.key=value[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        sys.exit(0)

    elif args.list_models:
        list_models(config_obj)
        sys.exit(0)
        return
    if args.delete_model:
        success = False
        try:
            success = delete_model(args.delete_model, config_obj)
            if success:
                console.print(f"[green]Model alias '{args.delete_model}' deleted successfully.[/green]")
        except Exception as e:
            console.print(f"[bold red]Error during delete model operation:[/bold red] {e}")
            if config_obj.VERBOSE:
                traceback.print_exc()
            success = False
        sys.exit(0 if success else 1)
        return
    if args.add_model:
        success = False
        try:
            if args.add_model == 'openai':
                success = update_models_interactive(config_obj, provider='openai')
            elif args.add_model == 'ollama':
                success = update_models_interactive(config_obj, provider='ollama')
            elif args.add_model == 'gguf':
                # Prompt for HuggingFace repo ID
                repo_id = Prompt.ask("Enter HuggingFace repo ID (e.g., TheBloke/Llama-2-7B-GGUF)")
                if repo_id and repo_id.strip():
                    success = handle_add_gguf(repo_id.strip(), config_obj)
                else:
                    console.print("[red]No repo ID provided. Cancelled.[/red]")
                    success = False
            if success:
                console.print(f"[green]Model add for '{args.add_model}' completed.[/green]")
            else:
                console.print(f"[red]Model add for '{args.add_model}' failed or was cancelled.[/red]")
        except KeyboardInterrupt:
            console.print("[bold red]Model add cancelled.[/bold red]")
            success = False # Ensure failure on interrupt
        except Exception as e:
            console.print(f"[bold red]Error during model add:[/bold red] {e}")
            if config_obj.VERBOSE:
                traceback.print_exc()
            success = False
        sys.exit(0 if success else 1)
        return

    ensure_default_user(config_obj, args)
    
    # Check if this is a history-only operation (doesn't need model)
    history_only = (args.delete_history or args.print_history is not None) and not args.question and not args.command
    
    # Check if using service mode (don't validate model locally - let service handle it)
    use_service = False
    if args.local:
        use_service = False
    elif getattr(args, 'service', False):
        use_service = True
    elif config_obj.USE_SERVICE:
        use_service = True
    
    # Determine which bot will be used (needed to get bot's default model)
    bot_manager = BotManager(config_obj)
    if args.bot:
        target_bot = bot_manager.get_bot(args.bot)
        if not target_bot:
            console.print(f"[bold red]Unknown bot: {args.bot}[/bold red]. Use --list-bots to see available bots.")
            sys.exit(1)
    elif args.local:
        target_bot = bot_manager.get_default_bot(local_mode=True)
    else:
        target_bot = bot_manager.get_bot(config_obj.DEFAULT_BOT) or bot_manager.get_default_bot()
    
    # Determine effective model alias (without validation for service mode)
    selection = bot_manager.select_model(args.model, bot_slug=target_bot.slug, local_mode=args.local)
    effective_model = selection.alias

    # Auto-switch to service when using non-OpenAI models and service is available
    defined_models = config_obj.defined_models.get("models", {})
    model_def = defined_models.get(effective_model, {}) if effective_model else {}
    effective_model_type = model_def.get("type")
    if not use_service and not args.local and effective_model_type and effective_model_type != "openai":
        service_client = get_service_client(config_obj)
        if service_client and service_client.is_available(force_check=True):
            use_service = True
            if config_obj.VERBOSE:
                console.print(f"[dim]Detected {effective_model_type} model; using service mode[/dim]")
        else:
            console.print(
                "[bold red]Service not available for local model. Start the service or pass --service when it is running.[/bold red]"
            )
            sys.exit(1)
    
    # For history-only or service mode, skip local model validation
    if history_only:
        resolved_alias = None
    elif use_service:
        # In service mode, pass the model alias directly - let the service validate it
        resolved_alias = effective_model
    else:
        # Local mode: validate model is available locally
        model_manager = ModelManager(config_obj)
        resolved_alias = model_manager.resolve_model_alias(effective_model)
        if not resolved_alias:
            sys.exit(1)
            return
    
    try:
        run_app(args, config_obj, resolved_alias)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during application execution:[/bold red] {e}")
        if config_obj.VERBOSE:
             traceback.print_exc()
        sys.exit(1)
    else:
        sys.exit(0)

def run_app(args: argparse.Namespace, config_obj: Config, resolved_alias: str):
    # Determine which bot to use
    bot_manager = BotManager(config_obj)
    
    # Handle --local with explicit --bot (conflicting intent warning)
    if args.local and args.bot:
        bot = bot_manager.get_bot(args.bot)
        if not bot:
            console.print(f"[bold red]Unknown bot: {args.bot}[/bold red]. Use --list-bots to see available bots.")
            sys.exit(1)
        # Only warn if the bot requires memory (e.g., nova)
        if bot.requires_memory and config_obj.VERBOSE:
            console.print(f"[dim]Note: Using --local with --bot {args.bot}. The bot will run without database memory.[/dim]")
        bot_id = bot.slug
    elif args.bot:
        # Explicit bot selection
        bot = bot_manager.get_bot(args.bot)
        if not bot:
            console.print(f"[bold red]Unknown bot: {args.bot}[/bold red]. Use --list-bots to see available bots.")
            sys.exit(1)
        bot_id = bot.slug
    elif args.local:
        # Local mode defaults to Spark
        bot = bot_manager.get_default_bot(local_mode=True)
        bot_id = bot.slug
    else:
        # Default bot from config
        bot = bot_manager.get_bot(config_obj.DEFAULT_BOT)
        if not bot:
            bot = bot_manager.get_default_bot()
        bot_id = bot.slug
    
    # Use config default if --user not specified
    user_id = args.user if args.user else config_obj.DEFAULT_USER
    
    # Check if we're using service mode - explicit flag or config default
    # --local takes precedence and disables service mode
    use_service = False
    if args.local:
        use_service = False
    elif getattr(args, 'service', False):
        use_service = True
    elif config_obj.USE_SERVICE:
        use_service = True

    # Get model type for fallback logic (used when service unavailable)
    defined_models = config_obj.defined_models.get("models", {})
    model_def = defined_models.get(resolved_alias, {}) if resolved_alias else {}
    effective_model_type = model_def.get("type")

    # Auto-enable service mode when bot uses tools and not in local mode
    use_tools = getattr(bot, 'uses_tools', False)
    if use_tools and not use_service and not args.local:
        use_service = True
        if config_obj.VERBOSE:
            console.print(f"[dim]Bot '{bot.name}' uses tools; enabling service mode[/dim]")

    # Validate service availability upfront when service mode is enabled
    if use_service:
        service_client = get_service_client(config_obj)
        if not service_client or not service_client.is_available(force_check=True):
            if effective_model_type and effective_model_type != "openai":
                console.print(
                    "[bold red]Service not available.[/bold red] Start the service with: [yellow]llm-service[/yellow]"
                )
                sys.exit(1)
            else:
                console.print(
                    "[yellow]Warning: Service not available. Queries will use direct API calls.[/yellow]"
                )
                use_service = False

    llmbothub = None
    
    # For history operations, we can use a lightweight path that doesn't require model init
    history_only = (args.delete_history or args.print_history is not None) and not args.question and not args.command
    
    if history_only:
        # Use lightweight history manager without full model initialization
        from llmbothub.clients.base import StubClient
        from llmbothub.utils.history import HistoryManager
        from llmbothub.memory_server.client import get_memory_client
        
        stub_client = StubClient(config=config_obj, bot_name=bot.name)
        
        # Initialize memory client if database available
        memory = None
        if not args.local and has_database_credentials(config_obj):
            try:
                memory = get_memory_client(
                    config=config_obj,
                    bot_id=bot_id,
                    user_id=user_id,
                    server_url=getattr(config_obj, "MEMORY_SERVER_URL", None),
                )
            except Exception as e:
                logger.debug(f"Memory client init failed: {e}")
        
        history_manager = HistoryManager(
            client=stub_client,
            config=config_obj,
            db_backend=memory.get_short_term_manager() if memory else None,
            bot_id=bot_id,
        )
        history_manager.load_history()
        
        if args.delete_history:
            history_manager.clear_history()
            console.print("[green]Chat history cleared.[/green]")
        
        if args.print_history is not None:
            history_manager.print_history(args.print_history)
        
        console.print()
        return
    
    # For query operations, we need full LLMBotHub
    need_local_llmbothub = not use_service or args.delete_history or args.print_history is not None
    
    if need_local_llmbothub:
        try:
            llmbothub = LLMBotHub(
                resolved_model_alias=resolved_alias,
                config=config_obj,
                local_mode=args.local,
                bot_id=bot_id,
                user_id=user_id,
                verbose=args.verbose,
                debug=args.debug,
            )
        except (ImportError, FileNotFoundError, ValueError, Exception) as e:
             console.print(f"[bold red]Failed to initialize LLM client for '{resolved_alias}':[/bold red] {e}")
             if config_obj.VERBOSE:
                 traceback.print_exc()
             sys.exit(1)
             
    if args.delete_history:
        if llmbothub:
            llmbothub.history_manager.clear_history()
            console.print("[green]Chat history cleared.[/green]")
        else:
            console.print("[yellow]History operations require a valid model configuration.[/yellow]")
    
    if args.print_history is not None:
        if llmbothub:
            llmbothub.history_manager.print_history(args.print_history)
        else:
            console.print("[yellow]History operations require a valid model configuration.[/yellow]")
        if not args.question and not args.command:
             console.print()
             return
    console.print("")
    command_output_str = ""
    if args.command:
        if config_obj.VERBOSE:
            console.print(f"Executing command: [yellow]{args.command}[/yellow]", highlight=False)
        try:
            result = subprocess.run(args.command,shell=True,capture_output=True,text=True,check=False)
            output = result.stdout.strip()
            error = result.stderr.strip()
            command_prefix = f"Command `[cyan]{args.command}[/cyan]` executed.\n"
            command_output_str += command_prefix
            if output:
                command_output_str += f"\nOutput:\n```\n{output}\n```\n"
            if error:
                command_output_str += f"\nError Output:\n```\n{error}\n```\n"
            if result.returncode != 0:
                status_msg = f"\n(Command exited with status {result.returncode})"
                console.print(f"[yellow]Warning: Command exited with status {result.returncode}[/yellow]", highlight=False)
                command_output_str += status_msg
            command_output_str += "\n---\n"
        except Exception as e:
            error_msg = f"Error executing command '{args.command}': {e}"
            console.print(f"[bold red]{error_msg}[/bold red]", highlight=False)
            command_output_str += f"{error_msg}\n\n---\n"
        console.print()
    stream_flag = not config_obj.NO_STREAM
    plaintext_flag = config_obj.PLAIN_OUTPUT
    # use_tools was already set above and factored into use_service decision
    
    def do_query(prompt: str):
        """Execute query via service or local client."""
        nonlocal llmbothub
        if use_service:
            if query_via_service(
                prompt,
                model=resolved_alias,  # Use resolved model, not raw args
                bot_id=bot_id,  # Use resolved bot_id
                user_id=user_id,  # Use resolved user_id
                plaintext_output=plaintext_flag,
                stream=stream_flag,
                config=config_obj,
            ):
                return
            # Service unavailable or failed
            service_client = get_service_client(config_obj)
            error_detail = ""
            if service_client and hasattr(service_client, 'last_error') and service_client.last_error:
                error_detail = f"\n[dim]Error: {service_client.last_error}[/dim]"
            
            if effective_model_type and effective_model_type != "openai":
                if use_tools:
                    console.print(
                        f"[bold red]Error: Tool usage with local models requires the background service.[/bold red]{error_detail}"
                    )
                    console.print("[yellow]Start the service with:[/yellow] llm-service")
                    console.print("[yellow]Or use an OpenAI-compatible model:[/yellow] llm -m gpt4o -b proto")
                else:
                    console.print(
                        f"[bold red]Service unavailable for local model. Start the service or use an OpenAI-compatible model.[/bold red]{error_detail}"
                    )
                return
            # Fall back to local for OpenAI-compatible models
            console.print("[yellow]Service unavailable, falling back to direct API calls.[/yellow]")
            if llmbothub is None:
                try:
                    llmbothub = LLMBotHub(
                        resolved_model_alias=resolved_alias,
                        config=config_obj,
                        local_mode=args.local,
                        bot_id=bot_id,
                        user_id=user_id,
                        verbose=args.verbose,
                        debug=args.debug,
                    )
                except Exception as e:
                    console.print(f"[bold red]Failed to initialize LLM client:[/bold red] {e}")
                    return
        if llmbothub:
            # Standard mode: hardcoded memory retrieval
            llmbothub.query(prompt, plaintext_output=plaintext_flag, stream=stream_flag)

    if args.question:
        question_text = command_output_str + " ".join(args.question)
        do_query(question_text.strip())
    elif command_output_str:
        if config_obj.VERBOSE:
            console.print("Command output captured, querying LLM with it...", highlight=False)
        do_query(command_output_str.strip())
    elif not sys.stdin.isatty():
        # Handle piped input - read once and exit
        piped_input = sys.stdin.read().strip()
        if piped_input:
            do_query(piped_input)
    else:
        console.print("[bold green]Entering interactive mode. Type 'exit' or 'quit' to leave.[/bold green]", highlight=False)
        console.print("[bold green]Type '>' at the beginning of a line for multiline input mode (end with Ctrl+D or Ctrl+Z).[/bold green]", highlight=False)
        handler_console = console
        if llmbothub and hasattr(llmbothub.client, 'console') and llmbothub.client.console:
            handler_console = llmbothub.client.console
        input_handler = MultilineInputHandler(console=handler_console)
        while True:
            try:
                prompt_text, is_multiline = input_handler.get_input("Enter your question:")
                if prompt_text is None or prompt_text.strip().lower() in ["exit", "quit"]:
                    console.print("[bold red]Exiting interactive mode.[/bold red]", highlight=False)
                    console.print()
                    break
                if not prompt_text.strip():
                    console.print("[dim]Empty input received. Asking again...[/dim]")
                    continue
                console.print()
                if prompt_text.strip():
                    do_query(prompt_text)
                console.print()
            except (KeyboardInterrupt, EOFError):
                console.print("[bold red]Exiting interactive mode.[/bold red]", highlight=False)
                console.print()
                break
    console.print()
