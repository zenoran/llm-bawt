import argparse
import json
import subprocess
import traceback
import sys  # Import sys for exit codes
from rich.console import Console
from rich.prompt import Prompt
from llm_bawt.utils.config import Config, has_database_credentials
from llm_bawt.utils.config import set_config_value
from llm_bawt.utils.input_handler import MultilineInputHandler
from llm_bawt.core import LLMBawt
from llm_bawt.model_manager import (
    ModelManager,
    delete_model,
    list_models,
    set_model_context_window,
    update_models_interactive,
)
from llm_bawt.gguf_handler import handle_add_gguf
from llm_bawt.cli.vllm_handler import handle_add_vllm
from llm_bawt.bots import BotManager
from llm_bawt.profiles import ProfileManager, EntityType, AttributeCategory
from llm_bawt.utils.streaming import render_streaming_response, render_complete_response
from llm_bawt.shared.logging import LogConfig
import logging
from pathlib import Path
from rich.table import Table
from rich.panel import Panel
from typing import Iterator
import yaml

console = Console()


def _use_service(config: Config) -> bool:
    """Return True if the CLI should delegate to the service for DB operations."""
    from llm_bawt.model_manager import is_service_mode_enabled
    return is_service_mode_enabled(config)


# Cache for service client
_service_client = None

def get_service_client(config: Config | None = None):
    """Get or create the service client singleton."""
    global _service_client
    if _service_client is None:
        try:
            from llm_bawt.service import ServiceClient
            # Build service URL from config if not provided
            if config is None:
                config = Config()
            service_url = getattr(config, 'SERVICE_URL', None)
            if not service_url and hasattr(config, 'SERVICE_HOST') and hasattr(config, 'SERVICE_PORT'):
                host = "127.0.0.1" if config.SERVICE_HOST == "0.0.0.0" else config.SERVICE_HOST
                service_url = f"http://{host}:{config.SERVICE_PORT}"
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
    from rich.style import Style

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

    # Determine panel style from bot config color with safe fallback.
    configured_color = (getattr(bot, "color", None) if bot else None) or "green"
    try:
        Style.parse(configured_color)
        panel_color = configured_color
    except Exception:
        panel_color = "green"

    title_style, border_style = panel_color, panel_color

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
                    elif isinstance(item, dict) and "tool_call" in item:
                        tool_name = item.get("tool_call") or "tool"
                        console.print(f"[dim]· {tool_name}[/dim]")
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
    from llm_bawt.core.status import collect_system_status

    def make_table() -> Table:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Val")
        return table

    def _format_uptime(seconds: float) -> str:
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    # ── Collect all data via shared module ────────────────────────
    bot_slug = getattr(args, "bot", None) if args else None
    explicit_model = getattr(args, "model", None) if args else None
    user_id = getattr(args, "user", None) if args else None

    status = collect_system_status(
        config,
        bot_slug=bot_slug,
        model_alias=explicit_model,
        user_id=user_id,
    )

    s_cfg = status.config
    s_svc = status.service
    s_mcp = status.mcp
    s_model = status.model
    s_mem = status.memory
    s_dep = status.dependencies

    # ── Header panel ─────────────────────────────────────────────
    if s_cfg.mode == "service":
        if s_svc.available and s_svc.healthy:
            mode_text = "[green]✓ Service Mode[/green]"
        elif s_svc.available:
            mode_text = "[yellow]⚠ Service Mode[/yellow] [dim](unhealthy)[/dim]"
        else:
            mode_text = "[red]✗ Service Mode[/red] [dim](not reachable)[/dim]"
    else:
        mode_text = "[cyan]Direct Mode[/cyan]"

    env_label = "[dim](docker)[/dim]" if s_cfg.environment == "docker" else "[dim](local)[/dim]"

    header_parts = [f"[dim]v{s_cfg.version}[/dim]", mode_text]
    if s_cfg.service_url:
        header_parts.append(f"[dim]{s_cfg.service_url}[/dim]")
    header_parts.append(env_label)
    header_line = "  ".join(header_parts)
    console.print(Panel(header_line, title="[bold magenta]llm-bawt[/bold magenta]", border_style="grey39", padding=(0, 2)))
    console.print()

    # ── Config + Service (side-by-side) ─────────────────────────
    main_table = Table(show_header=False, box=None, padding=(0, 2))
    main_table.add_column("Key", style="cyan", no_wrap=True)
    main_table.add_column("Val")
    main_table.add_column("Key2", style="cyan", no_wrap=True)
    main_table.add_column("Val2")

    bot_display = f"[bold cyan]{s_cfg.bot_name}[/bold cyan] ({s_cfg.bot_slug})"

    # Model display
    if s_cfg.model_alias and s_model:
        type_suffix = f" [dim]{s_model.type}[/dim]" if s_model.type else ""
        model_display = f"[green]{s_cfg.model_alias}[/green]{type_suffix}"
    elif s_cfg.model_alias:
        model_display = f"[green]{s_cfg.model_alias}[/green]"
    else:
        model_display = "[dim]not set[/dim]"

    # Service status values
    if s_svc.available and s_svc.healthy:
        uptime_str = _format_uptime(s_svc.uptime_seconds) if s_svc.uptime_seconds else ""
        svc_status = "[green]✓ Up[/green]"
        if uptime_str:
            svc_status += f" [dim]({uptime_str})[/dim]"
        svc_loaded = f"[green]{s_svc.current_model}[/green]" if s_svc.current_model else "[dim]none[/dim]"
        svc_tasks = f"{s_svc.tasks_processed} / {s_svc.tasks_pending} pending"
    elif s_svc.available:
        svc_status = "[yellow]⚠ Unhealthy[/yellow]"
        svc_loaded = ""
        svc_tasks = ""
    elif s_cfg.mode == "service":
        svc_status = "[red]✗ Not reachable[/red]"
        svc_loaded = ""
        svc_tasks = ""
    else:
        svc_status = "[dim]○ Not running[/dim]"
        svc_loaded = ""
        svc_tasks = ""

    # MCP display
    if s_mcp.mode == "server":
        mcp_mode = "[green]Server[/green]"
        if s_mcp.status == "up":
            mcp_status = "[green]✓ Up[/green]"
        elif s_mcp.status == "error":
            mcp_status = f"[yellow]⚠ HTTP {s_mcp.http_status}[/yellow]"
        else:
            mcp_status = "[red]✗ Down[/red]"
        mcp_url_display = str(s_mcp.url)
    else:
        mcp_mode = "[cyan]Embedded[/cyan]"
        mcp_status = "[green]✓ In-process[/green]"
        mcp_url_display = "[dim]embedded[/dim]"

    # Scheduler / HA / models catalog
    scheduler_display = "[green]Enabled[/green]" if s_cfg.scheduler_enabled else "[dim]Disabled[/dim]"
    scheduler_display += f" [dim]({s_cfg.scheduler_interval}s)[/dim]"

    models_catalog = f"{s_cfg.models_defined} defined"
    if s_cfg.models_service is not None:
        models_catalog += f" [dim]({s_cfg.models_service} service)[/dim]"

    if s_cfg.ha_mcp_enabled:
        ha_display = f"[green]Enabled[/green] [dim]({s_cfg.ha_mcp_url})[/dim]"
    else:
        ha_display = "[dim]Disabled[/dim]"

    # Bots list
    bot_parts = []
    for b in s_cfg.all_bots:
        if b.is_default:
            bot_parts.append(f"[cyan]{b.slug}*[/cyan]")
        else:
            bot_parts.append(b.slug)
    bots_display = " ".join(bot_parts)

    main_table.add_row("Bot", bot_display, "LLM", svc_status)
    main_table.add_row("Model", model_display, "Loaded", svc_loaded)
    main_table.add_row("User", f"{s_cfg.user_id}" if s_cfg.user_id else "[dim]not set[/dim]", "Tasks", svc_tasks)
    main_table.add_row("Bots", bots_display, "Memory", mcp_mode)
    main_table.add_row("Models", models_catalog, "MCP", mcp_status)
    main_table.add_row("Scheduler", scheduler_display, "MCP URL", mcp_url_display)
    main_table.add_row("HA MCP", ha_display, "Bind Host", f"[dim]{s_cfg.bind_host}[/dim]")

    console.print(Panel(main_table, title="[bold]Config[/bold] / [bold]Service[/bold]", border_style="grey39"))
    console.print()

    # ── Model Info Panel ─────────────────────────────────────────
    if s_model:
        model_info_table = make_table()
        model_info_table.add_row("Max Tokens", f"{s_model.max_tokens} [dim]({s_model.max_tokens_source})[/dim]")

        if s_model.type == "gguf":
            if s_model.gpu_name:
                model_info_table.add_row("GPU", f"[green]{s_model.gpu_name}[/green]")
                model_info_table.add_row(
                    "VRAM",
                    f"{s_model.vram_total_gb:.1f}GB total, {s_model.vram_free_gb:.1f}GB free [dim]({s_model.vram_detection_method})[/dim]"
                )
            else:
                model_info_table.add_row("GPU", "[dim]Not detected[/dim]")

            if s_model.context_window:
                model_info_table.add_row("Context", f"{s_model.context_window:,} tokens [dim]({s_model.context_source})[/dim]")

            if s_model.n_gpu_layers:
                layers_display = "All" if s_model.n_gpu_layers == "all" else s_model.n_gpu_layers
                model_info_table.add_row("GPU Layers", f"{layers_display} [dim]({s_model.gpu_layers_source})[/dim]")

            if s_model.native_context_limit:
                model_info_table.add_row("Native Limit", f"{s_model.native_context_limit:,} tokens")
        else:
            if s_model.context_window:
                model_info_table.add_row("Context", f"{s_model.context_window:,} tokens [dim]({s_model.context_source})[/dim]")

        console.print(Panel(model_info_table, title=f"[bold]Model: {s_model.alias}[/bold] [dim]{s_model.type}[/dim]", border_style="grey39"))
        console.print()

    # ── Memory + Dependencies (side-by-side) ─────────────────────
    memory_table = make_table()

    if not s_mem.postgres_connected and s_mem.postgres_error is None and s_mem.postgres_host is None:
        memory_table.add_row("Postgres", "[yellow]Not configured[/yellow]")
    elif s_mem.postgres_connected:
        pg_host = s_mem.postgres_host or ""
        if len(pg_host) > 15:
            pg_host = pg_host[:12] + "..."
        memory_table.add_row("Postgres", f"[green]✓[/green] {pg_host}")
    else:
        memory_table.add_row("Postgres", "[red]✗ Error[/red]")

    messages_display = f"[green]{s_mem.messages_count}[/green]" if s_mem.messages_count else "[dim]0[/dim]"
    memories_display = f"[green]{s_mem.memories_count}[/green]" if s_mem.memories_count else "[dim]0[/dim]"
    memory_table.add_row("Messages", messages_display)
    memory_table.add_row("Memories", memories_display)

    if s_mem.pgvector_available:
        memory_table.add_row("pgvector", "[green]✓[/green]")
    elif s_mem.postgres_connected or s_mem.postgres_error:
        memory_table.add_row("pgvector", "[red]✗ Missing[/red]")
    else:
        memory_table.add_row("pgvector", "[dim]○ N/A[/dim]")

    if s_mem.embeddings_available:
        memory_table.add_row("Embeddings", "[green]✓[/green] [dim]MiniLM[/dim]")
    else:
        memory_table.add_row("Embeddings", "[dim]○ N/A[/dim]")

    # Dependencies
    deps_table = make_table()
    missing_deps: list[str] = []

    if s_dep.cuda_version and s_dep.cuda_version not in ("error",):
        deps_table.add_row("CUDA", f"[green]✓[/green] {s_dep.cuda_version}")
    elif s_dep.cuda_version == "error":
        deps_table.add_row("CUDA", "[yellow]⚠ error[/yellow]")
    else:
        deps_table.add_row("CUDA", "[dim]○ N/A[/dim]")

    if s_dep.llama_cpp_available:
        llama_label = "[green]✓[/green]"
        if s_dep.llama_cpp_gpu is True:
            llama_label += " (GPU)"
        elif s_dep.llama_cpp_gpu is False:
            llama_label += " [dim](CPU)[/dim]"
        deps_table.add_row("llama-cpp", llama_label)
    else:
        deps_table.add_row("llama-cpp", "[red]✗ Missing[/red]")
        missing_deps.append("[cyan]make install-extras-llama[/cyan]")

    if s_dep.hf_hub_available:
        deps_table.add_row("hf-hub", "[green]✓[/green]")
    else:
        deps_table.add_row("hf-hub", "[red]✗ Missing[/red]")
        missing_deps.append("[cyan]make install-extras-hf[/cyan]")

    if s_dep.torch_available:
        deps_table.add_row("torch", "[green]✓[/green]")
    else:
        deps_table.add_row("torch", "[dim]○ Optional[/dim]")

    if s_dep.openai_key_set:
        deps_table.add_row("OpenAI", "[green]✓ Key set[/green]")
    else:
        deps_table.add_row("OpenAI", "[dim]○ No key[/dim]")

    if s_dep.newsapi_key_set:
        deps_table.add_row("NewsAPI", "[green]✓ Key set[/green]")
        deps_table.add_row("News Tool", "[green]✓ Enabled[/green]")
    else:
        deps_table.add_row("NewsAPI", "[dim]○ No key[/dim]")
        deps_table.add_row("News Tool", "[yellow]⚠ Disabled[/yellow] [dim](set LLM_BAWT_NEWSAPI_API_KEY)[/dim]")

    if s_dep.search_provider:
        deps_table.add_row("Search", f"[green]✓[/green] {s_dep.search_provider}")
    else:
        deps_table.add_row("Search", "[dim]○ None[/dim]")

    row2 = Table.grid(expand=True)
    row2.add_column(ratio=1)
    row2.add_column(ratio=1)
    row2.add_row(
        Panel(memory_table, title="[bold]Memory[/bold]", border_style="grey39"),
        Panel(deps_table, title="[bold]Dependencies[/bold]", border_style="grey39"),
    )
    console.print(row2)

    console.print()
    console.print(
        "[dim]Useful:[/dim] "
        "[cyan]--job-status[/cyan]  "
        "[cyan]--list-models[/cyan]  "
        "[cyan]--list-bots[/cyan]  "
        "[cyan]--list-users[/cyan]  "
        "[cyan]--user-profile[/cyan]  "
        "[cyan]--config-list[/cyan]"
    )

    if missing_deps:
        console.print()
        console.print(f"  [dim]Install missing:[/dim] {', '.join(missing_deps)}")


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
    table.add_column("Color")
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
            f"[{bot.color}]{bot.color}[/]" if bot.color else "[dim]default[/dim]",
            bot.description,
            default_model,
            memory_icon
        )
    
    console.print(table)
    console.print()
    console.print(f"[dim]⭐ = default bot | ✓ = requires database | Use -b/--bot <slug> to select[/dim]")
    console.print()


def _render_job_status_from_service(config: Config) -> bool:
    """Render job status via the service API. Returns True if successful."""
    client = get_service_client(config)
    if not client or not client.is_available():
        return False

    from datetime import datetime

    jobs_resp = client.get_jobs()
    runs_resp = client.get_job_runs(limit=10)
    if jobs_resp is None and runs_resp is None:
        return False

    jobs = (jobs_resp or {}).get("jobs", [])
    runs = (runs_resp or {}).get("runs", [])

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

        now = datetime.utcnow()
        for job in jobs:
            enabled_icon = "[green]✓[/green]" if job.get("enabled") else "[red]✗[/red]"
            interval_str = f"{job.get('interval_minutes', 0)}m"

            last_run_str = "[dim]never[/dim]"
            if job.get("last_run_at"):
                try:
                    last_run = datetime.fromisoformat(job["last_run_at"].replace("Z", "+00:00")).replace(tzinfo=None)
                    ago = now - last_run
                    if ago.total_seconds() < 3600:
                        last_run_str = f"{int(ago.total_seconds() // 60)}m ago"
                    elif ago.total_seconds() < 86400:
                        last_run_str = f"{int(ago.total_seconds() // 3600)}h ago"
                    else:
                        last_run_str = f"{int(ago.total_seconds() // 86400)}d ago"
                except Exception:
                    last_run_str = "[dim]?[/dim]"

            next_run_str = "[dim]pending[/dim]"
            if job.get("next_run_at"):
                try:
                    next_run = datetime.fromisoformat(job["next_run_at"].replace("Z", "+00:00")).replace(tzinfo=None)
                    until = next_run - now
                    if until.total_seconds() < 0:
                        next_run_str = "[yellow]overdue[/yellow]"
                    elif until.total_seconds() < 3600:
                        next_run_str = f"in {int(until.total_seconds() // 60)}m"
                    else:
                        next_run_str = f"in {int(until.total_seconds() // 3600)}h"
                except Exception:
                    next_run_str = "[dim]?[/dim]"

            job_table.add_row(
                job.get("job_type", "?"),
                job.get("bot_id", "?"),
                enabled_icon,
                interval_str,
                last_run_str,
                next_run_str,
            )
        console.print(job_table)

    console.print()
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
            status_val = run.get("status", "")
            if status_val == "success":
                status_str = "[green]✓ success[/green]"
            elif status_val == "failed":
                status_str = "[red]✗ failed[/red]"
            elif status_val == "running":
                status_str = "[yellow]⟳ running[/yellow]"
            elif status_val == "skipped":
                status_str = "[dim]○ skipped[/dim]"
            else:
                status_str = "[dim]○ pending[/dim]"

            duration_ms = run.get("duration_ms")
            duration_str = f"{duration_ms}ms" if duration_ms else "[dim]-[/dim]"
            error_msg = run.get("error_message") or ""
            error_str = (error_msg[:40] + "...") if len(error_msg) > 40 else (error_msg or "[dim]-[/dim]")
            started_str = "[dim]-[/dim]"
            if run.get("started_at"):
                try:
                    started_str = datetime.fromisoformat(run["started_at"].replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pass

            run_table.add_row(started_str, run.get("bot_id", "?"), status_str, duration_str, error_str)
        console.print(run_table)

    console.print()
    console.print(f"[dim]Scheduler enabled: {config.SCHEDULER_ENABLED} | Check interval: {config.SCHEDULER_CHECK_INTERVAL_SECONDS}s[/dim]")
    console.print()
    return True


def show_job_status(config: Config):
    """Display scheduled background jobs and recent run history."""
    if _use_service(config):
        if _render_job_status_from_service(config):
            return
        console.print("[yellow]Service is configured but not reachable. Falling back to direct DB.[/yellow]")

    if not has_database_credentials(config):
        console.print("[yellow]Job scheduler requires database connection.[/yellow]")
        console.print("[dim]Set LLM_BAWT_POSTGRES_PASSWORD in ~/.config/llm-bawt/.env[/dim]")
        return

    try:
        from sqlmodel import Session, create_engine, select
        from urllib.parse import quote_plus
        from llm_bawt.service.scheduler import ScheduledJob, JobRun, JobStatus, create_scheduler_tables
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
    if _use_service(config):
        # No service endpoint for trigger yet; direct DB access required
        console.print("[yellow]Job trigger requires direct DB access (not available via service).[/yellow]")
        console.print("[dim]Run this from the machine hosting the database, or add a trigger endpoint to the service.[/dim]")
        return

    if not has_database_credentials(config):
        console.print("[yellow]Job scheduler requires database connection.[/yellow]")
        console.print("[dim]Set LLM_BAWT_POSTGRES_PASSWORD in ~/.config/llm-bawt/.env[/dim]")
        return

    valid_types = ["profile_maintenance", "memory_consolidation", "memory_decay"]
    if job_type not in valid_types:
        console.print(f"[red]Unknown job type: {job_type}[/red]")
        console.print(f"[dim]Valid types: {', '.join(valid_types)}[/dim]")
        return

    try:
        from sqlmodel import Session, create_engine, select
        from urllib.parse import quote_plus
        from llm_bawt.service.scheduler import ScheduledJob, JobType, create_scheduler_tables
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
    if _use_service(config):
        client = get_service_client(config)
        if client and client.is_available():
            data = client.get_profile(user_id, entity_type="user")
            if data:
                console.print(Panel.fit(f"[bold cyan]User Profile: {user_id}[/bold cyan]", border_style="cyan"))
                console.print()
                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column("Field", style="cyan")
                table.add_column("Value")
                table.add_row("[bold]Identity[/bold]", "")
                table.add_row("  Display Name", data.get("display_name") or "[dim]not set[/dim]")
                table.add_row("  Description", data.get("description") or "[dim]not set[/dim]")

                by_category: dict[str, list] = {}
                for attr in data.get("attributes", []):
                    cat = attr.get("category", "other")
                    by_category.setdefault(cat, []).append(attr)

                category_names = {"preference": "Preferences", "fact": "Facts", "interest": "Interests", "communication": "Communication", "context": "Context"}
                for category, attrs in sorted(by_category.items()):
                    table.add_row("", "")
                    table.add_row(f"[bold]{category_names.get(category, category.title())}[/bold]", "")
                    for attr in sorted(attrs, key=lambda a: a.get("key", "")):
                        value_str = str(attr.get("value", ""))
                        if len(value_str) > 60:
                            value_str = value_str[:57] + "..."
                        conf = attr.get("confidence", 1.0)
                        conf_str = f" [dim]({conf:.0%})[/dim]" if conf < 1.0 else ""
                        table.add_row(f"  {attr.get('key', '?')}", f"{value_str}{conf_str}")

                console.print(table)
                console.print()
                console.print("[dim]Add attributes: llm --user-profile-set category.key=value[/dim]")
                console.print()
                return
            else:
                console.print(f"[yellow]No profile found for user '{user_id}' on service.[/yellow]")
                return

    if not has_database_credentials(config):
        console.print("[yellow]User profiles require database connection.[/yellow]")
        console.print("[dim]Set LLM_BAWT_POSTGRES_PASSWORD in ~/.config/llm-bawt/.env[/dim]")
        return

    try:
        from llm_bawt.profiles import ProfileManager, EntityType

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
    if _use_service(config):
        client = get_service_client(config)
        if client and client.is_available():
            data = client.list_profiles(entity_type="user")
            profiles_list = (data or {}).get("profiles", []) if data else []

            if not profiles_list:
                console.print("[yellow]No user profiles found.[/yellow]")
                console.print("[dim]Create one with: llm --user-profile-setup[/dim]")
                return

            console.print(Panel.fit("[bold cyan]User Profiles[/bold cyan]", border_style="cyan"))
            console.print()
            table = Table(show_header=True, box=None, padding=(0, 2))
            table.add_column("User ID", style="cyan")
            table.add_column("Display Name")
            table.add_column("Description")

            for p in profiles_list:
                desc = p.get("description") or ""
                table.add_row(
                    p.get("entity_id", "?"),
                    p.get("display_name") or "[dim]-[/dim]",
                    (desc[:50] + "..." if len(desc) > 50 else desc) or "[dim]-[/dim]",
                )

            console.print(table)
            console.print()
            console.print("[dim]Use --user <id> to select a user profile[/dim]")
            console.print()
            return

    if not has_database_credentials(config):
        console.print("[yellow]User profiles require database connection.[/yellow]")
        console.print("[dim]Set LLM_BAWT_POSTGRES_PASSWORD in ~/.config/llm-bawt/.env[/dim]")
        return

    try:
        from llm_bawt.profiles import ProfileManager, EntityType

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


def _parse_runtime_setting_value(raw: str):
    """Parse runtime setting value from CLI string input."""
    text = (raw or "").strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except Exception:
        lowered = text.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            if "." in text:
                return float(text)
            return int(text)
        except Exception:
            return text


def show_runtime_settings(config: Config, scope: str, bot_id: str | None) -> bool:
    """Display runtime settings from DB for global or bot scope."""
    if scope == "global":
        scope_id = "*"
        title = "Runtime Settings (global)"
    else:
        target_bot = bot_id or config.DEFAULT_BOT
        scope_id = target_bot
        title = f"Runtime Settings (bot: {target_bot})"

    if _use_service(config):
        client = get_service_client(config)
        if client and client.is_available():
            resp = client.get_settings(scope_type=scope, scope_id=scope_id if scope != "global" else None)
            if resp is not None:
                items = resp.get("settings", [])
                if not items:
                    console.print("[yellow]No runtime settings found for this scope.[/yellow]")
                    console.print("[dim]Tip: run `llm --settings-bootstrap` to seed DB settings from current bot/template config.[/dim]")
                    return True
                table = Table(title=title)
                table.add_column("Key", style="cyan")
                table.add_column("Value")
                for item in sorted(items, key=lambda i: i.get("key", "")):
                    table.add_row(item.get("key", "?"), repr(item.get("value")))
                console.print(table)
                return True

    from llm_bawt.runtime_settings import RuntimeSettingsStore

    store = RuntimeSettingsStore(config)
    if store.engine is None:
        console.print("[red]Runtime settings DB unavailable.[/red]")
        return False

    data = store.get_scope_settings(scope, scope_id)
    if not data:
        console.print("[yellow]No runtime settings found for this scope.[/yellow]")
        console.print("[dim]Tip: run `llm --settings-bootstrap` to seed DB settings from current bot/template config.[/dim]")
        return True

    table = Table(title=title)
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for key, value in sorted(data.items()):
        table.add_row(key, repr(value))
    console.print(table)
    return True


def set_runtime_setting(config: Config, scope: str, bot_id: str | None, key: str, raw_value: str) -> bool:
    """Set runtime setting in DB."""
    scope_id = "*" if scope == "global" else (bot_id or config.DEFAULT_BOT)
    value = _parse_runtime_setting_value(raw_value)

    if _use_service(config):
        client = get_service_client(config)
        if client and client.is_available():
            resp = client.set_setting(scope, scope_id if scope != "global" else None, key, value)
            if resp and resp.get("success"):
                console.print(f"[green]Set runtime setting[/green] {scope}:{scope_id} {key}={value!r}")
                return True
            console.print("[red]Failed to set runtime setting via service.[/red]")
            return False

    from llm_bawt.runtime_settings import RuntimeSettingsStore

    store = RuntimeSettingsStore(config)
    if store.engine is None:
        console.print("[red]Runtime settings DB unavailable.[/red]")
        return False

    store.set_value(scope, scope_id, key, value)
    console.print(f"[green]Set runtime setting[/green] {scope}:{scope_id} {key}={value!r}")
    return True


def delete_runtime_setting(config: Config, scope: str, bot_id: str | None, key: str) -> bool:
    """Delete runtime setting from DB."""
    scope_id = "*" if scope == "global" else (bot_id or config.DEFAULT_BOT)

    if _use_service(config):
        client = get_service_client(config)
        if client and client.is_available():
            resp = client.delete_setting(scope, key, scope_id=scope_id if scope != "global" else None)
            if resp and resp.get("success"):
                if resp.get("deleted"):
                    console.print(f"[green]Deleted runtime setting[/green] {scope}:{scope_id} {key}")
                else:
                    console.print("[yellow]No matching runtime setting found.[/yellow]")
                return True
            console.print("[red]Failed to delete runtime setting via service.[/red]")
            return False

    from llm_bawt.runtime_settings import RuntimeSettingsStore

    store = RuntimeSettingsStore(config)
    if store.engine is None:
        console.print("[red]Runtime settings DB unavailable.[/red]")
        return False

    deleted = store.delete_value(scope, scope_id, key)
    if not deleted:
        console.print("[yellow]No matching runtime setting found.[/yellow]")
        return True
    console.print(f"[green]Deleted runtime setting[/green] {scope}:{scope_id} {key}")
    return True


def bootstrap_runtime_settings(config: Config, bot_id: str | None, overwrite: bool = False) -> bool:
    """Seed DB runtime settings from current config/bot effective settings."""
    from llm_bawt.bots import BotManager

    bot_manager = BotManager(config)
    bots = bot_manager.list_bots()
    if bot_id:
        target = bot_manager.get_bot(bot_id)
        if not target:
            console.print(f"[red]Unknown bot: {bot_id}[/red]")
            return False
        bots = [target]

    global_map = {
        "max_context_tokens": config.MAX_CONTEXT_TOKENS,
        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        "history_duration_seconds": config.HISTORY_DURATION_SECONDS,
        "history_bridge_messages": config.HISTORY_BRIDGE_MESSAGES,
        "history_reload_ttl_seconds": config.HISTORY_RELOAD_TTL_SECONDS,
        "summarization_session_gap_seconds": config.SUMMARIZATION_SESSION_GAP_SECONDS,
        "summarization_min_messages": config.SUMMARIZATION_MIN_MESSAGES,
        "summarization_skip_low_signal": config.SUMMARIZATION_SKIP_LOW_SIGNAL,
        "summarization_min_user_messages": config.SUMMARIZATION_MIN_USER_MESSAGES,
        "summarization_min_content_chars": config.SUMMARIZATION_MIN_CONTENT_CHARS,
        "summarization_min_meaningful_turns": config.SUMMARIZATION_MIN_MEANINGFUL_TURNS,
        "summarization_max_in_context": config.SUMMARIZATION_MAX_IN_CONTEXT,
        "summarization_compact_context": config.SUMMARIZATION_COMPACT_CONTEXT,
        "memory_n_results": config.MEMORY_N_RESULTS,
        "memory_protected_recent_turns": config.MEMORY_PROTECTED_RECENT_TURNS,
        "memory_min_relevance": config.MEMORY_MIN_RELEVANCE,
        "memory_max_token_percent": config.MEMORY_MAX_TOKEN_PERCENT,
        "memory_dedup_similarity": config.MEMORY_DEDUP_SIMILARITY,
        "profile_maintenance_interval_minutes": config.PROFILE_MAINTENANCE_INTERVAL_MINUTES,
        "history_summarization_interval_minutes": config.HISTORY_SUMMARIZATION_INTERVAL_MINUTES,
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
    }

    if _use_service(config):
        client = get_service_client(config)
        if client and client.is_available():
            # Fetch current global settings to check for existing keys
            existing_resp = client.get_settings(scope_type="global")
            global_existing_keys = set()
            if existing_resp:
                for item in existing_resp.get("settings", []):
                    global_existing_keys.add(item.get("key"))

            batch_items: list[dict] = []
            set_count = 0
            skipped_count = 0

            for key, value in global_map.items():
                if key in global_existing_keys and not overwrite:
                    skipped_count += 1
                    continue
                batch_items.append({"scope_type": "global", "scope_id": "*", "key": key, "value": value})
                set_count += 1

            for bot in bots:
                bot_resp = client.get_settings(scope_type="bot", scope_id=bot.slug)
                bot_existing_keys = set()
                if bot_resp:
                    for item in bot_resp.get("settings", []):
                        bot_existing_keys.add(item.get("key"))
                for key, value in (bot.settings or {}).items():
                    if key in bot_existing_keys and not overwrite:
                        skipped_count += 1
                        continue
                    batch_items.append({"scope_type": "bot", "scope_id": bot.slug, "key": key, "value": value})
                    set_count += 1

            if batch_items:
                client.batch_set_settings(batch_items)

            scope_label = f"bot '{bot_id}'" if bot_id else "all bots"
            console.print(
                f"[green]Bootstrapped runtime settings[/green] ({scope_label}): "
                f"set={set_count}, skipped={skipped_count}, overwrite={overwrite}"
            )
            return True

    from llm_bawt.runtime_settings import RuntimeSettingsStore

    store = RuntimeSettingsStore(config)
    if store.engine is None:
        console.print("[red]Runtime settings DB unavailable.[/red]")
        return False

    global_existing = store.get_scope_settings("global", "*")

    set_count = 0
    skipped_count = 0

    for key, value in global_map.items():
        if (key in global_existing) and not overwrite:
            skipped_count += 1
            continue
        store.set_value("global", "*", key, value)
        set_count += 1

    for bot in bots:
        bot_existing = store.get_scope_settings("bot", bot.slug)
        for key, value in (bot.settings or {}).items():
            if (key in bot_existing) and not overwrite:
                skipped_count += 1
                continue
            store.set_value("bot", bot.slug, key, value)
            set_count += 1

    scope_label = f"bot '{bot_id}'" if bot_id else "all bots"
    console.print(
        f"[green]Bootstrapped runtime settings[/green] ({scope_label}): "
        f"set={set_count}, skipped={skipped_count}, overwrite={overwrite}"
    )
    return True


def migrate_bots_to_db(config: Config) -> bool:
    """Migrate merged YAML bot personalities to bot_profiles and settings to runtime_settings."""
    from llm_bawt.bots import (
        get_repo_bots_yaml_path,
        get_user_bots_yaml_path,
        _deep_merge,
    )

    def _load_yaml(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            console.print(f"[yellow]Skipping invalid YAML {path}: {e}[/yellow]")
            return {}

    repo_data = _load_yaml(get_repo_bots_yaml_path())
    user_data = _load_yaml(get_user_bots_yaml_path())
    merged = _deep_merge(repo_data, user_data)
    bots_data = merged.get("bots", {}) if isinstance(merged, dict) else {}

    if not isinstance(bots_data, dict) or not bots_data:
        console.print("[yellow]No YAML bots found to migrate.[/yellow]")
        return True

    if _use_service(config):
        client = get_service_client(config)
        if client and client.is_available():
            migrated_profiles = 0
            settings_items: list[dict] = []

            for slug, bot_data in bots_data.items():
                if not isinstance(bot_data, dict):
                    continue
                normalized_slug = str(slug).strip().lower()
                if not normalized_slug:
                    continue

                client.upsert_bot_profile(normalized_slug, {
                    "name": bot_data.get("name", normalized_slug.title()),
                    "description": bot_data.get("description", ""),
                    "system_prompt": bot_data.get("system_prompt", "You are a helpful assistant."),
                    "requires_memory": bot_data.get("requires_memory", True),
                    "voice_optimized": bot_data.get("voice_optimized", False),
                    "uses_tools": bot_data.get("uses_tools", False),
                    "uses_search": bot_data.get("uses_search", False),
                    "uses_home_assistant": bot_data.get("uses_home_assistant", False),
                    "default_model": bot_data.get("default_model"),
                    "nextcloud_config": bot_data.get("nextcloud"),
                })
                migrated_profiles += 1

                bot_settings = bot_data.get("settings", {}) or {}
                if isinstance(bot_settings, dict):
                    for key, value in bot_settings.items():
                        settings_items.append({"scope_type": "bot", "scope_id": normalized_slug, "key": key, "value": value})

            if settings_items:
                client.batch_set_settings(settings_items)

            console.print(
                "[green]Bot migration complete.[/green] "
                f"profiles={migrated_profiles}, bot_settings={len(settings_items)}"
            )
            return True

    from llm_bawt.runtime_settings import BotProfileStore, RuntimeSettingsStore

    profile_store = BotProfileStore(config)
    settings_store = RuntimeSettingsStore(config)
    if profile_store.engine is None or settings_store.engine is None:
        console.print("[red]Migration requires database access.[/red]")
        return False

    migrated_profiles = 0
    migrated_settings = 0

    for slug, bot_data in bots_data.items():
        if not isinstance(bot_data, dict):
            continue
        normalized_slug = str(slug).strip().lower()
        if not normalized_slug:
            continue

        profile_store.upsert(
            {
                "slug": normalized_slug,
                "name": bot_data.get("name", normalized_slug.title()),
                "description": bot_data.get("description", ""),
                "system_prompt": bot_data.get("system_prompt", "You are a helpful assistant."),
                "requires_memory": bot_data.get("requires_memory", True),
                "voice_optimized": bot_data.get("voice_optimized", False),
                "uses_tools": bot_data.get("uses_tools", False),
                "uses_search": bot_data.get("uses_search", False),
                "uses_home_assistant": bot_data.get("uses_home_assistant", False),
                "default_model": bot_data.get("default_model"),
                "nextcloud_config": bot_data.get("nextcloud"),
            }
        )
        migrated_profiles += 1

        bot_settings = bot_data.get("settings", {}) or {}
        if isinstance(bot_settings, dict):
            for key, value in bot_settings.items():
                settings_store.set_value("bot", normalized_slug, key, value)
                migrated_settings += 1

    console.print(
        "[green]Bot migration complete.[/green] "
        f"profiles={migrated_profiles}, bot_settings={migrated_settings}"
    )

    try:
        from llm_bawt.bots import _load_bots_config

        _load_bots_config()
    except Exception:
        pass

    return True


def run_user_profile_setup(config: Config, user_id: str) -> bool:
    """Run interactive user profile setup wizard."""
    use_svc = _use_service(config)

    if not use_svc and not has_database_credentials(config):
        console.print("[yellow]User profiles require database connection.[/yellow]")
        console.print("[dim]Set LLM_BAWT_POSTGRES_PASSWORD in ~/.config/llm-bawt/.env[/dim]")
        return False

    console.print(Panel.fit("[bold cyan]User Profile Setup[/bold cyan]", border_style="cyan"))
    console.print()
    console.print(f"[dim]Setting up profile for user: {user_id}[/dim]")
    console.print(f"[dim]Press Enter to skip any field.[/dim]")
    console.print()

    try:
        # Fetch existing profile for defaults
        existing_name = ""
        existing_attrs: dict[str, str] = {}
        if use_svc:
            client = get_service_client(config)
            if client and client.is_available():
                data = client.get_profile(user_id, entity_type="user")
                if data:
                    existing_name = data.get("display_name") or ""
                    for attr in data.get("attributes", []):
                        existing_attrs[attr.get("key", "")] = attr.get("value", "")
        else:
            from llm_bawt.profiles import ProfileManager, EntityType
            manager = ProfileManager(config)
            profile, _ = manager.get_or_create_profile(EntityType.USER, user_id)
            existing_name = profile.display_name or ""
            existing_attrs = {attr.key: attr.value for attr in manager.get_all_attributes(EntityType.USER, user_id)}

        name = Prompt.ask("What's your name?", default=existing_name or existing_attrs.get("name", ""))
        occupation = Prompt.ask("What do you do? (occupation)", default=existing_attrs.get("occupation", ""))
        console.print()

        if use_svc:
            client = get_service_client(config)
            if client and client.is_available():
                if name:
                    client.upsert_profile_attribute("user", user_id, "fact", "name", name)
                if occupation:
                    client.upsert_profile_attribute("user", user_id, "fact", "occupation", occupation)
                console.print(f"[green]✓ User profile saved for '{user_id}'[/green]")
                console.print()
                return True
        else:
            from llm_bawt.profiles import ProfileManager, EntityType, AttributeCategory
            manager = ProfileManager(config)
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
    if _use_service(config):
        try:
            client = get_service_client(config)
            if client and client.is_available():
                data = client.get_profile(user_id, entity_type="user")
                if data and data.get("display_name"):
                    return True
                # No profile or no name — run setup
                console.print()
                console.print("[yellow]Welcome! Let's set up your user profile.[/yellow]")
                console.print()
                return run_user_profile_setup(config, user_id)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not ensure user profile via service: {e}")
            return True

    if not has_database_credentials(config):
        logging.getLogger(__name__).debug("Database credentials not configured - skipping user profile")
        return True  # Not an error, just no profile

    try:
        from llm_bawt.profiles import ProfileManager, EntityType

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
        console.print("[dim]Set LLM_BAWT_DEFAULT_USER or pass --user.[/dim]")
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
    parser.add_argument("--add-model",type=str,choices=['ollama', 'openai', 'gguf', 'vllm'],metavar="TYPE",help="Add models: 'ollama' (refresh from server), 'openai' (query API), 'gguf' (add from HuggingFace repo), 'vllm' (add vLLM model from HuggingFace)")
    parser.add_argument("--delete-model",type=str,metavar="ALIAS",help="Delete the specified model alias from the configuration file after confirmation.")
    parser.add_argument(
        "--set-context-window",
        nargs=2,
        metavar=("ALIAS", "TOKENS"),
        help="Set per-model context window in models.yaml. Creates missing aliases (e.g., grok-* from --list-models).",
    )
    parser.add_argument("--config-set", nargs=2, metavar=("KEY", "VALUE"), help="Set a configuration value (e.g., DEFAULT_MODEL_ALIAS) in the .env file.")
    parser.add_argument("--config-list", action="store_true", help="List the current effective configuration settings.")
    parser.add_argument("--setup", action="store_true", help="Walk through client .env settings interactively (pre-filled with current values).")
    parser.add_argument("--settings-scope", choices=["bot", "global"], default="bot", help="Scope for runtime settings operations (default: bot)")
    parser.add_argument("--settings-list", action="store_true", help="List runtime settings from DB for selected scope")
    parser.add_argument("--settings-set", nargs=2, metavar=("KEY", "VALUE"), help="Set runtime setting in DB for selected scope")
    parser.add_argument("--settings-delete", metavar="KEY", help="Delete runtime setting from DB for selected scope")
    parser.add_argument("--settings-bootstrap", action="store_true", help="Seed DB runtime settings from current bot/template/env config")
    parser.add_argument("--settings-bootstrap-overwrite", action="store_true", help="When bootstrapping, overwrite existing DB keys")
    parser.add_argument("--settings-edit", action="store_true", help="Edit global runtime settings in your editor and save to DB")
    parser.add_argument("--bot-edit", metavar="SLUG", help="Edit a single bot config in your editor and save to DB")
    parser.add_argument("--migrate-bots", action="store_true", help="Migrate merged YAML bots to bot_profiles and runtime_settings DB")
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

    # Interactive config setup walkthrough
    if getattr(args, 'setup', False):
        from llm_bawt.cli.config_setup import run_config_setup
        sys.exit(run_config_setup(config_obj, console))

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
    elif getattr(args, "set_context_window", None):
        alias, token_value = args.set_context_window
        try:
            context_window = int(token_value)
        except ValueError:
            console.print(f"[bold red]Invalid token count:[/bold red] '{token_value}' is not an integer.")
            sys.exit(1)

        success = set_model_context_window(alias, context_window, config_obj)
        sys.exit(0 if success else 1)
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
                env_var = f"LLM_BAWT_{field_name}"
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
    elif getattr(args, "settings_list", False):
        ok = show_runtime_settings(config_obj, scope=args.settings_scope, bot_id=args.bot)
        sys.exit(0 if ok else 1)
    elif getattr(args, "settings_set", None):
        key, value = args.settings_set
        ok = set_runtime_setting(
            config_obj,
            scope=args.settings_scope,
            bot_id=args.bot,
            key=key,
            raw_value=value,
        )
        sys.exit(0 if ok else 1)
    elif getattr(args, "settings_delete", None):
        ok = delete_runtime_setting(
            config_obj,
            scope=args.settings_scope,
            bot_id=args.bot,
            key=args.settings_delete,
        )
        sys.exit(0 if ok else 1)
    elif getattr(args, "settings_bootstrap", False):
        ok = bootstrap_runtime_settings(
            config_obj,
            bot_id=args.bot if args.settings_scope == "bot" else None,
            overwrite=bool(getattr(args, "settings_bootstrap_overwrite", False)),
        )
        sys.exit(0 if ok else 1)
    elif getattr(args, "settings_edit", False):
        from llm_bawt.cli.bot_editor import edit_global_settings

        ok = edit_global_settings(config_obj)
        sys.exit(0 if ok else 1)
    elif getattr(args, "bot_edit", None):
        from llm_bawt.cli.bot_editor import edit_bot_yaml
        ok = edit_bot_yaml(config_obj, args.bot_edit)
        sys.exit(0 if ok else 1)
    elif getattr(args, "migrate_bots", False):
        ok = migrate_bots_to_db(config_obj)
        sys.exit(0 if ok else 1)

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
        try:
            field, value = args.user_profile_set.split("=", 1)
            field = field.strip()
            value = value.strip().strip('"').strip("'")
            user_id = args.user if args.user else config_obj.DEFAULT_USER

            # Parse field: category.key or just key
            valid_categories = {"preference", "fact", "interest", "communication", "context"}
            if "." in field:
                category_str, key = field.split(".", 1)
                if category_str.lower() not in valid_categories:
                    console.print(f"[red]Invalid category: {category_str}[/red]")
                    console.print("[dim]Valid categories: preference, fact, interest, communication, context[/dim]")
                    sys.exit(1)
                category_str = category_str.lower()
            else:
                key = field
                category_str = "fact"

            if _use_service(config_obj):
                client = get_service_client(config_obj)
                if client and client.is_available():
                    client.upsert_profile_attribute("user", user_id, category_str, key, value)
                    console.print(f"[green]Set {category_str}.{key} = {value}[/green]")
                    sys.exit(0)

            if not has_database_credentials(config_obj):
                console.print("[yellow]User profiles require database connection.[/yellow]")
                console.print("[dim]Set LLM_BAWT_POSTGRES_PASSWORD in ~/.config/llm-bawt/.env[/dim]")
                sys.exit(1)

            from llm_bawt.profiles import ProfileManager, EntityType, AttributeCategory
            category_map = {"preference": AttributeCategory.PREFERENCE, "fact": AttributeCategory.FACT, "interest": AttributeCategory.INTEREST, "communication": AttributeCategory.COMMUNICATION, "context": AttributeCategory.CONTEXT}
            category = category_map[category_str]
            manager = ProfileManager(config_obj)
            manager.set_attribute(EntityType.USER, user_id, category, key, value, source="explicit")
            console.print(f"[green]Set {category_str}.{key} = {value}[/green]")
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
            elif args.add_model == 'vllm':
                success = handle_add_vllm(config_obj)
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

    llm_bawt = None
    
    # For history operations, we can use a lightweight path that doesn't require model init
    history_only = (args.delete_history or args.print_history is not None) and not args.question and not args.command
    
    if history_only:
        # Prefer service-backed history in service mode so unconfigured local hosts still work.
        if use_service and not args.local:
            service_client = get_service_client(config_obj)
            if service_client and service_client.is_available(force_check=True):
                from llm_bawt.clients.base import StubClient
                from llm_bawt.models.message import Message
                from llm_bawt.utils.history import HistoryManager

                if args.delete_history:
                    clear_resp = service_client.clear_history(bot_id=bot_id)
                    if clear_resp and clear_resp.get("success"):
                        console.print("[green]Chat history cleared.[/green]")
                    else:
                        detail = service_client.last_error or "service did not return success"
                        console.print(f"[yellow]Could not clear history via service: {detail}[/yellow]")

                if args.print_history is not None:
                    pair_limit = args.print_history
                    if pair_limit == -1:
                        messages = service_client.get_all_history(bot_id=bot_id)
                        history_resp = {"messages": messages or []} if messages is not None else None
                    else:
                        message_limit = max(1, pair_limit * 2)
                        history_resp = service_client.get_history(bot_id=bot_id, limit=message_limit)

                    if history_resp is None:
                        detail = service_client.last_error or "service unavailable"
                        console.print(f"[yellow]Could not load history via service: {detail}[/yellow]")
                    else:
                        stub_client = StubClient(config=config_obj, bot_name=bot.name)
                        history_manager = HistoryManager(
                            client=stub_client,
                            config=config_obj,
                            db_backend=None,
                            bot_id=bot_id,
                        )
                        messages = history_resp.get("messages", [])
                        history_manager.messages = [
                            Message(
                                role=str(msg.get("role", "")),
                                content=str(msg.get("content", "")),
                                timestamp=float(msg.get("timestamp") or 0.0),
                                db_id=str(msg.get("id") or ""),
                            )
                            for msg in messages
                        ]
                        history_manager.print_history(pair_limit)

                        message_ids = [str(msg.get("id")) for msg in messages if msg.get("id")]
                        if message_ids:
                            tool_events = service_client.get_tool_call_events(
                                bot_id=bot_id,
                                message_ids=message_ids,
                                limit=200,
                            )
                            if tool_events and tool_events.get("events"):
                                console.print("[bold]Tool Activity:[/bold]")
                                for event in tool_events.get("events", []):
                                    msg_id = str(event.get("message_id", ""))[:8]
                                    tools = ", ".join(
                                        str(tc.get("name", "tool"))
                                        for tc in (event.get("tool_calls") or [])
                                        if isinstance(tc, dict)
                                    )
                                    if tools:
                                        console.print(f"[dim]· {msg_id}: {tools}[/dim]")
            else:
                console.print("[yellow]Service not available for history operations.[/yellow]")
        else:
            # Local fallback: lightweight history manager without full model initialization
            from llm_bawt.clients.base import StubClient
            from llm_bawt.utils.history import HistoryManager
            from llm_bawt.memory_server.client import get_memory_client

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
                    logging.getLogger(__name__).debug(f"Memory client init failed: {e}")

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
    
    # For query operations, we need full LLMBawt
    need_local_llm_bawt = not use_service or args.delete_history or args.print_history is not None
    
    if need_local_llm_bawt:
        try:
            llm_bawt = LLMBawt(
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
        if llm_bawt:
            llm_bawt.history_manager.clear_history()
            console.print("[green]Chat history cleared.[/green]")
        else:
            console.print("[yellow]History operations require a valid model configuration.[/yellow]")
    
    if args.print_history is not None:
        if llm_bawt:
            llm_bawt.history_manager.print_history(args.print_history)
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
        nonlocal llm_bawt
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
                        f"[bold red]Service call failed for model '{resolved_alias}' (bot '{bot_id}').[/bold red]{error_detail}"
                    )
                else:
                    console.print(
                        f"[bold red]Service call failed for model '{resolved_alias}' (bot '{bot_id}').[/bold red]{error_detail}"
                    )
                return
            # Fall back to local for OpenAI-compatible models
            console.print("[yellow]Service unavailable, falling back to direct API calls.[/yellow]")
            if llm_bawt is None:
                try:
                    llm_bawt = LLMBawt(
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
        if llm_bawt:
            # Standard mode: hardcoded memory retrieval
            llm_bawt.query(prompt, plaintext_output=plaintext_flag, stream=stream_flag)

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
        if llm_bawt and hasattr(llm_bawt.client, 'console') and llm_bawt.client.console:
            handler_console = llm_bawt.client.console
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
