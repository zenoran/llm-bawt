"""Handler for adding vLLM models from HuggingFace."""

from __future__ import annotations

import importlib.util
import pathlib
import re
from typing import Any

import yaml
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from ..utils.config import Config

console = Console()


def detect_quantization(model_id: str, config_data: dict[str, Any] | None = None) -> str | None:
    """Auto-detect quantization type from model ID or config."""
    model_id_lower = model_id.lower()

    if "awq" in model_id_lower:
        return "awq"
    if "gptq" in model_id_lower:
        return "gptq"
    if "fp8" in model_id_lower or "f8" in model_id_lower:
        return "fp8"
    if ".gguf" in model_id_lower or "gguf" in model_id_lower:
        return "gguf"

    if config_data and "quantization_config" in config_data:
        quant_method = config_data["quantization_config"].get("quant_method")
        if quant_method:
            return quant_method.lower()

    return None


def generate_vllm_alias(model_id: str, existing_aliases: list[str]) -> str:
    """Generate a clean alias from HuggingFace model ID."""
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id

    alias_base = model_name.lower()
    alias_base = re.sub(r"^(meta-)?llama-?", "llama-", alias_base)
    alias_base = re.sub(r"-instruct(-awq|-gptq|-fp8)?$", r"\1", alias_base)
    alias_base = re.sub(r"-chat(-awq|-gptq|-fp8)?$", r"\1", alias_base)
    alias_base = re.sub(r"[^a-z0-9]+", "-", alias_base).strip("-")

    alias = alias_base
    counter = 1
    while alias in existing_aliases:
        alias = f"{alias_base}-{counter}"
        counter += 1

    return alias


def _format_downloads(downloads: int) -> str:
    """Format download count with k/M suffixes."""
    if downloads >= 1_000_000:
        return f"{downloads / 1_000_000:.1f}M"
    if downloads >= 1_000:
        return f"{downloads / 1_000:.1f}k"
    return str(downloads)


def _estimate_model_size(siblings: list[Any]) -> str | None:
    """Estimate model size from safetensors files."""
    total_size = 0
    has_safetensors = False

    for sibling in siblings:
        filename = sibling.rfilename.lower()
        if filename.endswith(".safetensors"):
            has_safetensors = True
            if hasattr(sibling, "size") and sibling.size:
                total_size += sibling.size

    if not has_safetensors or total_size == 0:
        return None

    if total_size >= 1024**3:
        return f"{total_size / (1024 ** 3):.1f}GB"
    if total_size >= 1024**2:
        return f"{total_size / (1024 ** 2):.0f}MB"
    return f"{total_size / 1024:.0f}KB"


def _search_huggingface(query: str, limit: int = 10) -> list[dict[str, Any]] | None:
    """Search HuggingFace for models matching the query."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        console.print(f"Searching HuggingFace for: [cyan]{query}[/cyan]...")

        models = api.list_models(
            search=query,
            sort="downloads",
            limit=limit * 2,
            cardData=False,
        )

        results: list[dict[str, Any]] = []
        for model in models:
            if len(results) >= limit:
                break

            try:
                model_info = api.model_info(model.id, files_metadata=True)

                siblings = model_info.siblings or []
                has_valid_files = any(
                    sibling.rfilename.lower().endswith((".safetensors", ".gguf"))
                    for sibling in siblings
                )

                if has_valid_files:
                    results.append({
                        "id": model_info.id,
                        "downloads": model_info.downloads or 0,
                        "quantization": detect_quantization(model_info.id),
                        "size": _estimate_model_size(siblings) or "?",
                    })
            except Exception as e:
                if "gated" not in str(e).lower():
                    console.print(f"[dim]Skipping {model.id}: {e}[/dim]")
                continue

        return results if results else None

    except ImportError:
        console.print("[bold red]Error:[/bold red] huggingface-hub is required for search.")
        console.print("Install with: `pip install huggingface-hub`")
        return None
    except Exception as e:
        console.print(f"[bold red]Search failed:[/bold red] {e}")
        return None


def _display_search_results(results: list[dict[str, Any]]) -> None:
    """Display search results in a formatted table."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="bold cyan", width=3)
    table.add_column("Model", style="white", width=50)
    table.add_column("Quant", style="yellow", width=8)
    table.add_column("Size", style="green", width=8, justify="right")
    table.add_column("Downloads", style="magenta", width=10, justify="right")

    for i, model in enumerate(results, 1):
        model_style = "white" if i % 2 == 0 else "dim"
        quant_str = model["quantization"] or "BF16"

        table.add_row(
            str(i),
            f"[{model_style}]{model['id']}[/{model_style}]",
            f"[yellow]{quant_str}[/yellow]",
            f"[green]{model['size']}[/green]",
            f"[magenta]{_format_downloads(model['downloads'])}[/magenta]",
        )

    console.print(table)
    console.print()


def _validate_model_exists(model_id: str) -> bool:
    """Validate that a HuggingFace model exists and is accessible."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        console.print(f"Validating model: [cyan]{model_id}[/cyan]...")

        model_info = api.model_info(repo_id=model_id, files_metadata=True)
        siblings = model_info.siblings or []

        has_valid_files = any(
            sibling.rfilename.lower().endswith((".safetensors", ".gguf"))
            for sibling in siblings
        )

        if not has_valid_files:
            console.print("[bold red]Error:[/bold red] Model has no safetensors or GGUF files.")
            console.print("This model may not be compatible with vLLM.")
            return False

        console.print("[green]Model validated successfully![/green]")
        return True

    except Exception as e:
        error_str = str(e).lower()
        if "404" in error_str or "not found" in error_str:
            console.print(f"[bold red]Error:[/bold red] Model '{model_id}' not found on HuggingFace.")
        elif "401" in error_str or "unauthorized" in error_str:
            console.print(f"[bold red]Error:[/bold red] Model '{model_id}' is private or requires authentication.")
        else:
            console.print(f"[bold red]Error validating model:[/bold red] {e}")
        return False


def _update_models_yaml(model_id: str, config: Config) -> bool:
    """Add vLLM model to models.yaml configuration."""
    yaml_path = pathlib.Path(config.MODELS_CONFIG_PATH)
    models_data: dict[str, Any] = {"models": {}}

    try:
        if yaml_path.is_file():
            with open(yaml_path, encoding="utf-8") as f:
                loaded_data = yaml.safe_load(f)
                if isinstance(loaded_data, dict):
                    models_data = loaded_data
                elif loaded_data is not None:
                    console.print(f"[yellow]Warning:[/yellow] Existing content in {yaml_path} is not a dictionary. Overwriting structure.")
        else:
            console.print(f"[yellow]Models file {yaml_path} not found. Creating a new one.[/yellow]")

        if "models" not in models_data or not isinstance(models_data.get("models"), dict):
            models_data["models"] = {}

        # Check for duplicates
        existing_aliases = list(models_data["models"].keys())
        for alias, definition in models_data["models"].items():
            if definition.get("type") == "vllm" and definition.get("model_id") == model_id:
                console.print(f"[yellow]Model definition already exists under alias '[cyan]{alias}[/cyan]'. Not adding duplicate.[/yellow]")
                return True

        # Generate and confirm alias
        suggested_alias = generate_vllm_alias(model_id, existing_aliases)
        confirmed_alias = Prompt.ask(
            f"Proposed alias [cyan]{suggested_alias}[/cyan]. Use this or enter a new one?",
            default=suggested_alias,
        )

        while not confirmed_alias or confirmed_alias in existing_aliases:
            if not confirmed_alias:
                console.print("[yellow]Alias cannot be empty.[/yellow]")
            else:
                console.print(f"[yellow]Alias '[cyan]{confirmed_alias}[/cyan]' already exists.[/yellow]")
            confirmed_alias = Prompt.ask(
                "Please enter a unique alias",
                default=generate_vllm_alias(model_id, existing_aliases + [confirmed_alias]),
            )

        quant = detect_quantization(model_id)

        model_entry: dict[str, Any] = {
            "type": "vllm",
            "model_id": model_id,
            "description": f"{model_id} (vLLM)",
        }

        if quant:
            model_entry["quantization"] = quant

        console.print("\n[dim]Optional: You can add vLLM parameters to models.yaml manually:[/dim]")
        console.print("[dim]  - max_model_len: Maximum context length[/dim]")
        console.print("[dim]  - gpu_memory_utilization: GPU memory fraction (0.0-1.0)[/dim]")
        console.print("[dim]  - dtype: Data type (auto, half, float16, bfloat16)[/dim]")

        models_data["models"][confirmed_alias] = model_entry

        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(models_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

        console.print("\n[green]Success![/green]")
        console.print(f"Added model to [bold]{yaml_path}[/bold]")
        console.print(f"Use with: [cyan]llm --model {confirmed_alias} \"your prompt\"[/cyan]")
        return True

    except yaml.YAMLError as e:
        console.print(f"[bold red]Error processing YAML file {yaml_path}:[/bold red] {e}")
        return False
    except Exception as e:
        console.print(f"[bold red]Unexpected error updating {yaml_path}:[/bold red] {e}")
        return False


def handle_add_vllm(config: Config) -> bool:
    """Add a vLLM model from HuggingFace interactively."""
    if importlib.util.find_spec("huggingface_hub") is None:
        console.print("[bold red]Error:[/bold red] huggingface-hub is required to add vLLM models.")
        console.print("Install with: [cyan]pip install huggingface-hub[/cyan]")
        return False

    try:
        console.print("\n[bold]How would you like to add a vLLM model?[/bold]\n")
        console.print("  [cyan]1[/cyan]  Paste a HuggingFace model ID")
        console.print("  [cyan]2[/cyan]  Search HuggingFace\n")

        choice = Prompt.ask("Choose an option", choices=["1", "2"], default="1")

        model_id: str | None = None

        if choice == "1":
            console.print("\n[bold]Paste mode[/bold]")
            console.print("[dim]Enter a HuggingFace model ID (e.g., Qwen/Qwen2.5-32B-Instruct-AWQ)[/dim]\n")

            model_id = Prompt.ask("Model ID").strip()

            if not model_id:
                console.print("[yellow]No model ID provided. Cancelled.[/yellow]")
                return False

            if not _validate_model_exists(model_id):
                return False

        else:
            console.print("\n[bold]Search mode[/bold]")
            console.print("[dim]Enter search terms (e.g., 'qwen 32b instruct awq')[/dim]\n")

            query = Prompt.ask("Search query").strip()

            if not query:
                console.print("[yellow]No search query provided. Cancelled.[/yellow]")
                return False

            results = _search_huggingface(query, limit=10)

            if not results:
                console.print("[bold red]No matching models found.[/bold red]")
                console.print("[dim]Try a different search query or use paste mode.[/dim]")
                return False

            console.print(f"\n[bold]Found {len(results)} models:[/bold]\n")
            _display_search_results(results)

            while True:
                try:
                    choice_str = Prompt.ask("Enter the number of the model to add", default="1")

                    if not choice_str.isdigit():
                        console.print("[yellow]Invalid input. Please enter a number.[/yellow]")
                        continue

                    choice_int = int(choice_str)
                    if 1 <= choice_int <= len(results):
                        model_id = results[choice_int - 1]["id"]
                        break
                    else:
                        console.print(f"[yellow]Invalid selection. Please enter a number between 1 and {len(results)}.[/yellow]")

                except (ValueError, KeyError):
                    console.print("[yellow]Invalid selection.[/yellow]")
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[red]Selection cancelled.[/red]")
                    return False

        if model_id:
            console.print(f"\n[bold]Adding model:[/bold] [cyan]{model_id}[/cyan]\n")
            return _update_models_yaml(model_id, config)

        return False

    except (EOFError, KeyboardInterrupt):
        console.print("\n\n[red]Operation cancelled.[/red]")
        return False
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")
        return False
