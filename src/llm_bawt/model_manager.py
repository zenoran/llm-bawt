"""
Manages AI model configurations for the llm-bawt tool.

This module provides functionality to list, update, and delete model aliases
stored in the models configuration file (typically `~/.config/llm-bawt/models.yaml`).

External Usage:
    - Use `list_models(config)` to display available model aliases.
    - Use `delete_model(alias, config)` to remove a specific model alias.
    - Use `update_models_interactive(config, provider=None)` to fetch the latest models
      from providers (OpenAI, Ollama) and interactively add new ones.

The `ModelManager` class can be used directly for more granular control over
model configurations.
"""

import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytz
import yaml
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.rule import Rule

from llm_bawt.utils.config import (
    PROVIDER_GGUF,
    PROVIDER_GROK,
    PROVIDER_HF,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    PROVIDER_UNKNOWN,
)
from .utils.config import Config


def is_service_mode_enabled(config: Config) -> bool:
    """
    Check if service mode is enabled via environment variable or config.
    
    This is the SINGLE source of truth for determining if service mode should be used.
    All CLI commands should use this function to check service mode status.
    
    Returns:
        True if USE_SERVICE is enabled via LLM_BAWT_USE_SERVICE env var or config.USE_SERVICE
    """
    use_service_env = os.getenv("LLM_BAWT_USE_SERVICE", "").lower() in ("true", "1", "yes")
    use_service_config = getattr(config, "USE_SERVICE", False)
    return use_service_env or use_service_config

console = Console()


def normalize_for_match(text: str) -> str:
    """Normalize text for fuzzy matching by lowercasing and removing special characters."""
    return re.sub(r'[^a-z0-9]', '', text.lower())

class _PathWrapper:
    """Wraps a pathlib.Path to delegate file operations while allowing dynamic attributes."""
    __slots__ = ('_path', '__dict__')
    def __init__(self, path: Path):
        self._path = path
    @property
    def parent(self):
        return _PathWrapper(self._path.parent)
    def __getattr__(self, name):
        return getattr(self._path, name)
    def __fspath__(self):  # support os.PathLike
        return str(self._path)
    def __str__(self):
        return str(self._path)
    def __repr__(self):
        return repr(self._path)


class ModelManager:
    """Manages models defined in the configuration file."""

    def __init__(self, config: Config):
        self.config = config
        raw_path = Path(self.config.MODELS_CONFIG_PATH)
        self.config_path = _PathWrapper(raw_path)
        self.models_data: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> bool:
        if not self.config_path.is_file():
            console.print(f"[yellow]Warning:[/yellow] Config file {self.config_path} not found. Assuming empty structure.")
            self.models_data = {"models": {}}
            return True
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_data = yaml.safe_load(f) or {}
            if not isinstance(loaded_data, dict) or any(
                not isinstance(k, str) or not k.isidentifier()
                for k in loaded_data.keys() if k != 'models'
            ):
                raise Exception(f"Invalid format in {self.config_path}")
            if 'models' not in loaded_data or not isinstance(loaded_data.get('models'), dict):
                console.print(f"[yellow]Warning:[/yellow] Invalid format in {self.config_path}. Resetting 'models' dictionary.")
                loaded_data['models'] = {}
            self.models_data = loaded_data
            return True
        except Exception as e:
            console.print(f"[bold red]Error loading config {self.config_path}:[/bold red] {e}")
            self.models_data = {"models": {}}
            return False

    def save_config(self, added: int = 0, updated: int = 0, deleted: int = 0) -> bool:
        console.print(f"\nSaving configuration to {self.config_path}...")
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            defined_models = self.models_data.get('models', {})
            sorted_models = dict(sorted(
                defined_models.items(),
                key=lambda item: (item[1].get('type', PROVIDER_UNKNOWN), item[0])
            ))
            self.models_data['models'] = sorted_models
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.models_data, f, sort_keys=False, indent=2, default_flow_style=False, allow_unicode=True)
            messages = []
            if added > 0:
                messages.append(f"{added} added")
            if updated > 0:
                messages.append(f"{updated} updated")
            if deleted > 0:
                messages.append(f"{deleted} deleted")
            if messages:
                console.print(f"[bold green]Successfully processed config: {', '.join(messages)}.[/bold green]")
            else:
                console.print(f"[green]Config saved to {self.config_path}.[/green]")
            self.config._load_models_config()
            return True
        except Exception as e:
            console.print(f"[bold red]Error saving configuration to {self.config_path}:[/bold red] {e}")
            return False

    def list_available_models(self):
        console.print(f"Available model aliases from [cyan]{self.config_path}[/cyan]:")
        defined_models = self.models_data.get("models", {})
        available_aliases = set(self.config.get_model_options())
        local_aliases = set(defined_models.keys())
        
        # Only check service availability if service mode is enabled
        # This prevents "service not reachable" errors when user isn't using service mode
        use_service = is_service_mode_enabled(self.config)
        service_available = False
        service_models: list[str] = []
        service_url: str | None = None
        
        if use_service:
            try:
                from llm_bawt.service.client import get_service_client

                service_client = get_service_client()
                service_url = service_client.http_url
                if service_client.is_available(force_check=True):
                    service_available = True
                    service_models = service_client.list_models() or []
            except Exception:
                service_available = False
        
        if not defined_models:
            console.print("  [yellow]No models defined in the configuration file.[/yellow]")
        if hasattr(console, 'rule'):
            console.rule("[bold green]Local Models (config)[/bold green]")
        else:
            console.print(Rule("[bold green]Local Models (config)[/bold green]"))
        if not defined_models:
            console.print()
        else:
            models_by_type = defaultdict(list)
            for alias, info in defined_models.items():
                models_by_type[info.get("type", PROVIDER_UNKNOWN)].append(alias)
            for mtype in sorted(models_by_type):
                title = f"[bold magenta]{mtype.upper()} Models[/bold magenta]"
                if hasattr(console, 'rule'):
                    console.rule(title)
                else:
                    console.print(Rule(title))
                for alias in sorted(models_by_type[mtype]):
                    info = defined_models[alias]
                    details = self._format_model_details(mtype, info)
                    marker = "[green]✓[/green]" if alias in available_aliases else "[red]✗[/red]"
                    note = self._get_dependency_note(mtype) if alias not in available_aliases else ""
                    console.print(f"  {marker} [bold][bright_blue]{alias}[/bright_blue][/bold]: {details}{note}")
                console.print()
        
        # Only show service section if service mode is enabled
        if use_service:
            if hasattr(console, 'rule'):
                console.rule("[bold cyan]Service Models[/bold cyan]")
            else:
                console.print(Rule("[bold cyan]Service Models[/bold cyan]"))
            if service_available:
                if service_models:
                    for model_id in sorted(service_models):
                        overlap_note = " [dim](also local config)[/dim]" if model_id in local_aliases else ""
                        console.print(f"  [bold][bright_blue]{model_id}[/bright_blue][/bold]{overlap_note}")
                else:
                    console.print("  [yellow]No models reported by service.[/yellow]")
                if service_url:
                    console.print(f"  [dim]Source: {service_url}[/dim]")
            else:
                console.print("  [yellow]⚠ Service not reachable.[/yellow]")
                if service_url:
                    console.print(f"  [dim]Expected at: {service_url}[/dim]")
            console.print()
        if hasattr(console, 'rule'):
            console.rule(style="#777777")
        else:
            console.print(Rule(style="#777777"))
        console.print("[green]✓[/green] = Dependencies met | [red]✗[/red] = Dependencies missing")
        if self.config.DEFAULT_MODEL_ALIAS:
            status = "[green]✓[/green]" if self.config.DEFAULT_MODEL_ALIAS in available_aliases else "[red]✗[/red]"
            console.print(f"Default alias: {status} [cyan]{self.config.DEFAULT_MODEL_ALIAS}[/cyan]")

    def resolve_model_alias(self, requested_alias: Optional[str]) -> Optional[str]:
        defined = self.models_data.get("models", {})
        available_set = set(self.config.get_model_options())
        if not requested_alias:
            default = self.config.DEFAULT_MODEL_ALIAS
            if default and default in available_set:
                return default
            console.print("[bold red]Error:[/bold red] No model specified or default unavailable.")
            self.list_available_models()
            return None
        norm = normalize_for_match(requested_alias)
        if requested_alias in available_set:
            return requested_alias
        if requested_alias in defined:
            console.print(f"[bold red]Error:[/bold red] '{requested_alias}' defined but unavailable.")
            return None
        matches = [a for a in self.config.get_model_options() if norm in normalize_for_match(a)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return self._prompt_for_match(requested_alias, matches)
        console.print(f"[bold red]Error:[/bold red] Alias '{requested_alias}' not found.")
        self.list_available_models()
        return None

    def delete_model_alias(self, alias: str) -> bool:
        console.print(Rule(f"[bold yellow]Deleting Model Alias: {alias}[/bold yellow]"))
        models = self.models_data.get('models', {})
        if alias not in models:
            console.print(f"[bold red]Error:[/bold red] Alias '{alias}' not found.")
            return False
        info = models[alias]
        confirm = Confirm.ask(f"Delete '{alias}' (type: {info.get('type')})?", default=False)
        if not confirm:
            console.print("[info]Deletion cancelled.[/info]")
            return True
        del models[alias]
        return self.save_config(deleted=1)

    def update_models(self, provider_type: Optional[str] = None):
        targets = [provider_type] if provider_type else [PROVIDER_OPENAI, PROVIDER_OLLAMA]
        success = True # Track overall success
        for prov in targets:
            console.print(Rule(f"Updating models for provider: {prov}"))
            try:
                prov_success = self._update_provider_models(prov)
                if not prov_success:
                    success = False # Mark failure if any provider fails
            except Exception as e:
                console.print(f"[bold red]Error updating {prov}:[/bold red] {e}")
                success = False # Mark failure on exception
        return success

    def _update_provider_models(self, provider_type: str):
        success, api_models = self._fetch_api_models(provider_type)
        if not success:
            console.print(f"[yellow]Skipping {provider_type}, failed to fetch API models.[/yellow]")
            return False

        local_models = self.models_data.get('models', {})
        existing_map = {m['model_id']: alias for alias, m in local_models.items() if m.get('type') == provider_type}
        updated_count = self._update_existing_descriptions(provider_type, api_models, existing_map)
        new_models = [m for m in api_models if m['id'] not in existing_map]

        added_count = 0
        if new_models:
            if provider_type == PROVIDER_OLLAMA:
                console.print(f"[info]Detected {len(new_models)} new Ollama models. Adding automatically...[/info]")
                selected = new_models
                added_count = self._prepare_new_model_entries(provider_type, selected)
            else:
                selected = self._prompt_for_new_models(new_models, provider_type)
                added_count = self._prepare_new_model_entries(provider_type, selected)

        if updated_count or added_count:
            save_ok = self.save_config(added=added_count, updated=updated_count)
            if save_ok:
                self.config._load_models_config()
                if provider_type == PROVIDER_OLLAMA:
                    self.config.force_ollama_check()
            return save_ok
        else:
            console.print("[green]Local configuration is up-to-date.[/green]")
            return True

    def _fetch_api_models(self, provider_type: str) -> Tuple[bool, List[Dict[str, Any]]]:
        if provider_type == PROVIDER_OPENAI:
            return fetch_openai_api_models()
        if provider_type == PROVIDER_OLLAMA:
            ollama_url = getattr(self.config, 'OLLAMA_URL', None)
            if not ollama_url:
                console.print("[bold red]Error:[/bold red] OLLAMA_URL not set in config.")
                return False, []
            return fetch_ollama_api_models(ollama_url)
        console.print(f"[yellow]Provider '{provider_type}' not supported for update.[/yellow]")
        return False, []

    def _update_existing_descriptions(self, provider_type: str, api_models: List[Dict[str, Any]], existing_map: Dict[str, str]) -> int:
        tz = pytz.timezone('US/Eastern')
        count = 0
        for m in api_models:
            mid = m['id']
            if mid in existing_map:
                alias = existing_map[mid]
                entry = self.models_data['models'][alias]
                ts = self._format_model_timestamp_str(provider_type, m, tz)
                if not ts:
                    continue
                label = 'Created' if provider_type == PROVIDER_OPENAI else 'Modified'
                expected = f"{mid} ({label}: {ts})"
                if ts not in entry.get('description', '') or mid not in entry.get('description', ''):
                    entry['description'] = expected
                    count += 1
                    console.print(f"Updated description for alias '{alias}'.")
        return count

    def _prompt_for_new_models(
        self,
        new_models: List[Dict[str, Any]],
        provider_type: str = PROVIDER_OPENAI,
    ) -> List[Dict[str, Any]]:
        tz = pytz.timezone('US/Eastern')
        console.print("New models detected:")
        new_models.sort(key=lambda m: m['id'])
        choices_map = {str(idx + 1): model for idx, model in enumerate(new_models)}

        for idx, m in enumerate(new_models, start=1):
            ts = ModelManager._format_model_timestamp_str(None, provider_type, m, tz)
            model_id_colored = f"[bright_blue]{m['id']}[/bright_blue]"
            console.print(f"  [cyan]{idx}[/cyan]: {model_id_colored} ({ts})")
        sep = ',' if provider_type == PROVIDER_OPENAI else ' '
        prompt = (
            f"Enter numbers to add ({'comma' if sep == ',' else 'space'}-separated), or press Enter to skip"
        )
        resp = Prompt.ask(prompt, default="")
        if not resp.strip():
            return []
        parts = resp.split(sep)
        selected: List[Dict[str, Any]] = []
        for p in parts:
            i = p.strip()
            if i.isdigit() and i in choices_map:
                selected.append(choices_map[i])
        return selected

    def _prepare_new_model_entries(self, provider_type: str, selected_models: List[Dict[str, Any]]) -> int:
        tz = pytz.timezone('US/Eastern')
        existing = set(self.models_data['models'].keys())
        count = 0
        for m in selected_models:
            alias = self._generate_alias(provider_type, m, existing)
            existing.add(alias)
            ts = self._format_model_timestamp_str(provider_type, m, tz)
            label = 'Created' if provider_type == PROVIDER_OPENAI else 'Modified'
            desc = f"{m['id']} ({label}: {ts})"
            entry = {'type': provider_type, 'model_id': m['id'], 'description': desc}
            self.models_data['models'][alias] = entry
            count += 1
            console.print(f"Added new alias '{alias}'.")
        return count

    def _generate_alias(self, provider_type: str, model_info: Dict[str, Any], existing_aliases: set) -> str:
        base = model_info['id'].lower().replace('/', '-').replace(':', '-')
        alias = base
        i = 1
        while alias in existing_aliases:
            alias = f"{base}{i}"
            i += 1
        return alias

    def _format_model_timestamp_str(self, provider_type: str, model_info: Dict[str, Any], tz: Any) -> Optional[str]:
        dt = model_info.get('created') if provider_type == PROVIDER_OPENAI else model_info.get('modified_at')
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        try:
            local = dt.astimezone(tz)
            return local.strftime("%Y-%m-%d %H:%M %Z")
        except Exception:
            return None

    def _format_model_details(self, model_type: str, model_info: Dict[str, Any]) -> str:
        parts = []
        mid = None

        if model_type == PROVIDER_GGUF:
            repo = model_info.get('repo_id')
            filename = model_info.get('filename')
            desc = model_info.get('description')
            chat_format = model_info.get('chat_format')
            # Avoid duplicating repo info if already in description
            if repo and (not desc or repo not in desc):
                parts.append(f"Repo: {repo}")
            if chat_format:
                parts.append(f"Format: {chat_format}")
            if desc:
                parts.append(desc)
            # Add file size if the file exists
            if repo and filename:
                file_size = self._get_gguf_file_size(repo, filename)
                if file_size:
                    parts.append(f"[dim]{file_size}[/dim]")

        elif model_type in (PROVIDER_HF, PROVIDER_OLLAMA, PROVIDER_OPENAI, PROVIDER_GROK):
            mid = model_info.get('model_id')
            desc = model_info.get('description')
            formatted_id_str = ""
            raw_id_str = ""
            processed_desc = desc  # Work with a modifiable copy
            if processed_desc:
                provider_prefix = f"{model_type.capitalize()} "
                if processed_desc.lower().startswith(provider_prefix.lower()):
                    processed_desc = processed_desc[len(provider_prefix):].strip()
            if mid:
                formatted_id_str = f"[bright_blue]{mid}[/bright_blue]"
                raw_id_str = mid # Keep raw ID for matching
            if processed_desc and raw_id_str and raw_id_str in processed_desc:
                escaped_raw_id = re.escape(raw_id_str)
                final_desc = re.sub(rf'\\b{escaped_raw_id}\\b', formatted_id_str, processed_desc, count=1)
                parts = [final_desc]
            else:
                if mid:
                    parts.append(f"ID: {formatted_id_str}")
                if processed_desc: # Use potentially prefix-removed description
                    parts.append(processed_desc)

        return ", ".join(parts) or "No details"

    def _get_gguf_file_size(self, repo_id: str, filename: str) -> Optional[str]:
        """Get the file size of a GGUF model file if it exists locally."""
        from pathlib import Path
        
        # Check in the model cache directory
        file_path = Path(self.config.MODEL_CACHE_DIR) / repo_id / filename
        if not file_path.exists():
            return None
        
        try:
            size_bytes = file_path.stat().st_size
            return self._format_size(size_bytes)
        except Exception:
            return None
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format bytes as human-readable size (e.g., 14.5G, 3.2M)."""
        size = float(size_bytes)
        for unit in ['B', 'K', 'M', 'G', 'T']:
            if abs(size) < 1024:
                if unit == 'B':
                    return f"{int(size)}{unit}"
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}P"

    @staticmethod
    def _get_dependency_note(model_type: str) -> str:
        """Return a note about missing dependencies for the given model type."""
        if model_type == PROVIDER_HF:
            return " (Requires: [dim]llm-bawt[huggingface][/dim])"
        if model_type == PROVIDER_GGUF:
            return " (Requires: [dim]llama-cpp-python, huggingface-hub[/dim])"
        return ''

    def _prompt_for_match(self, requested_alias: str, matches: List[str]) -> Optional[str]:
        console.print(f"Multiple matches for '{requested_alias}':")
        choices = {str(i+1): m for i, m in enumerate(matches)}
        for k, m in choices.items():
            console.print(f"  [cyan]{k}[/cyan]: {m}")
        choice = Prompt.ask("Enter number", choices=list(choices.keys()), default="1")
        return choices.get(choice)


def fetch_ollama_api_models(ollama_url: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """Fetches model list from Ollama API."""
    import requests # Import lazily
    models_details = []
    start = time.time()
    endpoint = f"{ollama_url.rstrip('/')}/api/tags"
    try:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data.get('models'), list):
            for d in data['models']:
                mid = d.get('name')
                mat = d.get('modified_at')
                mod_dt = None
                if mat:
                    try:
                        mod_dt = _parse_iso(mat)
                    except Exception:
                        console.print(f"[yellow]Warning:[/] Could not parse timestamp {mat}")
                models_details.append({'id': mid, 'modified_at': mod_dt})
        console.print(f"⏱️ Ollama query took {(time.time()-start)*1000:.0f}ms, found {len(models_details)} models")
        return True, models_details
    except Exception as e:
        console.print(f"[bold red]Ollama API error:[/bold red] {e}")
        return False, []

def fetch_openai_api_models() -> Tuple[bool, List[Dict[str, Any]]]:
    """Fetches model list from OpenAI API."""
    from openai import OpenAI # Import lazily
    from openai import APIConnectionError, AuthenticationError, RateLimitError
    details = []
    start = time.time()
    try:
        client = OpenAI()
        client.models.list()
    except Exception as e:
        console.print(f"[bold red]OpenAI init error:[/bold red] {e}")
        return False, []
    try:
        resp = client.models.list()
        for m in resp.data:
            if 'gpt' in m.id or m.owned_by in ('openai', 'system'):
                ts = getattr(m, 'created', None)
                dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
                details.append({'id': m.id, 'created': dt})
        console.print(f"⏱️ OpenAI query took {(time.time()-start)*1000:.0f}ms, found {len(details)} models")
        return True, details
    except Exception as e:
        console.print(f"[bold red]OpenAI fetch error:[/bold red] {e}")
        return False, []


def list_models(config: Config):
    """CLI wrapper: list models using ModelManager, with fallback for tests."""
    try:
        manager = ModelManager(config)
    except Exception:
        manager = ModelManager.__new__(ModelManager)
        manager.config = config
    manager.list_available_models()

def delete_model(alias: str, config: Config) -> bool:
    manager = ModelManager(config)
    # Get model info before deletion from config
    model_info = manager.models_data.get('models', {}).get(alias)
    
    delete_ok = manager.delete_model_alias(alias)
    if delete_ok:
        config._load_models_config()
        # Avoid probing Ollama during unrelated operations.
        # Ollama connectivity/model checks happen only when explicitly refreshing Ollama models.
        
        # If the model is a GGUF model, check if we should delete the model files as well
        if model_info and model_info.get('type') == PROVIDER_GGUF:
            repo_id = model_info.get('repo_id')
            if repo_id:
                from pathlib import Path
                import shutil
                
                # Get the entire repo directory path
                repo_dir = Path(config.MODEL_CACHE_DIR) / repo_id
                
                if repo_dir.exists():
                    console.print(f"\nFound model directory at: [cyan]{repo_dir}[/cyan]")
                    confirm = Confirm.ask("Do you want to delete the entire model directory?", default=False)
                    if confirm:
                        try:
                            # Delete the entire directory tree
                            shutil.rmtree(repo_dir)
                            console.print(f"[green]Successfully deleted model directory: {repo_dir}[/green]")
                        except Exception as e:
                            console.print(f"[bold red]Error deleting model directory:[/bold red] {e}")
    
    return delete_ok

def update_models_interactive(config: Config, provider: Optional[str] = None):
    return ModelManager(config).update_models(provider)


def _parse_iso(ts_str: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except Exception:
        return None
