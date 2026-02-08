from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple
from urllib.parse import urlparse, urlunparse, quote_plus
import urllib.request
import urllib.error

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from llm_bawt.utils.config import Config, set_config_value
from llm_bawt.utils.env import set_env_value


@dataclass(frozen=True)
class OpenAIEndpoint:
    base_url: Optional[str]
    api_key: Optional[str]


def _normalize_openai_models_url(base_url: Optional[str]) -> str:
    if not base_url:
        return "https://api.openai.com/v1/models"

    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/models"):
        models_path = path
    elif path.endswith("/v1"):
        models_path = f"{path}/models"
    else:
        models_path = f"{path}/v1/models"

    normalized = parsed._replace(path=models_path)
    return urlunparse(normalized)


def _http_get_json(url: str, headers: dict[str, str] | None = None, timeout: float = 5.0) -> Tuple[bool, str]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return True, f"{resp.status} {resp.reason}"
            return False, f"{resp.status} {resp.reason}"
    except urllib.error.HTTPError as e:
        return False, f"{e.code} {e.reason}"
    except Exception as e:
        return False, str(e)


def _test_openai_endpoint(endpoint: OpenAIEndpoint) -> Tuple[bool, str]:
    url = _normalize_openai_models_url(endpoint.base_url)
    headers = {}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    return _http_get_json(url, headers=headers)


def _test_ollama(url: str) -> Tuple[bool, str]:
    clean = url.rstrip("/")
    return _http_get_json(f"{clean}/api/tags")


def _test_postgres(config: Config) -> Tuple[bool, str]:
    try:
        from sqlalchemy import create_engine, text
    except Exception as e:
        return False, f"sqlalchemy unavailable: {e}"

    password = getattr(config, "POSTGRES_PASSWORD", "")
    if not password:
        return False, "missing password"

    host = getattr(config, "POSTGRES_HOST", "localhost")
    port = int(getattr(config, "POSTGRES_PORT", 5432))
    user = getattr(config, "POSTGRES_USER", "llm_bawt")
    database = getattr(config, "POSTGRES_DATABASE", "llm_bawt")
    encoded_password = quote_plus(password)
    connection_url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{database}"

    try:
        engine = create_engine(connection_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "connected"
    except Exception as e:
        return False, str(e)


def _test_service(config: Config) -> Tuple[bool, str]:
    try:
        from llm_bawt.service import ServiceClient
    except Exception as e:
        return False, f"service client unavailable: {e}"

    http_url = f"http://{config.SERVICE_HOST}:{config.SERVICE_PORT}"
    try:
        client = ServiceClient(http_url=http_url)
        if not client.is_available(force_check=True):
            return False, "not reachable"
        status = client.get_status(silent=True)
        if getattr(status, "available", False):
            return True, "available"
        return False, "unhealthy"
    except Exception as e:
        return False, str(e)


def _openai_endpoints_from_models(config: Config, fallback_key: Optional[str]) -> list[OpenAIEndpoint]:
    endpoints: list[OpenAIEndpoint] = []
    models = config.defined_models.get("models", {})
    for model_info in models.values():
        if model_info.get("type") != "openai":
            continue
        base_url = model_info.get("base_url")
        api_key = model_info.get("api_key") or fallback_key
        endpoints.append(OpenAIEndpoint(base_url=base_url, api_key=api_key))
    return endpoints


def _unique_openai_endpoints(endpoints: Iterable[OpenAIEndpoint]) -> list[OpenAIEndpoint]:
    seen: set[tuple[Optional[str], Optional[str]]] = set()
    unique: list[OpenAIEndpoint] = []
    for endpoint in endpoints:
        key = (endpoint.base_url, endpoint.api_key)
        if key in seen:
            continue
        seen.add(key)
        unique.append(endpoint)
    return unique


def _needs_openai_key(config: Config) -> bool:
    models = config.defined_models.get("models", {})
    for model_info in models.values():
        if model_info.get("type") != "openai":
            continue
        if model_info.get("api_key"):
            continue
        if model_info.get("base_url"):
            continue
        return True
    return False


def _prompt_and_set_config(
    config: Config,
    key: str,
    value: str,
    console: Console,
) -> bool:
    if not set_config_value(key, value, config):
        console.print(f"[red]Failed to save {key} to config.[/red]")
        return False
    setattr(config, key, value)
    return True


def run_config_wizard(config: Config, console: Console) -> int:
    """Interactive config setup with connectivity checks."""
    env_path = Path(config.model_config["env_file"])
    console.print(Panel.fit("llm-bawt Config Wizard", border_style="cyan"))

    updated = False
    env_openai_key = os.getenv("OPENAI_API_KEY")

    if _needs_openai_key(config) and not env_openai_key:
        console.print("[yellow]OpenAI API key is required for OpenAI models.[/yellow]")
        key = Prompt.ask("Enter OPENAI_API_KEY", password=True)
        if key.strip():
            if set_env_value(env_path, "OPENAI_API_KEY", key.strip()):
                os.environ["OPENAI_API_KEY"] = key.strip()
                env_openai_key = key.strip()
                updated = True
                console.print("[green]Saved OPENAI_API_KEY to .env[/green]")
            else:
                console.print("[red]Failed to save OPENAI_API_KEY to .env[/red]")

    openai_endpoints = _unique_openai_endpoints(_openai_endpoints_from_models(config, env_openai_key))
    if openai_endpoints:
        console.print("\n[bold]OpenAI-compatible endpoints[/bold]")
        for endpoint in openai_endpoints:
            ok, msg = _test_openai_endpoint(endpoint)
            label = endpoint.base_url or "https://api.openai.com"
            if ok:
                console.print(f"[green]✓ {label} ({msg})[/green]")
            else:
                console.print(f"[red]✗ {label} ({msg})[/red]")

    models = config.defined_models.get("models", {})
    has_ollama = any(info.get("type") == "ollama" for info in models.values())
    if has_ollama:
        console.print("\n[bold]Ollama[/bold]")
        ok, msg = _test_ollama(config.OLLAMA_URL)
        if ok:
            console.print(f"[green]✓ {config.OLLAMA_URL} ({msg})[/green]")
        else:
            console.print(f"[red]✗ {config.OLLAMA_URL} ({msg})[/red]")
            if Confirm.ask("Update OLLAMA_URL and retry?", default=False):
                new_url = Prompt.ask("OLLAMA_URL", default=config.OLLAMA_URL)
                if new_url.strip():
                    if _prompt_and_set_config(config, "OLLAMA_URL", new_url.strip(), console):
                        updated = True
                        ok, msg = _test_ollama(config.OLLAMA_URL)
                        status = "✓" if ok else "✗"
                        color = "green" if ok else "red"
                        console.print(f"[{color}]{status} {config.OLLAMA_URL} ({msg})[/{color}]")

    use_service_env = os.getenv("LLM_BAWT_USE_SERVICE", "").lower() in ("true", "1", "yes")
    use_service = use_service_env or config.USE_SERVICE
    if use_service:
        console.print("\n[bold]Background service[/bold]")
        ok, msg = _test_service(config)
        if ok:
            console.print(f"[green]✓ http://{config.SERVICE_HOST}:{config.SERVICE_PORT} ({msg})[/green]")
        else:
            console.print(f"[red]✗ http://{config.SERVICE_HOST}:{config.SERVICE_PORT} ({msg})[/red]")
            if Confirm.ask("Update SERVICE_HOST/SERVICE_PORT and retry?", default=False):
                host = Prompt.ask("SERVICE_HOST", default=str(config.SERVICE_HOST))
                port = Prompt.ask("SERVICE_PORT", default=str(config.SERVICE_PORT))
                if host.strip():
                    updated = _prompt_and_set_config(config, "SERVICE_HOST", host.strip(), console) or updated
                if port.strip():
                    updated = _prompt_and_set_config(config, "SERVICE_PORT", port.strip(), console) or updated
                ok, msg = _test_service(config)
                status = "✓" if ok else "✗"
                color = "green" if ok else "red"
                console.print(f"[{color}]{status} http://{config.SERVICE_HOST}:{config.SERVICE_PORT} ({msg})[/{color}]")

    console.print("\n[bold]PostgreSQL memory backend[/bold]")
    if not config.POSTGRES_PASSWORD:
        if Confirm.ask("Configure PostgreSQL credentials?", default=True):
            password = Prompt.ask("POSTGRES_PASSWORD", password=True)
            if password.strip():
                if _prompt_and_set_config(config, "POSTGRES_PASSWORD", password.strip(), console):
                    updated = True
    if config.POSTGRES_PASSWORD:
        ok, msg = _test_postgres(config)
        if ok:
            console.print(f"[green]✓ Connected ({msg})[/green]")
        else:
            console.print(f"[red]✗ Connection failed ({msg})[/red]")
            if Confirm.ask("Edit PostgreSQL connection settings and retry?", default=False):
                host = Prompt.ask("POSTGRES_HOST", default=str(config.POSTGRES_HOST))
                port = Prompt.ask("POSTGRES_PORT", default=str(config.POSTGRES_PORT))
                user = Prompt.ask("POSTGRES_USER", default=str(config.POSTGRES_USER))
                database = Prompt.ask("POSTGRES_DATABASE", default=str(config.POSTGRES_DATABASE))
                password = Prompt.ask("POSTGRES_PASSWORD", password=True)
                if host.strip():
                    updated = _prompt_and_set_config(config, "POSTGRES_HOST", host.strip(), console) or updated
                if port.strip():
                    updated = _prompt_and_set_config(config, "POSTGRES_PORT", port.strip(), console) or updated
                if user.strip():
                    updated = _prompt_and_set_config(config, "POSTGRES_USER", user.strip(), console) or updated
                if database.strip():
                    updated = _prompt_and_set_config(config, "POSTGRES_DATABASE", database.strip(), console) or updated
                if password.strip():
                    updated = _prompt_and_set_config(config, "POSTGRES_PASSWORD", password.strip(), console) or updated
                ok, msg = _test_postgres(config)
                status = "✓" if ok else "✗"
                color = "green" if ok else "red"
                console.print(f"[{color}]{status} PostgreSQL ({msg})[/{color}]")

    if updated:
        console.print(f"\n[dim]Updated config file: {env_path}[/dim]")
    return 0
