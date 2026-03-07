"""Handler for adding OpenClaw bot profiles."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib import request as urlrequest
from urllib import error as urlerror

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from ..service.client import get_service_client
from ..utils.config import Config

console = Console()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    text = value.strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "on"}


def _read_dotenv_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        if not path.is_file():
            return values
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, val = text.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key:
                values[key] = val
    except Exception:
        return {}
    return values


def _load_openclaw_env_defaults(config: Config) -> dict[str, str]:
    defaults: dict[str, str] = {}

    env_file = config.model_config.get("env_file")
    env_candidates: list[Path] = []
    if isinstance(env_file, (str, Path)):
        env_candidates.append(Path(env_file))
    elif isinstance(env_file, (list, tuple)):
        env_candidates.extend(Path(p) for p in env_file if p)

    # Always include user config dotenv as explicit fallback.
    env_candidates.append(Path.home() / ".config" / "llm-bawt" / ".env")

    seen: set[Path] = set()
    for candidate in env_candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        defaults.update(_read_dotenv_values(candidate))

    return defaults


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "openclaw"


def _call_tools_invoke(base_url: str, token: str, tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
    payload = {"tool": tool, "args": args}
    req = urlrequest.Request(
        f"{base_url.rstrip('/')}/tools/invoke",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlrequest.urlopen(req, timeout=12) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def _fetch_sessions(base_url: str, token: str) -> list[dict[str, Any]]:
    data = _call_tools_invoke(
        base_url=base_url,
        token=token,
        tool="sessions_list",
        args={"limit": 30, "messageLimit": 0},
    )
    if not data:
        return []
    return (data.get("result", {}).get("details", {}) or {}).get("sessions", []) or []


def _fetch_sessions_via_ssh(base_url: str, token: str, ssh_user: str) -> list[dict[str, Any]]:
    """Fetch full sessions list via remote `openclaw gateway call sessions.list`.

    This path is more complete than HTTP `/tools/invoke` in some gateway policy
    configurations and includes channel sessions (e.g. nextcloud-talk groups).
    """
    host = urlparse(base_url).hostname
    if not host:
        return []
    remote_script = (
        "python3 - <<'PY'\n"
        "import json, subprocess\n"
        f"token={token!r}\n"
        "cmd=['openclaw','gateway','call','sessions.list','--json','--timeout','8000']\n"
        "if token:\n"
        "    cmd.extend(['--token', token])\n"
        "p=subprocess.run(cmd, capture_output=True, text=True)\n"
        "if p.returncode != 0 or not p.stdout.strip():\n"
        "    print('[]')\n"
        "    raise SystemExit(0)\n"
        "obj=json.loads(p.stdout)\n"
        "print(json.dumps(obj.get('sessions', [])))\n"
        "PY"
    )
    try:
        proc = subprocess.run(
            ["ssh", f"{ssh_user}@{host}", remote_script],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout.strip() or "[]")
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _fetch_channels_via_cli() -> list[dict[str, Any]]:
    """Get channels from `openclaw gateway call channels.status --json` when available."""
    try:
        proc = subprocess.run(
            ["openclaw", "gateway", "call", "channels.status", "--json", "--timeout", "8000"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout.strip() or "{}")
        channel_meta = data.get("channelMeta") or []
        if isinstance(channel_meta, list):
            return [item for item in channel_meta if isinstance(item, dict)]
        return []
    except Exception:
        return []


def _fetch_channels_via_ssh(base_url: str, token: str, ssh_user: str) -> list[dict[str, Any]]:
    """Fetch channel metadata via remote `openclaw gateway call channels.status`."""
    host = urlparse(base_url).hostname
    if not host:
        return []
    remote_script = (
        "python3 - <<'PY'\n"
        "import json, subprocess\n"
        f"token={token!r}\n"
        "cmd=['openclaw','gateway','call','channels.status','--json','--timeout','8000']\n"
        "if token:\n"
        "    cmd.extend(['--token', token])\n"
        "p=subprocess.run(cmd, capture_output=True, text=True)\n"
        "if p.returncode != 0 or not p.stdout.strip():\n"
        "    print('[]')\n"
        "    raise SystemExit(0)\n"
        "obj=json.loads(p.stdout)\n"
        "print(json.dumps(obj.get('channelMeta', [])))\n"
        "PY"
    )
    try:
        proc = subprocess.run(
            ["ssh", f"{ssh_user}@{host}", remote_script],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout.strip() or "[]")
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _check_gateway_health(base_url: str, token: str) -> bool:
    req = urlrequest.Request(
        f"{base_url.rstrip('/')}/health",
        method="GET",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urlrequest.urlopen(req, timeout=8) as resp:
            return int(getattr(resp, "status", 0)) < 500
    except urlerror.HTTPError as e:
        return e.code < 500
    except Exception:
        return False


def handle_add_openclaw(config: Config) -> bool:
    """Interactive wizard for creating OpenClaw bot profiles."""
    console.print("\n[bold magenta]Add OpenClaw Bot[/bold magenta]")
    console.print(
        "[dim]Session key controls memory/context continuity; channel is transport metadata.[/dim]"
    )

    env_defaults = _load_openclaw_env_defaults(config)

    default_url = (
        os.getenv("OPENCLAW_GATEWAY_URL", "").strip()
        or env_defaults.get("OPENCLAW_GATEWAY_URL", "").strip()
        or "http://127.0.0.1:18789"
    )
    gateway_url = default_url.rstrip("/")

    token_env_default = (
        os.getenv("OPENCLAW_GATEWAY_TOKEN_ENV", "").strip()
        or env_defaults.get("OPENCLAW_GATEWAY_TOKEN_ENV", "").strip()
        or "OPENCLAW_GATEWAY_TOKEN"
    )
    token_env = token_env_default
    token_value = os.getenv(token_env, "").strip() or env_defaults.get(token_env, "").strip()

    # Prompt for config details only when defaults are missing or invalid.
    needs_config_prompt = not gateway_url or not token_value
    if not needs_config_prompt:
        if not _check_gateway_health(gateway_url, token_value):
            console.print("[yellow]Configured OpenClaw gateway settings failed health/auth check.[/yellow]")
            needs_config_prompt = True

    if needs_config_prompt:
        gateway_url = Prompt.ask("OpenClaw gateway URL", default=default_url).strip().rstrip("/")
        token_env = Prompt.ask("Token env var name", default=token_env_default).strip()
        token_value = os.getenv(token_env, "").strip() or env_defaults.get(token_env, "").strip()

    if not token_value:
        console.print(f"[yellow]Env var {token_env} is not set.[/yellow]")
        token_value = Prompt.ask("Gateway token (used now for discovery; not stored)", password=True).strip()

    if not token_value:
        console.print("[red]Token is required to discover sessions/channels.[/red]")
        return False

    if not _check_gateway_health(gateway_url, token_value):
        console.print("[red]Gateway not reachable/authenticated at that URL.[/red]")
        return False

    ssh_user = (
        os.getenv("OPENCLAW_SSH_USER", "").strip()
        or env_defaults.get("OPENCLAW_SSH_USER", "").strip()
        or "vex"
    )

    discovery_mode = (
        os.getenv("OPENCLAW_DISCOVERY_MODE", "").strip().lower()
        or env_defaults.get("OPENCLAW_DISCOVERY_MODE", "").strip().lower()
        or "hybrid"
    )
    use_ssh_fallback = discovery_mode != "api"
    if discovery_mode not in {"api", "hybrid"}:
        discovery_mode = "hybrid"

    # Backward-compatible override flag.
    if "OPENCLAW_USE_SSH_FALLBACK" in os.environ or "OPENCLAW_USE_SSH_FALLBACK" in env_defaults:
        raw = os.getenv("OPENCLAW_USE_SSH_FALLBACK")
        if raw is None:
            raw = env_defaults.get("OPENCLAW_USE_SSH_FALLBACK")
        use_ssh_fallback = _as_bool(raw, default=use_ssh_fallback)

    console.print(
        f"[dim]Discovery mode: {'API + SSH fallback' if use_ssh_fallback else 'API-only'}[/dim]"
    )

    channels = _fetch_channels_via_cli()
    if not channels and use_ssh_fallback:
        console.print(f"[dim]Using SSH fallback for channel discovery via {ssh_user}@{urlparse(gateway_url).hostname}[/dim]")
        channels = _fetch_channels_via_ssh(gateway_url, token_value, ssh_user)

    sessions = _fetch_sessions(gateway_url, token_value)
    if len(sessions) <= 1 and use_ssh_fallback:
        console.print(f"[dim]Using SSH fallback for full sessions list via {ssh_user}@{urlparse(gateway_url).hostname}[/dim]")
        ssh_sessions = _fetch_sessions_via_ssh(gateway_url, token_value, ssh_user)
        if len(ssh_sessions) > len(sessions):
            sessions = ssh_sessions

    if channels:
        table = Table(title="OpenClaw channels", show_header=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("id", style="green")
        table.add_column("label", style="white")
        for idx, item in enumerate(channels, 1):
            table.add_row(str(idx), str(item.get("id", "")), str(item.get("label", "")))
        console.print(table)

    if sessions:
        table = Table(title="Recent sessions", show_header=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("session key", style="green")
        table.add_column("channel", style="white")
        table.add_column("model", style="magenta")
        for idx, item in enumerate(sessions, 1):
            table.add_row(
                str(idx),
                str(item.get("key", "")),
                str(item.get("channel", "")),
                str(item.get("model", "")),
            )
        console.print(table)

    # Session selection: pick from list, enter custom key, or create new
    session_mode_choices = []
    if sessions:
        session_mode_choices.append("list")
    session_mode_choices.extend(["custom", "new"])
    default_mode = "list" if sessions else "custom"

    if len(session_mode_choices) == 1:
        session_mode = session_mode_choices[0]
    else:
        labels = []
        if "list" in session_mode_choices:
            labels.append("[bold]list[/bold] (pick from above)")
        labels.append("[bold]custom[/bold] (enter session key)")
        labels.append("[bold]new[/bold] (create agent:id:slug)")
        console.print("Session mode: " + ", ".join(labels))
        session_mode = Prompt.ask(
            "Mode",
            choices=session_mode_choices,
            default=default_mode,
        ).strip()

    selected_session_key = ""
    agent_id = "main"

    if session_mode == "list" and sessions:
        choices = [str(i) for i in range(1, len(sessions) + 1)]
        selected = Prompt.ask("Pick session #", choices=choices, default="1")
        row = sessions[int(selected) - 1]
        selected_session_key = str(row.get("key", "")).strip()
        parts = selected_session_key.split(":")
        if len(parts) >= 2 and parts[0] == "agent":
            agent_id = parts[1]
    elif session_mode == "custom":
        selected_session_key = Prompt.ask(
            "Session key (e.g. agent:codebitch:main)",
        ).strip()
        if not selected_session_key:
            console.print("[red]Session key is required.[/red]")
            return False
        parts = selected_session_key.split(":")
        if len(parts) >= 2 and parts[0] == "agent":
            agent_id = parts[1]
    else:
        agent_id = Prompt.ask("OpenClaw agent id", default="main").strip() or "main"
        suggested_slug = "llm-bawt-channel"
        if channels:
            first_channel = str(channels[0].get("id", "")).strip()
            if first_channel:
                suggested_slug = f"{first_channel}:llm-bawt"
        session_slug = Prompt.ask(
            "Session/channel slug (suffix after agent:<id>:)",
            default=suggested_slug,
        ).strip()
        selected_session_key = f"agent:{agent_id}:{session_slug}"

    # --- Create bot profile ---
    default_bot_slug = _slugify(agent_id) if agent_id != "main" else _slugify(selected_session_key)
    bot_slug = Prompt.ask("Bot slug", default=default_bot_slug).strip().lower()
    if not bot_slug:
        console.print("[red]Bot slug is required.[/red]")
        return False

    bot_name = Prompt.ask("Bot display name", default=bot_slug.replace("-", " ").title()).strip()
    bot_system_prompt = Prompt.ask(
        "System prompt (leave blank for none)",
        default="",
    ).strip()

    bot_payload = {
        "name": bot_name,
        "description": f"OpenClaw agent ({selected_session_key})",
        "system_prompt": bot_system_prompt,
        "requires_memory": True,
        "agent_backend": "openclaw",
        "agent_backend_config": {
            "session_key": selected_session_key,
            "timeout_seconds": 600,
        },
    }

    try:
        service_client = get_service_client(config)
        if service_client and service_client.is_available(force_check=True):
            result = service_client.upsert_bot_profile(bot_slug, bot_payload)
            if result:
                console.print(f"[green]Created OpenClaw bot '{bot_slug}'[/green]")
                console.print(f"[dim]Session key: {selected_session_key}[/dim]")
                console.print(f"[dim]Usage: llm -b {bot_slug} \"your message\"[/dim]")

                # Refresh service model catalog to register the virtual model
                try:
                    reloaded = service_client.reload_models()
                    if isinstance(reloaded, dict) and reloaded.get("ok"):
                        console.print("[dim]Refreshed service model catalog.[/dim]")
                except Exception:
                    pass
            else:
                console.print("[yellow]Service did not confirm bot creation.[/yellow]")
                return False
        else:
            console.print("[yellow]Service not available — bot not created. Start the service first.[/yellow]")
            return False
    except Exception as e:
        console.print(f"[red]Could not create bot:[/red] {e}")
        return False

    return True
