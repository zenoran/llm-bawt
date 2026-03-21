"""Interactive workflows for creating bot profiles."""

from __future__ import annotations

from rich.console import Console
from rich.prompt import Confirm, Prompt

from ..service.client import get_service_client
from ..utils.config import Config

console = Console()


def handle_add_chat_bot(config: Config) -> bool:
    """Interactive wizard for creating a chat bot profile."""
    console.print("\n[bold magenta]Add Chat Bot[/bold magenta]")

    service_client = get_service_client(config)
    if not service_client or not service_client.is_available(force_check=True):
        console.print("[yellow]Service not available — bot not created. Start the service first.[/yellow]")
        return False

    bot_slug = Prompt.ask("Bot slug", default="my-bot").strip().lower()
    if not bot_slug:
        console.print("[red]Bot slug is required.[/red]")
        return False

    bot_name = Prompt.ask("Bot display name", default=bot_slug.replace("-", " ").title()).strip()
    description = Prompt.ask("Description", default="Custom chat bot").strip()
    system_prompt = Prompt.ask(
        "System prompt",
        default="You are a helpful assistant.",
    ).strip()
    default_model = Prompt.ask(
        "Default model alias (leave blank for bot/global default)",
        default="",
    ).strip()

    requires_memory = Confirm.ask("Enable memory?", default=True)
    uses_tools = Confirm.ask("Enable tool use?", default=True)
    uses_search = Confirm.ask("Enable web search?", default=uses_tools)
    uses_home_assistant = Confirm.ask("Enable Home Assistant tools?", default=False)

    payload = {
        "slug": bot_slug,
        "name": bot_name,
        "description": description,
        "system_prompt": system_prompt,
        "requires_memory": requires_memory,
        "uses_tools": uses_tools,
        "uses_search": uses_search,
        "uses_home_assistant": uses_home_assistant,
        "bot_type": "chat",
    }
    if default_model:
        payload["default_model"] = default_model

    result = service_client.create_bot(payload)
    if not result:
        detail = getattr(service_client, "last_error", None)
        if detail:
            console.print(f"[red]Could not create bot:[/red] {detail}")
        else:
            console.print("[red]Could not create bot.[/red]")
        return False

    console.print(f"[green]Created chat bot '{bot_slug}'[/green]")
    detail = getattr(service_client, "last_error", None)
    if detail and "legacy PUT" in detail:
        console.print(f"[yellow]{detail}[/yellow]")
    console.print(f"[dim]Usage: llm -b {bot_slug} \"your message\"[/dim]")
    return True
