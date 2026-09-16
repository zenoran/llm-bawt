"""Scheduled prompt target validation shared by management and dispatch.

This is capability validation, not authentication: BotManager exposes the same
LAN-trusted registry used by ordinary chat, without a per-user access API.
"""
from __future__ import annotations


def resolve_prompt_target(service, bot_id: str, requested_model: str | None, owner: str) -> str:
    """Return a canonical endpoint ref or raise ValueError for an invalid target.

    Bot defaults are resolved now; callers that persist a run must bind the
    returned ref once, then revalidate that bound ref before claiming a turn.
    """
    from ..bots import BotManager
    from ..model_catalog import bot_model_ref

    if not isinstance(owner, str) or not owner.strip():
        raise ValueError("Scheduled prompts require an owning user")
    if not isinstance(bot_id, str) or not bot_id.strip() or bot_id.strip() == "*":
        raise ValueError("Scheduled prompts require a concrete target bot")
    config = service.config
    bot = BotManager(config).get_bot(bot_id)
    if bot is None or not getattr(bot, "enabled", True):
        raise ValueError("Scheduled prompt target bot is missing or disabled")
    backend = getattr(bot, "agent_backend", None)
    harness = getattr(bot, "harness", None) or backend
    if backend != "claude-code" or harness not in {"claude-code", "claude-proxy"}:
        raise ValueError("Scheduled prompts require Claude Code explicit thread/model isolation")
    catalog = config.ensure_model_catalog()
    if catalog is None:
        raise ValueError("Scheduled prompt model catalog unavailable")
    requested = requested_model if requested_model is not None else bot_model_ref(config, bot)
    if not requested:
        raise ValueError("Scheduled prompt model reference unavailable")
    return catalog.resolve_endpoint(requested, harness=harness).ref
