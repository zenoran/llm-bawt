"""MCP tool for agent self-recap.

Registers ``self_recap`` on the shared FastMCP server. When an agent (or the
user, via the agent) asks for a "self recap", this:

  1. Pulls the agent's last N hours of conversation (sent + received).
  2. Ships that transcript to Grok with a structured handoff prompt.
  3. Stores the result as a ``role='summary'`` record (the existing summary
     system) so it survives the session.
  4. Returns the recap text — which, as an MCP tool result, lands in the
     agent's SDK transcript and is replayed into context on the next
     ``resume=`` turn. That is the injection: we lean on the SDK's own
     transcript mechanism rather than fighting it.

Imported by server.py to trigger tool registration on startup.

The prompt has two co-equal priorities: an EXHAUSTIVE itemized TOPIC LOG (every
topic discussed, done or not) and the PENDING / INCOMPLETE work a cold-context
agent must finish. Claimed-vs-verified is enforced throughout.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone

from .recap_prompt import RECAP_SYSTEM_PROMPT
from .server import mcp

logger = logging.getLogger(__name__)

# Default Grok model for the recap. Override with LLM_BAWT_SELF_RECAP_MODEL.
# Must be a real xAI API model_id (see `model_definitions` rows, type='grok').
_DEFAULT_RECAP_MODEL = "grok-4-fast-non-reasoning"

# Roles that represent real conversation bubbles (what the user sees in the app).
# Everything else — notably role='summary' (prior recaps) — is excluded so the
# recap summarizes the actual transcript, not a pile of old summaries.
_RECAP_ROLES = {"user", "assistant"}

# Cap on transcript size shipped to Grok. Oldest messages are dropped first if
# exceeded, and the drop is reported (never silent) so the recap can't silently
# claim to have seen the whole window. Default 600k chars (~150k tokens) keeps a
# busy 24h intact while staying well under grok-4-fast's context. Override with
# LLM_BAWT_SELF_RECAP_MAX_CHARS for very heavy windows.
try:
    _MAX_TRANSCRIPT_CHARS = int(os.getenv("LLM_BAWT_SELF_RECAP_MAX_CHARS", "600000"))
except ValueError:
    _MAX_TRANSCRIPT_CHARS = 600_000


# The recap system prompt lives in the prompt registry (key 'self_recap.system')
# so it is editable from the prompts API/UI with version history, exactly like
# every other LLM prompt in the system. The canonical default body is defined in
# recap_prompt.py (a side-effect-free module the registry can import without
# triggering MCP tool registration). This alias is the last-resort fallback used
# only if the registry is unavailable.
_RECAP_SYSTEM_PROMPT = RECAP_SYSTEM_PROMPT


def _resolve_recap_system_prompt(bot_id: str, config) -> str:
    """Resolve the live recap system prompt: DB override → code default.

    Resolves the ``self_recap.system`` key from the prompt registry, preferring a
    bot-scoped override, then global, then the in-code default. Falls back to the
    code default on any registry error so the tool never breaks on a store hiccup.
    """
    try:
        from llm_bawt.prompt_registry import get_prompt_resolver

        resolved = get_prompt_resolver(config).resolve(
            "self_recap.system", scope_type="bot", scope_id=bot_id
        )
        if resolved and (resolved.body or "").strip():
            return resolved.body
    except Exception as e:
        logger.warning(
            "self_recap prompt registry resolve failed, using code default: %s", e
        )
    return RECAP_SYSTEM_PROMPT


def _format_transcript(messages: list[dict]) -> tuple[str, int, bool]:
    """Render messages into a readable transcript.

    Returns (transcript_text, used_count, truncated). Drops oldest messages
    first if the char cap is exceeded.
    """
    lines: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "?").upper()
        content = msg.get("content") or ""
        ts = msg.get("timestamp")
        when = ""
        if ts:
            try:
                when = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
            except (ValueError, OSError, OverflowError):
                when = ""
        header = f"[{role}{(' @ ' + when) if when else ''}]"
        lines.append(f"{header}\n{content}")

    truncated = False
    # Trim from the oldest end until under the cap.
    while lines and sum(len(x) for x in lines) > _MAX_TRANSCRIPT_CHARS:
        lines.pop(0)
        truncated = True

    return "\n\n".join(lines), len(lines), truncated


@mcp.tool(name="self_recap")
async def self_recap(
    bot_id: str = "default",
    hours: int = 24,
    days: int = 0,
    model: str | None = None,
    store: bool = True,
) -> dict:
    """Summarize this agent's recent history into a cold-start continuation briefing.

    Pulls the last ``hours`` of conversation (sent + received), sends it to Grok
    with a structured handoff prompt, optionally stores the result as a summary
    record, and returns the recap. The briefing leads with an exhaustive itemized
    TOPIC LOG of everything discussed, then details the PENDING / INCOMPLETE work
    so an agent with completely clear context can pick up and continue.

    The returned recap text lands in this agent's SDK transcript (it's a tool
    result), so it is replayed into context on the next resumed turn.

    Args:
        bot_id: Bot namespace whose history to recap (pass your own slug).
        hours: Look-back window in hours (default 24).
        days: Additional look-back in days, stacked on top of ``hours``
            (e.g. days=7 → last week; days=7, hours=0 → also last week).
        model: Override the Grok model_id (default env LLM_BAWT_SELF_RECAP_MODEL
            or grok-4-fast-non-reasoning).
        store: Whether to persist the recap as a role='summary' record.

    Returns:
        Dict with: recap (text), stored (bool), summary_id, window_hours,
        messages_analyzed, model, truncated, and on failure, error.
    """
    # hours and days stack into one effective look-back window (min 1h).
    window_hours = max(1, int(hours) + int(days) * 24)
    window_label = (
        f"{days}d{f' {hours}h' if hours else ''}" if days else f"{hours}h"
    )
    logger.debug(
        "MCP tool invoked: self_recap bot_id=%s hours=%s days=%s window=%s",
        bot_id, hours, days, window_label,
    )

    from .server import _get_storage

    storage = _get_storage()
    since_seconds = window_hours * 3600

    # raw=True bypasses the summary-aware manager and reads the messages table
    # directly: real bubbles regardless of their `summarized` flag, with
    # role='summary' husks excluded. Without this, bubbles already rolled into a
    # session summary are invisible here and Grok ends up summarizing summaries.
    messages = await storage.get_messages(
        bot_id=bot_id, since_seconds=since_seconds, raw=True
    )

    # Keep only the conversation bubbles the user actually sees in the app
    # (raw mode already drops summaries; this also drops any system/tool rows).
    messages = [
        m for m in messages if (m.get("role") or "").lower() in _RECAP_ROLES
    ]

    if not messages:
        return {
            "recap": f"No messages found for '{bot_id}' in the last {window_label} — nothing to recap.",
            "stored": False,
            "summary_id": None,
            "window_hours": window_hours,
            "messages_analyzed": 0,
            "model": None,
            "truncated": False,
        }

    transcript, used_count, truncated = _format_transcript(messages)
    message_ids = [m["id"] for m in messages if m.get("id")]

    resolved_model = model or os.getenv("LLM_BAWT_SELF_RECAP_MODEL", _DEFAULT_RECAP_MODEL)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    user_message = (
        "=== CONTEXT ===\n"
        f"Agent being recapped : {bot_id}\n"
        f"Recap generated at   : {now}\n"
        f"Look-back window     : {window_label} ({used_count} messages"
        f"{', oldest dropped to fit size cap' if truncated else ''})\n\n"
        "=== RAW AGENT TRANSCRIPT ===\n"
        f"{transcript}"
    )

    try:
        from llm_bawt.clients.grok_client import GrokClient
        from llm_bawt.models.message import Message
        from llm_bawt.utils.config import Config

        config = Config()
        system_prompt = _resolve_recap_system_prompt(bot_id, config)
        client = GrokClient(model=resolved_model, config=config, api_key=config.XAI_API_KEY)

        # query() is synchronous and does network IO — keep it off the event loop.
        recap_text = await asyncio.to_thread(
            client.query,
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_message),
            ],
            plaintext_output=True,
            stream=False,
        )
    except Exception as e:
        logger.error("self_recap Grok call failed: %s", e)
        return {
            "recap": None,
            "stored": False,
            "summary_id": None,
            "window_hours": window_hours,
            "messages_analyzed": used_count,
            "model": resolved_model,
            "truncated": truncated,
            "error": f"Grok call failed: {e}",
        }

    recap_text = (recap_text or "").strip()
    if not recap_text:
        return {
            "recap": None,
            "stored": False,
            "summary_id": None,
            "window_hours": window_hours,
            "messages_analyzed": used_count,
            "model": resolved_model,
            "truncated": truncated,
            "error": "Grok returned an empty recap.",
        }

    summary_id = None
    if store:
        summary_id = await storage.store_recap_summary(
            bot_id=bot_id,
            content=recap_text,
            window_start=float(messages[0].get("timestamp") or 0.0),
            window_end=float(messages[-1].get("timestamp") or 0.0),
            message_ids=message_ids,
            model=resolved_model,
        )

    return {
        "recap": recap_text,
        "stored": bool(summary_id),
        "summary_id": summary_id,
        "window_hours": window_hours,
        "messages_analyzed": used_count,
        "model": resolved_model,
        "truncated": truncated,
    }


@mcp.tool(name="self_tail")
async def self_tail(
    bot_id: str = "default",
    count: int = 20,
) -> dict:
    """Return the agent's last N raw conversation bubbles — no LLM, no storage.

    The lightweight sibling of ``self_recap``: where recap ships history to Grok
    for a summary, tail just hands back the most recent raw messages (exactly the
    user/assistant bubbles seen in the app), so an agent can pull recent context
    straight back into its transcript. Read-only; nothing is persisted.

    role='summary' rows are excluded, so ``count`` always means N real bubbles,
    not N rows that might be old summaries.

    Args:
        bot_id: Bot namespace whose history to tail (pass your own slug).
        count: Number of most-recent bubbles to return (default 20, min 1).

    Returns:
        Dict with: messages (list of {role, content, timestamp}), transcript
        (rendered text), count_returned, total_available, truncated.
    """
    count = max(1, int(count))
    logger.debug("MCP tool invoked: self_tail bot_id=%s count=%s", bot_id, count)

    from .server import _get_storage

    storage = _get_storage()
    # raw=True reads the messages table directly so already-summarized bubbles
    # are still visible (the manager would hide them) and summary husks are
    # excluded. Filter to real bubbles BEFORE slicing the last `count`.
    messages = await storage.get_messages(bot_id=bot_id, raw=True)
    bubbles = [m for m in messages if (m.get("role") or "").lower() in _RECAP_ROLES]
    total_available = len(bubbles)
    tail = bubbles[-count:]

    transcript, used_count, truncated = _format_transcript(tail)
    return {
        "messages": [
            {
                "role": m.get("role"),
                "content": m.get("content"),
                "timestamp": m.get("timestamp"),
            }
            for m in tail
        ],
        "transcript": transcript,
        "count_returned": used_count,
        "total_available": total_available,
        "truncated": truncated,
    }
