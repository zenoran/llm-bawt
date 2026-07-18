from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from agent_bridge.events import AgentEvent, AgentEventKind, synthesize_event_id
from agent_bridge.publisher import COMMANDS_STREAM, RedisPublisher
from agent_bridge.session_queue import SessionQueue

from .transport import CodexTransport, validate_auth_json
from ._bridge_helpers import (
    CONSUMER_GROUP,
    CONSUMER_NAME,
    MCP_TOOL_CONTEXT_KEY,
    _MCP_TOOL_CONTEXT_FALLBACK,
    _ModelInfoCache,
    _bot_slug_from_session_key,
    _is_auth_failure,
    _is_codex_session_error,
)

logger = logging.getLogger("codex_bridge.bridge")


class CodexSessionMixin:
    """Codex session persistence + MCP tool context (TASK-555).

    Split out of ``CodexBridge`` (TASK-555). Composed back on via
    inheritance, so ``self.*`` state set in ``CodexBridge.__init__`` and
    methods on sibling mixins all resolve on the assembled instance.
    """

    async def _get_mcp_tool_context(self, bot_slug: str) -> str:
        """Return the MCP tool context block for a bot (TASK-490).

        Fetches the (bot-overridable) template body from the app registry via
        GET /v1/prompts/{key}, cached per bot, with a byte-identical local
        fallback if the app is unreachable. Returns the block WITHOUT the leading
        separator; the caller prepends ``\\n\\n``.
        """
        cache = getattr(self, "_mcp_ctx_cache", None)
        if cache is None:
            cache = {}
            self._mcp_ctx_cache = cache
        if bot_slug in cache:
            return cache[bot_slug]

        body = _MCP_TOOL_CONTEXT_FALLBACK
        if self._app_api_url:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(
                        f"{self._app_api_url}/v1/prompts/{MCP_TOOL_CONTEXT_KEY}",
                        params={"scope_type": "bot", "scope_id": bot_slug},
                    )
                    resp.raise_for_status()
                    fetched = (resp.json() or {}).get("body")
                    if fetched:
                        body = fetched
            except Exception as e:
                logger.warning(
                    "MCP tool context fetch failed for %s (%s); using local fallback",
                    bot_slug, e,
                )
        rendered = body.replace("{bot_slug}", bot_slug)
        cache[bot_slug] = rendered
        return rendered

    async def _get_session(self, bot_id: str) -> tuple[str, str] | None:
        """Get (thread_id, model) from the bot's agent_backend_config."""
        if not self._app_api_url or not bot_id:
            return None
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._app_api_url}/v1/bots")
                resp.raise_for_status()
                for bot in resp.json().get("data", []):
                    if bot.get("slug") == bot_id:
                        bc = bot.get("agent_backend_config") or {}
                        sk = bc.get("session_key")
                        # session_model = bridge-owned record of which model
                        # the persisted thread was created with (drives
                        # resume-vs-reset). "model" is the pre-migration key.
                        model = bc.get("session_model") or bc.get("model", "")
                        if sk:
                            sk = str(sk).strip()
                            if ":" in sk:
                                # Legacy bug: routing keys like "snark:nick"
                                # were once stored as session ids. Reject.
                                logger.warning(
                                    "Ignoring invalid persisted session_key for %s: %s",
                                    bot_id, sk,
                                )
                                return None
                            return (sk, model)
                        return None
        except Exception as e:
            logger.warning("Failed to get session for %s: %s", bot_id, e)
        return None

    async def _set_session(self, bot_id: str, thread_id: str, model: str) -> None:
        if not self._app_api_url or not bot_id:
            logger.warning("No API URL or bot_id — session not persisted")
            return
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._app_api_url}/v1/bots")
                resp.raise_for_status()
                bc: dict = {}
                for bot in resp.json().get("data", []):
                    if bot.get("slug") == bot_id:
                        bc = dict(bot.get("agent_backend_config") or {})
                        break
                bc["session_key"] = thread_id
                # Bridge-owned session metadata. The user-facing model lives
                # on the bot's default_model (catalog alias); "model" is no
                # longer accepted in agent_backend_config by the profile API.
                bc.pop("model", None)
                bc["session_model"] = model
                patch_response = await client.patch(
                    f"{self._app_api_url}/v1/bots/{bot_id}/profile",
                    json={"agent_backend_config": bc},
                )
                patch_response.raise_for_status()
            logger.info("Session persisted: %s -> %s", bot_id, thread_id)
        except Exception as e:
            logger.warning("Failed to persist session for %s: %s", bot_id, e)

    async def _clear_session(self, bot_id: str) -> bool:
        if not self._app_api_url or not bot_id:
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._app_api_url}/v1/bots")
                resp.raise_for_status()
                bc: dict = {}
                had_session = False
                for bot in resp.json().get("data", []):
                    if bot.get("slug") == bot_id:
                        bc = dict(bot.get("agent_backend_config") or {})
                        had_session = "session_key" in bc
                        bc.pop("session_key", None)
                        break
                patch_response = await client.patch(
                    f"{self._app_api_url}/v1/bots/{bot_id}/profile",
                    json={"agent_backend_config": bc},
                )
                patch_response.raise_for_status()
                logger.info("Session cleared: %s (had_session=%s)", bot_id, had_session)
                return had_session
        except Exception as e:
            logger.warning("Failed to clear session for %s: %s", bot_id, e)
            return False

    async def _bot_uses_codex(self, bot_id: str) -> bool:
        """Look up a bot's agent_backend; return True only if it's codex.

        Used to defensively skip legacy RPCs (no `backend` field on the
        message) that target bots owned by other bridges. Without this
        guard, the codex bridge would happily clear session_keys on
        claude-code / openclaw bots — a cross-backend interference bug.
        """
        if not self._app_api_url or not bot_id:
            # No way to verify — fall through to legacy behavior. The
            # operations the RPC triggers (clear_session, signal_cancel)
            # are no-ops on unknown bots/sessions anyway.
            return True
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._app_api_url}/v1/bots")
                resp.raise_for_status()
                for bot in resp.json().get("data", []):
                    if bot.get("slug") == bot_id:
                        return (bot.get("agent_backend") or "") == self._backend_name
        except Exception as e:
            logger.debug("agent_backend lookup failed for %s: %s", bot_id, e)
            # On lookup failure, default to True so we don't drop our own
            # RPCs because the API blipped.
            return True
        # Bot not found — definitely not ours.
        return False
