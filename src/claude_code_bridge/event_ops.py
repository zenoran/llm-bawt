from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterable
from datetime import datetime, timezone
from pathlib import Path

import httpx
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, StreamEvent
from claude_agent_sdk.types import (
    AssistantMessage,
    HookMatcher,
    MirrorErrorMessage,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from agent_bridge.approval import (
    ApprovalDecision,
    PolicyAction,
    PolicyBundle,
    evaluate as evaluate_policies,
)
from agent_bridge.events import AgentEvent, AgentEventKind, synthesize_event_id
from agent_bridge.publisher import COMMANDS_STREAM, RedisPublisher
from agent_bridge.session_queue import SessionQueue
from claude_code_bridge.tool_events import normalize_tool_result
from claude_code_bridge.tool_policy import effective_disallowed_tools

from ._bridge_helpers import (
    SESSION_PREFIX,
    MCP_TOOL_CONTEXT_KEY,
    _MCP_TOOL_CONTEXT_FALLBACK,
    _SEED_CLI_VERSION,
    _SEED_SANITIZE_RE,
    _XAI_RATES,
    _XAI_DEFAULT_RATES,
    _CREDENTIALS_PATH,
    _OAUTH_TOKEN_URL,
    _OAUTH_CLIENT_ID,
    _REFRESH_BUFFER_MS,
    _bot_slug_from_session_key,
    _fmt_tokens,
    _usage_input_total,
    _estimate_proxy_cost_usd,
    _pick_iteration_usage,
    _read_latest_compact_metadata,
    _load_oauth_bundle,
    _save_oauth_bundle,
    _token_expired_or_stale,
    _refresh_oauth_bundle,
    _get_fresh_oauth_token,
    _is_cli_crash,
    _is_auth_failure,
)

logger = logging.getLogger("claude_code_bridge.bridge")


class ClaudeEventMixin:
    """Claude output persistence + tool/event emission (TASK-555).

    Split out of ``ClaudeCodeBridge`` (TASK-555); composed back via
    inheritance so ``self.*`` state and sibling-mixin methods resolve
    on the assembled instance.
    """

    @classmethod
    def _read_persisted_output(cls, text: str) -> str:
        """If ``text`` is a <persisted-output> wrapper, return the full file body.

        The harness truncates oversized tool output to a wrapper containing
        ``Full output saved to: <path>`` plus a ~2KB preview, writing the real
        bytes to that path. We read the file back so the persisted payload holds
        the COMPLETE result. Non-fatal: any miss returns ``text`` unchanged.
        """
        if not isinstance(text, str) or cls._PERSISTED_OUTPUT_MARKER not in text:
            return text
        match = cls._PERSISTED_OUTPUT_PATH_RE.search(text)
        if not match:
            return text
        path = match.group(1)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                full = handle.read()
        except OSError:
            logger.warning(
                "persisted-output file unreadable (%s); using inline wrapper", path,
                exc_info=True,
            )
            return text
        logger.info(
            "persisted-output: re-hydrated %d chars from %s", len(full), path,
        )
        return full

    @classmethod
    def _resolve_persisted_output(cls, content: object) -> object:
        """Re-hydrate any <persisted-output> wrapper in a ToolResultBlock body.

        Handles both a raw string body and a list of content-block dicts (the
        text block is swapped for its re-hydrated full text). Never raises.
        """
        try:
            if isinstance(content, str):
                return cls._read_persisted_output(content)
            if isinstance(content, list):
                resolved: list = []
                changed = False
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                        and cls._PERSISTED_OUTPUT_MARKER in part["text"]
                    ):
                        full = cls._read_persisted_output(part["text"])
                        if full != part["text"]:
                            part = {**part, "text": full}
                            changed = True
                    resolved.append(part)
                return resolved if changed else content
        except Exception:
            logger.warning("persisted-output resolve failed; using inline content", exc_info=True)
        return content

    @classmethod
    def _is_image_result_tool(cls, tool_name: str | None) -> bool:
        """True for tools whose result carries an image we should persist
        (namespaced ``mcp__x__name`` or bare)."""
        if not tool_name:
            return False
        return tool_name.split("__")[-1] in cls._IMAGE_RESULT_TOOL_TAILS

    @staticmethod
    def _extract_image_block(block: object) -> tuple[str, str] | None:
        """Pull (base64_data, mime) from a tool-result image content block.

        Handles both the MCP shape ``{type:image, data, mimeType}`` and the
        Anthropic API shape ``{type:image, source:{data, media_type}}``.
        """
        if not isinstance(block, dict) or block.get("type") != "image":
            return None
        data = block.get("data")
        mime = block.get("mimeType") or block.get("mime_type")
        if not data:
            src = block.get("source")
            if isinstance(src, dict):
                data = src.get("data")
                mime = mime or src.get("media_type")
        if not data:
            return None
        return data, (mime or "image/png")

    async def _persist_screenshot_blocks(
        self, content: list, session_key: str, tool_use_id: str | None,
    ) -> list[dict]:
        """Upload each image block in a screenshot tool-result to the media
        store; return ``{asset_id, kind}`` refs. Best-effort — logs and skips
        anything it can't parse or upload."""
        refs: list[dict] = []
        if not self._app_api_url:
            return refs
        user_id = session_key.split(":", 1)[1] if ":" in session_key else "nick"
        for block in content:
            img = self._extract_image_block(block)
            if img is None:
                if isinstance(block, dict) and block.get("type") == "image":
                    logger.warning(
                        "Screenshot image block in unrecognised shape (keys=%s)",
                        list(block.keys()),
                    )
                continue
            data_b64, mime = img
            asset_id = await self._upload_data_url(
                f"data:{mime};base64,{data_b64}", user_id, tool_use_id,
            )
            if asset_id:
                refs.append({"asset_id": asset_id, "kind": "image"})
        return refs

    async def _upload_data_url(
        self, data_url: str, user_id: str, tool_use_id: str | None,
    ) -> str | None:
        """POST a ``data:`` URL to /v1/uploads as an agent attachment; return asset_id."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{self._app_api_url}/v1/uploads",
                    params={"source": "agent_attachment"},
                    headers={"X-Entity-Id": user_id},
                    json={
                        "data_url": data_url,
                        "filename": f"image-{tool_use_id or 'shot'}.png",
                    },
                )
                if resp.status_code >= 400:
                    logger.warning(
                        "Screenshot upload failed: %s %s",
                        resp.status_code, resp.text[:200],
                    )
                    return None
                return (resp.json() or {}).get("asset_id")
        except Exception:
            logger.warning("Screenshot upload error", exc_info=True)
            return None

    def _publish_event(
        self,
        request_id: str,
        session_key: str,
        seq: int,
        *,
        kind: AgentEventKind,
        text: str | None = None,
        tool_name: str | None = None,
        tool_arguments: dict | None = None,
        tool_result: str | None = None,
        tool_result_payload: dict | None = None,
        tool_error: bool | None = None,
        model: str | None = None,
        token_usage: dict | None = None,
        tool_use_id: str | None = None,
        parent_tool_use_id: str | None = None,
        attachments: list[dict] | None = None,
        extra_raw: dict | None = None,
    ) -> None:
        text = self._shorten_paths(text)
        tool_result = self._shorten_paths(tool_result)
        tool_arguments = self._shorten_paths_in_dict(tool_arguments)
        event_id = synthesize_event_id(
            session_key, kind.value,
            {"text": text, "tool": tool_name, "seq": seq},
            seq,
        )
        event = AgentEvent(
            event_id=event_id,
            session_key=session_key,
            run_id=request_id,
            kind=kind,
            origin="system",
            text=text,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            tool_result=tool_result,
            tool_result_payload=tool_result_payload,
            tool_error=tool_error,
            model=model,
            seq=seq,
            timestamp=datetime.now(timezone.utc),
            # APPROVAL_REQUIRED stashes policy metadata (policy_id, severity,
            # subject, prompt, grant_key) in raw so the app can persist the
            # request row and the UI can render the Approve/Deny card.
            raw=dict(extra_raw) if extra_raw else {},
            token_usage=token_usage,
            provider=self._backend_name,
            # Inherit from the active run so every event for this request
            # (TOOL_START, TOOL_END, ASSISTANT_*, RUN_*) carries the same
            # originating user-message UUID without each call site needing
            # to remember to thread it through.
            trigger_message_id=self._trigger_message_ids.get(request_id),
            tool_use_id=tool_use_id,
            parent_tool_use_id=parent_tool_use_id,
            attachments=attachments,
        )
        self._publisher.publish_run_event(request_id, event)

    def _publish_session_reset_unified(
        self,
        bot_id: str,
        session_key: str,
        *,
        had_session: bool = False,
        user_id: str = "nick",
    ) -> None:
        """Publish a SESSION_RESET event on the unified SSE stream.

        Deterministic signal for the frontend to clear its visible message
        buffer for ``bot_id``.  Carries the confirmation text so the UI can
        render it as a synthetic assistant message in one place instead of
        racing the HTTP run stream's ASSISTANT_DONE.  See TASK-249.
        """
        if not bot_id:
            return
        try:
            self._publisher.publish_unified_event(bot_id, user_id, {
                "_type": "session_reset",
                "bot_id": bot_id,
                "user_id": user_id,
                "session_key": session_key,
                "had_session": bool(had_session),
                "text": "Session reset. Ready for a new conversation.",
                "provider": self._backend_name,
                "ts": datetime.now(timezone.utc).timestamp(),
            })
        except Exception:
            logger.exception(
                "Failed to publish unified session_reset for bot=%s session=%s",
                bot_id, session_key,
            )

    @staticmethod
    def _shorten_paths(s: str | None) -> str | None:
        """Replace container home dir with ~ to save space in persisted logs."""
        if s is None:
            return None
        return s.replace("/home/bridge/", "~/").replace("/home/bridge", "~")

    @classmethod
    def _shorten_paths_in_dict(cls, d: dict | None) -> dict | None:
        if d is None:
            return None
        out = {}
        for k, v in d.items():
            if isinstance(v, str):
                out[k] = cls._shorten_paths(v)
            elif isinstance(v, dict):
                out[k] = cls._shorten_paths_in_dict(v)
            else:
                out[k] = v
        return out
