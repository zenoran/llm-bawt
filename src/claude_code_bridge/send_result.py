"""Successful Claude SDK ``ResultMessage`` finalization."""

from __future__ import annotations

import asyncio

from agent_bridge.events import AgentEventKind

from ._bridge_helpers import _fmt_tokens, _read_latest_compact_metadata


class ClaudeResultMixin:
    """Normalize successful terminal results and publish ``ASSISTANT_DONE``."""

    async def _finalize_result_message(
        self,
        msg,
        *,
        request_id: str,
        session_key: str,
        seq: int,
        text_parts: list[str],
        assistant_snapshot_text: str,
        api_retry_count: int,
        api_last_error: str | None,
        api_retry_surfaced: bool,
        actual_model: str,
        model: str,
        bot_context_window: int | None,
        latest_assistant_usage: dict | None,
        latest_stream_usage: dict | None,
        compact_status: str | None,
        compact_error_msg: str | None,
        turn_session_id: str | None,
        turn_screenshot_assets: list[dict],
    ) -> int:
        full_text = "".join(text_parts)
        if not full_text:
            result_text = getattr(msg, "text", "") or ""
            if not result_text:
                for block in getattr(msg, "content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        result_text += block.get("text", "")
            full_text = result_text
            if not full_text and assistant_snapshot_text:
                full_text = assistant_snapshot_text
        if not full_text and api_retry_count > 0:
            error_note = (
                f"\n\n❌ Upstream error after {api_retry_count} "
                f"retries: {api_last_error or 'unknown'}. "
                f"Try again in a moment."
            )
            full_text = (
                "".join(text_parts) + error_note
                if api_retry_surfaced
                else error_note.lstrip()
            )

        token_usage_payload, ctx_window, max_output = self._compute_result_usage(
            msg,
            actual_model=actual_model,
            model=model,
            bot_context_window=bot_context_window,
            latest_assistant_usage=latest_assistant_usage,
            latest_stream_usage=latest_stream_usage,
        )

        if compact_status == "success":
            cm = await asyncio.to_thread(
                _read_latest_compact_metadata, turn_session_id
            )
            pre = (cm or {}).get("preTokens")
            post = (cm or {}).get("postTokens")
            if post is not None:
                freed = (
                    f" ({round(100 * (pre - post) / pre)}% freed)"
                    if pre
                    else ""
                )
                note = (
                    f"\n\n✅ Compacted: {_fmt_tokens(pre)} → "
                    f"{_fmt_tokens(post)} tokens{freed}."
                )
                token_usage_payload = {
                    "input_tokens": int(post),
                    "cache_read_tokens": 0,
                    "cache_creation_tokens": 0,
                    "output_tokens": 0,
                    "context_window": ctx_window,
                    "max_output_tokens": max_output,
                    "total_cost_usd": getattr(msg, "total_cost_usd", None),
                }
            else:
                note = "\n\n✅ Conversation compacted."
            seq += 1
            text_parts.append(note)
            self._publish_event(
                request_id, session_key, seq,
                kind=AgentEventKind.ASSISTANT_DELTA,
                text=note,
            )
            full_text = "".join(text_parts)
        elif compact_status == "failed":
            note = f"\n\nℹ️ Nothing to compact — {compact_error_msg}"
            seq += 1
            text_parts.append(note)
            self._publish_event(
                request_id, session_key, seq,
                kind=AgentEventKind.ASSISTANT_DELTA,
                text=note,
            )
            full_text = "".join(text_parts)

        seq += 1
        self._publish_event(
            request_id, session_key, seq,
            kind=AgentEventKind.ASSISTANT_DONE,
            text=full_text,
            model=actual_model,
            token_usage=token_usage_payload,
            attachments=turn_screenshot_assets or None,
        )
        return seq
