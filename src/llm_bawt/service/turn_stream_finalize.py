"""Terminal lifecycle coordinator for one streaming chat turn."""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from .chat_stream_worker import put_queue_item_threadsafe
from .logging import get_service_logger
from .turn_stream_context import TurnStreamContext

log = get_service_logger(__name__)


class TurnStreamFinalizer:
    """Persist and publish the terminal state for a streaming turn."""

    def __init__(
        self,
        ctx: TurnStreamContext,
        *,
        publish_event_direct: Callable[[dict[str, Any]], Any],
        enrich_attachment_refs: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    ) -> None:
        self.ctx = ctx
        self._publish_event_direct = publish_event_direct
        self._enrich_attachment_refs = enrich_attachment_refs

    def _turn_was_aborted(self) -> bool:
        try:
            current_turn = self.ctx.svc._turn_log_store.get_turn(self.ctx.turn_log_id)
            return current_turn is not None and current_turn.status == "aborted"
        except Exception:
            return False

    def _usage_so_far(self) -> dict | None:
        ctx = self.ctx
        return ctx.svc._resolve_turn_token_usage(
            ctx.llm_bawt,
            ctx.token_usage_holder[0],
        )

    def finalize(self, *, prepared_messages: list[Any]) -> None:
        """Finalize persistence, public events, and generation bookkeeping."""
        ctx = self.ctx
        svc = ctx.svc
        end_time = ctx.timing_holder[1] or time.time()
        start_time = ctx.timing_holder[0] or end_time
        elapsed_ms = (end_time - start_time) * 1000
        externally_aborted = self._turn_was_aborted()
        if externally_aborted:
            ctx.cancelled_holder[0] = True

        # Wrap finalization so the sentinel, turn_complete event, and generation
        # cleanup still fire when persistence raises (for example, DB failure).
        try:
            if externally_aborted:
                # /v1/chat/abort owns this terminal state. Keep any partial reply
                # as a truncated assistant message without flipping the turn to ok.
                if ctx.full_response_holder[0]:
                    svc._finalize_turn(
                        llm_bawt=ctx.llm_bawt,
                        turn_id=ctx.turn_log_id,
                        response_text=ctx.full_response_holder[0],
                        tool_context=ctx.tool_context_holder[0],
                        tool_call_details=ctx.tool_call_details_holder,
                        prepared_messages=prepared_messages,
                        user_prompt=ctx.user_prompt,
                        model=ctx.model_alias,
                        bot_id=ctx.bot_id,
                        user_id=ctx.user_id,
                        elapsed_ms=elapsed_ms,
                        stream=True,
                        animation=ctx.animation_holder[0],
                        token_usage=ctx.token_usage_holder[0],
                        attachments=ctx.agent_attachments_holder or None,
                        reasoning=ctx.reasoning_holder[0] or None,
                        status="aborted",
                        end_reason="aborted",
                        assistant_message_id=ctx.assistant_message_id,
                    )
                else:
                    svc._update_turn_log(
                        turn_id=ctx.turn_log_id,
                        latency_ms=elapsed_ms,
                        tool_calls=ctx.tool_call_details_holder or None,
                        end_reason="aborted",
                    )
            elif ctx.full_response_holder[0]:
                svc._finalize_turn(
                    llm_bawt=ctx.llm_bawt,
                    turn_id=ctx.turn_log_id,
                    response_text=ctx.full_response_holder[0],
                    tool_context=ctx.tool_context_holder[0],
                    tool_call_details=ctx.tool_call_details_holder,
                    prepared_messages=prepared_messages,
                    user_prompt=ctx.user_prompt,
                    model=ctx.model_alias,
                    bot_id=ctx.bot_id,
                    user_id=ctx.user_id,
                    elapsed_ms=elapsed_ms,
                    stream=True,
                    animation=ctx.animation_holder[0],
                    token_usage=ctx.token_usage_holder[0],
                    attachments=ctx.agent_attachments_holder or None,
                    reasoning=ctx.reasoning_holder[0] or None,
                    assistant_message_id=ctx.assistant_message_id,
                )
            else:
                # Persist any tool details captured before a failed follow-up.
                svc._update_turn_log(
                    turn_id=ctx.turn_log_id,
                    status="timeout",
                    latency_ms=elapsed_ms,
                    tool_calls=ctx.tool_call_details_holder or None,
                    token_usage=self._usage_so_far(),
                )
        except Exception as fin_err:
            log.error("Finalization failed (turn %s): %s", ctx.turn_log_id, fin_err)
            try:
                svc._update_turn_log(
                    turn_id=ctx.turn_log_id,
                    status="error",
                    latency_ms=elapsed_ms,
                    response_text=ctx.full_response_holder[0] or None,
                    error_text=f"finalize_error: {fin_err}",
                    token_usage=self._usage_so_far(),
                )
            except Exception:
                pass

        status = "completed" if ctx.full_response_holder[0] else "timeout"
        if ctx.cancelled_holder[0]:
            status = "cancelled"

        question_id = ctx.question_id_holder[0]
        approval_id = ctx.approval_id_holder[0]
        approval_persist_failed = ctx.approval_persist_failed_holder[0]
        if question_id:
            end_reason = "question"
        elif approval_id:
            end_reason = "approval"
        elif approval_persist_failed:
            end_reason = "approval_persist_failed"
        elif ctx.cancelled_holder[0]:
            end_reason = "aborted"
        elif status == "timeout":
            end_reason = "error"
        else:
            end_reason = "stop"

        try:
            svc._turn_log_store.update_turn(
                turn_id=ctx.turn_log_id,
                end_reason=end_reason,
                question_id=question_id,
                error_text=(
                    json.dumps(approval_persist_failed)
                    if approval_persist_failed else None
                ),
                tts_scrubbed=ctx.tts_scrub,
            )
        except Exception as end_reason_err:
            log.debug(
                "update_turn end_reason failed for %s: %s",
                ctx.turn_log_id,
                end_reason_err,
            )

        # The TTS consumer closes input on turn_complete, so commit the final
        # scrubbed tail before publishing that terminal event.
        if ctx.tts_scrubber is not None and status not in ("cancelled", "aborted"):
            tts_tail = ctx.tts_scrubber.flush()
            if tts_tail:
                tts_tail_future = self._publish_event_direct({
                    "_type": "tts_delta",
                    "turn_id": ctx.turn_log_id,
                    "bot_id": ctx.bot_id,
                    "user_id": ctx.user_id,
                    "delta": tts_tail,
                    "ts": time.time(),
                })
                if tts_tail_future is not None:
                    try:
                        tts_tail_future.result(timeout=5)
                    except Exception as order_err:
                        log.warning(
                            "Final tts_delta publish did not complete before "
                            "turn_complete for turn %s: %s",
                            ctx.turn_log_id,
                            order_err,
                        )

        try:
            changed_files = svc._turn_log_store.changed_files_summary(ctx.turn_log_id)
        except Exception:
            changed_files = None

        try:
            completed_attachments = (
                self._enrich_attachment_refs(ctx.agent_attachments_holder)
                if ctx.agent_attachments_holder else None
            )
        except Exception as attachment_err:
            completed_attachments = None
            log.warning("turn_complete attachment enrichment failed: %s", attachment_err)

        # TASK-709: session id for external-turn adoption. Read from the
        # request-local thread binding — the same source that populated
        # turn_start's session_id in chat_streaming.py. Falls back to null on
        # non-agent turns or when the turn was never thread-bound.
        _sid: str | None = None
        try:
            _binding = getattr(ctx, "thread_binding", None)
            if isinstance(_binding, dict):
                _rsid = str(_binding.get("thread_session_id") or "").strip()
                _sid = _rsid or None
        except Exception:
            _sid = None

        # TASK-779: server-authoritative final text on the STREAMING turn
        # finalize path (symmetric with background_service.py's nonstream
        # emit). Streaming turns normally commit from the client-side
        # partial (accumulated text_delta), but if enough deltas drop
        # (subscriber gap, packed flush, network hiccup) the partial can
        # be empty at finalize and the reply vanishes. Carrying the full
        # server-side text lets commitServerOriginatedTurn fall back to
        # the same bytes the DB persisted. Additive optional field —
        # `partial || event.response_text` ordering means healthy
        # streaming still commits from the live partial. Only emitted on
        # successful completion; cancelled/timeout paths already own
        # their terminal state via _finalize_turn above.
        _response_text = (
            ctx.full_response_holder[0] if status == "completed" else None
        )
        self._publish_event_direct({
            "_type": "turn_complete",
            "turn_id": ctx.turn_log_id,
            # TASK-784: symmetric with turn_start / background_service.py nonstream
            # emits — carry the assistant_message_id on the streaming success path
            # too, so commitServerOriginatedTurn on the client can adopt the reply
            # even if it never saw turn_start (mid-turn page join, HMR reconnect
            # after the run began). Additive optional field; consumers that ignore
            # it keep working.
            "assistant_message_id": ctx.assistant_message_id,
            "bot_id": ctx.bot_id,
            "user_id": ctx.user_id,
            # TASK-709: match turn_start's session id (see block above).
            "session_id": _sid,
            "status": status,
            "end_reason": end_reason,
            "question_id": question_id,
            "approval_id": approval_id,
            "approval_persist_failed": approval_persist_failed,
            "animation": ctx.animation_holder[0],
            "token_usage": self._usage_so_far(),
            "changed_files": changed_files,
            "attachments": completed_attachments,
            "response_text": _response_text,
            "model": ctx.model_alias,
            "ts": time.time(),
        })

        put_queue_item_threadsafe(ctx.loop, ctx.chunk_queue, None)

        if ctx.is_agent_backend:
            ctx.done_event.set()
        else:
            svc._end_generation(ctx.cancel_event, ctx.done_event, ctx.bot_id)
