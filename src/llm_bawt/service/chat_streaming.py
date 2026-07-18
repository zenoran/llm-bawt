"""Chat completion streaming mixin (native + OpenClaw bridge).

Extracted from background_service.py — handles SSE streaming for
chat completions including tool call handling and bridge integration.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import Any, AsyncIterator

from ..approval_policies import ApprovalPersistError
from ..bots import (
    StreamingEmoteFilter,
    StreamingTTSScrubber,
    should_scrub_for_tts,
)
# TASK-225: MediaStore is used to (a) resolve new-style ``attachment_ids``
# into inline image bytes for the LLM call and (b) auto-upload legacy
# inline ``image_url`` base64 so the asset gets a persistent id.
from ..media import MediaAssetNotFound, get_media_store
from ..media.serializers import build_agent_image_manifest
from .chat_stream_worker import consume_stream_chunks, put_queue_item_threadsafe
from .chat_streaming_bridge import ChatStreamingBridgeMixin
from .logging import RequestContext, generate_request_id, get_service_logger
from .schemas import ChatCompletionRequest

log = get_service_logger(__name__)

# TASK-286: assistant text deltas are coalesced into chunks of at least this
# many characters before being published to the unified Redis stream as
# ``text_delta`` events. The original HTTP body still streams every token to
# the requesting client; this parallel channel exists only so a refreshed /
# secondary client can recover (and keep streaming) the response TEXT — not
# just tool calls. Coalescing keeps the capped Redis stream (MAXLEN 5000,
# shared per bot+user) from churning on per-token writes.
_TEXT_DELTA_FLUSH_CHARS = 80
# TASK-306 Section A: bounded retry for the confirmed approval-row commit.
# A gated tool call MUST persist a durable approval row before the turn ends;
# on transient DB failure we retry a small bounded number of times, then fail
# closed + informed rather than swallow.
_APPROVAL_PERSIST_MAX_ATTEMPTS = 3
_APPROVAL_PERSIST_BACKOFF_S = 0.25


class ChatStreamingMixin(ChatStreamingBridgeMixin):
    """Mixin providing streaming chat completions for BackgroundService.

    Bridge/session responsibilities (``_stream_via_bridge`` and the
    conversation-offset / OpenClaw session helpers) live on the inherited
    :class:`ChatStreamingBridgeMixin` (TASK-554).
    """


    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """Handle a streaming chat completion request.

        Uses llm_bawt's internal history + memory for context.
        Yields Server-Sent Events (SSE) formatted chunks.
        """
        # Flush an SSE comment IMMEDIATELY so the reverse proxy (Traefik) and
        # the Next.js /api/chat passthrough see bytes the instant the response
        # opens, before any of the slow setup below (model resolution, history
        # build, object-store attachment fetch, and — for agent bots — the
        # upstream image upload/analysis that delays the first real delta by
        # several seconds).  Without an early byte, a slow first token can trip
        # a gateway timeout and surface as a 502 on the client even though the
        # turn completes server-side.  Comment lines (": ...") are part of the
        # SSE spec and are ignored by every consumer — browser EventSource and
        # our own reader (ChatStreamContext skips lines starting with ":").
        yield ": connected\n\n"

        # Create request context for logging
        if request.client_system_context is not None:
            req_path = f"/v1/botchat/{request.bot_id}/{request.user}/chat/completions"
        else:
            req_path = "/v1/chat/completions"
        ctx = RequestContext(
            request_id=generate_request_id(),
            method="POST",
            path=req_path,
            model=request.model,
            bot_id=request.bot_id,
            user_id=request.user,
            stream=True,
        )

        # Log incoming request (verbose mode will show the full payload)
        log.api_request(ctx, request.model_dump(exclude_none=True))

        bot_id = request.bot_id or self._default_bot
        user_id = request.user or self.config.DEFAULT_USER
        local_mode = not request.augment_memory

        # Resolve model using shared bot/config logic
        # Agent-backend bots resolve to their virtual model (e.g. "openclaw")
        try:
            model_alias, model_warnings = self._resolve_request_model(request.model, bot_id, local_mode)
        except Exception as e:
            log.api_error(ctx, str(e), 400)
            raise

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # Get cached LLMBawt instance
        try:
            llm_bawt = self._get_llm_bawt(model_alias, bot_id, user_id, local_mode)
        except Exception as e:
            log.api_error(ctx, str(e), 500)
            warning_data = {
                "object": "service.warning",
                "model": model_alias,
                "warnings": [f"request_failed: {e}"],
            }
            yield f"data: {json.dumps(warning_data)}\n\n"
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_alias,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Get the user's prompt (last user message).
        #
        # TASK-225: attachment handling has two input shapes that converge
        # on the same in-memory ``user_attachments`` list (the LLM-call
        # form, ``{mimeType, content=naked-b64}``) and on a tiny
        # ``attachments_to_persist`` list (the JSONB row form,
        # ``{asset_id, kind}``):
        #
        #   1. NEW STYLE — ``attachment_ids: ["ma_xxx", ...]`` references
        #      ``media_assets`` rows.  Resolved via
        #      ``MediaStore.read_original_as_data_url`` (cap-bounded WebP).
        #
        #   2. LEGACY STYLE — ``content: [{type:"image_url", image_url:
        #      {url:"data:..."}}]`` (OpenAI multimodal).  Bytes go to the
        #      LLM exactly as before.  Additionally the server uploads
        #      them to MediaStore so legacy clients get persisted
        #      attachments "for free" — a rolling deploy never regresses
        #      to "image dropped".
        #
        # Only the trailing user message is examined; both shapes can
        # appear on the same message — order is preserved (legacy inline
        # parts first, then new-style attachment_ids).
        user_prompt = ""
        user_attachments: list[dict] = []
        attachments_to_persist: list[dict] = []
        # A degraded MediaStore (stale NFS handle, DB outage) must never
        # kill the chat stream — attachments are dropped for the turn
        # instead.
        try:
            media_store = get_media_store()
        except Exception as e:
            media_store = None
            log.error(
                "MediaStore unavailable — attachments disabled for this turn: %s",
                e,
            )

        for m in reversed(request.messages):
            if m.role != "user":
                continue

            # ---- Resolve text + legacy inline images from content shape ----
            if isinstance(m.content, list):
                for part in m.content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        user_prompt += part.get("text", "")
                    elif part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url", "")
                        if not url.startswith("data:"):
                            continue
                        try:
                            header, data = url.split(",", 1)
                            mime = header.split(":")[1].split(";")[0]
                        except Exception:
                            continue

                        # Feed the LLM exactly as before — bytes are
                        # already inline, no point re-reading from disk.
                        user_attachments.append({"mimeType": mime, "content": data})

                        # Back-compat auto-upload so the inline image
                        # gets a persistent asset_id.  Failures are
                        # non-fatal: a degraded MediaStore must not
                        # break an otherwise-valid chat — the LLM still
                        # sees the image; we just won't persist a ref.
                        try:
                            if media_store is None:
                                raise RuntimeError("MediaStore unavailable")
                            import base64 as _b64

                            raw_bytes = _b64.b64decode(data, validate=False)
                            asset = media_store.upload(
                                raw_bytes=raw_bytes,
                                original_mime=mime,
                                source="chat_upload",
                                owner_user_id=user_id,
                            )
                            attachments_to_persist.append(
                                {"asset_id": asset.id, "kind": "image"}
                            )
                        except Exception as e:
                            log.warning(
                                "TASK-225: failed to auto-upload legacy inline image: %s",
                                e,
                            )
            else:
                user_prompt = m.content or ""

            # ---- Resolve new-style attachment_ids ----
            requested_ids = list(getattr(m, "attachment_ids", None) or [])
            for asset_id in requested_ids:
                if not isinstance(asset_id, str) or not asset_id.strip():
                    continue
                asset_id = asset_id.strip()
                if media_store is None:
                    log.error(
                        "TASK-225: dropping attachment_id %s — MediaStore unavailable",
                        asset_id,
                    )
                    continue
                try:
                    # Off-load to a thread: read_preview_as_data_url is a
                    # SYNCHRONOUS object-store fetch (Garage/S3) + base64
                    # encode.  Calling it inline blocked the event loop for the
                    # whole download — stalling every other in-flight turn and
                    # padding this turn's time-to-first-byte enough to trip the
                    # reverse proxy (the image-paste 502).
                    #
                    # Inline the 1024px PREVIEW (not the 1568px original): ~55%
                    # fewer pixels at Q82 cuts the per-image vision-token bill
                    # with only a modest fidelity loss. The full-res original
                    # stays available via the asset_id for anything that needs
                    # it (lightbox, download).
                    data_url = await asyncio.to_thread(
                        media_store.read_preview_as_data_url, asset_id
                    )
                except MediaAssetNotFound:
                    log.warning(
                        "TASK-225: attachment_id not found in media_assets: %s",
                        asset_id,
                    )
                    continue
                except FileNotFoundError as e:
                    # Preview blob missing (older asset predating preview
                    # generation, or a failed derive) — fall back to the
                    # full-res original so we degrade to "bigger image"
                    # rather than silently dropping the attachment.
                    log.warning(
                        "TASK-225: preview blob missing for asset_id=%s, "
                        "falling back to original: %s",
                        asset_id, e,
                    )
                    try:
                        data_url = await asyncio.to_thread(
                            media_store.read_original_as_data_url, asset_id
                        )
                    except Exception as e2:
                        log.error(
                            "TASK-225: original fallback also failed for "
                            "asset_id=%s: %s",
                            asset_id, e2,
                        )
                        continue
                except Exception as e:
                    log.warning(
                        "TASK-225: MediaStore.read_preview failed for asset_id=%s: %s",
                        asset_id, e,
                    )
                    continue

                # The ``user_attachments`` contract is {mimeType, content
                # = naked-b64} — strip the ``data:<mime>;base64,`` prefix
                # here.  The prefix is re-added at the LLM boundary
                # inside :meth:`prepare_messages_for_query`.
                try:
                    header, payload = data_url.split(",", 1)
                    mime = header.split(":")[1].split(";")[0]
                except Exception:
                    continue
                user_attachments.append({"mimeType": mime, "content": payload})
                attachments_to_persist.append({"asset_id": asset_id, "kind": "image"})

            break

        # Extract the user message ID so tool-call events and turn logs are
        # joinable with the frontend's history rendering.  Prefer the explicit
        # request.user_message_id (frontend-supplied UUID) and fall back to
        # scanning the trailing user message in request.messages.
        trigger_message_id = getattr(request, "user_message_id", None)
        if not trigger_message_id:
            for _m in reversed(request.messages):
                if _m.role != "user":
                    continue
                _mid = getattr(_m, "id", None) or getattr(_m, "message_id", None)
                if isinstance(_mid, str) and _mid.strip():
                    trigger_message_id = _mid.strip()
                    break

        # Frontend-minted UUID for the ASSISTANT reply row. Persisting the
        # assistant message under this id makes the live streaming bubble and the
        # reloaded history row share ONE id, so they merge into a single bubble
        # (closes the EPIC TASK-217 assistant-identity gap). None → history's
        # add_message mints a server UUID (server-originated turns).
        _amid = getattr(request, "assistant_message_id", None)
        assistant_message_id = _amid.strip() if isinstance(_amid, str) and _amid.strip() else None

        if not user_prompt:
            warning_data = {
                "object": "service.warning",
                "model": model_alias,
                "warnings": ["request_failed: No user message found in request"],
            }
            yield f"data: {json.dumps(warning_data)}\n\n"
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_alias,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # OpenClaw bots now flow through the normal AgentBackendClient.stream_raw()
        # pipeline, which routes via Redis -> Bridge -> WS automatically.

        chunk_queue: asyncio.Queue = asyncio.Queue()
        full_response_holder = [""]  # Use list to allow mutation in nested function
        # TTS scrub decision — computed ONCE here (single source of truth via
        # should_scrub_for_tts), tracked on the turn, and shared by every
        # downstream consumer: the turn_start flag, the block-boundary scrubber
        # that emits `tts_delta` events, and the persisted `tts_scrubbed` column.
        tts_scrub = should_scrub_for_tts(llm_bawt.bot)
        tts_scrubber = StreamingTTSScrubber() if tts_scrub else None
        reasoning_holder = [""]  # TASK-301: accumulated model reasoning, persisted display-only
        animation_holder = [None]   # Populated by the embedding classifier (TASK-215)
        tool_context_holder = [""]  # Store tool context from native tool calls
        tool_call_details_holder: list[dict] = []
        timing_holder = [0.0, 0.0]  # [start_time, end_time]
        cancelled_holder = [False]  # Track if we were cancelled
        token_usage_holder: list[dict | None] = [None]  # Captures upstream SDK token usage for turn_complete
        # {asset_id, kind} refs for media the agent backend persisted during the
        # turn (e.g. Playwright screenshots) — attached to the assistant reply.
        agent_attachments_holder: list[dict] = []
        # TASK-269: tool_use_id of an AskUserQuestion the agent deferred this
        # turn.  Set when an await_tool_result chunk arrives; consumed at turn
        # completion to mark end_reason="question" + question_id on the turn log
        # and in the turn_complete event so the UI renders a QuestionMessage.
        question_id_holder: list[str | None] = [None]
        # TASK-292: request_id of an approval-gated tool call deferred this turn.
        # Set when an approval_required chunk arrives; consumed at turn
        # completion to mark end_reason="approval" so the UI keeps the card.
        approval_id_holder: list[str | None] = [None]
        # TASK-306 Section A: structured failure when a gated tool's approval
        # row could NOT be durably committed. When set, the turn ends honestly
        # (end_reason="approval_persist_failed") and the user is told the gate
        # did not reach them — never a silent success.
        approval_persist_failed_holder: list[dict | None] = [None]

        # Capture the event loop before entering the thread
        loop = asyncio.get_running_loop()

        # Start new generation.
        # For agent_backend (OpenClaw) bots the gateway queues messages as
        # separate runs, so we must NOT cancel a previous generation — each
        # request streams independently.  For native models we cancel the
        # previous generation so only the latest request runs.
        is_agent_backend = llm_bawt.client.model_definition.get("type") in ("agent_backend", "claude-code")

        # ---- /new command for CHAT bots --------------------------------
        # Agent bots already handle /new at the bridge (clears the SDK
        # session). Chat bots have no such concept, so /new here moves a
        # per-bot "conversation offset" marker to now: prior raw messages
        # drop out of the live context, but summaries + long-term memory are
        # kept (see HistoryManager/get_messages). Nothing is deleted — it's a
        # pointer move, and the dropped transcript still gets summarized by
        # the background job and reappears as a summary.
        if not is_agent_backend:
            _stripped = (user_prompt or "").lstrip()
            _low = _stripped.lower()
            if _low == "/new" or _low.startswith("/new ") or _low.startswith("/new\n"):
                self._set_conversation_offset(llm_bawt, bot_id, time.time())
                remainder = _stripped[len("/new"):].strip()
                if not remainder:
                    confirm = (
                        "Fresh start. I've set down the recent back-and-forth — "
                        "I still keep the longer-term summaries and what I know "
                        "about you, just not this last thread. What's on your mind?"
                    )
                    yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_alias, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': confirm}, 'finish_reason': None}]})}\n\n"
                    yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_alias, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                # `/new <message>`: start fresh, then answer the message in the
                # clean context.
                user_prompt = remainder

        if is_agent_backend:
            cancel_event = threading.Event()
            done_event = threading.Event()
        else:
            cancel_event, done_event = await self._start_generation(bot_id)
        turn_log_id = f"turn-{uuid.uuid4().hex}"

        # Resolve agent session_key for abort support.
        #
        # The bridge tracks active streams keyed by the chat.send command's
        # ``session_key`` field. chat.abort RPC must arrive with the same
        # key or the bridge can't find the active controller (results in
        # "no_active_task" and the abort is a no-op server-side).
        #
        # claude-code and codex both produce the routing key as
        # ``f"{bot_id}:{user_id}"`` in their respective backends — match that
        # here. The persisted ``session_key`` in their agent_backend_config
        # is the SDK-internal thread/session id, NOT a routing key.
        #
        # OpenClaw (and any other backend) keeps using its persisted
        # session_key (e.g. ``agent:main:main``), which IS the routing key
        # for that backend.
        oc_session_key: str | None = None
        # TASK-501: seed history the app pre-assembles and pushes to the bridge
        # so it need not call back to /v1/history/context-seed. Stays None
        # unless this is a claude-code agent turn that will open a fresh session.
        inject_seed_messages: list | None = None
        if is_agent_backend:
            bc = getattr(llm_bawt.bot, "agent_backend_config", None) or {}
            backend_name = getattr(llm_bawt.bot, "agent_backend", "")
            # TASK-276: the "local" backend routes the same way claude-code /
            # codex do — f"{bot_id}:{user_id}" — so chat.abort RPCs reach the
            # right active stream on the local-model-bridge.  Fall back to the
            # client's resolved backend when the bot has no agent_backend set
            # (local GPU bots are ordinary chat bots whose model_definition was
            # rewritten to type=agent_backend/backend=local in core.py).
            client_backend = ""
            try:
                client_backend = str(
                    llm_bawt.client.model_definition.get("backend") or ""
                ).strip()
            except Exception:
                client_backend = ""
            if backend_name in ("claude-code", "codex") or client_backend == "local":
                oc_session_key = f"{bot_id}:{user_id}"
            else:
                oc_session_key = bc.get("session_key")

            # TASK-501: pre-assemble the session seed app-side and push it to
            # the bridge (inject_messages) so the bridge need not call back to
            # /v1/history/context-seed. Shared decision helper — SAME logic the
            # non-streaming path uses (background_service.chat_completion) so the
            # two dispatch routes stay consistent.
            from .routes.history import maybe_build_session_seed
            from .dependencies import get_service
            inject_seed_messages = maybe_build_session_seed(
                llm_bawt, bot_id, model_alias, user_prompt, get_service()
            )

        # Persist turn log immediately so the user's prompt is recorded
        # even if the backend times out or errors before responding.
        # TASK-269: a continuation turn answering a deferred question carries
        # the awaiting turn as parent_turn_id (threads the chain + drives
        # cross-tab resolution via turn_start{parent_turn_id}).
        parent_turn_id = getattr(request, "parent_turn_id", None)
        answered_question_id = getattr(request, "answered_question_id", None)
        self._persist_turn_log(
            turn_id=turn_log_id,
            request_id=ctx.request_id,
            path=ctx.path,
            stream=True,
            model=model_alias,
            bot_id=bot_id,
            user_id=user_id,
            status="streaming",
            latency_ms=None,
            user_prompt=user_prompt,
            prepared_messages=[],
            response_text="",
            agent_session_key=oc_session_key,
            trigger_message_id=trigger_message_id,
            parent_turn_id=parent_turn_id,
        )
        # Record which continuation turn carried the answer back to the agent.
        if answered_question_id:
            try:
                self._pending_question_store.set_answered_turn(
                    answered_question_id, turn_log_id,
                )
            except Exception as _link_err:
                log.debug("set_answered_turn failed for %s: %s", answered_question_id, _link_err)

        # Capture Redis subscriber for direct publish from the background
        # thread — ensures tool events (and turn_complete) reach Redis even
        # if the SSE generator is cancelled (client disconnect / page refresh).
        _redis_sub = getattr(self, "_redis_subscriber", None)

        async def _publish_unified(event_dict):
            """Single source of truth for publishing to the unified SSE stream.

            Race-free when awaited directly from the async generator (loop
            thread). Worker-thread callers must NOT call this directly —
            they go through _publish_event_direct, which marshals onto the
            loop with run_coroutine_threadsafe.
            """
            if not _redis_sub:
                return
            try:
                await _redis_sub.publish_tool_event(bot_id, user_id, event_dict)
            except Exception as pub_err:
                log.debug("Unified event publish failed: %s", pub_err)

        from .tool_event_coordinator import ToolEventCoordinator
        _tool_event_coordinator = ToolEventCoordinator(self._turn_log_store.engine)

        def _publish_event_direct(event_dict):
            """Publish to the unified stream FROM THE WORKER THREAD.

            Thin cross-thread wrapper over _publish_unified. ``run_coroutine_
            threadsafe`` is correct ONLY when called from a thread other than
            the loop's own — calling it from the loop thread schedules a future
            that races request teardown and is silently dropped (this was the
            TASK-305 approval-card/bell/gold-lock bug). Loop-thread callers must
            ``await _publish_unified(...)`` instead.

            Returns the cross-thread future so callers that require strict event
            ordering can wait for the Redis write before publishing a terminal
            event such as ``turn_complete``.
            """
            if event_dict.get("_type") == "tool_event":
                if event_dict.get("event") == "tool_start":
                    event_dict = _tool_event_coordinator.start(event_dict)
                elif event_dict.get("event") == "tool_end":
                    event_dict = _tool_event_coordinator.end(event_dict)
            if not _redis_sub:
                return None
            try:
                return asyncio.run_coroutine_threadsafe(
                    _publish_unified(event_dict),
                    loop,
                )
            except Exception as pub_err:
                log.debug("Direct event publish failed: %s", pub_err)
                return None

        # Announce turn kickoff to the unified SSE stream BEFORE the worker
        # thread starts.  Other clients subscribed to this bot (other browser
        # tabs, the dashboard, CLI tools, scheduled jobs) flip their "bot is
        # active" indicators on the turn_start event — without it they'd only
        # find out a turn was running when the first tool_start arrived
        # (could be many seconds for non-tool turns) or on the next page
        # refresh via /api/chat/active-bots.  turn_complete (line ~1435)
        # is the matching teardown event.
        #
        # TASK-358: the payload also carries the user message itself
        # (``role``/``content``/``attachments``) so a SECOND window with no
        # prior bubble for this turn can render the USER BUBBLE live — not
        # just the "bot is active" rail. ``content`` is already rendered
        # client-side elsewhere (history + the originating tab's own optimistic
        # bubble), so this is not new exposure. ``attachments`` is enriched to
        # the SAME shape /v1/history emits ({asset_id, kind, mime_type, width,
        # height, urls}) via the canonical serializer, so the client can reuse
        # ``normalizeHistoryAttachment`` verbatim. Enrichment does a synchronous
        # media_assets SELECT, so it is offloaded to a thread and only when the
        # turn actually carries attachments (the common no-attachment path skips
        # all of it).
        turn_start_attachments: list[dict] = []
        if attachments_to_persist:
            try:
                from ..media.assets import MediaAssetStore
                from ..media.serializers import enrich_attachments_for_messages

                def _enrich_turn_start_attachments() -> list[dict]:
                    shell = [{"attachments": list(attachments_to_persist)}]
                    enrich_attachments_for_messages(
                        shell, MediaAssetStore(self.config)
                    )
                    return shell[0].get("attachments") or []

                turn_start_attachments = await asyncio.to_thread(
                    _enrich_turn_start_attachments
                )
            except Exception as _att_err:
                # A degraded MediaStore must never block the turn_start
                # announcement — fall back to no attachments (text bubble
                # still renders; images fill in on the receiving tab's next
                # history load).
                log.warning(
                    "TASK-358: turn_start attachment enrichment failed: %s",
                    _att_err,
                )
                turn_start_attachments = []

        await _publish_unified({
            "_type": "turn_start",
            "turn_id": turn_log_id,
            "trigger_message_id": trigger_message_id,
            # Canonical assistant row id (frontend-minted). Lets a non-originating
            # window key its assistant bubble on the SAME id the reload will carry,
            # so live→reload is a merge, not a duplicate.
            "assistant_message_id": assistant_message_id,
            "bot_id": bot_id,
            "user_id": user_id,
            "parent_turn_id": parent_turn_id,
            "role": "user",
            "content": user_prompt,
            "attachments": turn_start_attachments,
            # TTS consumers (chat_tts_driver): when true, IGNORE raw text_delta
            # and synthesize the scrubbed `tts_delta` events instead. Markdown
            # is scrubbed at block boundaries (not per-token), so audio still
            # streams — just at paragraph granularity.
            "tts_scrubbed": tts_scrub,
            "ts": time.time(),
        })

        # ── Durable approval handling (TASK-292/306 race hardening) ──
        # The approval row + its surfacing MUST NOT depend on the SSE async
        # generator draining the chunk. A gated tool ends the turn and the
        # generator is torn down (client disconnect) before it dequeues the
        # already-delivered approval chunk — so the row never persisted and the
        # user saw nothing (a silent vacuum). We persist + publish from the
        # streaming WORKER THREAD (_intercept_tool_events), which runs to
        # completion regardless of client teardown. Idempotent: record_request
        # is keyed on tool_use_id and _approval_handled guards the async-gen
        # path from double-handling.
        _approval_handled: set[str] = set()

        def _persist_publish_approval(chunk: dict) -> None:
            """Durably record + surface one approval_required chunk (worker thread).

            Synchronous DB write with bounded retry; on success publishes the
            live unified ``tool_approval_required`` event, on a confirmed
            failure publishes ``tool_approval_persist_failed`` so the error
            reaches the user instead of vanishing.
            """
            req_id = chunk.get("tool_use_id") or ""
            if not req_id or req_id in _approval_handled:
                return
            _approval_handled.add(req_id)
            tool_args = chunk.get("arguments") or {}
            store = getattr(self, "_tool_approval_policy_store", None)

            persist_failure: dict | None = None
            if store is not None:
                committed = False
                last_err: Exception | None = None
                for _attempt in range(_APPROVAL_PERSIST_MAX_ATTEMPTS):
                    try:
                        store.record_request(
                            request_id=req_id,
                            bot_id=bot_id,
                            user_id=user_id,
                            turn_id=turn_log_id,
                            backend=chunk.get("provider") or "claude-code",
                            tool_name=chunk.get("tool_name") or "",
                            tool_arguments=tool_args if isinstance(tool_args, dict) else {"value": tool_args},
                            subject=chunk.get("subject") or "",
                            grant_key=chunk.get("grant_key") or "",
                            policy_id=chunk.get("policy_id"),
                            severity=chunk.get("severity") or "medium",
                            prompt=chunk.get("prompt") or "",
                            trigger_message_id=trigger_message_id,
                            session_key=chunk.get("session_key") or None,
                        )
                        committed = True
                        break
                    except ApprovalPersistError as _persist_err:
                        last_err = _persist_err
                        log.warning(
                            "approval persist attempt %d/%d failed id=%s: %s",
                            _attempt + 1, _APPROVAL_PERSIST_MAX_ATTEMPTS,
                            req_id, _persist_err,
                        )
                        if _attempt + 1 < _APPROVAL_PERSIST_MAX_ATTEMPTS:
                            time.sleep(_APPROVAL_PERSIST_BACKOFF_S * (_attempt + 1))
                if committed:
                    approval_id_holder[0] = req_id
                else:
                    persist_failure = {
                        "kind": "side_effect_failed",
                        "effect": "approval_request_persist",
                        "request_id": req_id,
                        "tool_name": chunk.get("tool_name") or "",
                        "reachable_to_user": False,
                        "error_class": type(last_err).__name__ if last_err else "ApprovalPersistError",
                        "detail": str(last_err) if last_err else "unknown",
                        "attempts": _APPROVAL_PERSIST_MAX_ATTEMPTS,
                    }
            else:
                persist_failure = {
                    "kind": "side_effect_failed",
                    "effect": "approval_request_persist",
                    "request_id": req_id,
                    "tool_name": chunk.get("tool_name") or "",
                    "reachable_to_user": False,
                    "error_class": "NoApprovalStore",
                    "detail": "approval policy store is not configured on this service",
                    "attempts": 0,
                }

            if persist_failure is None:
                _publish_event_direct({
                    "_type": "tool_approval_required",
                    "turn_id": turn_log_id,
                    "trigger_message_id": trigger_message_id,
                    "bot_id": bot_id,
                    "user_id": user_id,
                    "request_id": req_id,
                    "tool_name": chunk.get("tool_name", ""),
                    "arguments": tool_args,
                    "subject": chunk.get("subject", ""),
                    "label": chunk.get("label", ""),
                    "prompt": chunk.get("prompt", ""),
                    "severity": chunk.get("severity", "medium"),
                    "policy_id": chunk.get("policy_id"),
                    "session_key": chunk.get("session_key", ""),
                    "provider": chunk.get("provider", ""),
                    "ts": time.time(),
                })
            else:
                approval_persist_failed_holder[0] = persist_failure
                log.error(
                    "approval persist FAILED id=%s tool=%s — surfacing failure to user: %s",
                    req_id, persist_failure["tool_name"], persist_failure["detail"],
                )
                _publish_event_direct({
                    "_type": "tool_approval_persist_failed",
                    "turn_id": turn_log_id,
                    "trigger_message_id": trigger_message_id,
                    "bot_id": bot_id,
                    "user_id": user_id,
                    "request_id": req_id,
                    "tool_name": persist_failure["tool_name"],
                    "detail": persist_failure["detail"],
                    "ts": time.time(),
                })

        # TASK-370: tool_use_ids of AskUserQuestion awaits already persisted +
        # surfaced by the streaming WORKER THREAD (_intercept_tool_events), which
        # runs to completion regardless of client/SSE teardown. Before this, a
        # deferred question was persisted INLINE in the async generator — which
        # can be torn down on turn-end before it drains the await chunk on a LONG
        # turn, silently dropping the question (turn ends end_reason="stop", no
        # chat_pending_questions row, bot believes it asked and never retries).
        # This is the exact durability fix approvals got in TASK-292/306.
        # _await_handled guards the async-gen path from double-handling.
        _await_handled: set[str] = set()

        def _persist_publish_await(chunk: dict) -> None:
            """Durably record + surface one deferred AskUserQuestion (worker thread).

            Mirrors _persist_publish_approval: persists the pending question and
            publishes the live unified ``tool_await_result`` event from a thread
            that survives async-gen teardown, and sets ``question_id_holder`` so
            the turn finalizes with end_reason="question" even if the async
            generator never drains the chunk.
            """
            tool_use_id = chunk.get("tool_use_id") or ""
            if not tool_use_id or tool_use_id in _await_handled:
                return
            _await_handled.add(tool_use_id)
            tool_args = chunk.get("arguments") or {}
            origin_harness = (chunk.get("provider") or "claude") or "claude"
            # Set BEFORE the persist so the turn ends as a question regardless of
            # a transient store failure (matches the prior inline semantics).
            question_id_holder[0] = tool_use_id
            try:
                # A turn that re-asks supersedes its earlier pending question so
                # only the latest stays answerable.
                self._pending_question_store.supersede_awaiting_for_turn(
                    turn_log_id, keep=tool_use_id,
                )
                self._pending_question_store.upsert_awaiting(
                    tool_use_id=tool_use_id,
                    bot_id=bot_id,
                    user_id=user_id,
                    turn_id=turn_log_id,
                    arguments=tool_args if isinstance(tool_args, dict) else {"value": tool_args},
                    tool_name=chunk.get("tool_name") or "AskUserQuestion",
                    trigger_message_id=trigger_message_id,
                    session_key=chunk.get("session_key") or None,
                    origin_harness=origin_harness,
                )
            except Exception as _persist_err:
                log.warning(
                    "Failed to persist pending question tool_use_id=%s: %s",
                    tool_use_id, _persist_err,
                )
            _publish_event_direct({
                "_type": "tool_await_result",
                "turn_id": turn_log_id,
                "trigger_message_id": trigger_message_id,
                "bot_id": bot_id,
                "user_id": user_id,
                "tool_use_id": tool_use_id,
                "tool_name": chunk.get("tool_name", ""),
                "arguments": tool_args,
                "session_key": chunk.get("session_key", ""),
                "provider": chunk.get("provider", ""),
                "ts": time.time(),
            })

        def _stream_to_queue():
            """Run streaming in a thread and push chunks to the async queue."""
            def _turn_was_aborted() -> bool:
                try:
                    current_turn = self._turn_log_store.get_turn(turn_log_id)
                    return current_turn is not None and current_turn.status == "aborted"
                except Exception:
                    return False

            try:
                # Check if already cancelled before starting
                if cancel_event.is_set():
                    cancelled_holder[0] = True
                    return

                # Inject client-supplied system context (e.g. HA device list)
                llm_bawt._client_system_context = request.client_system_context
                llm_bawt._ha_mode = request.ha_mode
                # Bot's include_summaries=false overrides request default
                if hasattr(llm_bawt.bot, 'include_summaries') and not llm_bawt.bot.include_summaries:
                    llm_bawt._include_summaries = False
                else:
                    llm_bawt._include_summaries = request.include_summaries
                # Agent backends use a per-turn user-message prefix for
                # voice mode (chat.agent_voice_prefix), so tts_mode must
                # reflect the UI toggle only — bot.tts_mode is a profile
                # default for chatbots whose system prompt carries the TTS
                # instructions. Mixing both double-doses the constraint and
                # cripples agent tool chaining.
                _is_agent = llm_bawt.client.model_definition.get("type") in (
                    "agent_backend", "claude-code",
                )
                if _is_agent:
                    llm_bawt._tts_mode = request.tts_mode
                else:
                    llm_bawt._tts_mode = request.tts_mode or llm_bawt.bot.tts_mode
                llm_bawt._inject_user_prefix = bool(request.inject_user_prefix)

                # TASK-214: animations now arrive on the request payload from
                # bawthub (Prisma is the source of truth). llm-bawt no longer
                # owns the catalog. Caller pre-filters to enabled rows.
                #
                # TASK-215: animation selection no longer happens via an
                # injected tool call. The main LLM prompt is left untouched
                # (no "you MUST call trigger_animation" hack). Instead, we
                # run a local embedding classifier on the assistant's
                # response text *after* it finishes streaming — see the hook
                # below, before the sentinel is enqueued. The classifier
                # only runs when an avatar is actually visible so we don't
                # waste CPU on a response no one can see animated.
                _animations: list = list(request.animations or []) if llm_bawt._tts_mode else []
                _avatar_visible: bool = bool(request.avatar_visible) if request.avatar_visible is not None else False
                _run_classifier = bool(
                    llm_bawt._tts_mode and _avatar_visible and _animations
                )
                llm_bawt._avatar_visible = _avatar_visible
                log.info(
                    "🎬 tts_mode=%s (req=%s bot=%s) avatar_visible=%s %d animations classifier=%s",
                    llm_bawt._tts_mode, request.tts_mode, llm_bawt.bot.tts_mode,
                    _avatar_visible, len(_animations), _run_classifier,
                )

                # Use llm_bawt.prepare_messages_for_query to get full context
                # (history from DB + memory + system prompt).
                #
                # TASK-225: ``attachments`` is the tiny ``{asset_id, kind}``
                # JSONB payload persisted on the user-message row.  Bytes
                # for the LLM live on ``user_attachments`` (separate
                # argument) and never touch the DB.
                # TASK-391: agent backends can *see* attached images inline but
                # their shell/tools can't fetch blob: URLs. Build a curlable
                # "Attached Images" manifest and hand it to the agent ONLY, via
                # context_suffix — it rides the outbound user message (like the
                # per-turn agent prefixes) and is NEVER persisted. The durable
                # record of what was attached is the tiny {asset_id, kind} refs
                # on the message row (``attachments`` below); the manifest is
                # just a per-turn, absolute-URL rendering of those same handles
                # so a sibling-container tool can curl them. user_prompt stays
                # clean in history + turn log; non-agent chat is unaffected.
                agent_context_suffix = None
                if is_agent_backend and attachments_to_persist and media_store is not None:
                    try:
                        manifest = build_agent_image_manifest(
                            attachments_to_persist,
                            media_store.db,
                            origin=getattr(self.config, "AGENT_ORIGIN", "") or "",
                        )
                    except Exception as _manifest_err:
                        manifest = ""
                        log.warning(
                            "TASK-391: failed to build attachment manifest: %s",
                            _manifest_err,
                        )
                    if manifest:
                        agent_context_suffix = manifest

                messages = llm_bawt.prepare_messages_for_query(
                    user_prompt,
                    user_attachments=user_attachments or None,
                    message_id=trigger_message_id,
                    attachments=attachments_to_persist or None,
                    context_suffix=agent_context_suffix,
                )

                # TASK-215: the "AVATAR ANIMATION: You MUST call trigger_animation"
                # prompt-injection hack has been removed. Animation selection
                # now runs as a post-hoc embedding classifier (see hook after
                # consume_stream_chunks below) — the main LLM call no longer
                # has to manage a tool call on every turn.

                # Backfill prepared messages into the turn log.
                # TASK-501: when the app injected a session seed for this turn,
                # splice it into the logged prompt so the turn log reflects what
                # the harness session actually received: [system, ...seed
                # history..., user]. Additive — nothing dropped. The seed dicts
                # and Message objects both serialize via _message_to_dict.
                _logged_messages = messages
                if inject_seed_messages:
                    _system = [m for m in messages if getattr(m, "role", None) == "system"]
                    _rest = [m for m in messages if getattr(m, "role", None) != "system"]
                    _logged_messages = [*_system, *inject_seed_messages, *_rest]
                self._update_turn_log(
                    turn_id=turn_log_id,
                    prepared_messages=_logged_messages,
                )

                # Log what we're sending to the LLM (verbose mode)
                log.llm_context(messages)

                # Track when first token arrives
                timing_holder[0] = time.time()

                # Choose streaming method based on whether bot uses tools.
                # Agent-backend models (e.g. claude-code/openclaw) already handle
                # tool execution in their own bridge/runtime and may report
                # tool_format="none" at the llm-bawt model layer, so they must not
                # be forced through llm-bawt's text/native tool loop here.
                should_use_llm_bawt_tool_streaming = (
                    not is_agent_backend
                    and llm_bawt.bot.uses_tools
                    and (llm_bawt.memory or llm_bawt.home_client or llm_bawt.ha_native_client)
                )
                if should_use_llm_bawt_tool_streaming:
                    # Check if client supports native streaming with tools (OpenAI)
                    use_native_streaming = (
                        llm_bawt.client.supports_native_tools()
                        and llm_bawt.tool_format in ("native", "NATIVE_OPENAI")
                        and hasattr(llm_bawt.client, "stream_with_tools")
                    )

                    # Resolve per-bot generation parameters
                    gen_kwargs = llm_bawt._get_generation_kwargs()

                    if use_native_streaming:
                        # Native streaming with tools - streams content AND handles tool calls
                        from ..tools.executor import ToolExecutor
                        from ..tools.formats import get_format_handler
                        from ..models.message import Message as Msg

                        log.debug("Using native streaming with tools")

                        tool_definitions = llm_bawt._get_tool_definitions() if llm_bawt.bot.uses_tools else []
                        handler = get_format_handler(llm_bawt.tool_format)
                        tools_schema = handler.get_tools_schema(tool_definitions)

                        # TASK-215: the virtual trigger_animation tool has been
                        # removed. The classifier picks the animation post-hoc.

                        # Log tool names for debugging
                        tool_names = [t.get("function", {}).get("name", "?") for t in tools_schema]
                        ha_tools = [n for n in tool_names if n.startswith("Hass") or n in ("GetLiveContext",)]
                        log.info(f"📋 {len(tools_schema)} tools in schema ({len(ha_tools)} HA: {', '.join(ha_tools[:5])}{'...' if len(ha_tools) > 5 else ''})")

                        executor = ToolExecutor(
                            memory_client=llm_bawt.memory,
                            profile_manager=llm_bawt.profile_manager,
                            search_client=llm_bawt.search_client,
                            home_client=llm_bawt.home_client,
                            ha_native_client=llm_bawt.ha_native_client,
                            model_lifecycle=llm_bawt.model_lifecycle,
                            config=llm_bawt.config,
                            user_id=llm_bawt.user_id,
                            bot_id=llm_bawt.bot_id,
                        )

                        def native_stream_with_tool_loop():
                            """Stream with native tool support, handling tool calls inline."""
                            import json as _json
                            from ..tools.parser import ToolCall, strip_tool_result_tags

                            current_msgs = list(messages)
                            max_iterations = 3 if llm_bawt._ha_mode else 5
                            has_executed_tools = False

                            for iteration in range(max_iterations):
                                # After first tool execution, keep tools_schema
                                # but force tool_choice="none" so the model
                                # generates a text response.  Dropping the schema
                                # entirely causes some providers (xAI) to ignore
                                # function_call_output items and return nothing.
                                if has_executed_tools:
                                    current_tools = tools_schema
                                    current_tool_choice = "none"
                                else:
                                    current_tools = tools_schema
                                    current_tool_choice = "auto"

                                # TASK-216: stream text tokens immediately instead
                                # of buffering until we know whether a tool will
                                # fire. The original buffer was a workaround for
                                # models that emit preamble before tool calls
                                # ("I don't have live access, let me check…").
                                # Modern Claude / GPT-5 / Grok rarely do that, but
                                # the buffering imposed 700-2700ms TTFT on every
                                # tool-capable bot — which is what made voice mode
                                # feel laggy (see scripts/probe_voice_latency.py).
                                #
                                # Trade-off: in the rare case a model does emit
                                # preamble before a tool call, those tokens will
                                # leak to the client. For voice that's harmless
                                # (TTS speaks naturally); for chat the preamble
                                # text shows up alongside the tool result, which
                                # is acceptable.
                                #
                                # We still track yielded text in `yielded_text` so
                                # downstream code that previously consumed
                                # text_buffer (e.g. the no-tool-call early return)
                                # can be reasoned about.
                                yielded_text: list[str] = []

                                for item in llm_bawt.client.stream_with_tools(
                                    current_msgs,
                                    tools_schema=current_tools,
                                    tool_choice=current_tool_choice,
                                    **gen_kwargs,
                                ):
                                    if isinstance(item, str):
                                        # Stream immediately — no buffering.
                                        yielded_text.append(item)
                                        yield item
                                    elif isinstance(item, dict) and "tool_calls" in item:
                                        tool_calls = item["tool_calls"]

                                        if not tool_calls:
                                            # Empty tool_calls list — treat as pure text response.
                                            # Text was already streamed above; nothing more to flush.
                                            return

                                        # TASK-215: trigger_animation interception removed.
                                        # All tool calls in this stream are real.
                                        real_tool_calls = tool_calls

                                        # Tools follow streamed text. The text has
                                        # already been delivered to the client; we
                                        # can't take it back. (No-op kept here so
                                        # downstream diff is small.)
                                        yielded_text.clear()

                                        # Emit tool calls as OpenAI delta.tool_calls, then execute
                                        tool_results = []
                                        for idx, tc in enumerate(real_tool_calls):
                                            func = tc.get("function", {})
                                            name = func.get("name", "")
                                            call_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                                            args_str = func.get("arguments", "{}")
                                            try:
                                                args = _json.loads(args_str) if args_str else {}
                                            except _json.JSONDecodeError:
                                                args = {}

                                            log.info(f"🔧 {name}({args})")
                                            # Push OpenAI-format tool_call delta to queue
                                            put_queue_item_threadsafe(
                                                loop,
                                                chunk_queue,
                                                {
                                                    "_type": "tool_call_delta",
                                                    "index": idx,
                                                    "id": call_id,
                                                    "name": name,
                                                    "arguments": args_str,
                                                },
                                            )
                                            # Publish tool_start directly to Redis
                                            # (bypasses chunk_queue so events reach
                                            # Redis even if the SSE generator is dead)
                                            _publish_event_direct({
                                                "_type": "tool_event",
                                                "event": "tool_start",
                                                "turn_id": turn_log_id,
                                                "trigger_message_id": trigger_message_id,
                                                "bot_id": bot_id,
                                                "user_id": user_id,
                                                "tool_name": name,
                                                "arguments": args,
                                                "call_id": call_id,
                                                "iteration": iteration + 1,
                                                "ts": time.time(),
                                            })

                                            tool_call_obj = ToolCall(name=name, arguments=args, raw_text="")
                                            raw_result = executor.execute(tool_call_obj)
                                            # Strip <tool_result> XML tags — native tool path
                                            # sends results as structured function_call_output
                                            # items, not inline text.
                                            result = strip_tool_result_tags(raw_result)
                                            tool_results.append({
                                                "tool_call_id": tc.get("id", ""),
                                                "content": result,
                                            })
                                            # Publish tool_end directly to Redis
                                            _publish_event_direct({
                                                "_type": "tool_event",
                                                "event": "tool_end",
                                                "turn_id": turn_log_id,
                                                "trigger_message_id": trigger_message_id,
                                                "bot_id": bot_id,
                                                "user_id": user_id,
                                                "tool_name": name,
                                                "arguments": args,
                                                "call_id": call_id,
                                                "result": result or "",
                                                "iteration": iteration + 1,
                                                "ts": time.time(),
                                            })

                                            # Persist for TurnLog.tool_calls_json
                                            tool_call_details_holder.append({
                                                "tool": name,
                                                "arguments": args,
                                                "call_id": call_id,
                                                "result": result or "",
                                                "iteration": iteration + 1,
                                            })

                                        has_executed_tools = True

                                        # Signal finish_reason: "tool_calls" to consumer
                                        put_queue_item_threadsafe(
                                            loop,
                                            chunk_queue,
                                            {"_type": "tool_calls_finish"},
                                        )

                                        # Store tool context
                                        tool_context_holder[0] = "\n\n".join(
                                            f"[{tc['function']['name']}]\n{tr['content']}"
                                            for tc, tr in zip(real_tool_calls, tool_results)
                                        )

                                        # Build continuation messages — empty content so the
                                        # follow-up pass is grounded on tool results, not the
                                        # discarded pre-tool fallback text.
                                        assistant_msg = Msg(
                                            role="assistant",
                                            content="",
                                            tool_calls=[
                                                {"id": tc.get("id"), "name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                                                for tc in real_tool_calls
                                            ],
                                        )
                                        current_msgs.append(assistant_msg)

                                        for tc, tr in zip(real_tool_calls, tool_results):
                                            current_msgs.append(Msg(
                                                role="tool",
                                                content=tr["content"],
                                                tool_call_id=tr["tool_call_id"],
                                            ))

                                        # Ground the synthesis pass on tool results.
                                        # Without this, stale history summaries (e.g.
                                        # "assistant lacks live access") can override
                                        # the actual retrieved data in the model's response.
                                        current_msgs.append(Msg(
                                            role="system",
                                            content="You have retrieved live data using tools above. Answer the user's question directly and confidently using those results. Do not claim to lack live access.",
                                        ))

                                        # Continue to next iteration (will stream the follow-up)
                                        # Yield paragraph break so post-tool text is
                                        # separated from any prior content.  This flows
                                        # into full_response_holder (persisted to DB)
                                        # and the SSE stream.
                                        yield "\n\n"
                                        break
                                else:
                                    # Stream finished without a tool_calls dict
                                    # (pure content response). TASK-216: tokens
                                    # were already yielded inline above, so
                                    # nothing left to flush.
                                    return

                            log.warning(f"Tool loop: max iterations ({max_iterations}) reached")

                        stream_iter = native_stream_with_tool_loop()
                    else:
                        # Fall back to text-based streaming for non-native models (GGUF, etc.)
                        from ..tools import stream_with_tools
                        log.debug(f"Using stream_with_tools for tool format: {llm_bawt.tool_format}")

                        adapter = getattr(llm_bawt, 'adapter', None)
                        if adapter:
                            log.debug(f"Passing adapter '{adapter.name}' to stream_with_tools")
                        else:
                            log.warning("No adapter found on llm_bawt instance")

                        def stream_fn(msgs, stop_sequences=None):
                            return llm_bawt.client.stream_raw(msgs, stop=stop_sequences, **gen_kwargs)

                        stream_iter = stream_with_tools(
                            messages=messages,
                            stream_fn=stream_fn,
                            memory_client=llm_bawt.memory,
                            profile_manager=llm_bawt.profile_manager,
                            search_client=llm_bawt.search_client,
                            home_client=llm_bawt.home_client,
                            ha_native_client=llm_bawt.ha_native_client,
                            model_lifecycle=llm_bawt.model_lifecycle,
                            config=llm_bawt.config,
                            user_id=llm_bawt.user_id,
                            bot_id=llm_bawt.bot_id,
                            tool_format=llm_bawt.tool_format,
                            adapter=adapter,
                            history_manager=llm_bawt.history_manager,
                            tool_call_details=tool_call_details_holder,
                        )
                else:
                    # Resolve per-bot generation parameters
                    gen_kwargs = llm_bawt._get_generation_kwargs()
                    # Pass adapter stop sequences even without tools
                    adapter = getattr(llm_bawt, 'adapter', None)
                    adapter_stops = adapter.get_stop_sequences() if adapter else []
                    extra_kwargs = {}
                    if is_agent_backend and user_attachments:
                        extra_kwargs["attachments"] = user_attachments
                    if is_agent_backend and trigger_message_id:
                        # Forward the frontend user-message UUID to the bridge
                        # so every emitted tool event carries it (frontend
                        # buckets tool activity by trigger_message_id).
                        extra_kwargs["trigger_message_id"] = trigger_message_id
                    if is_agent_backend and inject_seed_messages:
                        # TASK-501: push the pre-assembled seed so the bridge
                        # seeds the fresh session without the context-seed callback.
                        extra_kwargs["inject_messages"] = inject_seed_messages
                    stream_iter = llm_bawt.client.stream_raw(
                        messages, stop=adapter_stops or None, **gen_kwargs, **extra_kwargs
                    )

                # Wrap stream to publish tool events directly to Redis
                # from this thread (survives SSE generator cancellation).
                _oc_call_index = [0]
                # Stack of (call_id, tool_name) for in-flight OpenClaw tool
                # calls.  Each tool_result pops by name match so nested calls
                # (Agent → Grep → Read) pair correctly with their tool_call.
                _oc_call_stack: list[tuple[str, str]] = []

                _oc_request_id_captured = [False]
                # Fires once, on the SDK/bridge-CONFIRMED first output of this
                # turn, to reap any other still-open turns for this bot.
                _confirmed_start_reaped = [False]

                def _intercept_tool_events(inner):
                    _saw_tool = False  # Track tool→content transitions
                    _saw_text = False  # Track whether any text has been yielded
                    # Cumulative count of assistant text chars yielded so far.
                    # Stamped onto each tool_call as text_offset so the frontend
                    # can split response_text at tool boundaries and render an
                    # interleaved transcript (text → tool → text) that survives
                    # reload. MUST match what consume_stream_chunks accumulates
                    # into full_response_holder, so count EVERY str yielded
                    # (including the injected "\n\n" breaks below).
                    _text_chars = [0]

                    # TASK-286: buffer assistant text and publish it to the
                    # unified Redis stream as coalesced ``text_delta`` events so
                    # a refreshed / secondary client recovers (and keeps
                    # streaming) the response TEXT alongside tool calls. Buffer
                    # must stay in lock-step with ``_text_chars`` so each delta's
                    # ``text_offset`` (chars emitted before it) is exact.
                    #
                    # TASK-456: this was originally gated to agent backends
                    # (claude-code / openclaw) — the only path whose text lacked a
                    # durable/resumable channel. The gate is now removed so NATIVE
                    # chat bots also publish coalesced text_delta. Two reasons:
                    #   (1) it gives native bots the same cross-tab / cold-reload
                    #       text recovery agent bots already have, and
                    #   (2) it is the token source the server-side per-turn chat
                    #       TTS driver subscribes to (TASK-453/457).
                    # Safe against double-render: the frontend routes text_delta
                    # into a SEPARATE ``partialResponseByUserMsgId`` bucket that
                    # ``useInFlightTurn`` only surfaces when NO live HTTP stream
                    # exists (refreshed/secondary tab); the originating tab's HTTP
                    # body wins (useMessageStreamState.ts:122-146). OQ-1 verified.
                    _text_buf: list[str] = []

                    def _flush_text(min_chars: int = 0):
                        if not _text_buf:
                            return
                        s = "".join(_text_buf)
                        if len(s) < min_chars:
                            return
                        _text_buf.clear()
                        _publish_event_direct({
                            "_type": "text_delta",
                            "turn_id": turn_log_id,
                            "trigger_message_id": trigger_message_id,
                            "bot_id": bot_id,
                            "user_id": user_id,
                            # Chars of assistant text emitted BEFORE this delta —
                            # lets the client splice it at the right position and
                            # dedupe an overlapping/replayed delta regardless of
                            # arrival order vs the cold-reload partial fetch.
                            "text_offset": _text_chars[0] - len(s),
                            "delta": s,
                            "ts": time.time(),
                        })
                        # Voice-optimized turns: feed the SAME raw text through the
                        # block-boundary scrubber and emit any completed, scrubbed
                        # block(s) as `tts_delta`. The raw text_delta above still
                        # carries markdown for the visual stream; only the audio
                        # path consumes tts_delta. (No-op for non-voice bots.)
                        if tts_scrubber is not None:
                            _tts_block = tts_scrubber.feed(s)
                            if _tts_block:
                                _publish_event_direct({
                                    "_type": "tts_delta",
                                    "turn_id": turn_log_id,
                                    "bot_id": bot_id,
                                    "user_id": user_id,
                                    "delta": _tts_block,
                                    "ts": time.time(),
                                })

                    for item in inner:
                        # CONFIRMED turn start. This first item is the backend/
                        # SDK's first real output (native first chunk, or an
                        # agent-bridge ASSISTANT_DELTA/tool event) — proof the
                        # turn is genuinely running, not the optimistic
                        # save_turn insert or turn_start publish that fire
                        # before anything is produced. On this confirmed signal,
                        # reap any OTHER still-open turns for this bot as
                        # timeouts and clear their UI indicators. Runs once.
                        if not _confirmed_start_reaped[0]:
                            _confirmed_start_reaped[0] = True
                            try:
                                _reaped = self._turn_log_store.reap_other_open_turns(
                                    bot_id=bot_id, current_turn_id=turn_log_id,
                                )
                                for _rt in _reaped:
                                    _ruid = _rt.get("user_id") or user_id
                                    _evt = {
                                        "_type": "turn_complete",
                                        "turn_id": _rt["id"],
                                        "bot_id": bot_id,
                                        "user_id": _ruid,
                                        "status": "timeout",
                                        "end_reason": "timeout",
                                        "ts": time.time(),
                                    }
                                    # Publish to the reaped turn's OWN
                                    # {bot_id}:{user_id} stream so the right
                                    # client clears (its user may differ from
                                    # the current turn's).
                                    if _redis_sub:
                                        try:
                                            asyncio.run_coroutine_threadsafe(
                                                _redis_sub.publish_tool_event(
                                                    bot_id, _ruid, _evt,
                                                ),
                                                loop,
                                            )
                                        except Exception as _pub_err:
                                            log.debug(
                                                "reap turn_complete publish failed for %s: %s",
                                                _rt["id"], _pub_err,
                                            )
                                if _reaped:
                                    log.info(
                                        "Reaped %d stale open turn(s) for bot %s on confirmed start of %s",
                                        len(_reaped), bot_id, turn_log_id,
                                    )
                            except Exception as _reap_err:
                                log.debug(
                                    "confirmed-start reap failed for %s: %s",
                                    bot_id, _reap_err,
                                )

                        # Capture agent_request_id on first yielded item
                        if is_agent_backend and not _oc_request_id_captured[0]:
                            _oc_request_id_captured[0] = True
                            backend = getattr(llm_bawt.client, "_backend", None)
                            oc_req_id = getattr(backend, "_active_request_id", None)
                            if oc_req_id:
                                self._update_turn_log(
                                    turn_id=turn_log_id,
                                    agent_request_id=oc_req_id,
                                )

                        # Inject paragraph break when transitioning from
                        # tool events back to text content.  This flows into
                        # full_response_holder (persisted to DB) and the SSE
                        # stream, ensuring paragraph breaks survive reload.
                        if isinstance(item, str) and item.strip():
                            if _saw_tool and _saw_text:
                                _text_chars[0] += 2
                                _text_buf.append("\n\n")
                                yield "\n\n"
                            _saw_tool = False
                            _saw_text = True

                        if isinstance(item, dict):
                            evt = item.get("event")
                            if evt == "approval_required":
                                # Persist + surface durably from this worker
                                # thread so it survives the async-gen teardown
                                # that drops the chunk on turn-end (TASK-292/306
                                # race). Falls through to `yield item`; the
                                # async-gen branch dedups via _approval_handled.
                                _persist_publish_approval(item)
                            if evt == "await_tool_result":
                                # TASK-370: persist + surface the deferred
                                # AskUserQuestion from this worker thread so it
                                # survives async-gen teardown on long turns (the
                                # same fix approvals got in TASK-292/306). Falls
                                # through to `yield item`; the async-gen branch is
                                # now just the originating-client SSE sidecar and
                                # dedups via _await_handled.
                                _persist_publish_await(item)
                            if evt == "reasoning":
                                # Model reasoning ("thinking"). Two destinations,
                                # both dict-only so it NEVER enters
                                # full_response_holder / the saved message
                                # (TASK-301):
                                #   1. unified Redis stream → live cross-tab +
                                #      cold-reload recovery of the lane.
                                #   2. yielded downstream → OpenAI-compat SSE as
                                #      choices[].delta.reasoning_content.
                                # Orthogonal to the answer transcript: does NOT
                                # touch _text_chars/_text_buf/_saw_text/_saw_tool.
                                _rtext = item.get("text", "")
                                if _rtext:
                                    # TASK-301: accumulate for display-only
                                    # persistence on the assistant row. Stays out
                                    # of full_response_holder so it never enters
                                    # the answer transcript or LLM context.
                                    # Absolute reasoning-char offset (chars of
                                    # reasoning emitted BEFORE this delta) —
                                    # parallels text_offset. Lets a refreshed
                                    # client splice each live delta at its true
                                    # position against the cold-reload seed
                                    # (resumeScan -> turn_logs.reasoning) with no
                                    # gap and no double-count, regardless of
                                    # connect-vs-snapshot ordering. Without it the
                                    # reasoning lane freezes at the seed until
                                    # turn_complete (TASK-506).
                                    _r_offset = len(reasoning_holder[0])
                                    reasoning_holder[0] += _rtext
                                    _publish_event_direct({
                                        "_type": "reasoning_delta",
                                        "turn_id": turn_log_id,
                                        "trigger_message_id": trigger_message_id,
                                        "bot_id": bot_id,
                                        "user_id": user_id,
                                        "delta": _rtext,
                                        "reasoning_offset": _r_offset,
                                        "ts": time.time(),
                                    })
                                    yield {"_type": "reasoning_delta", "delta": _rtext}
                                continue
                            if evt == "metadata":
                                if item.get("upstream_model"):
                                    _upstream_model[0] = item["upstream_model"]
                                continue
                            if evt == "token_usage":
                                # Capture for turn_complete payload — frontend
                                # surfaces input/cache/output + context window
                                # in the assistant bubble's lower-right pill.
                                tu = item.get("token_usage")
                                if isinstance(tu, dict):
                                    token_usage_holder[0] = tu
                                continue
                            if evt == "attachments":
                                refs = item.get("attachments")
                                if isinstance(refs, list):
                                    agent_attachments_holder.extend(
                                        r for r in refs if isinstance(r, dict)
                                    )
                                continue
                            if evt in ("subagent_started", "subagent_progress", "subagent_done"):
                                # TASK-344: sub-agent lifecycle events. Publish
                                # to unified stream for cross-tab visibility, then
                                # skip downstream (not part of the text/tool response).
                                _publish_event_direct({
                                    "_type": evt,
                                    "turn_id": turn_log_id,
                                    "trigger_message_id": item.get("trigger_message_id") or trigger_message_id,
                                    "bot_id": bot_id,
                                    "user_id": user_id,
                                    "task_id": item.get("task_id", ""),
                                    "description": item.get("description", ""),
                                    "task_type": item.get("task_type"),
                                    "status": item.get("status"),
                                    "summary": item.get("summary"),
                                    "last_tool_name": item.get("last_tool_name"),
                                    "usage": item.get("usage"),
                                    "tool_use_id": item.get("tool_use_id"),
                                    "provider": item.get("provider"),
                                    "ts": time.time(),
                                })
                                continue
                            if evt in ("tool_call", "tool_result"):
                                _saw_tool = True
                                # Flush buffered text BEFORE the tool event so
                                # the resume channel preserves text→tool order.
                                _flush_text()
                            if evt == "tool_call":
                                _oc_call_index[0] += 1
                                cid = f"call_{uuid.uuid4().hex[:8]}"
                                _oc_tool_name = item.get("name", "unknown")
                                _oc_call_stack.append((cid, _oc_tool_name))
                                # Inject call_id into the dict so the queue consumer
                                # uses the same ID as the SSE event (no double-ID).
                                item["_call_id"] = cid
                                # Stash the AUTHORITATIVE text_offset (chars of
                                # assistant text before this tool) onto the same
                                # dict so the downstream OpenAI-compat emitter can
                                # put it on the live SSE wire. Without this the
                                # client recomputes the offset from its own running
                                # accumulatedContent.length, which drifts from the
                                # server's _text_chars count and splits bubbles
                                # mid-sentence live while refresh (which uses this
                                # same value, persisted below) renders correctly
                                # (TASK-367).
                                item["_text_offset"] = _text_chars[0]
                                tool_call_details_holder.append({
                                    'tool': item.get('name', 'unknown'),
                                    'parameters': item.get('arguments', {}),
                                    'call_id': cid,
                                    # Chars of assistant text before this tool —
                                    # drives interleaved-transcript reconstruction.
                                    'text_offset': _text_chars[0],
                                    # Persist the originating backend so when the
                                    # turn is recalled from tool_calls_json, the
                                    # frontend's (provider, tool_name) renderer
                                    # registry can dispatch to FileChangeBody /
                                    # BashBody / WebSearchBody. Without this,
                                    # recall falls back to GenericClaudeBody.
                                    'provider': item.get('provider'),
                                    # SDK tool_use id + parent id so sub-agent
                                    # nesting (child card under its Agent card)
                                    # survives reload from tool_calls_json, not just
                                    # live rendering (TASK-344).
                                    'tool_use_id': item.get('tool_use_id'),
                                    'parent_tool_use_id': item.get('parent_tool_use_id'),
                                })
                                _publish_event_direct({
                                    "_type": "tool_event",
                                    "event": "tool_start",
                                    "turn_id": turn_log_id,
                                    # Prefer the bridge-stamped value from the
                                    # event itself (authoritative for that turn),
                                    # falling back to the request-scope value
                                    # for backends that haven't been updated yet.
                                    "trigger_message_id": item.get("trigger_message_id") or trigger_message_id,
                                    "bot_id": bot_id,
                                    "user_id": user_id,
                                    "tool_name": item.get("name", "unknown"),
                                    "arguments": item.get("arguments", {}),
                                    "call_id": cid,
                                    # SDK tool_use id: the stable anchor a sub-agent's
                                    # child cards match their parent_tool_use_id against
                                    # (the Agent card is keyed by this). Was never on
                                    # the live wire before — without it, nesting can't
                                    # resolve live (TASK-344).
                                    "tool_use_id": item.get("tool_use_id"),
                                    "parent_tool_use_id": item.get("parent_tool_use_id"),
                                    "iteration": 1,
                                    "provider": item.get("provider"),
                                    "ts": time.time(),
                                    "text_offset": _text_chars[0],
                                })
                            elif evt == "tool_result":
                                # Pair tool_result with its in-flight tool_call.
                                # Match by tool_name (handles nested calls
                                # finishing out of declared order); fall back
                                # to popping the innermost entry.
                                _result_name = item.get("name") or ""
                                _end_cid = ""
                                if _oc_call_stack:
                                    _matched_idx = -1
                                    for _i in range(len(_oc_call_stack) - 1, -1, -1):
                                        _cid_i, _name_i = _oc_call_stack[_i]
                                        if _name_i == _result_name:
                                            _matched_idx = _i
                                            break
                                    if _matched_idx < 0:
                                        _matched_idx = len(_oc_call_stack) - 1
                                    _end_cid, _ = _oc_call_stack.pop(_matched_idx)
                                item["_call_id"] = _end_cid
                                # TASK-483: enrich this call's screenshot refs
                                # ({asset_id, kind}, bridge-stamped on TOOL_END)
                                # into the canonical envelope (mime_type/width/
                                # height/urls) — the SAME shape /v1/history
                                # emits — so the live tool card can render the
                                # thumbnail inline the moment the tool finishes.
                                # We're in the worker thread here, so the sync
                                # media_assets SELECT is fine (mirrors the
                                # turn_start enrichment, which offloads only
                                # because it runs on the event loop). Failure
                                # degrades to no inline image; the end-of-turn
                                # grid still covers it.
                                _tool_end_attachments = None
                                if item.get("attachments"):
                                    try:
                                        from ..media.assets import MediaAssetStore
                                        from ..media.serializers import (
                                            enrich_attachments_for_messages,
                                        )
                                        _shell = [{
                                            "attachments": list(item["attachments"]),
                                        }]
                                        enrich_attachments_for_messages(
                                            _shell, MediaAssetStore(self.config)
                                        )
                                        _tool_end_attachments = (
                                            _shell[0].get("attachments") or None
                                        )
                                    except Exception as _enrich_err:
                                        log.warning(
                                            "TASK-483: tool_end attachment "
                                            "enrichment failed: %s",
                                            _enrich_err,
                                        )
                                _publish_event_direct({
                                    "_type": "tool_event",
                                    "event": "tool_end",
                                    "turn_id": turn_log_id,
                                    # Prefer bridge-stamped value (see tool_start above).
                                    "trigger_message_id": item.get("trigger_message_id") or trigger_message_id,
                                    "bot_id": bot_id,
                                    "user_id": user_id,
                                    "tool_name": _result_name or "unknown",
                                    "call_id": _end_cid,
                                    "result": item.get("result", ""),
                                    "tool_result_payload": item.get("tool_result_payload"),
                                    # Same ids as tool_start so a reconnect that only
                                    # sees the tool_end can still place the card under
                                    # its parent Agent (TASK-344).
                                    "tool_use_id": item.get("tool_use_id"),
                                    "parent_tool_use_id": item.get("parent_tool_use_id"),
                                    "iteration": 1,
                                    "provider": item.get("provider"),
                                    "ts": time.time(),
                                    # SDK failure flag threaded from the agent bridge
                                    # tool_result event → persisted by api.py so the
                                    # red error ring survives reload.
                                    "is_error": item.get("is_error"),
                                    # Enriched screenshot refs for inline render in
                                    # the live tool card (TASK-483). None when the
                                    # tool produced no media.
                                    "attachments": _tool_end_attachments,
                                })
                                # Update the matching detail entry with result +
                                # failure flag so the finalized tool_calls_json
                                # (read on reload for completed turns) keeps the
                                # red error ring, not just the live stream.
                                for _td in reversed(tool_call_details_holder):
                                    if _td.get("call_id") == _end_cid:
                                        from agent_bridge.tool_results import payload_from_event
                                        _payload = payload_from_event(
                                            item.get("tool_result_payload"), item.get("result", "")
                                        )
                                        _td["result"] = _payload.preview
                                        _td["is_error"] = item.get("is_error")
                                        break
                        if isinstance(item, str):
                            _text_chars[0] += len(item)
                            _text_buf.append(item)
                            _flush_text(min_chars=_TEXT_DELTA_FLUSH_CHARS)
                        yield item

                    # Flush trailing buffered text so the tail of the response
                    # reaches the resume channel even if it never crossed the
                    # size threshold (e.g. a short final paragraph after a tool).
                    _flush_text()

                # Stream chunks to queue
                cancelled_holder[0] = consume_stream_chunks(
                    _intercept_tool_events(stream_iter),
                    cancel_event=cancel_event,
                    loop=loop,
                    chunk_queue=chunk_queue,
                    full_response_holder=full_response_holder,
                )
                if cancelled_holder[0]:
                    log.info("Generation cancelled - newer request received")

                # TASK-215: post-hoc embedding classifier. Runs only when
                # tts_mode is on, the avatar is actually visible, and we
                # have a non-empty response. Misses (no match above the
                # similarity threshold) leave animation_holder[0] as None —
                # the frontend then stays on its idle animation rather than
                # forcing an awkward fallback gesture.
                if _run_classifier and full_response_holder[0] and not animation_holder[0]:
                    try:
                        from .animation_classifier import classify_animation
                        picked = classify_animation(full_response_holder[0], _animations)
                        if picked:
                            animation_holder[0] = picked
                    except Exception as cls_err:
                        log.debug("animation classifier failed: %s", cls_err)

                timing_holder[1] = time.time()

            except Exception as e:
                externally_aborted = _turn_was_aborted()
                if externally_aborted:
                    cancelled_holder[0] = True
                elif not cancel_event.is_set():
                    put_queue_item_threadsafe(loop, chunk_queue, e)
                elapsed_ms = (time.time() - timing_holder[0]) * 1000 if timing_holder[0] else None
                if not externally_aborted:
                    self._update_turn_log(
                        turn_id=turn_log_id,
                        status="error",
                        latency_ms=elapsed_ms,
                        response_text=full_response_holder[0] or None,
                        error_text=str(e),
                    )
            finally:
                end_time = timing_holder[1] or time.time()
                start_time = timing_holder[0] or end_time
                elapsed_ms = (end_time - start_time) * 1000
                externally_aborted = _turn_was_aborted()
                if externally_aborted:
                    cancelled_holder[0] = True

                # Wrap finalization in try/except so that the sentinel,
                # turn_complete event, and generation cleanup always fire
                # even if persistence raises (e.g. database failure).
                try:
                    if externally_aborted:
                        # /v1/chat/abort owns this terminal state. Do not let
                        # worker cleanup overwrite it as completed/timeout.
                        #
                        # TASK-286: a Stop must NOT delete the in-progress reply.
                        # If any assistant text streamed before the abort, COMMIT
                        # it to history (so it stays in the chat as a truncated
                        # turn) — _finalize_turn writes the assistant row +
                        # response_text. Pass status="aborted" so it keeps the
                        # aborted terminal state instead of flipping to "ok".
                        # With no partial text, just stamp the turn log aborted.
                        if full_response_holder[0]:
                            self._finalize_turn(
                                llm_bawt=llm_bawt,
                                turn_id=turn_log_id,
                                response_text=full_response_holder[0],
                                tool_context=tool_context_holder[0],
                                tool_call_details=tool_call_details_holder,
                                prepared_messages=_logged_messages if "_logged_messages" in locals() else (messages if "messages" in locals() else []),
                                user_prompt=user_prompt,
                                model=model_alias,
                                bot_id=bot_id,
                                user_id=user_id,
                                elapsed_ms=elapsed_ms,
                                stream=True,
                                animation=animation_holder[0],
                                token_usage=token_usage_holder[0],
                                attachments=agent_attachments_holder or None,
                                reasoning=reasoning_holder[0] or None,
                                status="aborted",
                                end_reason="aborted",
                                assistant_message_id=assistant_message_id,
                            )
                        else:
                            self._update_turn_log(
                                turn_id=turn_log_id,
                                latency_ms=elapsed_ms,
                                tool_calls=tool_call_details_holder or None,
                                end_reason="aborted",
                            )
                    elif full_response_holder[0]:
                        self._finalize_turn(
                            llm_bawt=llm_bawt,
                            turn_id=turn_log_id,
                            response_text=full_response_holder[0],
                            tool_context=tool_context_holder[0],
                            tool_call_details=tool_call_details_holder,
                            prepared_messages=_logged_messages if "_logged_messages" in locals() else (messages if "messages" in locals() else []),
                            user_prompt=user_prompt,
                            model=model_alias,
                            bot_id=bot_id,
                            user_id=user_id,
                            elapsed_ms=elapsed_ms,
                            stream=True,
                            animation=animation_holder[0],
                            token_usage=token_usage_holder[0],
                            attachments=agent_attachments_holder or None,
                            reasoning=reasoning_holder[0] or None,
                            assistant_message_id=assistant_message_id,
                        )
                    else:
                        # No response received — mark as timeout so turn doesn't
                        # stay stuck at status='streaming' forever.
                        # Persist any tool_call_details that were collected before
                        # the follow-up model call failed.
                        self._update_turn_log(
                            turn_id=turn_log_id,
                            status="timeout",
                            latency_ms=elapsed_ms,
                            tool_calls=tool_call_details_holder or None,
                        )
                except Exception as fin_err:
                    log.error("Finalization failed (turn %s): %s", turn_log_id, fin_err)
                    try:
                        self._update_turn_log(
                            turn_id=turn_log_id,
                            status="error",
                            latency_ms=elapsed_ms,
                            response_text=full_response_holder[0] or None,
                            error_text=f"finalize_error: {fin_err}",
                        )
                    except Exception:
                        pass

                # Notify SSE consumers that the turn is done so they
                # can finalize without polling the turn-log API.
                status = "completed" if full_response_holder[0] else "timeout"
                if cancelled_holder[0]:
                    status = "cancelled"
                # TASK-269: classify why the turn ended.  A deferred
                # AskUserQuestion ends the turn cleanly with end_reason
                # "question" — the persisted question stays "awaiting" (NOT
                # abandoned; the user answers later via a continuation turn).
                question_id = question_id_holder[0]
                approval_id = approval_id_holder[0]
                approval_persist_failed = approval_persist_failed_holder[0]
                if question_id:
                    end_reason = "question"
                elif approval_id:
                    # TASK-292: gated tool awaiting approval — same clean-end
                    # semantics as a question; the request stays "pending" and
                    # the user resolves it via a continuation turn.
                    end_reason = "approval"
                elif approval_persist_failed:
                    # TASK-306 Section A: a gated tool required approval but the
                    # row could not be durably committed. End honestly so the
                    # turn record reflects that the gate never reached the user.
                    end_reason = "approval_persist_failed"
                elif cancelled_holder[0]:
                    end_reason = "aborted"
                elif status == "timeout":
                    end_reason = "error"
                else:
                    end_reason = "stop"
                # Stamp the turn log so hydration/turn-log APIs can render a
                # first-class QuestionMessage for end_reason="question" turns.
                # For a persist failure, record the structured reason as
                # error_text so the agent (on continuation) and history see the
                # truth instead of a phantom success.
                try:
                    self._turn_log_store.update_turn(
                        turn_id=turn_log_id,
                        end_reason=end_reason,
                        question_id=question_id,
                        error_text=(
                            json.dumps(approval_persist_failed)
                            if approval_persist_failed else None
                        ),
                        tts_scrubbed=tts_scrub,
                    )
                except Exception as _er_err:
                    log.debug("update_turn end_reason failed for %s: %s", turn_log_id, _er_err)
                # Voice-optimized turns: flush the scrubber's final (partial)
                # block as one last tts_delta BEFORE turn_complete, so the TTS
                # driver speaks the tail then hits EOS. Skipped on cancel/abort.
                if tts_scrubber is not None and status not in ("cancelled", "aborted"):
                    _tts_tail = tts_scrubber.flush()
                    if _tts_tail:
                        _tts_tail_future = _publish_event_direct({
                            "_type": "tts_delta",
                            "turn_id": turn_log_id,
                            "bot_id": bot_id,
                            "user_id": user_id,
                            "delta": _tts_tail,
                            "ts": time.time(),
                        })
                        # The TTS consumer closes its text input as soon as it
                        # receives turn_complete. Ensure Redis has committed the
                        # final scrubber tail first; otherwise these independent
                        # cross-thread publishes may arrive in reverse order and
                        # consistently drop the response's last sentence.
                        if _tts_tail_future is not None:
                            try:
                                _tts_tail_future.result(timeout=5)
                            except Exception as _tts_order_err:
                                log.warning(
                                    "Final tts_delta publish did not complete before "
                                    "turn_complete for turn %s: %s",
                                    turn_log_id,
                                    _tts_order_err,
                                )
                _publish_event_direct({
                    "_type": "turn_complete",
                    "turn_id": turn_log_id,
                    "bot_id": bot_id,
                    "user_id": user_id,
                    "status": status,
                    "end_reason": end_reason,
                    "question_id": question_id,
                    "approval_id": approval_id,
                    "approval_persist_failed": approval_persist_failed,
                    "animation": animation_holder[0],
                    "token_usage": token_usage_holder[0],
                    # Catalog alias for this turn (matches the persisted turn-log
                    # `model` and the /v1/models pricing key) so the client can
                    # cost a live turn without waiting on the turn-log backfill.
                    "model": model_alias,
                    "ts": time.time(),
                })

                put_queue_item_threadsafe(loop, chunk_queue, None)  # Sentinel

                # Signal generation complete from the worker thread so
                # that _start_generation() properly waits for us before
                # starting a new generation for the same bot.
                if is_agent_backend:
                    done_event.set()
                else:
                    self._end_generation(cancel_event, done_event, bot_id)

        try:
            # Visual-stream emote filtering shares the SAME decision as TTS
            # scrubbing (should_scrub_for_tts), computed once above as tts_scrub.
            emote_filter = StreamingEmoteFilter() if tts_scrub else None
            oc_tool_call_index = 0  # OpenClaw agent backend tool call index
            _upstream_model = [None]  # Actual model reported by agent backend

            # TASK-367: release-at-boundary buffer for agent-backend assistant
            # text. Holds the still-forming tail of the response so the SSE wire
            # only ever advances delta.content to a CLEAN boundary (sentence /
            # clause punctuation, newline, or — as a safety valve — a word
            # boundary once the tail grows long). A tool card anchors at the
            # client's running content length; if that length can end mid-word,
            # the tool slices a half-typed token into its own bubble that only
            # heals a frame later — the live-only split Nick keeps seeing.
            # Boundary-gating the wire makes that length always land on a real
            # boundary, so the split can't form — independent of whether the
            # authoritative text_offset reached the client. Buffers ONLY the
            # wire; _text_chars, persisted tool offsets and Redis events are
            # produced upstream in the worker thread and stay untouched, so the
            # DB-refresh render is unchanged and the live render converges to it.
            _content_buf: list[str] = []
            _CONTENT_MAX_HOLD = 120  # force a word-boundary release past this
            _HARD_BOUNDARY = frozenset(".!?…\n")       # sentence / hard stops
            _SOFT_BOUNDARY = frozenset(",;:)]}\"'’”")   # clause / closing punct

            def _content_release_cut(s: str) -> int:
                """Slice index: release s[:cut], hold s[cut:]. 0 ⇒ hold all."""
                # Release through the last hard boundary (sentence end / newline).
                for i in range(len(s) - 1, -1, -1):
                    if s[i] in _HARD_BOUNDARY:
                        return i + 1
                # Else through the last clause-closing punctuation.
                for i in range(len(s) - 1, -1, -1):
                    if s[i] in _SOFT_BOUNDARY:
                        return i + 1
                # No punctuation yet: keep holding until we've buffered enough
                # that a long, punctuation-free run (code, URLs, lists) would
                # stall — then release up to the last whitespace so we never
                # leave a mid-word tail on the wire.
                if len(s) >= _CONTENT_MAX_HOLD:
                    ws = s.rfind(" ")
                    return ws + 1 if ws > 0 else len(s)
                return 0

            def _drain_content_buf(flush_all: bool):
                """SSE line for releasable buffered content, or None. Mutates buf."""
                if not _content_buf:
                    return None
                s = "".join(_content_buf)
                cut = len(s) if flush_all else _content_release_cut(s)
                if cut <= 0:
                    return None
                _content_buf.clear()
                if cut < len(s):
                    _content_buf.append(s[cut:])
                data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_alias,
                    "choices": [{"index": 0, "delta": {"content": s[:cut]}, "finish_reason": None}],
                }
                return f"data: {json.dumps(data)}\n\n"

            # Send service warnings (e.g. model fallback) before content
            if model_warnings:
                warning_data = {
                    "object": "service.warning",
                    "model": model_alias,
                    "warnings": model_warnings,
                }
                yield f"data: {json.dumps(warning_data)}\n\n"

            # In-process GPU inference is gone (local models run in the
            # standalone local_model_bridge, TASK-276/278), so every model
            # streams on the default thread pool — multiple bots run
            # concurrently and nothing is serialized on a single GPU thread.
            loop.run_in_executor(None, _stream_to_queue)

            # Yield SSE chunks (with keepalive to prevent client timeout
            # during slow backends like vLLM first-inference)
            while True:
                try:
                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Send SSE comment as keepalive to prevent client/proxy timeout
                    yield ": keepalive\n\n"
                    continue

                # DEBUG-292: trace every dict chunk through the async generator
                if isinstance(chunk, dict) and chunk.get("event") == "approval_required":
                    log.info(
                        "DEBUG-292 async_gen: approval_required chunk DEQUEUED "
                        "bot=%s turn=%s req_id=%s",
                        bot_id, turn_log_id,
                        chunk.get("tool_use_id", "?"),
                    )

                if chunk is None:
                    # Release the held text tail before closing (TASK-367).
                    sse = _drain_content_buf(flush_all=True)
                    if sse:
                        yield sse
                    # Stream complete - flush any buffered content from emote filter
                    if emote_filter:
                        final_chunk = emote_filter.flush()
                        if final_chunk:
                            data = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_alias,
                                "choices": [{"index": 0, "delta": {"content": final_chunk}, "finish_reason": None}],
                            }
                            yield (f"data: {json.dumps(data)}\n\n")

                    # Emit animation event (tts_mode only, normal completion only)
                    if animation_holder[0]:
                        anim_data = {"object": "service.animation", "animation": animation_holder[0]}
                        yield (f"data: {json.dumps(anim_data)}\n\n")

                    # Use upstream model name if available (e.g. "claude-opus-4-6[1m]")
                    final_model = _upstream_model[0] or model_alias

                    # Send final chunk
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": final_model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield (f"data: {json.dumps(data)}\n\n")
                    yield ("data: [DONE]\n\n")
                    break

                if isinstance(chunk, Exception):
                    # Upstream stream failed; close SSE cleanly so downstream clients
                    # don't see an incomplete chunked read protocol error.
                    log.error("Streaming backend failed: %s", chunk)
                    # Release the held text tail before the interruption notice
                    # so no buffered prose is lost on error (TASK-367).
                    sse = _drain_content_buf(flush_all=True)
                    if sse:
                        yield sse
                    warning_data = {
                        "object": "service.warning",
                        "model": model_alias,
                        "warnings": [f"stream_interrupted: {chunk}"],
                    }
                    yield (f"data: {json.dumps(warning_data)}\n\n")

                    # Flush any buffered content from emote filter.
                    if emote_filter:
                        final_chunk = emote_filter.flush()
                        if final_chunk:
                            data = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_alias,
                                "choices": [{"index": 0, "delta": {"content": final_chunk}, "finish_reason": None}],
                            }
                            yield (f"data: {json.dumps(data)}\n\n")

                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_alias,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield (f"data: {json.dumps(data)}\n\n")
                    yield ("data: [DONE]\n\n")
                    break

                # Past the None/Exception guards, a dict chunk is an event (tool
                # call, tool result, reasoning, terminal marker…) — NOT text.
                # Release the held text tail FIRST so buffered prose always
                # reaches the client before the event that follows it; this is
                # what anchors a tool card after its complete lead-in text
                # instead of mid-word (TASK-367). Guarded to dicts so a normal
                # str text chunk keeps accumulating in the boundary buffer below
                # instead of force-flushing every token.
                if isinstance(chunk, dict):
                    sse = _drain_content_buf(flush_all=True)
                    if sse:
                        yield sse

                # OpenAI-compatible delta.tool_calls streaming
                if isinstance(chunk, dict) and chunk.get("_type") == "tool_call_delta":
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_alias,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": chunk["index"],
                                    "id": chunk["id"],
                                    "type": "function",
                                    "function": {
                                        "name": chunk["name"],
                                        "arguments": chunk["arguments"],
                                    },
                                }],
                            },
                            "finish_reason": None,
                        }],
                    }
                    yield (f"data: {json.dumps(data)}\n\n")
                    continue

                if isinstance(chunk, dict) and chunk.get("_type") == "tool_event":
                    # Already published directly from the background thread
                    # via _publish_tool_event_direct — just skip.
                    continue

                # Model reasoning ("thinking") → OpenAI-compat reasoning lane.
                # Emitted as choices[].delta.reasoning_content so it lands in a
                # separate UI channel, never in delta.content / the saved
                # message. The chunk is a dict, so consume_stream_chunks already
                # kept it out of full_response_holder (TASK-301).
                if isinstance(chunk, dict) and chunk.get("_type") == "reasoning_delta":
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_alias,
                        "choices": [{
                            "index": 0,
                            "delta": {"reasoning_content": chunk.get("delta", "")},
                            "finish_reason": None,
                        }],
                    }
                    yield (f"data: {json.dumps(data)}\n\n")
                    continue

                # OpenClaw agent backend tool events (from openclaw.py queue)
                if isinstance(chunk, dict) and chunk.get("event") == "tool_call":
                    tc_name = chunk.get("name", "unknown")
                    tc_args = chunk.get("arguments", {})
                    # Use the call_id injected by _intercept_tool_events so
                    # HTTP stream and SSE events share the same ID.
                    tc_id = chunk.get("_call_id") or f"call_{uuid.uuid4().hex[:8]}"
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_alias,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": oc_tool_call_index,
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {
                                        "name": tc_name,
                                        "arguments": json.dumps(tc_args, ensure_ascii=False) if isinstance(tc_args, dict) else str(tc_args),
                                    },
                                    # Authoritative char offset (chars of assistant
                                    # text before this tool), stamped by
                                    # _intercept_tool_events from the SAME _text_chars
                                    # counter persisted to tool_calls_json. Lets the
                                    # client anchor the tool at the byte-identical
                                    # position it will land on refresh, instead of its
                                    # own drifting local count (TASK-367). Non-standard
                                    # field — other OpenAI consumers ignore it.
                                    "text_offset": chunk.get("_text_offset"),
                                }],
                            },
                            "finish_reason": None,
                        }],
                    }
                    yield (f"data: {json.dumps(data)}\n\n")
                    oc_tool_call_index += 1
                    # Tool events already published from background thread
                    continue

                if isinstance(chunk, dict) and chunk.get("event") == "metadata":
                    if chunk.get("upstream_model"):
                        _upstream_model[0] = chunk["upstream_model"]
                    continue

                if isinstance(chunk, dict) and chunk.get("event") == "tool_result":
                    if oc_tool_call_index > 0:
                        data = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_alias,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                        }
                        yield (f"data: {json.dumps(data)}\n\n")
                        oc_tool_call_index = 0
                    continue

                if isinstance(chunk, dict) and chunk.get("event") == "await_tool_result":
                    # TASK-269/370 — deferred AskUserQuestion from the agent
                    # bridge. Durable persistence (chat_pending_questions),
                    # question_id_holder, and the unified tool_await_result event
                    # are now owned ENTIRELY by the streaming worker thread
                    # (_persist_publish_await), which runs to completion
                    # regardless of client/SSE teardown — the fix for a question
                    # emitted at the tail of a LONG turn being dropped when this
                    # async generator is torn down before draining the chunk
                    # (TASK-370). By the time this branch runs the worker has
                    # already persisted the row, set question_id_holder, and
                    # published the cross-tab event. This branch is now a pure
                    # sidecar: emit the originating client's OpenAI-compat SSE
                    # chunk so the live (still-connected) tab renders the question
                    # inline without waiting for a reload. (Mirrors the
                    # approval_required async-gen branch, likewise a dedup skip
                    # after _persist_publish_approval.)
                    tool_use_id = chunk.get("tool_use_id") or ""
                    tool_args = chunk.get("arguments") or {}
                    await_payload = {
                        "object": "tool.await_result",
                        "tool_use_id": tool_use_id,
                        "tool_name": chunk.get("tool_name", ""),
                        "arguments": tool_args,
                        "session_key": chunk.get("session_key", ""),
                        "provider": chunk.get("provider", ""),
                        "trigger_message_id": chunk.get("trigger_message_id"),
                        "bot_id": bot_id,
                        "turn_id": turn_log_id,
                    }
                    yield (f"data: {json.dumps(await_payload)}\n\n")
                    continue

                if isinstance(chunk, dict) and chunk.get("event") == "approval_required":
                    # TASK-292/306 — approval persistence + surfacing is owned
                    # ENTIRELY by the streaming worker thread
                    # (_persist_publish_approval). That path is the single source
                    # of truth: it commit-confirms the durable row (bounded retry),
                    # fails closed + informed on a confirmed write failure, and
                    # publishes the live event — all from a thread that runs to
                    # completion regardless of client/SSE teardown. By the time an
                    # approval_required chunk reaches this async generator the
                    # worker has already handled it, so this branch is a pure
                    # dedup skip. (Previously this branch carried a second, full
                    # copy of the persist/retry/emit logic — deleted: duplicated
                    # persistence is exactly the failure mode TASK-306 set out to
                    # remove.)
                    req_id = chunk.get("tool_use_id") or ""
                    if not req_id or req_id in _approval_handled:
                        continue
                    # Reaching here means the chunk bypassed _intercept_tool_events
                    # (a wiring bug) — the worker is wired unconditionally for every
                    # backend, so this should never happen. Log loudly rather than
                    # silently re-implementing persistence a second time.
                    log.error(
                        "approval_required reached async-gen UNHANDLED id=%s — "
                        "worker thread never persisted it; check "
                        "_intercept_tool_events wiring",
                        req_id,
                    )
                    continue

                if isinstance(chunk, dict) and chunk.get("event") == "tool_preapproved":
                    # TASK-305 — a previously approval-gated tool was re-attempted
                    # on this continuation turn and consumed a live one-shot grant.
                    # Publish to the unified stream so the tab(s) showing this turn
                    # mark that exact tool card as pre-approved (gold/lock). No HTTP
                    # yield needed: it's a live activity-card flag, and every chat
                    # tab is on the unified stream.
                    _preapproved_tuid = chunk.get("tool_use_id", "")
                    await _publish_unified({
                        "_type": "tool_preapproved",
                        "turn_id": turn_log_id,
                        "trigger_message_id": trigger_message_id,
                        "bot_id": bot_id,
                        "user_id": user_id,
                        "tool_use_id": _preapproved_tuid,
                        "tool_name": chunk.get("tool_name", ""),
                        "policy_id": chunk.get("policy_id"),
                        "severity": chunk.get("severity", "medium"),
                        "provider": chunk.get("provider", ""),
                        "ts": time.time(),
                    })
                    # Persist the preapproved flag on the tool_call_record so the
                    # gold badge survives a page reload (TASK-305 persistence).
                    if _preapproved_tuid and self._turn_log_store:
                        try:
                            self._turn_log_store.set_preapproved(_preapproved_tuid)
                        except Exception:
                            pass  # best-effort; live event is the primary surface
                    continue

                if isinstance(chunk, dict) and chunk.get("_type") == "tool_calls_finish":
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_alias,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    }
                    yield (f"data: {json.dumps(data)}\n\n")
                    continue

                # Apply emote filter for voice_optimized bots
                if emote_filter:
                    chunk = emote_filter.process(chunk)
                    if not chunk:
                        # Chunk was filtered out or buffered
                        continue

                # Agent-backend text: hold the forming tail, release only at a
                # clean boundary (TASK-367). Every dict/terminal handler above
                # already flushed the buffer, so text→tool order is preserved.
                if is_agent_backend:
                    _content_buf.append(chunk)
                    sse = _drain_content_buf(flush_all=False)
                    if sse:
                        yield sse
                    continue

                # Normal chunk
                data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_alias,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                yield (f"data: {json.dumps(data)}\n\n")
        finally:
            # Generation lifecycle (done_event / _end_generation) is now
            # signalled from the worker thread's finally block so that it
            # fires only after _finalize_turn completes.  The worker runs
            # independently of this async generator, so a client disconnect
            # (GeneratorExit here) does not interrupt persistence.
            pass
