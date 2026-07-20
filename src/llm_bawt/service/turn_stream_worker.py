"""TASK-622: the per-turn streaming worker thread.

``TurnStreamWorker`` runs the blocking stream (native tool loop or agent-bridge
dispatch), consumes/forwards events, and finalizes the turn. Extracted verbatim
from ``chat_streaming.chat_completion_stream``; the method body is unchanged
except for the leading preamble that rebinds captured closure variables from the
shared :class:`TurnStreamContext` and the sibling publish helpers (inherited
from :class:`TurnStreamPublishMixin`).
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid

from ..media.serializers import build_agent_image_manifest
from .chat_stream_worker import consume_stream_chunks, put_queue_item_threadsafe
from .logging import get_service_logger
from .turn_stream_context import TurnStreamContext, _TEXT_DELTA_FLUSH_CHARS
from .turn_stream_publish import TurnStreamPublishMixin

log = get_service_logger(__name__)


class TurnStreamWorker(TurnStreamPublishMixin):
    """Owns the worker-thread streaming + finalize for one chat turn."""

    def __init__(self, ctx: TurnStreamContext) -> None:
        self.ctx = ctx

    def _stream_to_queue(self):
        ctx = self.ctx
        _persist_publish_approval = self._persist_publish_approval
        _persist_publish_await = self._persist_publish_await
        _publish_event_direct = self._publish_event_direct
        _redis_sub = ctx._redis_sub
        _upstream_model = ctx._upstream_model
        agent_attachments_holder = ctx.agent_attachments_holder
        animation_holder = ctx.animation_holder
        approval_id_holder = ctx.approval_id_holder
        approval_persist_failed_holder = ctx.approval_persist_failed_holder
        assistant_message_id = ctx.assistant_message_id
        attachments_to_persist = ctx.attachments_to_persist
        bot_id = ctx.bot_id
        cancel_event = ctx.cancel_event
        cancelled_holder = ctx.cancelled_holder
        chunk_queue = ctx.chunk_queue
        done_event = ctx.done_event
        full_response_holder = ctx.full_response_holder
        inject_seed_messages = ctx.inject_seed_messages
        is_agent_backend = ctx.is_agent_backend
        llm_bawt = ctx.llm_bawt
        loop = ctx.loop
        media_store = ctx.media_store
        model_alias = ctx.model_alias
        question_id_holder = ctx.question_id_holder
        reasoning_holder = ctx.reasoning_holder
        request = ctx.request
        timing_holder = ctx.timing_holder
        token_usage_holder = ctx.token_usage_holder
        tool_call_details_holder = ctx.tool_call_details_holder
        tool_context_holder = ctx.tool_context_holder
        trigger_message_id = ctx.trigger_message_id
        tts_scrub = ctx.tts_scrub
        tts_scrubber = ctx.tts_scrubber
        turn_log_id = ctx.turn_log_id
        user_attachments = ctx.user_attachments
        user_id = ctx.user_id
        user_prompt = ctx.user_prompt
        self = ctx.svc
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
