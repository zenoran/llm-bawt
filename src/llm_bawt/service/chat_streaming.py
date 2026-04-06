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

from ..bots import StreamingEmoteFilter, get_bot
from .chat_stream_worker import consume_stream_chunks, put_queue_item_threadsafe
from .logging import RequestContext, generate_request_id, get_service_logger
from .schemas import ChatCompletionRequest

log = get_service_logger(__name__)


class ChatStreamingMixin:
    """Mixin providing streaming chat completions for BackgroundService."""

    def _is_openclaw_bot(self, model_alias: str) -> bool:
        """Check if this model alias maps to an openclaw agent backend."""
        model_def = self.config.defined_models.get("models", {}).get(model_alias, {})
        if model_def.get("type") == "agent_backend" and model_def.get("backend") == "openclaw":
            return True
        return model_alias == "openclaw"

    def _get_openclaw_session_key(self, model_alias: str) -> str:
        """Get the session_key for an OpenClaw model from its bot_config."""
        model_def = self.config.defined_models.get("models", {}).get(model_alias, {})
        return (
            (model_def.get("bot_config") or {}).get("session_key")
            or "agent:main:main"
        )

    async def _stream_via_bridge(
        self,
        *,
        user_prompt: str,
        session_key: str,
        response_id: str,
        created: int,
        model_alias: str,
        bot_id: str,
        user_id: str,
        ctx: Any,
        turn_log_id: str,
        llm_bawt: Any,
    ) -> AsyncIterator[str]:
        """Stream an OpenClaw response via the WS SessionBridge."""
        bridge = self._session_bridge
        start_time = time.time()

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
        )

        # Accumulate response data outside try so the finally block can
        # always access them (including on GeneratorExit from client disconnect).
        full_text_parts: list[str] = []
        tool_call_details: list[dict] = []
        _finalized = False

        try:
            # Persist user message to bot history so it appears in /v1/history
            llm_bawt.history_manager.add_message("user", user_prompt)

            # Send user message and subscribe to events
            redis_sub = getattr(self, '_redis_subscriber', None)
            if redis_sub:
                # Redis mode: send via Gateway HTTP, subscribe via Redis
                gateway = getattr(self, '_gateway_http', None)
                if not gateway:
                    from openclaw_bridge.gateway_http import GatewayHttpClient
                    gateway = GatewayHttpClient(
                        self.config.OPENCLAW_GATEWAY_URL,
                        self.config.OPENCLAW_GATEWAY_TOKEN,
                    )
                    self._gateway_http = gateway
                run_id = await gateway.send_user_message(session_key, user_prompt)
                log.info("OpenClaw gateway HTTP: sent message, run_id=%s", run_id)
                event_source = redis_sub.subscribe(session_key, run_id=run_id)
            else:
                # In-process mode: send via WS bridge, subscribe via fanout
                run_id = await bridge.send_user_message(session_key, user_prompt)
                bridge._api_run_ids.add(run_id)
                log.info("OpenClaw bridge: sent message, run_id=%s", run_id)
                event_source = bridge._fanout.subscribe(session_key)

            from openclaw_bridge.events import OpenClawEventKind

            # Track tool call state for OpenAI-compat delta.tool_calls
            _tool_call_index = 0
            _in_tool_calls = False
            _last_call_id = ""

            async for event in event_source:
                if event.kind == OpenClawEventKind.ASSISTANT_DELTA:
                    delta = event.text or ""
                    if delta:
                        # If transitioning from tool calls to content, emit finish_reason first
                        if _in_tool_calls:
                            finish_data = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_alias,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                            }
                            yield f"data: {json.dumps(finish_data)}\n\n"
                            _in_tool_calls = False
                            _tool_call_index = 0
                            _last_call_id = ""
                            # Insert paragraph break between tool-call
                            # segment and resumed content so it persists
                            # to DB and survives page reload.
                            full_text_parts.append("\n\n")
                            delta = "\n\n" + delta

                        full_text_parts.append(delta)
                        data = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_alias,
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                elif event.kind == OpenClawEventKind.TOOL_START:
                    tool_name = event.tool_name or "unknown"
                    tool_args = event.tool_arguments or {}
                    call_id = f"call_{uuid.uuid4().hex[:8]}"
                    _last_call_id = call_id
                    # Emit as OpenAI delta.tool_calls
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_alias,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": _tool_call_index,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps(tool_args, ensure_ascii=False) if isinstance(tool_args, dict) else str(tool_args),
                                    },
                                }],
                            },
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    _in_tool_calls = True
                    _tool_call_index += 1
                    tool_call_details.append({
                        "iteration": 1,
                        "tool": tool_name,
                        "parameters": tool_args,
                        "result": "",
                        "call_id": call_id,
                    })
                    # Publish tool_start to unified event stream
                    if redis_sub:
                        try:
                            await redis_sub.publish_tool_event(bot_id, user_id, {
                                "_type": "tool_event",
                                "event": "tool_start",
                                "turn_id": turn_log_id,
                                "bot_id": bot_id,
                                "user_id": user_id,
                                "tool_name": tool_name,
                                "arguments": tool_args,
                                "call_id": call_id,
                                "iteration": _tool_call_index,
                                "ts": time.time(),
                            })
                        except Exception:
                            pass

                elif event.kind == OpenClawEventKind.TOOL_END:
                    tool_result = str(event.tool_result or "")
                    # Recover the call_id from the matching tool_start.
                    # Tool calls can nest (e.g. Agent > Grep > Read), so pop
                    # the stack to pair each tool_end with its tool_start.
                    end_call_id = _last_call_id or ""
                    if tool_call_details:
                        matched = tool_call_details.pop()
                        matched["result"] = tool_result
                        end_call_id = matched.get("call_id", end_call_id)
                    # Publish tool_end to unified event stream
                    if redis_sub:
                        try:
                            await redis_sub.publish_tool_event(bot_id, user_id, {
                                "_type": "tool_event",
                                "event": "tool_end",
                                "turn_id": turn_log_id,
                                "bot_id": bot_id,
                                "user_id": user_id,
                                "tool_name": event.tool_name or "unknown",
                                "call_id": end_call_id,
                                "iteration": _tool_call_index,
                                "result": tool_result[:2000],
                                "ts": time.time(),
                            })
                        except Exception:
                            pass

                elif event.kind == OpenClawEventKind.ASSISTANT_DONE:
                    # ASSISTANT_DONE carries the complete response text.
                    # Yield any portion not already streamed as deltas.
                    done_text = event.text or ""
                    if done_text:
                        accumulated = "".join(full_text_parts)
                        extra = ""
                        if done_text.startswith(accumulated):
                            extra = done_text[len(accumulated):]
                        elif len(done_text) > len(accumulated):
                            extra = done_text[len(accumulated):] if accumulated else done_text
                        if extra.strip():
                            if _in_tool_calls:
                                finish_data = {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_alias,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                                }
                                yield f"data: {json.dumps(finish_data)}\n\n"
                                _in_tool_calls = False
                                full_text_parts.append("\n\n")
                                extra = "\n\n" + extra
                            full_text_parts.append(extra)
                            data = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_alias,
                                "choices": [{"index": 0, "delta": {"content": extra}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(data)}\n\n"

                elif event.kind == OpenClawEventKind.RUN_COMPLETED:
                    if event.run_id == run_id or not run_id:
                        break

                elif event.kind == OpenClawEventKind.ERROR:
                    warning_data = {
                        "object": "service.warning",
                        "model": model_alias,
                        "warnings": [f"openclaw_error: {event.text}"],
                    }
                    yield f"data: {json.dumps(warning_data)}\n\n"
                    break

            # Final chunk
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_alias,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

            # Fetch tool calls from chat history (gateway doesn't stream them)
            # Only fetch if streaming didn't already capture tool calls —
            # otherwise this duplicates them (they lack call_id so dedup fails).
            if not tool_call_details:
                try:
                    if redis_sub and gateway:
                        history_msgs = await gateway.get_chat_history(session_key, limit=20)
                    elif bridge:
                        history_msgs = await bridge._ws_client.get_chat_history(session_key, limit=20)
                    else:
                        history_msgs = []
                    for msg in reversed(history_msgs):
                        if not isinstance(msg, dict):
                            continue
                        # Look for tool_calls in the most recent assistant message
                        if msg.get("role") == "assistant":
                            for tc in msg.get("tool_calls") or []:
                                fn = tc.get("function") or {}
                                tool_call_details.append({
                                    "iteration": 1,
                                    "tool": fn.get("name") or tc.get("name") or "unknown",
                                    "parameters": fn.get("arguments") or {},
                                    "result": "",
                                })
                            break
                        # Also capture tool results
                        if msg.get("role") == "tool":
                            # Will be matched to tool_call above
                            pass
                except Exception as e:
                    log.debug("Could not fetch tool calls from chat.history: %s", e)

            # Finalize turn
            full_text = "".join(full_text_parts)
            elapsed_ms = (time.time() - start_time) * 1000
            if full_text:
                self._finalize_turn(
                    llm_bawt=llm_bawt,
                    turn_id=turn_log_id,
                    response_text=full_text,
                    tool_context="",
                    tool_call_details=tool_call_details,
                    prepared_messages=[],
                    user_prompt=user_prompt,
                    model=model_alias,
                    bot_id=bot_id,
                    user_id=user_id,
                    elapsed_ms=elapsed_ms,
                    stream=True,
                )
            else:
                self._update_turn_log(
                    turn_id=turn_log_id,
                    status="timeout",
                    latency_ms=elapsed_ms,
                )
            _finalized = True

        except Exception as e:
            log.error("OpenClaw bridge stream failed: %s", e)
            elapsed_ms = (time.time() - start_time) * 1000
            self._update_turn_log(
                turn_id=turn_log_id,
                status="error",
                latency_ms=elapsed_ms,
                error_text=str(e),
            )
            _finalized = True
            warning_data = {
                "object": "service.warning",
                "model": model_alias,
                "warnings": [f"bridge_error: {e}"],
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

        finally:
            # Ensure turn is finalized even on GeneratorExit (client disconnect).
            # GeneratorExit inherits from BaseException, not Exception, so the
            # except block above does not catch it.
            if not _finalized:
                full_text = "".join(full_text_parts)
                elapsed_ms = (time.time() - start_time) * 1000
                try:
                    if full_text:
                        self._finalize_turn(
                            llm_bawt=llm_bawt,
                            turn_id=turn_log_id,
                            response_text=full_text,
                            tool_context="",
                            tool_call_details=tool_call_details,
                            prepared_messages=[],
                            user_prompt=user_prompt,
                            model=model_alias,
                            bot_id=bot_id,
                            user_id=user_id,
                            elapsed_ms=elapsed_ms,
                            stream=True,
                        )
                    else:
                        self._update_turn_log(
                            turn_id=turn_log_id,
                            status="disconnected",
                            latency_ms=elapsed_ms,
                            tool_calls=tool_call_details or None,
                        )
                except Exception as fin_err:
                    log.error("Bridge finalization failed (turn %s): %s", turn_log_id, fin_err)
                    try:
                        self._update_turn_log(
                            turn_id=turn_log_id,
                            status="error",
                            latency_ms=elapsed_ms,
                            error_text=f"finalize_error: {fin_err}",
                        )
                    except Exception:
                        pass

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """Handle a streaming chat completion request.

        Uses llm_bawt's internal history + memory for context.
        Yields Server-Sent Events (SSE) formatted chunks.
        """
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

        # Get the user's prompt (last user message)
        user_prompt = ""
        user_attachments: list[dict] = []
        for m in reversed(request.messages):
            if m.role == "user":
                if isinstance(m.content, list):
                    for part in m.content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text":
                            user_prompt += part.get("text", "")
                        elif part.get("type") == "image_url":
                            url = (part.get("image_url") or {}).get("url", "")
                            if url.startswith("data:"):
                                try:
                                    header, data = url.split(",", 1)
                                    mime = header.split(":")[1].split(";")[0]
                                    user_attachments.append({"mimeType": mime, "content": data})
                                except Exception:
                                    pass
                else:
                    user_prompt = m.content or ""
                break

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
        animation_holder = [None]   # Captures trigger_animation choice (tts_mode only)
        tool_context_holder = [""]  # Store tool context from native tool calls
        tool_call_details_holder: list[dict] = []
        timing_holder = [0.0, 0.0]  # [start_time, end_time]
        cancelled_holder = [False]  # Track if we were cancelled

        # Capture the event loop before entering the thread
        loop = asyncio.get_running_loop()

        # Start new generation.
        # For agent_backend (OpenClaw) bots the gateway queues messages as
        # separate runs, so we must NOT cancel a previous generation — each
        # request streams independently.  For native models we cancel the
        # previous generation so only the latest request runs.
        is_agent_backend = llm_bawt.client.model_definition.get("type") == "agent_backend"
        if is_agent_backend:
            cancel_event = threading.Event()
            done_event = threading.Event()
        else:
            cancel_event, done_event = await self._start_generation(bot_id)
        turn_log_id = f"turn-{uuid.uuid4().hex}"

        # Resolve agent session_key for abort support
        oc_session_key: str | None = None
        if is_agent_backend:
            bc = getattr(llm_bawt.bot, "agent_backend_config", None) or {}
            backend_name = getattr(llm_bawt.bot, "agent_backend", "")
            if backend_name == "claude-code":
                # Claude-code uses bot_id as the routing key
                oc_session_key = bot_id
            else:
                oc_session_key = bc.get("session_key")

        # Persist turn log immediately so the user's prompt is recorded
        # even if the backend times out or errors before responding.
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
        )

        # Capture Redis subscriber for direct publish from the background
        # thread — ensures tool events (and turn_complete) reach Redis even
        # if the SSE generator is cancelled (client disconnect / page refresh).
        _redis_sub = getattr(self, "_redis_subscriber", None)

        def _publish_event_direct(event_dict):
            """Publish an event to Redis from the worker thread."""
            if not _redis_sub:
                return
            try:
                asyncio.run_coroutine_threadsafe(
                    _redis_sub.publish_tool_event(bot_id, user_id, event_dict),
                    loop,
                )
            except Exception as pub_err:
                log.debug("Direct event publish failed: %s", pub_err)

        def _stream_to_queue():
            """Run streaming in a thread and push chunks to the async queue."""
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
                llm_bawt._tts_mode = request.tts_mode or llm_bawt.bot.tts_mode

                # Load animations for tts_mode — must happen before streaming starts
                _animations: list = []
                if llm_bawt._tts_mode:
                    try:
                        from .animation_tool import AvatarAnimationStore
                        _animations = AvatarAnimationStore(self.config).list_enabled()
                    except Exception as _e:
                        log.debug("Could not load animations: %s", _e)
                _use_animation_tool = bool(
                    _animations
                    and llm_bawt.client.supports_native_tools()
                    and hasattr(llm_bawt.client, "stream_with_tools")
                )
                log.info("🎬 tts_mode=%s (req=%s bot=%s), %d animations, use_animation_tool=%s",
                         llm_bawt._tts_mode, request.tts_mode, llm_bawt.bot.tts_mode, len(_animations), _use_animation_tool)

                # Use llm_bawt.prepare_messages_for_query to get full context
                # (history from DB + memory + system prompt)
                messages = llm_bawt.prepare_messages_for_query(
                    user_prompt,
                    user_attachments=user_attachments or None,
                )

                # When animation tool is active, append a system instruction
                # so the model always calls trigger_animation.
                if _use_animation_tool:
                    from ..models.message import Message as _Msg
                    anim_list = "\n".join(
                        f'- "{a.name}": {a.description or a.name}'
                        for a in _animations
                    )
                    messages.append(_Msg(
                        role="system",
                        content=(
                            "AVATAR ANIMATION: You MUST call trigger_animation exactly once "
                            "at the end of every response — no exceptions. Write your full "
                            "text reply first, then call the function. Pick the animation "
                            "that best matches your response's emotional tone. If none of "
                            "the other gestures fit, the minimum should be a talking "
                            "animation like 'Acknowledging' or 'Head Nod Yes'. Never skip "
                            "calling this function.\n\n"
                            f"Available animations:\n{anim_list}"
                        ),
                    ))

                # Backfill prepared messages into the turn log
                self._update_turn_log(
                    turn_id=turn_log_id,
                    prepared_messages=messages,
                )

                # Log what we're sending to the LLM (verbose mode)
                log.llm_context(messages)

                # Track when first token arrives
                timing_holder[0] = time.time()

                # Choose streaming method based on whether bot uses tools
                if (llm_bawt.bot.uses_tools and (llm_bawt.memory or llm_bawt.home_client or llm_bawt.ha_native_client)) or _use_animation_tool:
                    # Check if client supports native streaming with tools (OpenAI)
                    use_native_streaming = (
                        llm_bawt.client.supports_native_tools()
                        and (llm_bawt.tool_format in ("native", "NATIVE_OPENAI") or _use_animation_tool)
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

                        # Inject trigger_animation virtual tool for tts_mode requests
                        if _use_animation_tool:
                            from .animation_tool import build_trigger_animation_tool
                            tools_schema = tools_schema + [build_trigger_animation_tool(_animations)]

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

                                # Buffer text until we know if tool_calls follow.
                                # If tools fire, discard the buffer (stale pre-tool
                                # text like "I don't have live access…").
                                # If no tools, flush the buffer to the client.
                                text_buffer: list[str] = []

                                for item in llm_bawt.client.stream_with_tools(
                                    current_msgs,
                                    tools_schema=current_tools,
                                    tool_choice=current_tool_choice,
                                    **gen_kwargs,
                                ):
                                    if isinstance(item, str):
                                        # Buffer — don't yield until we know no tools follow
                                        text_buffer.append(item)
                                    elif isinstance(item, dict) and "tool_calls" in item:
                                        tool_calls = item["tool_calls"]
                                        content = item.get("content", "")

                                        if not tool_calls:
                                            # Empty tool_calls list — treat as pure text response
                                            yield from text_buffer
                                            return

                                        # Separate virtual tools (trigger_animation) from real tools
                                        real_tool_calls = [
                                            tc for tc in tool_calls
                                            if tc.get("function", {}).get("name") != "trigger_animation"
                                        ]
                                        for tc in tool_calls:
                                            func = tc.get("function", {})
                                            if func.get("name") == "trigger_animation":
                                                try:
                                                    a_args = _json.loads(func.get("arguments", "{}") or "{}")
                                                except _json.JSONDecodeError:
                                                    a_args = {}
                                                if not animation_holder[0]:
                                                    # Try "name" first (matches tool schema), then
                                                    # common alternatives models sometimes use.
                                                    anim_name = (
                                                        a_args.get("name")
                                                        or a_args.get("animation")
                                                        or a_args.get("animation_name")
                                                    )
                                                    # Last resort: take the first string value
                                                    if not anim_name:
                                                        for v in a_args.values():
                                                            if isinstance(v, str):
                                                                anim_name = v
                                                                break
                                                    animation_holder[0] = anim_name
                                                log.info("🎭 trigger_animation: %s (raw args: %s)", animation_holder[0], a_args)

                                        if not real_tool_calls:
                                            # Only animation tool — flush buffered text.
                                            # Do NOT yield content here — it is the same
                                            # text already accumulated in text_buffer
                                            # (yielded chunk-by-chunk above), so yielding
                                            # it again would double the response.
                                            yield from text_buffer
                                            return

                                        # Real tool calls follow — discard stale pre-tool text
                                        text_buffer.clear()

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
                                                "bot_id": bot_id,
                                                "user_id": user_id,
                                                "tool_name": name,
                                                "arguments": args,
                                                "call_id": call_id,
                                                "result": result[:2000] if result else "",
                                                "iteration": iteration + 1,
                                                "ts": time.time(),
                                            })

                                            # Persist for TurnLog.tool_calls_json
                                            tool_call_details_holder.append({
                                                "tool": name,
                                                "arguments": args,
                                                "call_id": call_id,
                                                "result": result[:2000] if result else "",
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
                                    # Stream finished without tool calls dict (pure content response)
                                    # Flush any buffered text that wasn't already consumed
                                    yield from text_buffer
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
                    stream_iter = llm_bawt.client.stream_raw(
                        messages, stop=adapter_stops or None, **gen_kwargs, **extra_kwargs
                    )

                # Wrap stream to publish tool events directly to Redis
                # from this thread (survives SSE generator cancellation).
                _oc_call_index = [0]
                _oc_last_call_id = [""]

                _oc_request_id_captured = [False]

                def _intercept_tool_events(inner):
                    _saw_tool = False  # Track tool→content transitions
                    _saw_text = False  # Track whether any text has been yielded

                    for item in inner:
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
                                yield "\n\n"
                            _saw_tool = False
                            _saw_text = True

                        if isinstance(item, dict):
                            evt = item.get("event")
                            if evt == "metadata":
                                if item.get("upstream_model"):
                                    _upstream_model[0] = item["upstream_model"]
                                continue
                            if evt in ("tool_call", "tool_result"):
                                _saw_tool = True
                            if evt == "tool_call":
                                _oc_call_index[0] += 1
                                cid = f"call_{uuid.uuid4().hex[:8]}"
                                _oc_last_call_id[0] = cid
                                # Inject call_id into the dict so the queue consumer
                                # uses the same ID as the SSE event (no double-ID).
                                item["_call_id"] = cid
                                tool_call_details_holder.append({
                                    'tool': item.get('name', 'unknown'),
                                    'parameters': item.get('arguments', {}),
                                    'call_id': cid,
                                })
                                _publish_event_direct({
                                    "_type": "tool_event",
                                    "event": "tool_start",
                                    "turn_id": turn_log_id,
                                    "bot_id": bot_id,
                                    "user_id": user_id,
                                    "tool_name": item.get("name", "unknown"),
                                    "arguments": item.get("arguments", {}),
                                    "call_id": cid,
                                    "iteration": 1,
                                    "ts": time.time(),
                                })
                            elif evt == "tool_result":
                                item["_call_id"] = _oc_last_call_id[0]
                                _publish_event_direct({
                                    "_type": "tool_event",
                                    "event": "tool_end",
                                    "turn_id": turn_log_id,
                                    "bot_id": bot_id,
                                    "user_id": user_id,
                                    "tool_name": item.get("name", "unknown"),
                                    "call_id": _oc_last_call_id[0],
                                    "result": str(item.get("result", ""))[:2000],
                                    "iteration": 1,
                                    "ts": time.time(),
                                })
                                # Update the matching detail entry with result
                                for _td in reversed(tool_call_details_holder):
                                    if _td.get("call_id") == _oc_last_call_id[0]:
                                        _td["result"] = str(item.get("result", ""))[:2000]
                                        break
                        yield item

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

                # Fallback: if animation tool was active but model didn't
                # call it, default to "Acknowledging" so every tts_mode
                # response gets a gesture.
                if _use_animation_tool and not animation_holder[0] and full_response_holder[0]:
                    animation_holder[0] = "Acknowledging"
                    log.info("🎭 animation fallback → Acknowledging (model skipped trigger_animation)")

                timing_holder[1] = time.time()

            except Exception as e:
                if not cancel_event.is_set():
                    put_queue_item_threadsafe(loop, chunk_queue, e)
                elapsed_ms = (time.time() - timing_holder[0]) * 1000 if timing_holder[0] else None
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

                # Wrap finalization in try/except so that the sentinel,
                # turn_complete event, and generation cleanup always fire
                # even if persistence raises (e.g. database failure).
                try:
                    if full_response_holder[0]:
                        self._finalize_turn(
                            llm_bawt=llm_bawt,
                            turn_id=turn_log_id,
                            response_text=full_response_holder[0],
                            tool_context=tool_context_holder[0],
                            tool_call_details=tool_call_details_holder,
                            prepared_messages=messages if "messages" in locals() else [],
                            user_prompt=user_prompt,
                            model=model_alias,
                            bot_id=bot_id,
                            user_id=user_id,
                            elapsed_ms=elapsed_ms,
                            stream=True,
                            animation=animation_holder[0],
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
                _publish_event_direct({
                    "_type": "turn_complete",
                    "turn_id": turn_log_id,
                    "bot_id": bot_id,
                    "user_id": user_id,
                    "status": status,
                    "animation": animation_holder[0],
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
            # Check if this bot needs emote filtering for TTS
            bot = get_bot(bot_id)
            emote_filter = StreamingEmoteFilter() if (bot and bot.voice_optimized) else None
            oc_tool_call_index = 0  # OpenClaw agent backend tool call index
            oc_last_call_id = ""   # Track last call_id for tool_end matching
            _upstream_model = [None]  # Actual model reported by agent backend

            # Send service warnings (e.g. model fallback) before content
            if model_warnings:
                warning_data = {
                    "object": "service.warning",
                    "model": model_alias,
                    "warnings": model_warnings,
                }
                yield f"data: {json.dumps(warning_data)}\n\n"

            # Start streaming — API clients (openai/grok) use the default
            # thread pool so multiple bots can stream concurrently.  Local
            # models (gguf) keep the single-worker executor to avoid CUDA /
            # llama-cpp-python thread-safety issues.
            model_type = llm_bawt.client.model_definition.get("type", "")
            if model_type in ("openai", "grok", "agent_backend"):
                loop.run_in_executor(None, _stream_to_queue)
            else:
                loop.run_in_executor(self._llm_executor, _stream_to_queue)

            # Yield SSE chunks (with keepalive to prevent client timeout
            # during slow backends like vLLM first-inference)
            while True:
                try:
                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Send SSE comment as keepalive to prevent client/proxy timeout
                    yield ": keepalive\n\n"
                    continue

                if chunk is None:
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
                                }],
                            },
                            "finish_reason": None,
                        }],
                    }
                    yield (f"data: {json.dumps(data)}\n\n")
                    oc_tool_call_index += 1
                    oc_last_call_id = tc_id
                    # Tool events already published from background thread
                    continue

                if isinstance(chunk, dict) and chunk.get("event") == "metadata":
                    if chunk.get("upstream_model"):
                        _upstream_model[0] = chunk["upstream_model"]
                    continue

                if isinstance(chunk, dict) and chunk.get("event") == "tool_result":
                    tc_result = str(chunk.get("result", ""))
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
