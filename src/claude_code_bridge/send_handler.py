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
    _REFRESH_BUFFER_MS,
    _bot_slug_from_session_key,
    _fmt_tokens,
    _usage_input_total,
    _estimate_proxy_cost_usd,
    _pick_iteration_usage,
    _read_latest_compact_metadata,
    _token_expired_or_stale,
    _get_fresh_oauth_token,
    _is_cli_crash,
    _is_auth_failure,
)
from .send_request import SendRequest
from .send_stream import ClaudeStreamMixin
from .send_usage import ClaudeUsageMixin

logger = logging.getLogger("claude_code_bridge.bridge")


def _classify_send_error(exc_text: str) -> str:
    """TASK-637: tag credential-death errors with a structured marker.

    A turn that dies on an expired/revoked Claude credential (after the
    one-shot auth-retry above has already failed) gets a
    ``[credential_expired:claude]`` prefix so the chat UI can deterministically
    render the inline Reconnect flow (frontend: chat/CredentialErrorCard.tsx)
    instead of a dead error bubble. Reuses the same matcher the auth-retry
    path uses — one definition of "auth failure".
    """
    if _is_auth_failure(Exception(exc_text), []):
        return f"[credential_expired:claude] {exc_text}"
    return exc_text


class ClaudeSendMixin(ClaudeStreamMixin, ClaudeUsageMixin):
    """Claude agent send-path (TASK-555 quarantine; see TASK-622-style follow-up).

    Split out of ``ClaudeCodeBridge`` (TASK-555); composed back via
    inheritance so ``self.*`` state and sibling-mixin methods resolve
    on the assembled instance.
    """

    async def _handle_send(
        self, fields: dict, msg_id: str, async_redis,
    ) -> None:
        # TASK-623: field parsing / validation / normalization extracted into
        # SendRequest.from_fields (behavior-identical). Unpack into locals so
        # the rest of this method — which mutates ``message`` on /new — reads
        # exactly as before.
        req = SendRequest.from_fields(fields)
        request_id = req.request_id
        session_key = req.session_key
        bot_slug = req.bot_slug
        message = req.message
        system_prompt = req.system_prompt
        model = req.model
        inject_messages = req.inject_messages
        trigger_message_id = req.trigger_message_id
        bot_effort = req.bot_effort
        bot_max_turns = req.bot_max_turns
        subagent_model = req.subagent_model
        bot_context_window = req.bot_context_window
        configured_disallowed_tools = req.configured_disallowed_tools
        attachments = req.attachments

        if not request_id or not message:
            logger.warning("Invalid send command: missing request_id or message")
            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
            return

        if not model:
            # No silent fallback. The caller MUST pass an explicit model. Surface
            # the failure both to the log and to the originating chat so the user
            # immediately sees which bot's config is missing a model.
            err = (
                f"Claude Code bridge: missing 'model' field for bot={bot_slug or '?'} "
                f"session={session_key}. Set the bot's Model (default_model) to a "
                f"claude-code catalog entry on the bot's profile."
            )
            logger.error(err)
            self._publish_event(
                request_id, session_key, 1,
                kind=AgentEventKind.ERROR,
                text=err,
            )
            self._publisher.publish_run_done(request_id)
            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
            return

        if trigger_message_id:
            self._trigger_message_ids[request_id] = trigger_message_id

        # /new resets the session — strip it and start fresh
        if message.lstrip().startswith("/new"):
            cleared = await self._clear_session(bot_slug or session_key)
            logger.info("Session reset via /new: %s (had_session=%s)", bot_slug or session_key, cleared)
            # Publish a deterministic SESSION_RESET unified event so the
            # frontend can clear its visible buffer without racing
            # turn_complete timing.  See TASK-249.
            self._publish_session_reset_unified(
                bot_slug or session_key, session_key, had_session=cleared,
            )
            # TASK-445: optionally seed the fresh session from chat summary
            # history. Persists the minted session id so a trailing message
            # below resumes the seeded transcript (no double-seed).
            seed_stats = await self._seed_new_session(bot_slug, model, injected=inject_messages)
            message = message.lstrip().removeprefix("/new").strip()
            if not message:
                # Just "/new" with no follow-up — acknowledge and done
                self._publish_event(
                    request_id, session_key, 1,
                    kind=AgentEventKind.ASSISTANT_DONE,
                    text=self._format_seed_ack(seed_stats),
                    model=model,
                )
                self._publisher.publish_run_done(request_id)
                await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
                return

        if self._session_queue.is_busy(session_key):
            logger.info(
                "Session %s busy — queuing send request_id=%s",
                session_key, request_id,
            )

        async with self._session_queue.active(session_key):
            logger.info(
                "Handling send: request_id=%s session=%s model=%s system_prompt=%s msg=%.60s...",
                request_id, session_key, model,
                f"{len(system_prompt)} chars" if system_prompt else "none",
                message,
            )

            seq = 0
            text_parts: list[str] = []
            current_tool_name: str | None = None
            current_tool_input: str = ""
            actual_model: str = model  # updated from SystemMessage if available
            # Map tool_use_id -> tool name (the SDK's ToolResultBlock doesn't echo
            # the name, only the id) so we can recognise a Playwright screenshot
            # result and persist its image instead of letting the inline base64
            # ride in the model context forever.
            tool_names_by_id: dict[str, str] = {}
            # {asset_id, kind} refs for screenshots persisted to the media store
            # during this turn; stamped onto the terminal ASSISTANT_DONE event so
            # the app can attach them to the bot's reply message.
            turn_screenshot_assets: list[dict] = []

            try:
                # Inject MCP tool context so Claude passes the right identifiers.
                # Body comes from the registry (TASK-490) with a byte-identical
                # local fallback; separator added here.
                if system_prompt and self._mcp_servers and bot_slug:
                    mcp_ctx = await self._get_mcp_tool_context(bot_slug)
                    system_prompt += f"\n\n{mcp_ctx}"

                # Reuse SDK session for conversation continuity.
                # If the model changed, start a fresh session.
                existing = await self._get_session(bot_slug)
                resume_id = None
                if existing:
                    prev_sid, prev_model = existing
                    if prev_model == model:
                        resume_id = prev_sid
                    else:
                        logger.info(
                            "Model changed (%s -> %s), starting new session for %s",
                            prev_model, model, bot_slug or session_key,
                        )
                        await self._clear_session(bot_slug or session_key)

                # TASK-445: cold start with no session to resume — first-ever
                # run or post-model-switch. Seed from summary history so the new
                # SDK session opens with continuity. A /new above that already
                # seeded will have persisted a session, so _get_session found it
                # and resume_id is set — this block is skipped (no double-seed).
                if resume_id is None:
                    cold_seed = await self._seed_new_session(bot_slug, model, injected=inject_messages)
                    if cold_seed and cold_seed.get("seeded"):
                        resume_id = cold_seed["session_id"]
                        logger.info(
                            "Cold-start seeded session for %s: %s (%s summaries, %s msgs)",
                            bot_slug, resume_id,
                            cold_seed.get("summary_count"), cold_seed.get("message_count"),
                        )

                # Resolve settings file path
                settings_path = str(Path.home() / ".claude" / "settings.json")
                if not Path(settings_path).exists():
                    settings_path = None

                # Build prompt — multimodal if attachments present
                if attachments:
                    content: list[dict] = [{"type": "text", "text": message}]
                    for att in attachments:
                        mime = att.get("mimeType", "image/png")
                        data = att.get("content", "")
                        if data:
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime,
                                    "data": data,
                                },
                            })
                    logger.info("Multimodal prompt: %d text + %d images", 1, len(attachments))
                    user_content: str | list[dict] = content
                else:
                    user_content = message

                # The prompt MUST be an AsyncIterable: can_use_tool only works in
                # the SDK's streaming-input mode (a plain str raises "can_use_tool
                # callback requires streaming mode").
                #
                # Critically, the generator must stay OPEN for the whole turn.
                # Returning right after the single yield closes the subprocess
                # input stream, which tears down the bidirectional control channel
                # that can_use_tool (AskUserQuestion) rides on — the pending
                # permission request then dies with "Tool permission request
                # failed: Error: Stream closed", and the half-finished turn leaves
                # a dangling tool_use that makes the resumed session un-replayable
                # (it wedges every subsequent message on that session).
                #
                # So: yield the user message, then block on `done_event` until the
                # response loop signals the turn is complete (ResultMessage), and
                # only then return — letting the SDK close input and end the
                # output stream via StopAsyncIteration cleanly.
                def _make_prompt_input(done_event: asyncio.Event) -> AsyncIterable:
                    async def _prompt():
                        yield {
                            "type": "user",
                            "message": {"role": "user", "content": user_content},
                            "parent_tool_use_id": None,
                            "session_id": "default",
                        }
                        await done_event.wait()

                    return _prompt()

                auth_retry_attempted = False
                fresh_session_retry = False
                # Pull (or create) the cooperative cancel event for this session so a
                # chat.abort that arrives mid-loop is observed at the next message
                # boundary — without waiting for `task.cancel()` to fire CancelledError
                # at the next `await` (which can be tens of seconds inside a tool call).
                cancel_event = (
                    self._session_queue.cancel_event(session_key) if session_key else None
                )
                if cancel_event is not None and cancel_event.is_set():
                    # Stale signal from a previous run — clear so this turn can proceed.
                    cancel_event.clear()
                while True:
                    # An async generator is single-use and the auth/session retry
                    # paths below re-enter this loop, so build a fresh prompt +
                    # completion gate per attempt.  turn_done releases the prompt
                    # generator (closing SDK input) only once the turn finishes —
                    # see _make_prompt_input for why it must stay open until then.
                    turn_done = asyncio.Event()
                    prompt_input = _make_prompt_input(turn_done)
                    stderr_lines: list[str] = []
                    # Per-attempt: the auth/session retry paths re-run the turn
                    # from scratch, so reset screenshot tracking to avoid double-
                    # counting an earlier attempt's uploads on the final DONE event.
                    tool_names_by_id.clear()
                    turn_screenshot_assets.clear()

                    def _log_stderr(line: str) -> None:
                        line = line.rstrip()
                        stderr_lines.append(line)
                        logger.warning("CLI stderr: %s", line)

                    # TASK-270: route this turn to the in-process Anthropic-compat
                    # proxy when the model name carries a known provider prefix
                    # (e.g. "openai_chatgpt/gpt-5.4"). The proxy reads ChatGPT
                    # OAuth from ~/.codex/auth.json and forwards to OpenAI's
                    # Responses API. Otherwise fall through to Anthropic-direct.
                    use_proxy = (
                        self._proxy_base_url is not None
                        and self._model_provider_prefix(model) is not None
                    )

                    # TASK-593: the app resolves the DB-managed base list on each
                    # dispatch. Normalize again at this trust boundary and preserve
                    # the proxy transport rule: Anthropic server-side WebSearch /
                    # WebFetch cannot execute through non-Anthropic upstreams.
                    disallowed_tools = effective_disallowed_tools(
                        configured_disallowed_tools,
                        use_proxy=use_proxy,
                    )

                    # TASK-623: proxy-vs-direct SDK env construction extracted
                    # into _build_sdk_env (behavior-identical).
                    sdk_env = self._build_sdk_env(
                        use_proxy=use_proxy,
                        model=model,
                        subagent_model=subagent_model,
                        force_refresh=auth_retry_attempted,
                    )

                    # Pass `seq` to the can_use_tool factory by reference so it
                    # can keep the AWAIT_TOOL_RESULT event ordered in the same
                    # sequence as the surrounding ASSISTANT_DELTA / TOOL_START
                    # events.  Tuple-wrapped in a single-element list so the
                    # closure can mutate it without rebinding.
                    seq_holder = [seq]
                    can_use_tool_cb = self._make_can_use_tool(
                        request_id=request_id,
                        session_key=session_key,
                        seq_holder=seq_holder,
                    )
                    # TASK-292: the approval gate lives in a PreToolUse hook, NOT
                    # can_use_tool. Under permission_mode="bypassPermissions" (our
                    # standing config) the SDK auto-approves regular tools and
                    # never calls can_use_tool for them, so a can_use_tool-based
                    # gate is dead code. PreToolUse hooks are a separate control
                    # plane that fires regardless of permission_mode (verified
                    # live: hook fires + "deny" blocks under bypass). Shares
                    # seq_holder with can_use_tool — turns are sequential, so no
                    # concurrent mutation.
                    pre_tool_use_cb = self._make_pre_tool_use_hook(
                        request_id=request_id,
                        session_key=session_key,
                        seq_holder=seq_holder,
                    )

                    # TASK-288 observability: log the system_prompt value AS SENT
                    # to the SDK, paired with resume state. This is the only place
                    # the resume-gate decision is visible — the earlier "Handling
                    # send" log prints the pre-gate request value and cannot prove
                    # whether persona actually reached the agent on a resumed turn.
                    logger.info(
                        "SDK call: resume=%s system_prompt_sent=%s",
                        bool(resume_id),
                        f"{len(system_prompt)} chars" if system_prompt else "none",
                    )

                    # TASK-623: ClaudeAgentOptions construction moved to
                    # _build_agent_options (behavior-identical).
                    options = self._build_agent_options(
                        model=model,
                        system_prompt=system_prompt,
                        disallowed_tools=disallowed_tools,
                        resume_id=resume_id,
                        sdk_env=sdk_env,
                        settings_path=settings_path,
                        bot_effort=bot_effort,
                        bot_max_turns=bot_max_turns,
                        can_use_tool_cb=can_use_tool_cb,
                        pre_tool_use_cb=pre_tool_use_cb,
                        stderr=_log_stderr,
                    )

                    session_persisted = False
                    aborted = False
                    # Track the latest AssistantMessage.usage to surface "current
                    # context fullness" to the UI. ResultMessage.usage is
                    # cumulative across all internal API iterations in a turn,
                    # so cache_read_input_tokens can exceed contextWindow on
                    # multi-tool-use turns and produce >100% counters. The last
                    # AssistantMessage's usage reflects the actual final API
                    # call's view of the context.
                    latest_assistant_usage: dict | None = None
                    # Some synthetic Anthropic streams (notably the
                    # Responses-backed ChatGPT proxy) surface final input/cache
                    # usage only on StreamEvent(message_delta), while the
                    # AssistantMessage snapshot can remain at message_start's
                    # zeroed input fields. Track the last stream-level usage as
                    # a fallback so turn_logs get the real pill numbers.
                    latest_stream_usage: dict | None = None
                    # /compact lifecycle tracking for this turn. The SDK reports
                    # compaction via SystemMessage(subtype="status") — never a
                    # compact_boundary on the wire — so we watch the status
                    # payload to (a) give immediate feedback (a /compact can be
                    # ~50s of otherwise-silent work the UI reads as "hung") and
                    # (b) report the new resident size, since the /compact
                    # ResultMessage.usage is all-zeros.
                    compact_announced = False
                    compact_status: str | None = None  # None | "success" | "failed"
                    compact_error_msg: str | None = None
                    turn_session_id: str | None = resume_id
                    # Some providers / SDK paths terminate after an
                    # AssistantMessage snapshot with text/tool_use content and
                    # NEVER emit a trailing ResultMessage. If we only finalize
                    # on ResultMessage, the bridge logs "Send completed" but
                    # the app receives no ASSISTANT_DONE and the turn is saved
                    # as an empty timeout. Capture the latest assistant text
                    # snapshot so we can publish a fallback DONE on clean EOF.
                    assistant_snapshot_text: str = ""
                    assistant_done_emitted = False
                    # Track upstream API retries so we can (a) show live
                    # status in the UI ("z.ai overloaded, retrying…") and
                    # (b) include the error in the final DONE when all
                    # retries are exhausted and the turn ends empty.
                    api_retry_count = 0
                    api_last_error: str | None = None
                    api_retry_surfaced = False  # True once we've pushed a delta

                    sdk_client = None
                    msg_stream = None
                    try:
                        sdk_client = ClaudeSDKClient(options=options)
                        await sdk_client.connect(prompt_input)
                        msg_stream = sdk_client.receive_messages()
                        # Register the live client so `chat.abort` can call
                        # `disconnect()` on it — that closes the SDK Query and
                        # drives the subprocess transport through EOF/SIGTERM/
                        # SIGKILL teardown even mid-tool-call. `task.cancel()`
                        # alone is insufficient because CancelledError only fires
                        # at the next `await`, and the SDK is awaiting on subprocess
                        # output that doesn't arrive until the running tool exits.
                        if session_key:
                            self._session_queue.set_active_client(session_key, sdk_client)
                        while True:
                            # Cooperative abort check — runs before every SDK
                            # `__anext__`, so an abort signalled by chat.abort is
                            # observed even if the previous `await` was already
                            # past the cancel injection point.
                            if cancel_event is not None and cancel_event.is_set():
                                logger.info(
                                    "chat.abort signalled, halting SDK iteration: session=%s request=%s",
                                    session_key, request_id,
                                )
                                aborted = True
                                break
                            try:
                                msg = await asyncio.wait_for(
                                    msg_stream.__anext__(),
                                    timeout=self._request_timeout,
                                )
                            except StopAsyncIteration:
                                    break
                            except TimeoutError:
                                raise TimeoutError(
                                    f"No SDK messages for {self._request_timeout}s — CLI may be hung"
                                )

                            # Session-mirror write failure. MirrorErrorMessage is a
                            # SystemMessage subclass the SDK emits when its
                            # SessionStore.append fails — i.e. a turn's frame did not
                            # get persisted to the on-disk transcript. Left unhandled
                            # it slips into the generic SystemMessage branch below,
                            # matches none of its data conditions, and vanishes — so a
                            # persistence failure that can later wedge resume/replay
                            # goes completely unsignalled. Surface it as a structured
                            # warning (operational, not user-facing — no chat bubble).
                            if isinstance(msg, MirrorErrorMessage):
                                logger.warning(
                                    "SDK session-mirror append failed: key=%s error=%s session=%s",
                                    getattr(msg, "key", None),
                                    getattr(msg, "error", None),
                                    session_key,
                                )
                            if isinstance(msg, SystemMessage):
                                data = getattr(msg, "data", {}) or {}
                                # Capture session_id + actual model from the first
                                # SystemMessage (the init), then persist once.
                                if not session_persisted:
                                    if data.get("model"):
                                        actual_model = data["model"]
                                        logger.info("Actual model: %s", actual_model)
                                    if not resume_id:
                                        sid = data.get("session_id")
                                        if sid:
                                            await self._set_session(bot_slug or session_key, sid, model)
                                    session_persisted = True
                                # Track the session_id for this turn regardless of
                                # resume state — used to read the compaction result
                                # back from the transcript below.
                                if data.get("session_id"):
                                    turn_session_id = data["session_id"]
                                # Compaction lifecycle. A /compact turn emits
                                # SystemMessage(subtype="status"): first
                                # status="compacting", then a payload carrying
                                # compact_result ("success"/"failed") and, on
                                # failure, compact_error. There is NO
                                # compact_boundary on the wire. Surface the start
                                # immediately so the turn doesn't read as "hung",
                                # and record the outcome for the ResultMessage.
                                if data.get("status") == "compacting" and not compact_announced:
                                    compact_announced = True
                                    seq += 1
                                    note = "🗜️ Compacting conversation to free up context…"
                                    text_parts.append(note)
                                    self._publish_event(
                                        request_id, session_key, seq,
                                        kind=AgentEventKind.ASSISTANT_DELTA,
                                        text=note,
                                    )
                                cr = data.get("compact_result")
                                if cr == "success":
                                    compact_status = "success"
                                elif cr == "failed":
                                    compact_status = "failed"
                                    compact_error_msg = (
                                        data.get("compact_error") or "unknown error"
                                    )
                                # API retry lifecycle. The SDK CLI retries on
                                # upstream errors (429, 500, 529, etc.). Surface
                                # the retry to the UI as a live status delta so
                                # the user sees feedback instead of a dead bubble.
                                if data.get("subtype") == "api_retry":
                                    attempt = data.get("attempt", 0)
                                    max_retries = data.get("max_retries", 10)
                                    err_status = data.get("error_status", "?")
                                    err_text = data.get("error", "unknown")
                                    api_retry_count = attempt
                                    api_last_error = f"HTTP {err_status}: {err_text}"
                                    logger.warning(
                                        "API retry %d/%d: status=%s error=%s session=%s",
                                        attempt, max_retries, err_status, err_text, session_key,
                                    )
                                    # Push a live status on first retry so the
                                    # user immediately sees something.
                                    if not api_retry_surfaced:
                                        api_retry_surfaced = True
                                        seq += 1
                                        note = f"⏳ Upstream unavailable ({err_text}), retrying…"
                                        text_parts.append(note)
                                        self._publish_event(
                                            request_id, session_key, seq,
                                            kind=AgentEventKind.ASSISTANT_DELTA,
                                            text=note,
                                        )
                                # TASK-623: sub-agent (Task*) lifecycle event
                                # emission extracted into _emit_subagent_task_events.
                                seq = self._emit_subagent_task_events(
                                    msg, request_id=request_id,
                                    session_key=session_key, seq=seq,
                                )
                            msg_type = type(msg).__name__
                            if not isinstance(msg, (StreamEvent, SystemMessage)):
                                content = getattr(msg, "content", [])
                                logger.debug(
                                    "SDK msg: %s blocks=%d content_types=%s",
                                    msg_type, len(content) if isinstance(content, list) else 0,
                                    [getattr(b, "type", type(b).__name__) for b in content] if isinstance(content, list) else "n/a",
                                )

                            if isinstance(msg, StreamEvent):
                                event = msg.event
                                event_type = event.get("type", "")
                                if event_type == "message_delta":
                                    ev_usage = event.get("usage")
                                    if isinstance(ev_usage, dict) and ev_usage:
                                        latest_stream_usage = ev_usage

                                if event_type == "content_block_delta":
                                    delta = event.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text = delta.get("text", "")
                                        if text:
                                            seq += 1
                                            text_parts.append(text)
                                            self._publish_event(
                                                request_id, session_key, seq,
                                                kind=AgentEventKind.ASSISTANT_DELTA,
                                                text=text,
                                            )
                                    elif delta.get("type") == "thinking_delta":
                                        # Model reasoning ("thinking"). Surface on
                                        # the REASONING_DELTA channel for the UI's
                                        # collapsible lane. Deliberately NOT
                                        # appended to text_parts — reasoning must
                                        # never enter the final assistant message
                                        # body (TASK-301).
                                        thinking = delta.get("thinking", "")
                                        if thinking:
                                            seq += 1
                                            self._publish_event(
                                                request_id, session_key, seq,
                                                kind=AgentEventKind.REASONING_DELTA,
                                                text=thinking,
                                            )
                                    elif delta.get("type") == "signature_delta":
                                        # Opaque reasoning signature — no display
                                        # value; drop it.
                                        pass
                                    elif delta.get("type") == "input_json_delta":
                                        current_tool_input += delta.get("partial_json", "")

                                elif event_type == "content_block_start":
                                    block = event.get("content_block", {})
                                    if block.get("type") == "tool_use":
                                        current_tool_name = block.get("name", "unknown")
                                        current_tool_input = ""

                                elif event_type == "content_block_stop":
                                    if current_tool_name:
                                        current_tool_name = None
                                        current_tool_input = ""

                            elif isinstance(msg, AssistantMessage):
                                # TASK-623: AssistantMessage tool_use / snapshot
                                # handling extracted into _on_assistant_message.
                                seq, latest_assistant_usage, assistant_snapshot_text = (
                                    self._on_assistant_message(
                                        msg,
                                        request_id=request_id,
                                        session_key=session_key,
                                        seq=seq,
                                        tool_names_by_id=tool_names_by_id,
                                        latest_assistant_usage=latest_assistant_usage,
                                        assistant_snapshot_text=assistant_snapshot_text,
                                    )
                                )

                            elif isinstance(msg, UserMessage):
                                # TASK-623: UserMessage tool_result (TOOL_END)
                                # handling extracted into _on_user_message_tool_results.
                                seq = await self._on_user_message_tool_results(
                                    msg,
                                    request_id=request_id,
                                    session_key=session_key,
                                    seq=seq,
                                    tool_names_by_id=tool_names_by_id,
                                    turn_screenshot_assets=turn_screenshot_assets,
                                )

                            elif isinstance(msg, ResultMessage):
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
                                # If no text and retries happened, surface the error
                                if not full_text and api_retry_count > 0:
                                    error_note = (
                                        f"\n\n❌ Upstream error after {api_retry_count} "
                                        f"retries: {api_last_error or 'unknown'}. "
                                        f"Try again in a moment."
                                    )
                                    if api_retry_surfaced:
                                        # Already pushed "⏳ retrying" — append the outcome
                                        full_text = "".join(text_parts) + error_note
                                    else:
                                        full_text = error_note.lstrip()

                                # TASK-623: token-usage / context-window
                                # extraction moved to _compute_result_usage
                                # (behavior-identical; returns partial ctx_window/
                                # max_output on internal failure just as before).
                                token_usage_payload, ctx_window, max_output = (
                                    self._compute_result_usage(
                                        msg,
                                        actual_model=actual_model,
                                        model=model,
                                        bot_context_window=bot_context_window,
                                        latest_assistant_usage=latest_assistant_usage,
                                        latest_stream_usage=latest_stream_usage,
                                    )
                                )

                                # Compaction outcome. The /compact ResultMessage
                                # usage is all-zeros and the new resident size lives
                                # only in the transcript, so on success we read back
                                # compactMetadata.postTokens to (a) append a human
                                # summary to the reply and (b) OVERRIDE the usage
                                # gauge so the UI drops to the post-compaction size
                                # immediately instead of showing the stale
                                # pre-compaction number (the "reported context is
                                # exactly the same" symptom). On failure we explain
                                # why (e.g. "Not enough messages to compact.") so a
                                # no-op /compact isn't silent.
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
                                assistant_done_emitted = True
                                # Turn complete — release the prompt generator so
                                # the SDK closes its input stream.  Kept open until
                                # now so the can_use_tool control channel survived
                                # any AskUserQuestion pause earlier in the turn.
                                turn_done.set()
                                # ResultMessage is terminal for this send (one user
                                # message -> one assistant turn), so stop iterating
                                # NOW instead of looping back to await a trailing
                                # StopAsyncIteration. After a deferred
                                # AskUserQuestion the streaming-input session stays
                                # alive — heartbeat/stream events keep re-arming the
                                # per-message timeout — so that await can block
                                # indefinitely while STILL holding the per-session
                                # lock, deadlocking the next continuation turn on the
                                # same session (TASK-269). The `finally` below closes
                                # the stream and kills the subprocess cleanly.
                                break
                        if aborted:
                            # Cooperative abort fired — fall straight through to
                            # publish_run_done without retry. We deliberately drop
                            # any partial response: the user explicitly cancelled.
                            seq += 1
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=AgentEventKind.ASSISTANT_DONE,
                                text="".join(text_parts),
                                model=actual_model,
                            )
                            assistant_done_emitted = True
                        elif not assistant_done_emitted:
                            # Clean EOF without a ResultMessage. z.ai / GLM via
                            # the Claude SDK can end a turn after an
                            # AssistantMessage snapshot only (text and/or tool
                            # uses already captured above). Publish a fallback
                            # terminal DONE so the app persists the reply instead
                            # of timing out with response_chars=0.
                            full_text = "".join(text_parts)
                            if assistant_snapshot_text:
                                if not full_text:
                                    full_text = assistant_snapshot_text
                                elif assistant_snapshot_text.startswith(full_text):
                                    full_text = assistant_snapshot_text
                            # Surface retry errors when the turn ends empty
                            if not full_text and api_retry_count > 0:
                                error_note = (
                                    f"\n\n❌ Upstream error after {api_retry_count} "
                                    f"retries: {api_last_error or 'unknown'}. "
                                    f"Try again in a moment."
                                )
                                if api_retry_surfaced:
                                    full_text = "".join(text_parts) + error_note
                                else:
                                    full_text = error_note.lstrip()
                            # Always publish — even if full_text is empty.
                            # An empty ASSISTANT_DONE is far better than no
                            # DONE at all, which causes timeout + vanishing bubble.
                            seq += 1
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=AgentEventKind.ASSISTANT_DONE,
                                text=full_text,
                                model=actual_model,
                                attachments=turn_screenshot_assets or None,
                            )
                            assistant_done_emitted = True
                            logger.info(
                                "EOF fallback ASSISTANT_DONE: chars=%d request_id=%s session=%s",
                                len(full_text), request_id, session_key,
                            )
                        break
                    except asyncio.CancelledError:
                        # task.cancel() arrived from elsewhere (legacy path /
                        # belt-and-suspenders fallback). Make sure the run is
                        # finalized before we re-raise so the frontend doesn't
                        # see a stuck `streaming` turn.
                        logger.info(
                            "Send cancelled via task.cancel: request_id=%s session=%s",
                            request_id, session_key,
                        )
                        seq += 1
                        try:
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=AgentEventKind.ASSISTANT_DONE,
                                text="".join(text_parts),
                                model=actual_model,
                            )
                            assistant_done_emitted = True
                        except Exception:
                            logger.debug("Failed to publish ASSISTANT_DONE on cancel", exc_info=True)
                        try:
                            self._publisher.publish_run_done(request_id)
                        except Exception:
                            logger.debug("Failed to publish run_done on cancel", exc_info=True)
                        raise
                    except Exception as e:
                        if cancel_event is not None and cancel_event.is_set():
                            logger.info(
                                "Abort teardown surfaced %r; treating as abort: session=%s",
                                e, session_key,
                            )
                            aborted = True
                            seq += 1
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=AgentEventKind.ASSISTANT_DONE,
                                text="".join(text_parts),
                                model=actual_model,
                            )
                            assistant_done_emitted = True
                            break

                        # 1) Auth failure → refresh token and retry once
                        if (
                            not auth_retry_attempted
                            and not text_parts
                            and _is_auth_failure(e, stderr_lines)
                        ):
                            auth_retry_attempted = True
                            logger.warning(
                                "Auth failure for %s; refreshing token and retrying",
                                request_id,
                            )
                            continue

                        # 2) CLI crash or timeout before any text streamed →
                        #    clear stale session and retry once fresh
                        if (
                            not fresh_session_retry
                            and not text_parts
                            and (_is_cli_crash(e) or isinstance(e, TimeoutError))
                        ):
                            fresh_session_retry = True
                            reason = "timeout" if isinstance(e, TimeoutError) else "CLI crash (exit code 1)"
                            logger.warning(
                                "%s for %s; clearing session and retrying fresh",
                                reason, request_id,
                            )
                            if stderr_lines:
                                logger.warning("Captured stderr before retry: %s", stderr_lines)
                            await self._clear_session(bot_slug or session_key)
                            resume_id = None
                            continue

                        raise
                    finally:
                        # Guarantee the prompt generator is released on EVERY exit
                        # path (StopAsyncIteration, the ResultMessage break above,
                        # abort, or exception). If it stays parked on
                        # `await done_event.wait()` the SDK input stream never closes
                        # and this session's lock can never be reacquired — the
                        # TASK-269 deadlock. Event.set() is idempotent.
                        turn_done.set()
                        # Always deregister this iteration's client so a
                        # subsequent chat.abort doesn't try to disconnect()
                        # something that's already finished. We pop only if
                        # the registry still points at our client — concurrent
                        # aborts may have already popped it to disconnect().
                        if session_key and sdk_client is not None:
                            current = self._session_queue.get_active_client(session_key)
                            if current is sdk_client:
                                self._session_queue.pop_active_client(session_key)
                            try:
                                await asyncio.wait_for(sdk_client.disconnect(), timeout=20.0)
                            except (asyncio.TimeoutError, asyncio.CancelledError):
                                pass
                            except Exception:
                                logger.debug(
                                    "sdk_client.disconnect() raised", exc_info=True,
                                )

                self._publisher.publish_run_done(request_id)
                if aborted:
                    logger.info(
                        "Send aborted via chat.abort: request_id=%s session=%s",
                        request_id, session_key,
                    )
                else:
                    logger.info("Send completed: request_id=%s session=%s", request_id, session_key)

            except asyncio.CancelledError:
                # Already handled inside the inner try (we published run_done
                # before re-raising). Suppress here so the asyncio task ends
                # cleanly without the "Task was destroyed but it is pending"
                # noise.
                logger.debug("Send cancellation propagated past inner handler")
                raise
            except Exception as e:
                logger.exception("Send failed: request_id=%s", request_id)
                seq += 1
                self._publish_event(
                    request_id, session_key, seq,
                    kind=AgentEventKind.ERROR,
                    text=_classify_send_error(str(e)),
                )
                self._publisher.publish_run_done(request_id)
            finally:
                # Drop the per-run trigger_message_id mapping so we don't leak.
                self._trigger_message_ids.pop(request_id, None)
                await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
