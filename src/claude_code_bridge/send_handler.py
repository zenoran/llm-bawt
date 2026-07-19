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


class ClaudeSendMixin:
    """Claude agent send-path (TASK-555 quarantine; see TASK-622-style follow-up).

    Split out of ``ClaudeCodeBridge`` (TASK-555); composed back via
    inheritance so ``self.*`` state and sibling-mixin methods resolve
    on the assembled instance.
    """

    async def _handle_send(
        self, fields: dict, msg_id: str, async_redis,
    ) -> None:
        request_id = fields.get("request_id", "")
        session_key = fields.get("session_key", "")
        bot_slug = (fields.get("bot_id", "") or "").strip() or _bot_slug_from_session_key(session_key)
        message = fields.get("message", "")
        system_prompt = fields.get("system_prompt") or None
        model = (fields.get("model") or "").strip()
        # TASK-501: history the app pre-assembled for a fresh-session seed.
        # When present, the bridge seeds from THIS instead of calling back to
        # /v1/history/context-seed. JSON-encoded list of {role, content} dicts.
        inject_messages = None
        _raw_inject = fields.get("inject_messages")
        if _raw_inject:
            try:
                inject_messages = json.loads(_raw_inject)
            except (ValueError, TypeError) as _e:
                logger.warning("Failed to parse inject_messages: %s", _e)
        # Frontend-supplied user-message UUID; stamped on every emitted event
        # so the frontend can bucket tool activity under the originating user
        # message without falling back to turn_id heuristics.
        trigger_message_id = (fields.get("trigger_message_id") or "").strip() or None

        # Per-bot ClaudeAgentOptions tuning (TASK: bot config -> SDK).
        # ``effort`` constrains thinking depth; ``max_turns`` caps the
        # autonomous tool-loop length per dispatch. Both default to None
        # (SDK default) when the bot doesn't override.
        _allowed_effort = {"low", "medium", "high", "xhigh", "max"}
        effort_raw = (fields.get("effort") or "").strip().lower() or None
        bot_effort = effort_raw if effort_raw in _allowed_effort else None
        if effort_raw and bot_effort is None:
            logger.warning(
                "Ignoring invalid effort=%r for %s (allowed: %s)",
                effort_raw, bot_slug, sorted(_allowed_effort),
            )
        max_turns_raw = (fields.get("max_turns") or "").strip()
        bot_max_turns: int | None = None
        if max_turns_raw:
            try:
                mt = int(max_turns_raw)
                bot_max_turns = mt if mt > 0 else None
            except ValueError:
                logger.warning(
                    "Ignoring invalid max_turns=%r for %s (must be positive int)",
                    max_turns_raw, bot_slug,
                )

        # TASK-546: Per-bot subagent/background model override. When set,
        # the bridge injects it as ANTHROPIC_SMALL_FAST_MODEL and
        # CLAUDE_CODE_SUBAGENT_MODEL on proxy-routed turns so Claude Code's
        # internal background Haiku calls and subagent model resolution use
        # a provider-qualified model the proxy accepts instead of bare
        # Anthropic IDs (claude-haiku-4-5-...) that the proxy rejects.
        subagent_model = (fields.get("subagent_model") or "").strip() or None

        # TASK-609: app-resolved catalog context window. The app (single Tier-2
        # authority) resolves the true per-model window and sends it down; the
        # bridge consumes this scalar to report the real window for
        # proxy-routed models the Claude CLI defaults to 200k. No bridge-side
        # window table. Absent/invalid -> None -> defer to the SDK's own view.
        cw_raw = (fields.get("context_window") or "").strip()
        bot_context_window: int | None = None
        if cw_raw:
            try:
                cw = int(cw_raw)
                bot_context_window = cw if cw > 0 else None
            except ValueError:
                logger.warning(
                    "Ignoring invalid context_window=%r for %s (must be positive int)",
                    cw_raw, bot_slug,
                )

        # TASK-593: app-resolved, DB-managed base SDK tool policy. Keep the raw
        # field until proxy routing is known; the pure resolver below decodes it,
        # falls back safely, and adds proxy-only exclusions.
        configured_disallowed_tools = fields.get("disallowed_tools")

        attachments_raw = fields.get("attachments", "")
        attachments: list[dict] = []
        if attachments_raw:
            try:
                attachments = json.loads(attachments_raw)
            except json.JSONDecodeError:
                pass

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

                    sdk_env = {}
                    # Force Task/Agent subagents to run SYNCHRONOUSLY. CLI 2.1.x
                    # backgrounds agents by default: the tool returns immediately,
                    # the model ends its turn ("agents launched, standing by"),
                    # and the CLI re-invokes the model via task-notification when
                    # they finish — emitting a SECOND ResultMessage. This bridge
                    # is one-send-one-turn: it finalizes on the FIRST ResultMessage
                    # and disconnect()s the client, which kills the subprocess and
                    # orphans every still-running subagent, so their results are
                    # lost and the user sees a dead-end turn. With this flag the
                    # CLI awaits agent completion in-turn (verified live on
                    # 2.1.191: flag on → tool_result carries the agent's answer,
                    # single ResultMessage; flag off → "running in background"
                    # stub + task-notification + second ResultMessage).
                    sdk_env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] = "1"
                    if use_proxy:
                        sdk_env["ANTHROPIC_BASE_URL"] = self._proxy_base_url  # type: ignore[assignment]
                        # The SDK still requires *some* auth token to send; the
                        # proxy ignores it. Use a sentinel that obviously isn't
                        # a real Anthropic key so logs don't confuse anyone.
                        sdk_env["ANTHROPIC_AUTH_TOKEN"] = "proxy-routed"
                        # CLAUDE_CODE_OAUTH_TOKEN takes precedence over
                        # ANTHROPIC_AUTH_TOKEN inside the CLI; clear it so
                        # the SDK doesn't fall back to api.anthropic.com.
                        sdk_env["CLAUDE_CODE_OAUTH_TOKEN"] = ""
                        # TASK-546: Override the small/fast (Haiku) and subagent
                        # models so Claude Code's internal background calls
                        # (title gen, tool-use summaries, API verification) and
                        # Agent tool subagents route through the proxy with a
                        # provider-qualified model instead of sending bare
                        # Anthropic IDs that the proxy rejects (HTTP 400).
                        # If subagent_model is not configured, fall back to the
                        # main model — the proxy accepts it and the cost is
                        # acceptable for the low volume of background calls.
                        effective_subagent_model = subagent_model or model
                        sdk_env["ANTHROPIC_SMALL_FAST_MODEL"] = effective_subagent_model
                        sdk_env["CLAUDE_CODE_SUBAGENT_MODEL"] = effective_subagent_model
                        logger.debug(
                            "Routing turn through proxy: model=%s base=%s "
                            "subagent_model=%s",
                            model, self._proxy_base_url, effective_subagent_model,
                        )
                    else:
                        # Read fresh token on each request (auto-refresh from credentials file)
                        fresh_token = _get_fresh_oauth_token(force_refresh=auth_retry_attempted)
                        if fresh_token:
                            sdk_env["CLAUDE_CODE_OAUTH_TOKEN"] = fresh_token

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

                    options = ClaudeAgentOptions(
                        model=model,
                        # TASK-288: send the system prompt on EVERY turn, resume
                        # included. The SDK rebuilds and re-sends systemPrompt on
                        # every query() (it is NOT locked at session start), and
                        # the prompt is now byte-stable (temporal + response-style
                        # moved off it), so re-sending reads the full prefix from
                        # cache (~10% cost, POC-confirmed) while keeping persona
                        # alive on resume instead of decaying to the stock default.
                        system_prompt=system_prompt,
                        cwd=self._cwd,
                        disallowed_tools=disallowed_tools,
                        permission_mode=self._permission_mode,
                        include_partial_messages=True,
                        resume=resume_id,
                        add_dirs=self._add_dirs if self._add_dirs else [],
                        stderr=_log_stderr,
                        env=sdk_env,
                        settings=settings_path,
                        effort=bot_effort,
                        # Opt into summarized reasoning text. On Opus 4.7+ the
                        # thinking display defaults to "omitted" — the model
                        # thinks but Anthropic streams empty thinking blocks
                        # (signature only), so the UI reasoning lane (TASK-301)
                        # had nothing to render on the Anthropic-direct path.
                        # "summarized" returns a readable summary as
                        # thinking_delta text (raw CoT is never exposed on
                        # Opus). Proxy-routed models synthesize their own
                        # thinking and are unaffected by this flag.
                        thinking={"type": "adaptive", "display": "summarized"},
                        max_turns=bot_max_turns,
                        mcp_servers=self._mcp_servers if self._mcp_servers else {},
                        can_use_tool=can_use_tool_cb,
                        # TASK-292: matcher=None → fires for every tool. The hook
                        # (not can_use_tool) is the sole approval gate.
                        hooks={
                            "PreToolUse": [
                                HookMatcher(matcher=None, hooks=[pre_tool_use_cb]),
                            ],
                        },
                        # The SDK's stdio reader defaults to a 1 MiB JSON buffer
                        # (claude_agent_sdk subprocess_cli _DEFAULT_MAX_BUFFER_SIZE).
                        # A single Playwright screenshot or large browser_snapshot
                        # tool-result blows past that and the reader raises
                        # SDKJSONDecodeError mid-stream, which the bridge surfaces
                        # as an ERROR event and the app turns into a hard
                        # RuntimeError — i.e. the bot's turn silently dies. Raise
                        # the ceiling so those results flow; screenshots are then
                        # offloaded to the media store below.
                        max_buffer_size=32 * 1024 * 1024,
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
                                # ── Sub-agent task lifecycle (TASK-344) ──
                                # The SDK emits TaskStarted/Progress/Updated/
                                # Notification messages (SystemMessage subclasses)
                                # when the Agent or Workflow tool spawns sub-agents.
                                # Detect them and publish structured events so the
                                # app→frontend pipeline can show live progress.
                                if isinstance(msg, TaskStartedMessage):
                                    seq += 1
                                    logger.info(
                                        "Sub-agent started: task_id=%s desc=%s tool_use_id=%s",
                                        msg.task_id, msg.description, msg.tool_use_id,
                                    )
                                    self._publish_event(
                                        request_id, session_key, seq,
                                        kind=AgentEventKind.SUBAGENT_STARTED,
                                        tool_use_id=msg.tool_use_id,
                                        extra_raw={
                                            "task_id": msg.task_id,
                                            "description": msg.description or "",
                                            "task_type": getattr(msg, "task_type", None),
                                            "uuid": getattr(msg, "uuid", ""),
                                        },
                                    )
                                elif isinstance(msg, TaskProgressMessage):
                                    seq += 1
                                    usage = msg.usage if isinstance(msg.usage, dict) else {}
                                    self._publish_event(
                                        request_id, session_key, seq,
                                        kind=AgentEventKind.SUBAGENT_PROGRESS,
                                        tool_use_id=msg.tool_use_id,
                                        tool_name=getattr(msg, "last_tool_name", None),
                                        extra_raw={
                                            "task_id": msg.task_id,
                                            "description": msg.description or "",
                                            "usage": {
                                                "total_tokens": usage.get("total_tokens", 0),
                                                "tool_uses": usage.get("tool_uses", 0),
                                                "duration_ms": usage.get("duration_ms", 0),
                                            },
                                        },
                                    )
                                elif isinstance(msg, TaskUpdatedMessage):
                                    # Status updates (running, paused, etc.) —
                                    # surface as SUBAGENT_PROGRESS with the status.
                                    if msg.status:
                                        seq += 1
                                        self._publish_event(
                                            request_id, session_key, seq,
                                            kind=AgentEventKind.SUBAGENT_PROGRESS,
                                            extra_raw={
                                                "task_id": msg.task_id,
                                                "status": msg.status,
                                            },
                                        )
                                elif isinstance(msg, TaskNotificationMessage):
                                    seq += 1
                                    usage = msg.usage if isinstance(msg.usage, dict) else {}
                                    logger.info(
                                        "Sub-agent done: task_id=%s status=%s tool_use_id=%s",
                                        msg.task_id, msg.status, msg.tool_use_id,
                                    )
                                    self._publish_event(
                                        request_id, session_key, seq,
                                        kind=AgentEventKind.SUBAGENT_DONE,
                                        tool_use_id=msg.tool_use_id,
                                        text=getattr(msg, "summary", "") or "",
                                        extra_raw={
                                            "task_id": msg.task_id,
                                            "status": msg.status,
                                            "output_file": getattr(msg, "output_file", ""),
                                            "usage": {
                                                "total_tokens": usage.get("total_tokens", 0),
                                                "tool_uses": usage.get("tool_uses", 0),
                                                "duration_ms": usage.get("duration_ms", 0),
                                            } if usage else None,
                                        },
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
                                # Capture per-iteration usage — overwrites on
                                # each internal API call so the LAST one wins,
                                # giving the UI the model's true final view
                                # of the context (not a cumulative sum).
                                am_usage = getattr(msg, "usage", None)
                                if isinstance(am_usage, dict) and am_usage:
                                    latest_assistant_usage = am_usage
                                # When this AssistantMessage is a sub-agent's inner
                                # activity, the SDK stamps parent_tool_use_id with the
                                # spawning Agent/Workflow tool's id. Thread it onto the
                                # TOOL_START so the UI can nest this tool card under the
                                # parent Agent card. None for top-level calls (TASK-344).
                                parent_tuid = getattr(msg, "parent_tool_use_id", None)
                                snapshot_parts: list[str] = []
                                for block in getattr(msg, "content", []):
                                    if isinstance(block, ToolUseBlock):
                                        seq += 1
                                        tu_id = getattr(block, "id", None)
                                        if tu_id:
                                            tool_names_by_id[tu_id] = block.name
                                        # QDIAG (TASK-413): record when the MODEL
                                        # actually emits an AskUserQuestion tool_use
                                        # block. Pair this with the "QDIAG can_use_tool
                                        # ENTER" line below: block-seen WITHOUT a
                                        # matching ENTER == the bridge/SDK dropped the
                                        # question (control-channel wedge on a resumed
                                        # turn); block NOT seen at all == the model
                                        # narrated the question in prose without calling
                                        # the tool. Split model-vs-pipeline for real.
                                        if self._is_ask_user_question(block.name):
                                            logger.info(
                                                "QDIAG model-emitted AskUserQuestion "
                                                "tool_use_id=%s session=%s parent=%s — "
                                                "expect a matching 'QDIAG can_use_tool "
                                                "ENTER' next",
                                                tu_id, session_key, parent_tuid,
                                            )
                                        # TOOLMAP (TASK-414): the id the bridge stamps
                                        # on TOOL_START. Must equal the TOOL_END
                                        # tool_use_id for the same call; if the
                                        # frontend can't heal a card, compare this
                                        # id against the "TOOLMAP end" line and the
                                        # frontend "TOOLMAP" orphan line.
                                        logger.info(
                                            "TOOLMAP start tool=%s tool_use_id=%s parent=%s session=%s",
                                            block.name, tu_id, parent_tuid, session_key,
                                        )
                                        self._publish_event(
                                            request_id, session_key, seq,
                                            kind=AgentEventKind.TOOL_START,
                                            tool_name=block.name,
                                            tool_arguments=block.input if isinstance(block.input, dict) else {},
                                            tool_use_id=tu_id,
                                            parent_tool_use_id=parent_tuid,
                                        )
                                        continue
                                    btype = getattr(block, "type", None)
                                    if isinstance(block, dict):
                                        btype = block.get("type")
                                    if btype == "text":
                                        if isinstance(block, dict):
                                            btext = block.get("text", "")
                                        else:
                                            btext = getattr(block, "text", "") or ""
                                        if btext:
                                            snapshot_parts.append(str(btext))
                                if snapshot_parts:
                                    assistant_snapshot_text = "".join(snapshot_parts)

                            elif isinstance(msg, UserMessage):
                                # Mirror of the AssistantMessage path: a sub-agent's
                                # tool *result* arrives on a UserMessage carrying the
                                # same parent_tool_use_id, so the TOOL_END nests under
                                # the same parent Agent card as its TOOL_START.
                                parent_tuid = getattr(msg, "parent_tool_use_id", None)
                                for block in getattr(msg, "content", []):
                                    if isinstance(block, ToolResultBlock):
                                        seq += 1
                                        result_content = block.content or ""
                                        # Persist Playwright screenshots to the
                                        # media store and collect a {asset_id,
                                        # kind} ref so the app can attach them to
                                        # the bot's reply (browsable per turn).
                                        # Never let an upload failure break the
                                        # turn — the inline image still reaches
                                        # the model regardless.
                                        # Refs for THIS tool result only — stamped on
                                        # the TOOL_END below so the UI can render the
                                        # screenshot inline in the tool card the moment
                                        # it exists (TASK-483), instead of only in the
                                        # end-of-turn grid.
                                        tool_end_attachments: list[dict] | None = None
                                        if isinstance(result_content, list) and self._is_image_result_tool(
                                            tool_names_by_id.get(block.tool_use_id or "")
                                        ):
                                            try:
                                                refs = await self._persist_screenshot_blocks(
                                                    result_content, session_key, block.tool_use_id,
                                                )
                                                turn_screenshot_assets.extend(refs)
                                                tool_end_attachments = refs or None
                                            except Exception:
                                                logger.warning(
                                                    "Screenshot persist failed (tool_use_id=%s)",
                                                    block.tool_use_id, exc_info=True,
                                                )
                                        # TASK-594: the harness truncates large
                                        # tool output to a <persisted-output>
                                        # wrapper and writes the real bytes to a
                                        # tool-results/<id>.txt file. Re-hydrate
                                        # the full output from that file so the
                                        # payload's total_chars/preview reflect the
                                        # REAL result (non-fatal: falls back to the
                                        # inline wrapper if the file is missing).
                                        result_content = self._resolve_persisted_output(result_content)
                                        result_payload = normalize_tool_result(result_content)
                                        # TOOLMAP (TASK-414): log the id the bridge
                                        # stamps on TOOL_END. The frontend heals a
                                        # running card by call_id; the bridge only
                                        # emits tool_use_id here (no call_id — the SDK
                                        # ToolResultBlock has none). If a start/end
                                        # tool_use_id pair diverges, or the app fails
                                        # to translate id→call_id, the card orphans on
                                        # "running". Pair this with the frontend
                                        # "TOOLMAP" console line and the TOOL_START log
                                        # below (grep both for the same tool_use_id).
                                        logger.info(
                                            "TOOLMAP end tool_use_id=%s parent=%s session=%s is_error=%s",
                                            block.tool_use_id,
                                            parent_tuid,
                                            session_key,
                                            getattr(block, "is_error", None),
                                        )
                                        self._publish_event(
                                            request_id, session_key, seq,
                                            kind=AgentEventKind.TOOL_END,
                                            # NB: the SDK's ToolResultBlock does
                                            # not echo the tool *name* — only the
                                            # tool_use_id linking back to the
                                            # originating ToolUseBlock.  Surface
                                            # both: tool_use_id in its proper
                                            # field, and a placeholder name so
                                            # legacy consumers still see something.
                                            tool_name=block.tool_use_id or "unknown",
                                            tool_use_id=block.tool_use_id,
                                            parent_tool_use_id=parent_tuid,
                                            tool_result=result_payload.preview,
                                            tool_result_payload=result_payload.to_dict(),
                                            # The SDK marks failed tool runs with
                                            # is_error on the ToolResultBlock — the
                                            # single authoritative failure signal.
                                            # Thread it so the UI can tint the card.
                                            tool_error=bool(getattr(block, "is_error", False)),
                                            # Screenshot refs for this tool call so the
                                            # UI can show the image inline immediately
                                            # (TASK-483). turn_screenshot_assets still
                                            # flushes on ASSISTANT_DONE for history.
                                            attachments=tool_end_attachments,
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

                                # Extract token usage + context window for UI surfacing.
                                #
                                # IMPORTANT: ResultMessage.usage is CUMULATIVE across all
                                # internal API iterations in the turn — for a multi-tool-use
                                # turn that re-reads cached context on each call, the summed
                                # cache_read_input_tokens can exceed the context_window itself
                                # and produce nonsense >100% counters in the UI. We instead
                                # use the LAST AssistantMessage's per-iteration usage, which
                                # represents the model's final view of the context (what the
                                # user actually wants to see as "context fullness"). Cumulative
                                # output_tokens and total_cost_usd still come from ResultMessage
                                # since those genuinely accumulate across the turn.
                                #
                                # ResultMessage.model_usage is keyed by model id and exposes
                                # the model's contextWindow + maxOutputTokens.
                                token_usage_payload: dict | None = None
                                ctx_window = None
                                max_output = None
                                try:
                                    cumulative_usage = getattr(msg, "usage", None) or {}
                                    proxy_model = self._model_provider_prefix(
                                        actual_model or model
                                    ) is not None
                                    # Prefer stream message_delta for proxy providers
                                    # (xAI/ChatGPT/z.ai) — AssistantMessage often keeps
                                    # message_start zeros or a partial merge.
                                    iter_usage = _pick_iteration_usage(
                                        latest_assistant_usage,
                                        latest_stream_usage,
                                        cumulative_usage,
                                        proxy_model=proxy_model,
                                    )
                                    model_usage = getattr(msg, "model_usage", None) or {}
                                    ctx_window = None
                                    max_output = None
                                    if isinstance(model_usage, dict):
                                        # Prefer the actual model we ran on; fall back to any entry.
                                        mu_entry = (
                                            model_usage.get(actual_model)
                                            if actual_model
                                            else None
                                        )
                                        if mu_entry is None and model_usage:
                                            mu_entry = next(iter(model_usage.values()), None)
                                        if isinstance(mu_entry, dict):
                                            ctx_window = mu_entry.get("contextWindow")
                                            max_output = mu_entry.get("maxOutputTokens")
                                    # TASK-609: Claude Code defaults unknown (proxy-routed)
                                    # models to 200k. Override with the app-resolved catalog
                                    # window for proxy providers only — the CLI never knows
                                    # xAI/OpenAI/z.ai windows. Direct-Anthropic models keep the
                                    # SDK's own contextWindow (the app value would agree, and
                                    # the SDK is authoritative for its native models).
                                    if bot_context_window and proxy_model:
                                        if not ctx_window or int(ctx_window) in (200_000, 0):
                                            ctx_window = bot_context_window
                                        elif int(ctx_window) < bot_context_window:
                                            # e.g. CLI said 200k for a 500k/1M model
                                            ctx_window = bot_context_window
                                    if iter_usage or ctx_window:
                                        # z.ai reports input_tokens only in message_delta, so the
                                        # per-iteration AssistantMessage.usage (iter_usage) carries
                                        # the message_start value (0). Fall back to the cumulative
                                        # ResultMessage.usage, which via the SDK's last-non-zero merge
                                        # holds the real final-context input. No-op for Anthropic,
                                        # where iter_usage.input_tokens is already >0 (its
                                        # message_delta sends explicit 0s that updateUsage ignores).
                                        _input_tokens = int(iter_usage.get("input_tokens", 0) or 0)
                                        if _input_tokens == 0:
                                            _input_tokens = int(
                                                cumulative_usage.get("input_tokens", 0) or 0
                                            )
                                        _cache_read = int(
                                            iter_usage.get("cache_read_input_tokens", 0) or 0
                                        )
                                        _cache_create = int(
                                            iter_usage.get("cache_creation_input_tokens", 0) or 0
                                        )
                                        # If the chosen snapshot still has zero total input but
                                        # cumulative does not, take cache fields from cumulative too.
                                        if (
                                            _input_tokens + _cache_read + _cache_create
                                        ) == 0 and isinstance(cumulative_usage, dict):
                                            _cache_read = int(
                                                cumulative_usage.get(
                                                    "cache_read_input_tokens", 0
                                                )
                                                or 0
                                            )
                                            _cache_create = int(
                                                cumulative_usage.get(
                                                    "cache_creation_input_tokens", 0
                                                )
                                                or 0
                                            )
                                        _out_tokens = int(
                                            cumulative_usage.get("output_tokens", 0) or 0
                                        )
                                        if _out_tokens == 0:
                                            _out_tokens = int(
                                                iter_usage.get("output_tokens", 0) or 0
                                            )
                                        # Cost must be CUMULATIVE across the turn
                                        # (every internal API iteration billed). Context
                                        # fullness above is last-iteration only.
                                        if isinstance(cumulative_usage, dict) and (
                                            _usage_input_total(cumulative_usage) > 0
                                            or int(cumulative_usage.get("output_tokens", 0) or 0) > 0
                                        ):
                                            usage_for_cost = {
                                                "input_tokens": int(
                                                    cumulative_usage.get("input_tokens", 0) or 0
                                                ),
                                                "cache_read_input_tokens": int(
                                                    cumulative_usage.get(
                                                        "cache_read_input_tokens", 0
                                                    )
                                                    or 0
                                                ),
                                                "cache_creation_input_tokens": int(
                                                    cumulative_usage.get(
                                                        "cache_creation_input_tokens", 0
                                                    )
                                                    or 0
                                                ),
                                                "output_tokens": int(
                                                    cumulative_usage.get("output_tokens", 0)
                                                    or 0
                                                ),
                                            }
                                        else:
                                            usage_for_cost = {
                                                "input_tokens": _input_tokens,
                                                "cache_read_input_tokens": _cache_read,
                                                "cache_creation_input_tokens": _cache_create,
                                                "output_tokens": _out_tokens,
                                            }
                                        # Prefer provider-accurate cost for proxy models;
                                        # CLI prices unknowns as Opus-tier ($5/$25).
                                        cost = _estimate_proxy_cost_usd(
                                            actual_model or model, usage_for_cost
                                        )
                                        if cost is None:
                                            cost = getattr(msg, "total_cost_usd", None)
                                        token_usage_payload = {
                                            "input_tokens": _input_tokens,
                                            "cache_read_tokens": _cache_read,
                                            "cache_creation_tokens": _cache_create,
                                            # Output is still the cumulative turn total — that's
                                            # what the user generated overall, regardless of how
                                            # many internal iterations produced it.
                                            "output_tokens": _out_tokens,
                                            "context_window": ctx_window,
                                            "max_output_tokens": max_output,
                                            "total_cost_usd": cost,
                                        }
                                        if actual_model and (
                                            str(actual_model).startswith("zai/")
                                            or str(actual_model).startswith("xai/")
                                        ):
                                            logger.info(
                                                "proxy usage: model=%s iter_in=%s stream_in=%s "
                                                "cum_in=%s out=%s ctx=%s cost=%s",
                                                actual_model,
                                                (latest_assistant_usage or {}).get(
                                                    "input_tokens"
                                                )
                                                if isinstance(
                                                    latest_assistant_usage, dict
                                                )
                                                else None,
                                                (latest_stream_usage or {}).get(
                                                    "input_tokens"
                                                )
                                                if isinstance(latest_stream_usage, dict)
                                                else None,
                                                cumulative_usage.get("input_tokens")
                                                if isinstance(cumulative_usage, dict)
                                                else None,
                                                _out_tokens,
                                                ctx_window,
                                                cost,
                                            )
                                except Exception as _usage_err:
                                    logger.debug("Failed to extract token usage: %s", _usage_err)

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
                    text=str(e),
                )
                self._publisher.publish_run_done(request_id)
            finally:
                # Drop the per-run trigger_message_id mapping so we don't leak.
                self._trigger_message_ids.pop(request_id, None)
                await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
