"""SDK query setup + tool-loop message handlers for the Claude send path
(TASK-623).

Extracted from ``ClaudeSendMixin._handle_send``. These are verbatim moves of
cohesive sub-blocks of the message loop:

* ``_build_sdk_env`` — proxy-vs-direct environment for the SDK subprocess.
* ``_emit_subagent_task_events`` — the sub-agent (Task*) lifecycle branch.
* ``_on_assistant_message`` — AssistantMessage tool_use / snapshot handling.
* ``_on_user_message_tool_results`` — UserMessage tool_result (TOOL_END) handling.

Each threads the mutable ``seq`` counter in and out so ordering is byte-identical
to the pre-split loop; dicts / lists (``tool_names_by_id``,
``turn_screenshot_assets``) are mutated in place exactly as before. Composed onto
``ClaudeSendMixin`` so ``self._publish_event`` etc. resolve on the assembled
instance.
"""

from __future__ import annotations

import logging

from agent_bridge.events import AgentEventKind
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk.types import (
    HookMatcher,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    ToolResultBlock,
    ToolUseBlock,
)

from claude_code_bridge.tool_events import normalize_tool_result

from ._bridge_helpers import _get_fresh_oauth_token

logger = logging.getLogger("claude_code_bridge.bridge")


class ClaudeStreamMixin:
    """SDK env construction + per-message tool-loop handlers."""

    def _build_sdk_env(
        self,
        *,
        use_proxy: bool,
        model: str,
        subagent_model: str | None,
        force_refresh: bool,
    ) -> dict:
        """Build the environment dict handed to ``ClaudeAgentOptions(env=...)``."""
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
            fresh_token = _get_fresh_oauth_token(force_refresh=force_refresh)
            if fresh_token:
                sdk_env["CLAUDE_CODE_OAUTH_TOKEN"] = fresh_token
        return sdk_env

    def _build_agent_options(
        self,
        *,
        model: str,
        system_prompt: str | None,
        disallowed_tools,
        resume_id,
        sdk_env: dict,
        settings_path,
        bot_effort: str | None,
        bot_max_turns: int | None,
        can_use_tool_cb,
        pre_tool_use_cb,
        stderr,
    ) -> ClaudeAgentOptions:
        """Construct the per-attempt ``ClaudeAgentOptions`` (verbatim move)."""
        return ClaudeAgentOptions(
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
            stderr=stderr,
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

    def _emit_subagent_task_events(
        self, msg, *, request_id: str, session_key: str, seq: int,
    ) -> int:
        """Publish structured sub-agent lifecycle events; return the new ``seq``."""
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
        return seq

    def _on_assistant_message(
        self,
        msg,
        *,
        request_id: str,
        session_key: str,
        seq: int,
        tool_names_by_id: dict,
        latest_assistant_usage: dict | None,
        assistant_snapshot_text: str,
    ) -> tuple[int, dict | None, str]:
        """Handle an AssistantMessage; return
        ``(seq, latest_assistant_usage, assistant_snapshot_text)``."""
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
        return seq, latest_assistant_usage, assistant_snapshot_text

    async def _on_user_message_tool_results(
        self,
        msg,
        *,
        request_id: str,
        session_key: str,
        seq: int,
        tool_names_by_id: dict,
        turn_screenshot_assets: list[dict],
    ) -> int:
        """Handle a UserMessage's tool_result blocks (TOOL_END); return ``seq``."""
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
        return seq
