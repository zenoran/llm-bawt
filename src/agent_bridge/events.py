from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AgentEventKind(str, Enum):
    ASSISTANT_DELTA = "assistant_delta"
    ASSISTANT_DONE = "assistant_done"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    USER_MESSAGE = "user_message"
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    SYSTEM_NOTE = "system_note"
    ERROR = "error"
    # Emitted whenever a bridge clears its server-side session — via /new
    # in _handle_send or via the session.reset RPC.  Lets the frontend
    # clear its visible message buffer deterministically instead of
    # racing turn_complete timing.  Payload uses raw={"bot_id", "session_key",
    # "target", "had_session"} so the UI can scope the clear.
    SESSION_RESET = "session_reset"
    # Emitted when a bridge has paused the SDK on an interactive tool call
    # (currently only the Claude Agent SDK's built-in ``AskUserQuestion``).
    # The bridge holds an asyncio.Future keyed by ``tool_use_id``; the run
    # cannot continue until the app POSTs a chat.tool_result Redis command
    # carrying the user's answer.  Payload uses ``tool_name``,
    # ``tool_arguments`` (the original tool input — e.g. ``{questions: [...]}``
    # for AskUserQuestion), and the new ``tool_use_id`` field on AgentEvent
    # so the UI can echo it back when the user picks an answer.
    AWAIT_TOOL_RESULT = "await_tool_result"
    # Streamed model reasoning ("thinking") text, surfaced on a separate channel
    # from ASSISTANT_DELTA so the UI can render a collapsible "Thinking…" lane
    # without the reasoning ever entering the final assistant message body.
    # Emitted by the claude-code bridge when the proxy/native SDK yields Anthropic
    # ``thinking_delta`` frames (TASK-301). ``text`` carries the reasoning chunk.
    # Additive: consumers that don't know this kind ignore it (see from_dict).
    REASONING_DELTA = "reasoning_delta"
    # Emitted when an approval-gated tool policy matched a tool the model tried
    # to run (TASK-292). The bridge denies the call, emits this, and ends the
    # turn cleanly (same deferred/continuation model as AWAIT_TOOL_RESULT). The
    # app persists a tool_approval_requests row, the UI renders an Approve/Deny
    # card, and on Approve the app sends an approval.grant Redis command back to
    # the bridge + dispatches a continuation turn so the model re-issues the now
    # allow-listed call. Payload: ``tool_name``, ``tool_arguments`` (the original
    # tool input), ``tool_use_id`` (the SDK tool_use id), and ``raw`` carries
    # {policy_id, severity, subject, prompt, grant_key, action}.
    APPROVAL_REQUIRED = "approval_required"


@dataclass
class AgentEvent:
    event_id: str
    session_key: str
    run_id: str | None
    kind: AgentEventKind
    origin: str  # "user" | "system" | "heartbeat" | "cron" | "subagent"
    text: str | None = None
    tool_name: str | None = None
    tool_arguments: dict | None = None
    tool_result: Any | None = None
    model: str | None = None
    seq: int | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw: dict = field(default_factory=dict)
    db_id: int | None = None  # populated after store
    # Per-turn token accounting from the upstream SDK's ResultMessage.
    # Populated on ASSISTANT_DONE for backends that expose usage info
    # (e.g. claude_code_bridge surfaces Claude Code SDK usage + modelUsage).
    # Shape: {input_tokens, cache_read_tokens, cache_creation_tokens,
    #         output_tokens, context_window, total_cost_usd}.
    token_usage: dict | None = None
    # Identifies which agent backend produced this event ("claude-code",
    # "codex", "openclaw"). Lets the UI dispatch tool rendering by
    # (provider, tool_name) so each harness can show its own native tool
    # shapes (e.g. codex file_change carries only path+kind, no diff).
    provider: str | None = None
    # Frontend-supplied user-message UUID (or "local-user-*" placeholder)
    # that triggered this run.  Stamped on TOOL_START / TOOL_END events by
    # every bridge so the frontend can bucket tool activity under the
    # originating user message without relying on turn_id heuristics.
    # Optional because passive subscription paths (e.g. CLI sessions on
    # the OpenClaw gateway) have no originating frontend message.
    trigger_message_id: str | None = None
    # SDK-supplied tool_use id for the *specific* tool call this event refers
    # to.  Stamped on TOOL_START / TOOL_END / AWAIT_TOOL_RESULT events by the
    # claude-code bridge so the app can pair a user-supplied answer (via the
    # chat.tool_result Redis command) back to the paused SDK Future.  Other
    # bridges leave it unset.
    tool_use_id: str | None = None
    # Media refs ({"asset_id": "ma_...", "kind": "image"}) produced *during*
    # this turn and already persisted to the media store — e.g. Playwright
    # screenshots the claude-code bridge offloaded instead of leaving the inline
    # base64 in the model context.  Stamped on the terminal ASSISTANT_DONE event
    # so the app can attach them to the bot's reply message (browsable per turn).
    attachments: list[dict] | None = None

    def to_dict(self) -> dict:
        """Serialize for Redis/JSON transport."""
        return {
            "event_id": self.event_id,
            "session_key": self.session_key,
            "run_id": self.run_id,
            "kind": self.kind.value,
            "origin": self.origin,
            "text": self.text,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "tool_result": self.tool_result,
            "model": self.model,
            "seq": self.seq,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "db_id": self.db_id,
            "raw": self.raw,
            "token_usage": self.token_usage,
            "provider": self.provider,
            "trigger_message_id": self.trigger_message_id,
            "tool_use_id": self.tool_use_id,
            "attachments": self.attachments,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AgentEvent:
        """Deserialize from Redis/JSON transport."""
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif ts is None:
            ts = datetime.now(timezone.utc)
        # Tolerant kind decode: an additive kind emitted by a newer producer must
        # not crash an older consumer (rolling deploy). Unknown values degrade to
        # SYSTEM_NOTE — a benign kind every consumer already ignores for display —
        # instead of raising ValueError (TASK-301).
        raw_kind = data.get("kind")
        try:
            kind = AgentEventKind(raw_kind)
        except ValueError:
            kind = AgentEventKind.SYSTEM_NOTE
        return cls(
            event_id=data["event_id"],
            session_key=data["session_key"],
            run_id=data.get("run_id"),
            kind=kind,
            origin=data.get("origin", "system"),
            text=data.get("text"),
            tool_name=data.get("tool_name"),
            tool_arguments=data.get("tool_arguments"),
            tool_result=data.get("tool_result"),
            model=data.get("model"),
            seq=data.get("seq"),
            timestamp=ts,
            db_id=data.get("db_id"),
            raw=data.get("raw", {}),
            token_usage=data.get("token_usage"),
            provider=data.get("provider"),
            trigger_message_id=data.get("trigger_message_id"),
            tool_use_id=data.get("tool_use_id"),
            attachments=data.get("attachments"),
        )


def synthesize_event_id(session_key: str, event_type: str, data: dict, stream_seq: int) -> str:
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    payload = f"{session_key}:{event_type}:{canonical}:{stream_seq}"
    return hashlib.sha256(payload.encode("utf-8", errors="surrogateescape")).hexdigest()
