from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .events import OpenClawEvent, OpenClawEventKind, synthesize_event_id

logger = logging.getLogger(__name__)


@dataclass
class IngestFilterConfig:
    """Configurable filters applied at the ingest boundary.

    - drop_msg_types: top-level ``type`` values to silently discard (e.g. ping, pong).
    - drop_event_names: gateway event names to discard (e.g. health, tick).
    - drop_content_patterns: regex patterns matched against user-message text;
      if *any* pattern matches the full text, the message is dropped.
    """
    drop_msg_types: set[str] = field(default_factory=lambda: {"ping", "pong", "heartbeat"})
    drop_event_names: set[str] = field(default_factory=lambda: {"health", "tick", "heartbeat", "presence", "shutdown"})
    drop_content_patterns: list[re.Pattern[str]] = field(default_factory=list)

    @classmethod
    def from_env(cls, *, drop_patterns_csv: str = "", drop_events_csv: str = "", drop_msg_types_csv: str = "") -> IngestFilterConfig:
        """Build from comma-separated env-var strings.

        Values are *merged* with the built-in defaults, not replacing them.
        """
        cfg = cls()
        if drop_msg_types_csv:
            for t in drop_msg_types_csv.split(","):
                t = t.strip()
                if t:
                    cfg.drop_msg_types.add(t)
        if drop_events_csv:
            for e in drop_events_csv.split(","):
                e = e.strip()
                if e:
                    cfg.drop_event_names.add(e)
        if drop_patterns_csv:
            for pat in drop_patterns_csv.split(","):
                pat = pat.strip()
                if pat:
                    try:
                        cfg.drop_content_patterns.append(re.compile(pat, re.DOTALL))
                    except re.error as exc:
                        logger.warning("Invalid ingest drop pattern %r: %s", pat, exc)
        return cfg

    def should_drop_content(self, text: str) -> bool:
        return any(p.search(text) for p in self.drop_content_patterns)


class EventIngestPipeline:
    def __init__(self, filter_config: IngestFilterConfig | None = None) -> None:
        self._stream_seq = 0
        self._filter = filter_config or IngestFilterConfig()

    def parse(self, raw: dict, session_key: str) -> OpenClawEvent | None:
        msg_type = str(raw.get("type", ""))

        # Ignore infra noise
        if msg_type in self._filter.drop_msg_types:
            return None

        if msg_type == "event":
            event_name = str(raw.get("event", raw.get("event_type", "")))
            payload = raw.get("payload") if raw.get("payload") is not None else raw.get("data", {})
            payload = payload if isinstance(payload, dict) else {}

            # Gateway session/run fields are usually inside payload
            sk = str(payload.get("sessionKey") or payload.get("session_key") or raw.get("session_key") or session_key)
            # Use raw session key from gateway — no normalization
            run_id = payload.get("runId") or payload.get("run_id") or raw.get("run_id")
            seq = payload.get("seq") if isinstance(payload.get("seq"), int) else raw.get("seq")

            event_id = (
                raw.get("event_id")
                or payload.get("eventId")
                or synthesize_event_id(sk, event_name, payload, self._next_seq())
            )

            # --- Agent stream events (authoritative for run lifecycle + deltas) ---
            if event_name == "agent":
                stream = str(payload.get("stream", ""))
                data = payload.get("data") if isinstance(payload.get("data"), dict) else {}

                if stream == "lifecycle":
                    phase = str(data.get("phase", ""))
                    if phase == "start":
                        return OpenClawEvent(
                            event_id=event_id,
                            session_key=sk,
                            seq=seq,
                            run_id=run_id,
                            kind=OpenClawEventKind.RUN_STARTED,
                            origin="system",
                            raw=raw,
                        )
                    if phase == "end":
                        return OpenClawEvent(
                            event_id=event_id,
                            session_key=sk,
                            seq=seq,
                            run_id=run_id,
                            kind=OpenClawEventKind.RUN_COMPLETED,
                            origin="system",
                            raw=raw,
                        )
                    # other lifecycle phases are useful notes (fallback_cleared, etc)
                    return OpenClawEvent(
                        event_id=event_id,
                        session_key=sk,
                        seq=seq,
                        run_id=run_id,
                        kind=OpenClawEventKind.SYSTEM_NOTE,
                        origin="system",
                        text=json.dumps(data, ensure_ascii=False),
                        raw=raw,
                    )

                if stream == "assistant":
                    delta = str(data.get("delta") or data.get("text") or "")
                    return OpenClawEvent(
                        event_id=event_id,
                        session_key=sk,
                        seq=seq,
                        run_id=run_id,
                        kind=OpenClawEventKind.ASSISTANT_DELTA,
                        origin="system",
                        text=delta,
                        raw=raw,
                    )

                if stream == "tool":
                    phase = str(data.get("phase") or data.get("state") or "")
                    tool_name = str(data.get("name") or data.get("tool") or "")
                    if phase in ("start", "calling"):
                        return OpenClawEvent(
                            event_id=event_id,
                            session_key=sk,
                            seq=seq,
                            run_id=run_id,
                            kind=OpenClawEventKind.TOOL_START,
                            origin="system",
                            tool_name=tool_name,
                            tool_arguments=data.get("arguments") or data.get("args") or data.get("input"),
                            raw=raw,
                        )
                    if phase in ("end", "result", "done"):
                        return OpenClawEvent(
                            event_id=event_id,
                            session_key=sk,
                            seq=seq,
                            run_id=run_id,
                            kind=OpenClawEventKind.TOOL_END,
                            origin="system",
                            tool_name=tool_name,
                            tool_result=data.get("result") or data.get("output") or data.get("meta"),
                            raw=raw,
                        )

                if stream == "error":
                    return OpenClawEvent(
                        event_id=event_id,
                        session_key=sk,
                        seq=seq,
                        run_id=run_id,
                        kind=OpenClawEventKind.ERROR,
                        origin="system",
                        text=json.dumps(data, ensure_ascii=False),
                        raw=raw,
                    )

                # Log unknown agent streams for debugging
                if stream not in ("lifecycle", "assistant", "tool", "error"):
                    logger.debug("Unknown agent stream=%s data=%s", stream, json.dumps(data, default=str)[:200])

            # --- Chat events (delta/final assembled message) ---
            if event_name == "chat":
                state = str(payload.get("state", "")).lower()
                message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
                text = self._extract_message_text(message)
                if not text:
                    text = str(payload.get("text") or payload.get("chunk") or "")

                if state == "final":
                    return OpenClawEvent(
                        event_id=event_id,
                        session_key=sk,
                        seq=seq,
                        run_id=run_id,
                        kind=OpenClawEventKind.ASSISTANT_DONE,
                        origin="system",
                        text=text,
                        raw=raw,
                    )

                if state == "delta":
                    # Suppress chat deltas — agent.assistant stream already
                    # provides ASSISTANT_DELTA events with the same content.
                    return None

            # Ignore unrelated system events
            if event_name in self._filter.drop_event_names:
                return None

            # Unknown gateway event -> keep as system note for observability
            return OpenClawEvent(
                event_id=event_id,
                session_key=sk,
                seq=seq,
                run_id=run_id,
                kind=OpenClawEventKind.SYSTEM_NOTE,
                origin="system",
                raw=raw,
            )

        # Some gateways emit non-event frames like chat.sent
        if msg_type == "chat.sent":
            text = self._extract_message_text(raw.get("message", {})) or str(raw.get("content") or raw.get("text") or "")
            if text and self._filter.should_drop_content(text):
                logger.debug("Dropping chat.sent matching content filter: %.80s…", text)
                return None
            event_id = raw.get("event_id") or synthesize_event_id(session_key, "chat.sent", raw, self._next_seq())
            return OpenClawEvent(
                event_id=event_id,
                session_key=str(raw.get("session_key") or session_key),
                run_id=raw.get("run_id"),
                kind=OpenClawEventKind.USER_MESSAGE,
                origin="user",
                text=text or None,
                raw=raw,
            )

        # req/res/control frames are not stream events for ingest
        if msg_type in {"req", "res"}:
            return None

        # fallback unknown frame
        event_id = raw.get("event_id") or synthesize_event_id(session_key, msg_type, raw, self._next_seq())
        return OpenClawEvent(
            event_id=event_id,
            session_key=session_key,
            run_id=raw.get("run_id"),
            kind=OpenClawEventKind.SYSTEM_NOTE,
            origin="system",
            text=str(raw),
            raw=raw,
        )

    @property
    def filter_config(self) -> IngestFilterConfig:
        return self._filter

    def should_drop_content(self, text: str) -> bool:
        return self._filter.should_drop_content(text)

    @staticmethod
    def _extract_message_text(message: dict[str, Any]) -> str:
        if not message:
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    t = item.get("text")
                    if isinstance(t, str) and t:
                        parts.append(t)
            return "".join(parts)
        return ""

    def _next_seq(self) -> int:
        self._stream_seq += 1
        return self._stream_seq
