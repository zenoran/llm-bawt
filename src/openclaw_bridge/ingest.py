from __future__ import annotations

import json
import logging
from typing import Any

from .events import OpenClawEvent, OpenClawEventKind, synthesize_event_id

logger = logging.getLogger(__name__)


class EventIngestPipeline:
    def __init__(self) -> None:
        self._stream_seq = 0

    def parse(self, raw: dict, session_key: str) -> OpenClawEvent | None:
        msg_type = str(raw.get("type", ""))

        # Ignore infra noise
        if msg_type in ("ping", "pong", "heartbeat"):
            return None

        if msg_type == "event":
            event_name = str(raw.get("event", raw.get("event_type", "")))
            payload = raw.get("payload") if raw.get("payload") is not None else raw.get("data", {})
            payload = payload if isinstance(payload, dict) else {}

            # Gateway session/run fields are usually inside payload
            sk = str(payload.get("sessionKey") or payload.get("session_key") or raw.get("session_key") or session_key)
            sk = self._normalize_session_key(sk)
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
                            tool_result=data.get("result") or data.get("output"),
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

            # Ignore unrelated system events for now (health/tick/heartbeat/presence etc)
            if event_name in {"health", "tick", "heartbeat", "presence", "shutdown"}:
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
            event_id = raw.get("event_id") or synthesize_event_id(session_key, "chat.sent", raw, self._next_seq())
            return OpenClawEvent(
                event_id=event_id,
                session_key=self._normalize_session_key(str(raw.get("session_key") or session_key)),
                run_id=raw.get("run_id"),
                kind=OpenClawEventKind.USER_MESSAGE,
                origin="user",
                raw=raw,
            )

        # req/res/control frames are not stream events for ingest
        if msg_type in {"req", "res"}:
            return None

        # fallback unknown frame
        event_id = raw.get("event_id") or synthesize_event_id(session_key, msg_type, raw, self._next_seq())
        return OpenClawEvent(
            event_id=event_id,
            session_key=self._normalize_session_key(session_key),
            run_id=raw.get("run_id"),
            kind=OpenClawEventKind.SYSTEM_NOTE,
            origin="system",
            text=str(raw),
            raw=raw,
        )

    @staticmethod
    def _normalize_session_key(sk: str) -> str:
        """Normalize gateway session keys like 'agent:main:main' -> 'main'."""
        if sk.startswith("agent:"):
            parts = sk.split(":")
            if len(parts) >= 2:
                return parts[1]
        return sk

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
