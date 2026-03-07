from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class OpenClawEventKind(str, Enum):
    ASSISTANT_DELTA = "assistant_delta"
    ASSISTANT_DONE = "assistant_done"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    USER_MESSAGE = "user_message"
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    SYSTEM_NOTE = "system_note"
    ERROR = "error"


@dataclass
class OpenClawEvent:
    event_id: str
    session_key: str
    run_id: str | None
    kind: OpenClawEventKind
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
        }

    @classmethod
    def from_dict(cls, data: dict) -> OpenClawEvent:
        """Deserialize from Redis/JSON transport."""
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif ts is None:
            ts = datetime.now(timezone.utc)
        return cls(
            event_id=data["event_id"],
            session_key=data["session_key"],
            run_id=data.get("run_id"),
            kind=OpenClawEventKind(data["kind"]),
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
        )


def synthesize_event_id(session_key: str, event_type: str, data: dict, stream_seq: int) -> str:
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    payload = f"{session_key}:{event_type}:{canonical}:{stream_seq}"
    return hashlib.sha256(payload.encode()).hexdigest()
