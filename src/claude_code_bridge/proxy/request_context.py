"""Request-local metadata shared by the bridge and proxy adapters.

The Claude CLI supports ``ANTHROPIC_CUSTOM_HEADERS`` for requests sent to a
custom ``ANTHROPIC_BASE_URL``.  The bridge uses that channel to carry opaque,
non-secret routing metadata into the in-process proxy without putting volatile
values in the model prompt.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass

CONVERSATION_HEADER = "X-LLM-Bawt-Conversation-ID"
BOT_HEADER = "X-LLM-Bawt-Bot-ID"
REQUEST_HEADER = "X-LLM-Bawt-Request-ID"

_OPAQUE_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def durable_conversation_identity(
    *, bot_id: str, session_key: str, thread_session_id: str
) -> str:
    """Return a stable opaque identity for one durable conversation.

    ``session_key`` contains the app's bot/user routing scope and
    ``thread_session_id`` is the durable DB thread rotated by ``/new``.  A
    versioned domain separator prevents accidental collisions with unrelated
    hashes.  The 128-bit UUID-shaped result matches the format historically
    accepted by the ChatGPT backend's ``session_id`` header.
    """
    seed = "\0".join(
        ("llm-bawt-proxy-conversation-v1", bot_id, session_key, thread_session_id)
    ).encode("utf-8")
    return uuid.UUID(bytes=hashlib.sha256(seed).digest()[:16]).hex


def valid_conversation_identity(value: str | None) -> str | None:
    """Accept only the opaque format emitted by the bridge."""
    candidate = (value or "").strip().lower()
    return candidate if _OPAQUE_ID_RE.fullmatch(candidate) else None


@dataclass(slots=True)
class ProxyRequestContext:
    """Mutable request telemetry; one instance is never shared across calls."""

    request_id: str
    provider: str
    bot_id: str | None = None
    conversation_id: str | None = None
    started_at: float = 0.0
    account_hash: str = "default"
    active_provider: int = 0
    active_account: int = 0
    queue_wait_ms: float = 0.0
    local_setup_ms: float | None = None
    upstream_ttfb_ms: float | None = None
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = time.perf_counter()

    @property
    def session_hash(self) -> str:
        return (self.conversation_id or "missing")[:12]

    @property
    def cache_hit_pct(self) -> float:
        if not self.input_tokens:
            return 0.0
        return 100.0 * self.cached_tokens / self.input_tokens

    def record_usage(
        self, input_tokens: int, output_tokens: int, cached_tokens: int, _created: int
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_tokens = cached_tokens


def custom_header_env(context: ProxyRequestContext) -> str:
    """Serialize proxy metadata in Claude CLI's newline header format."""
    lines = [
        f"{CONVERSATION_HEADER}: {context.conversation_id}",
        f"{BOT_HEADER}: {context.bot_id or 'unknown'}",
        f"{REQUEST_HEADER}: {context.request_id}",
    ]
    return "\n".join(lines)
