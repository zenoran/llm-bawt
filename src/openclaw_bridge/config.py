"""Bridge service configuration — all from env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class BridgeConfig:
    # OpenClaw Gateway
    gateway_url: str = ""  # HTTP base URL (e.g. http://10.0.0.97:18789)
    ws_url: str = ""       # WS URL (e.g. ws://10.0.0.97:18789/v1/ws)
    ws_token: str = ""     # Bearer token for both HTTP and WS auth
    session_keys: list[str] = field(default_factory=lambda: ["main"])
    reconnect_max_delay: int = 60

    # PostgreSQL (for event store)
    postgres_user: str = "llm_bawt"
    postgres_password: str = ""
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "llm_bawt"

    # Redis (for event fanout)
    redis_url: str = "redis://localhost:6379/0"

    # Session -> bot_id mapping (JSON string from env, e.g. '{"main": "vex"}')
    session_to_bot_json: str = ""

    # Health check
    health_port: int = 8680

    # Ingest filters (comma-separated; merged with built-in defaults)
    ingest_drop_patterns: str = ""   # regex patterns to drop from user message content
    ingest_drop_events: str = ""     # additional gateway event names to drop
    ingest_drop_msg_types: str = ""  # additional top-level msg types to drop

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> BridgeConfig:
        sessions_str = os.environ.get("OPENCLAW_WS_SESSIONS", "main")
        session_keys = [s.strip() for s in sessions_str.split(",") if s.strip()]
        return cls(
            ws_url=os.environ.get("OPENCLAW_WS_URL", ""),
            ws_token=os.environ.get("OPENCLAW_GATEWAY_TOKEN", ""),
            session_keys=session_keys,
            reconnect_max_delay=int(os.environ.get("OPENCLAW_WS_RECONNECT_MAX_DELAY", "60")),
            postgres_user=os.environ.get("POSTGRES_USER", "llm_bawt"),
            postgres_password=os.environ.get("POSTGRES_PASSWORD", ""),
            postgres_host=os.environ.get("POSTGRES_HOST", "localhost"),
            postgres_port=int(os.environ.get("POSTGRES_PORT", "5432")),
            postgres_database=os.environ.get("POSTGRES_DATABASE", "llm_bawt"),
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            session_to_bot_json=os.environ.get("OPENCLAW_SESSION_TO_BOT", ""),
            health_port=int(os.environ.get("OPENCLAW_BRIDGE_HEALTH_PORT", "8680")),
            ingest_drop_patterns=os.environ.get("OPENCLAW_INGEST_DROP_PATTERNS", ""),
            ingest_drop_events=os.environ.get("OPENCLAW_INGEST_DROP_EVENTS", ""),
            ingest_drop_msg_types=os.environ.get("OPENCLAW_INGEST_DROP_MSG_TYPES", ""),
            log_level=os.environ.get("OPENCLAW_BRIDGE_LOG_LEVEL", "INFO"),
        )

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
        )

    @property
    def session_to_bot(self) -> dict[str, str]:
        if not self.session_to_bot_json:
            return {}
        import json
        try:
            return json.loads(self.session_to_bot_json)
        except (json.JSONDecodeError, TypeError):
            return {}
