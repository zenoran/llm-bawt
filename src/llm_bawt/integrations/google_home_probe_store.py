"""Bounded encrypted probe captures, separate from chat/history/agent execution."""
from __future__ import annotations

import time

from sqlalchemy import Column, Float, MetaData, String, Table, Text, delete, select
from sqlalchemy.exc import IntegrityError

from llm_bawt.service.providers import crypto
from llm_bawt.utils.schema import SchemaBootstrapGuard

_metadata = MetaData()
_captures = Table(
    "google_home_probe_captures", _metadata,
    Column("request_key", String(64), primary_key=True),
    Column("created_at", Float, nullable=False),
    Column("payload_enc", Text, nullable=False),
)


class ProbeCaptureStore:
    _guard = SchemaBootstrapGuard()

    def __init__(self, engine):
        self.engine = engine
        self._guard.run(engine, "google-home-probe-v1", lambda conn: _metadata.create_all(conn))

    def record(self, key: str, payload: str):
        encrypted = crypto.encrypt(payload)
        now = time.time()
        # Repeat deliveries do not create more captures. No operational effects.
        try:
            with self.engine.begin() as conn:
                conn.execute(_captures.insert().values(
                    request_key=key, created_at=now, payload_enc=encrypted,
                ))
        except IntegrityError:
            pass
        with self.engine.begin() as conn:
            conn.execute(delete(_captures).where(_captures.c.created_at < now - 86400))
            old = select(_captures.c.request_key).order_by(
                _captures.c.created_at.desc(), _captures.c.request_key
            ).offset(1000)
            conn.execute(delete(_captures).where(_captures.c.request_key.in_(old)))

    def recent(self, limit: int = 20) -> list[dict]:
        """Internal-only inspection. No public read endpoint."""
        import json

        with self.engine.connect() as conn:
            rows = conn.execute(select(_captures).where(
                _captures.c.created_at >= time.time() - 86400
            ).order_by(_captures.c.created_at.desc()).limit(min(max(limit, 1), 100))).mappings()
            return [{"created_at": row["created_at"],
                     "payload": json.loads(crypto.decrypt(row["payload_enc"]))} for row in rows]
