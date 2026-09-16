"""Check durable source-row consent before extracting from summary prose."""
from __future__ import annotations

import json

from sqlalchemy import bindparam, text


def _metadata(value):
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else {}


def summary_allows_extraction(backend, summary_id: str) -> bool:
    """A mixed summary is indivisible: skip it if any source opted out.

    Check canonical provenance rather than mutable schedule settings. Missing
    source rows/provenance cannot prove consent and therefore fail closed. This
    also covers user content copied into a summary and extraction after restart.
    """
    table = backend._messages_table_name
    with backend.engine.connect() as conn:
        value = conn.execute(text(
            f"SELECT summary_metadata FROM {table} WHERE id=:id AND role='summary'"
        ), {"id": summary_id}).scalar_one_or_none()
        metadata = _metadata(value)
        if metadata.get("extract_memory") is False:
            return False
        ids = metadata.get("message_ids")
        if not isinstance(ids, list) or not ids:
            return False
        rows = conn.execute(text(
            f"SELECT id, summary_metadata FROM {table} WHERE id IN :ids"
        ).bindparams(bindparam("ids", expanding=True)), {"ids": ids}).mappings().all()
        return (len(rows) == len(set(ids)) and all(
            _metadata(row["summary_metadata"]).get("extract_memory") is not False
            for row in rows
        ))
