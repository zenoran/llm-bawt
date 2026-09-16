"""Prompt scheduling safety and FK ordering for destructive bot maintenance."""
from __future__ import annotations

from sqlalchemy import inspect, text


class PromptPurgeGuard:
    """Use the caller's transaction; never create optional scheduling tables."""

    def __init__(self, conn, tables=None):
        self.conn = conn
        self.tables = set(tables if tables is not None else inspect(conn).get_table_names())

    def _json(self, column, key):
        if self.conn.dialect.name == "postgresql":
            return f"{column}->>'{key}'"
        return f"json_extract({column}, '$.{key}')"

    def guard(self, bot_ids):
        # Stabilize writers before checking, including an outbox whose delivery
        # committed before its occurrence acquired delivery_id. These locks last
        # until the enclosing purge commits/rolls back. No transport cancellation.
        if self.conn.dialect.name == "postgresql":
            for table in ("scheduled_jobs", "prompt_schedules", "prompt_occurrences", "inter_bot_deliveries"):
                if table in self.tables:
                    self.conn.execute(text(f"LOCK TABLE {table} IN SHARE ROW EXCLUSIVE MODE"))
        for bot_id in sorted(bot_ids):
            params = {"bot": bot_id}
            if {"prompt_occurrences", "scheduled_jobs"} <= self.tables:
                snapshot_bot = self._json("o.snapshot_json", "bot_id")
                blocked = self.conn.execute(text(f"""
                    SELECT 1 FROM prompt_occurrences o
                    JOIN scheduled_jobs j ON j.id=o.job_id
                    WHERE (j.bot_id=:bot OR {snapshot_bot}=:bot)
                      AND (o.state IS NULL OR o.state NOT IN
                           ('succeeded','failed','skipped','cancelled')) LIMIT 1
                """), params).first()
                if blocked:
                    raise ValueError(f"Cannot purge {bot_id}: prompt occurrences are active or unknown")
            if "inter_bot_deliveries" in self.tables:
                origin = self._json("metadata_json", "prompt_schedule")
                targets = ["target_bot_id=:bot"]
                if {"prompt_occurrences", "scheduled_jobs"} <= self.tables:
                    targets.append("id IN (SELECT o.delivery_id FROM prompt_occurrences o "
                                   "JOIN scheduled_jobs j ON j.id=o.job_id WHERE j.bot_id=:bot)")
                if "scheduled_jobs" in self.tables:
                    schedule_id = ("metadata_json->'prompt_schedule'->>'schedule_id'"
                                   if self.conn.dialect.name == "postgresql" else
                                   "json_extract(metadata_json, '$.prompt_schedule.schedule_id')")
                    targets.append(f"{schedule_id} IN (SELECT id FROM scheduled_jobs WHERE bot_id=:bot)")
                blocked = self.conn.execute(text(f"""
                    SELECT 1 FROM inter_bot_deliveries
                    WHERE ({' OR '.join(targets)}) AND {origin} IS NOT NULL
                      AND (status IS NULL OR status NOT IN ('DELIVERED','FAILED','CANCELLED')
                           OR (transport_accepted_at IS NOT NULL AND status != 'DELIVERED'))
                    LIMIT 1
                """), params).first()
                if blocked:
                    raise ValueError(f"Cannot purge {bot_id}: prompt deliveries are active or unknown")

    def delete_jobs(self, bot_id):
        """Delete companions first, then run parents, then job parents."""
        counts = {}
        if "scheduled_jobs" not in self.tables:
            return counts
        for table in ("prompt_occurrences", "prompt_schedules", "job_runs"):
            if table in self.tables:
                result = self.conn.execute(text(
                    f"DELETE FROM {table} WHERE job_id IN "
                    "(SELECT id FROM scheduled_jobs WHERE bot_id=:bot)"
                ), {"bot": bot_id})
                counts[table] = result.rowcount
        result = self.conn.execute(text(
            "DELETE FROM scheduled_jobs WHERE bot_id=:bot"
        ), {"bot": bot_id})
        counts["scheduled_jobs"] = result.rowcount
        return counts
