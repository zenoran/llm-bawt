# Background Job Scheduler

The `JobScheduler` (in `src/llm_bawt/service/scheduler.py`) runs recurring maintenance jobs from inside the FastAPI service process. It wakes every ~30 seconds, picks up any `scheduled_jobs` row whose `next_run_at` is due (or NULL), and records the outcome in `job_runs`.

Default jobs are registered idempotently at service start via `init_default_jobs()` so a fresh database boots with everything wired and a stable boot is safe to repeat.

## Job types

| `JobType` value | Cadence | Scope | Purpose |
|---|---|---|---|
| `profile_maintenance` | per `PROFILE_MAINTENANCE_INTERVAL_MINUTES` | wildcard `*` (per bot) | Rebuild entity-profile summaries when attributes have changed since the last build |
| `history_summarization` | per `HISTORY_SUMMARIZATION_INTERVAL_MINUTES` | wildcard `*` (per bot) | Compress eligible session history into summaries |
| `memory_extraction` | per `MEMORY_EXTRACTION_INTERVAL_MINUTES` | wildcard `*` (per bot) | LLM/heuristic fact extraction from unprocessed messages |
| `memory_consolidation` | manual / per-bot | per bot | Merge duplicate memories |
| `memory_decay` | manual / per-bot | per bot | Prune low-importance memories whose decay window has elapsed |
| `media_gc` | nightly at 04:00 UTC (interval `MEDIA_GC_INTERVAL_MINUTES`, default 1440) | global `system` | Garbage-collect orphan `media_assets` rows + their on-disk blob variants (TASK-231) |

Wildcard (`bot_id="*"`) jobs are expanded at execution time into one run per bot — new bots are picked up automatically without restarting the service.

## Activity gating

Most jobs are skipped when there's no relevant new activity since the last run (see `_has_new_activity_for_job`):

- `history_summarization`: skipped unless there are pending unsummarized sessions *or* fresh message activity
- `memory_extraction`: skipped unless there are unprocessed messages
- `memory_consolidation` / `memory_decay`: skipped unless memories were updated
- `profile_maintenance`: skipped unless profile attributes were updated after the last summary build
- `media_gc`: **always runs** — the SQL probe is cheap and worst-case is a 24h pile-up

Skipped runs land in `job_runs` with `status='skipped'` and a `result_json.reason` explaining why.

## Manual triggers

Ops can force any registered job to run at the next scheduler tick:

```bash
curl -sS -X POST http://<service-host>:8642/v1/jobs/<JOB_TYPE>/trigger
```

The path segment is case-insensitive (normalized to the lower-case enum value). Triggering sets `next_run_at` to `now() - 1m` so the scheduler picks it up immediately; the response is `{"success": true, "job_type": "<normalized>"}`.

Examples:

```bash
# Force a media_assets GC sweep now (TASK-231)
curl -sS -X POST http://localhost:8642/v1/jobs/MEDIA_GC/trigger

# Force a profile-maintenance pass
curl -sS -X POST http://localhost:8642/v1/jobs/profile_maintenance/trigger
```

Inspect the result:

```bash
# Latest runs for a given job type, including parsed result payload
curl -sS 'http://localhost:8642/v1/jobs/runs?job_type=media_gc&include_result=true&limit=5'
```

Or query the DB directly:

```sql
SELECT started_at, finished_at, status, duration_ms, result_json, error_message
FROM job_runs
WHERE job_id = (
    SELECT id FROM scheduled_jobs WHERE job_type = 'media_gc'
)
ORDER BY started_at DESC
LIMIT 5;
```

## `media_gc` specifics

The `MEDIA_GC` job (see `service/jobs/media_gc.py`) sweeps `media_assets` for orphans and deletes both the row and the three on-disk blob variants (`originals/`, `thumb_256/`, `preview_1024/`) via `MediaStore.delete`.

An asset is considered an orphan when **either**:

- `expires_at IS NOT NULL AND expires_at < NOW()` — soft-deleted, GC'd immediately at any age; producers use `expires_at` to mark abandoned uploads, **or**
- `created_at < NOW() - INTERVAL '<grace_days> days'` (default 7) **and** no `{bot}_messages.attachments` row anywhere references the asset by `id`. The grace window lets clients retry / wire a paste cleanup into a follow-up message without yanking the blob.

Reference detection enumerates every `*_messages` table from `information_schema` and UNIONs the referenced `asset_id` set — new per-bot tables are picked up the next time the job runs, no code change required.

Result payload (visible in `job_runs.result_json`):

```json
{
  "orphan_count": 17,
  "deleted_count": 17,
  "freed_bytes": 8421120,
  "dry_run": false,
  "scanned_tables": 22,
  "errors": []
}
```

Tunable via the job's `config_json`:

- `grace_days` (int, default 7) — minimum age before an unreferenced asset is eligible
- `dry_run` (bool, default false) — compute the orphan list and `freed_bytes` without deleting anything

Each delete is idempotent, so a crash mid-sweep is safe — the next nightly run picks up where the previous one left off.
