# Background Scheduler

`llm-bawt` runs a DB-backed scheduler inside the FastAPI service process. The
implementation lives in [src/llm_bawt/service/scheduler.py](/home/bridge/dev/llm-bawt/src/llm_bawt/service/scheduler.py)
and the API surface is in
[src/llm_bawt/service/routes/jobs.py](/home/bridge/dev/llm-bawt/src/llm_bawt/service/routes/jobs.py).

## How it works

- The service creates `scheduled_jobs` and `job_runs` tables at startup.
- `init_default_jobs()` seeds the built-in recurring jobs if they do not exist.
- `JobScheduler` wakes every `SCHEDULER_CHECK_INTERVAL_SECONDS` seconds
  (default `30`) and executes rows whose `next_run_at` is due or `NULL`.
- Wildcard jobs with `bot_id="*"` are expanded at runtime so newly-created bots
  are picked up automatically.

## Built-in jobs

| Job type | Default scope | Default interval | Notes |
|---|---|---:|---|
| `profile_maintenance` | `*` | `60m` | Rebuilds user/entity profile summaries |
| `history_summarization` | `*` | `30m` | Summarizes eligible message history |
| `memory_extraction` | `*` | `30m` | Extracts memories from unprocessed messages |
| `memory_consolidation` | per-bot | manual | Supported by the scheduler code, not auto-seeded |
| `memory_decay` | per-bot | manual | Supported by the scheduler code, not auto-seeded |
| `media_gc` | `system` | `1440m` | Daily media garbage collection; first run is anchored to `04:00 UTC` |

## Activity gating

The scheduler skips work when there is nothing new to process. Current
examples:

- `history_summarization` skips unless there are unsummarized sessions or new messages.
- `memory_extraction` skips unless there are unprocessed messages.
- `memory_consolidation` and `memory_decay` skip unless memories changed.
- `profile_maintenance` skips unless profile attributes changed.
- `media_gc` always runs when due.

Skipped runs are still written to `job_runs` with `status="skipped"`.

## API

```bash
# List scheduled jobs
curl -s http://localhost:8642/v1/jobs | jq

# Trigger a job immediately
curl -s -X POST http://localhost:8642/v1/jobs/profile_maintenance/trigger | jq

# Inspect recent runs
curl -s 'http://localhost:8642/v1/jobs/runs?job_type=media_gc&include_result=true&limit=5' | jq
```

## Relevant config

Defined in [src/llm_bawt/utils/config.py](/home/bridge/dev/llm-bawt/src/llm_bawt/utils/config.py):

- `SCHEDULER_ENABLED`
- `SCHEDULER_CHECK_INTERVAL_SECONDS`
- `PROFILE_MAINTENANCE_INTERVAL_MINUTES`
- `HISTORY_SUMMARIZATION_INTERVAL_MINUTES`
- `MEMORY_EXTRACTION_INTERVAL_MINUTES`
- `MEDIA_GC_INTERVAL_MINUTES`
- `MEDIA_GC_GRACE_DAYS`

## Media GC result shape

The `media_gc` job writes a JSON result payload similar to:

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
