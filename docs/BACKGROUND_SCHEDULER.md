# Background Job Scheduler

## Overview

The background job scheduler runs recurring maintenance tasks on configurable intervals. It is integrated into the LLM service and uses PostgreSQL for job persistence and execution tracking.

## Architecture

```
JobScheduler (async watcher loop)
    ↓ checks scheduled_jobs table for due jobs
    ↓ creates Task via factory
    ↓ enqueues to TaskProcessor
TaskProcessor (existing service infrastructure)
    ↓ processes task
    ↓ calls appropriate service (e.g., ProfileMaintenanceService)
JobScheduler
    ↓ records result in job_runs table
```

## Job Types

| Job Type | Description | Default Interval |
|----------|-------------|-----------------|
| `PROFILE_MAINTENANCE` | LLM-based consolidation of user profile attributes | 60 minutes |
| `MEMORY_CONSOLIDATION` | Consolidate related memory entries | Configurable |
| `MEMORY_DECAY` | Prune low-relevance memories | Configurable |

## Configuration

Add to `~/.config/llm-bawt/.env`:

```bash
# Enable/disable the scheduler
LLM_BAWT_SCHEDULER_ENABLED=true

# How often to check for due jobs (seconds)
LLM_BAWT_SCHEDULER_CHECK_INTERVAL_SECONDS=30

# Profile maintenance interval (minutes)
LLM_BAWT_PROFILE_MAINTENANCE_INTERVAL_MINUTES=60

# Model to use for profile maintenance (empty = auto-select local model)
LLM_BAWT_PROFILE_MAINTENANCE_MODEL=
```

## Database Schema

### `scheduled_jobs` Table

Defines recurring jobs:

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `job_type` | enum | `profile_maintenance`, `memory_consolidation`, `memory_decay` |
| `bot_id` | string | Bot this job runs for, or `*` for all |
| `enabled` | bool | Whether job is active |
| `interval_minutes` | int | Run frequency |
| `last_run_at` | datetime | Last execution time |
| `next_run_at` | datetime | Next scheduled run |
| `config_json` | text | Job-specific configuration as JSON |

### `job_runs` Table

Execution history for debugging:

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `job_id` | UUID | Foreign key to `scheduled_jobs` |
| `bot_id` | string | Bot this run was for |
| `status` | enum | `pending`, `running`, `success`, `failed`, `skipped` |
| `started_at` | datetime | Execution start |
| `finished_at` | datetime | Execution end |
| `duration_ms` | int | Execution duration in milliseconds |
| `result_json` | text | Success result as JSON |
| `error_message` | text | Error details on failure |

## Profile Maintenance

The primary scheduled job. Uses an LLM to consolidate redundant or contradictory user profile attributes into clean summaries.

**How it works:**
1. Fetches all profile attributes for an entity
2. Sends them to an LLM with a consolidation prompt
3. LLM merges duplicates, resolves contradictions, removes noise
4. Consolidated summary is stored in `EntityProfile.summary`
5. Summary is used in system prompts for fast injection

**Model selection:** Prefers local models over OpenAI for privacy. Auto-selects an available local model if `PROFILE_MAINTENANCE_MODEL` is not set.

**Service:** `ProfileMaintenanceService` in `memory/profile_maintenance.py`

## Service Integration

The scheduler starts automatically with the LLM service when `SCHEDULER_ENABLED=true`:

1. On startup: creates scheduler tables, initializes default jobs, starts scheduler loop
2. During operation: checks for due jobs every `SCHEDULER_CHECK_INTERVAL_SECONDS`
3. On shutdown: gracefully stops the scheduler

Default jobs are created on first startup (e.g., profile maintenance for all bots at 60-minute intervals).

## File Locations

```
src/llm_bawt/
├── service/
│   └── scheduler.py           # JobScheduler, ScheduledJob, JobRun, JobStatus, JobType
├── memory/
│   └── profile_maintenance.py # ProfileMaintenanceService
└── utils/
    └── config.py              # Scheduler config settings
```

## Debugging

```bash
# Check job definitions
psql -d llm_bawt -c "SELECT * FROM scheduled_jobs;"

# Check recent job runs
psql -d llm_bawt -c "SELECT * FROM job_runs ORDER BY started_at DESC LIMIT 10;"

# Force immediate run (set next_run_at to past)
psql -d llm_bawt -c "UPDATE scheduled_jobs SET next_run_at = NOW() - INTERVAL '1 minute';"

# Watch service logs for job execution
./start.sh logs
```
