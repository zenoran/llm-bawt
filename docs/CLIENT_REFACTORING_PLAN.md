# Refactoring Plan: Strip Direct DB Access from CLI

## Context

llm-bawt started as a CLI-only tool that connected directly to PostgreSQL for memory, profiles, and settings. When the service layer was added, the CLI kept all its direct-DB paths as fallbacks — creating a complex dual-mode architecture with ~15 conditional branches ("if service available, use API; else fall back to direct DB"). The new direction is **CLI always proxies through the service API**. This cleanup removes ~640 net lines and eliminates an entire class of "is the DB reachable from the client?" bugs.

---

## Phase 1: Expand ServiceClient + Hard Gate at CLI Entry

**Goal:** Fill gaps in `ServiceClient` and add a single, clear "service required" gate.

### 1a. ServiceClient gaps (`src/llm_bawt/service/client.py`)
- Add `trigger_job(job_type: str)` method — currently `app.py:840` notes "No service endpoint for trigger yet". Add corresponding `POST /v1/jobs/{job_type}/trigger` route on the service side (in `service/routes/jobs.py`).
- Verify all other CLI-used operations are covered (they mostly are: history, memory, profiles, settings, bots, status, jobs listing).

### 1b. CLI entry gate (`src/llm_bawt/cli/app.py`)
Replace the complex decision tree (~lines 1890-1960) with a simple gate:
```
if not args.local:
    service_client = get_service_client(config)
    if not service_client or not service_client.is_available(force_check=True):
        print("Service not available. Start with: llm-service")
        print("Or use --local for direct API calls without memory/tools.")
        sys.exit(1)
```

**Files:** `service/client.py`, `service/routes/jobs.py` (minor), `cli/app.py`

---

## Phase 2: Remove Direct-DB Fallbacks from CLI Commands

**Goal:** Every CLI subcommand that has "service path + direct-DB fallback" gets its fallback deleted. ~500 lines removed.

### Functions to strip in `src/llm_bawt/cli/app.py`:

| Function | Lines (approx) | What to remove |
|----------|----------------|----------------|
| `show_job_status()` | 689-835 | Direct SQLModel/engine queries for ScheduledJob/JobRun |
| `trigger_job()` | 837-894 | Direct DB trigger, replace with ServiceClient call |
| `show_user_profile()` | 897-985 | Direct ProfileManager fallback |
| `show_users()` | 988-1059 | Direct ProfileManager fallback |
| `show_runtime_settings()` | 1081-1128 | Direct RuntimeSettingsStore fallback |
| `set_runtime_setting()` | 1131-1155 | Direct RuntimeSettingsStore fallback |
| `delete_runtime_setting()` | 1158-1187 | Direct RuntimeSettingsStore fallback |
| `bootstrap_runtime_settings()` | 1190-1305 | Direct RuntimeSettingsStore fallback |
| `migrate_bots_to_db()` | 1308-1428 | Direct BotProfileStore/RuntimeSettingsStore fallback |
| `run_user_profile_setup()` | 1431-1520+ | Direct ProfileManager fallback |

### Pattern for each:
**Before:** check service → try service → fall back to direct DB
**After:** check service → use service → if unavailable, print error and return

### Also: `src/llm_bawt/cli/bot_editor.py`
- Remove `_use_service()` helper and all conditional DB branches (~lines 28, 142-144, 266-268, 316-317, 372-373)
- Always use ServiceClient methods

**Files:** `cli/app.py`, `cli/bot_editor.py`

---

## Phase 3: Clean Up BaseLLMBawt, Bot Config, and Status

**Goal:** Remove DB initialization from client-reachable code paths.

### 3a. BaseLLMBawt (`src/llm_bawt/core/base.py`)
- Move `_init_memory()` DB logic (has_database_credentials check, MemoryClient creation, ProfileManager creation) into `ServiceLLMBawt._init_memory()` override in `service/core.py`
- Base class `_init_memory()` becomes a no-op (just sets `self._db_available = False`)
- Remove `has_database_credentials`, `get_memory_client` imports from `base.py`

### 3b. Bot config loading (`src/llm_bawt/bots.py`)
- Remove the direct-DB section from `_load_bots_config()` that imports `BotProfileStore`/`RuntimeSettingsStore` when `has_database_credentials()` is true
- YAML-only loading stays for basic bot metadata (slug, name, default_model, capabilities)
- Service process enriches bots with DB data during its own startup (move DB loading logic to service-side initialization)
- CLI gets full bot info from service API (`/v1/bots`) when needed

### 3c. Status collection (`src/llm_bawt/core/status.py`)
- Remove local-DB fallback in `collect_system_status()`
- Remove `_collect_memory_info()` direct PostgreSQLMemoryBackend usage
- When service unavailable, return status with `service.available=False` and empty sections

**Files:** `core/base.py`, `service/core.py`, `bots.py`, `core/status.py`

---

## Phase 4: Dead Code Cleanup and Test Updates

**Goal:** Remove orphaned imports, dead helpers, and update tests.

### Imports to remove from `cli/app.py`:
- `has_database_credentials`
- `ProfileManager`, `EntityType`, `AttributeCategory`
- `_use_service()` helper function

### Simplify `run_app()` in `cli/app.py`:
- Remove `use_service` variable and all branching
- `do_query()` no longer creates `LLMBawt` on service failure — error instead
- Remove `need_local_llm_bawt` logic
- Remove local history fallback that creates `StubClient` + `MemoryClient`

### `model_manager.py`:
- Deprecate or remove `is_service_mode_enabled()` — service mode is always on
- Update callers in `status.py` and `bot_editor.py`

### Tests:
- `tests/test_integration.py` — Mock ServiceClient instead of DB connections
- `tests/test_scheduler.py` — Update trigger_job tests to use service endpoint
- `tests/test_tool_formats.py` — Likely unaffected (tools layer is already clean)

**Files:** `cli/app.py`, `cli/bot_editor.py`, `model_manager.py`, `utils/config.py`, `tests/`

---

## Scope Summary

| Phase | Net Lines Removed | Key Risk |
|-------|-------------------|----------|
| 1 | ~0 (adds ~70, removes ~30) | New service endpoint for job trigger |
| 2 | ~450 | Many functions touched; per-function testing needed |
| 3 | ~130 | Bot loading split is architecturally significant |
| 4 | ~100 | Test coverage must be maintained |
| **Total** | **~640 net reduction** | |

## What Stays Unchanged
- **Service-side code** — all direct DB access in `service/`, `memory/`, `profiles.py`, `runtime_settings.py` remains
- **Tools layer** — already uses pure dependency injection, no changes needed
- **Pipeline** — only runs server-side, no changes needed
- **Integrations** (HA, NewsAPI, Nextcloud) — no DB access, unaffected
- **`--local` flag** — preserved as "no memory/profiles/tools, just direct OpenAI API call"

## Verification
- Run `llm-bawt --local "hello"` — should work without service (direct OpenAI)
- Run `llm-bawt "hello"` without service running — should fail with clear error message
- Run `llm-bawt "hello"` with service running — should work via service proxy
- Run all CLI subcommands (--jobs, --profile, --settings, etc.) — should work via service
- Run `pytest tests/` — all tests pass
- Grep for `has_database_credentials` in `cli/` — should find zero results
- Grep for `create_engine` in `cli/` — should find zero results
