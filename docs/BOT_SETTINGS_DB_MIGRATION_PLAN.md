# Bot Settings + DB Configuration Migration Plan

## Claude Handoff (2026-02-13)

### What Is Implemented
- Bot settings template overlay is live in bot loader:
  - top-level `bot_settings_template`
  - per-bot `bots.<slug>.settings`
  - effective bot settings = deep-merge(template, bot override)
- DB runtime settings layer exists:
  - `runtime_settings` table via `src/llm_bawt/runtime_settings.py`
  - scope model: `global/*` and `bot/<slug>`
  - resolver precedence:
    1. request override
    2. bot DB
    3. global DB
    4. bot YAML settings
    5. config fallback
- Request/runtime wiring (partial but active):
  - history/context now resolves per-bot keys:
    - `history_duration_seconds`
    - `history_bridge_messages`
    - `summarization_compact_context`
    - `summarization_max_in_context`
    - `memory_protected_recent_turns`
    - `history_reload_ttl_seconds`
    - `max_context_tokens`
    - `max_output_tokens` (used in context budget math)
    - `memory_n_results` (cold-start priming)
    - `memory_min_relevance`
  - generation parameters now resolve per-bot:
    - `temperature`
    - `top_p`
    - `max_output_tokens` (used for generation, not just context budget)
  - summarization tunables now resolve per-bot:
    - `summarization_session_gap_seconds`
    - `summarization_min_messages`
- Service settings API exists:
  - `GET /v1/settings`
  - `PUT /v1/settings`
  - `DELETE /v1/settings`
  - `POST /v1/settings/batch`
- CLI controls exist:
  - `--settings-list`
  - `--settings-set KEY VALUE`
  - `--settings-delete KEY`
  - `--settings-scope bot|global`
  - `--settings-bootstrap`
  - `--settings-bootstrap-overwrite`
  - `--bot-edit <slug>` (temp YAML edit + validation + diff preview + confirm)

### Important Clarification
- `--settings-list` reads DB rows only.
- If DB is empty, it will show no settings until seeded.
- Use:
  - `uv run llm --settings-bootstrap`
  - then `uv run llm --settings-list --settings-scope bot --bot mira`

### What Is NOT Migrated Yet
- Core bot profile fields are still YAML-driven:
  - `name`, `description`, `system_prompt`
  - capability flags (`requires_memory`, `uses_tools`, etc.)
  - `nextcloud` block
- Only runtime tunables were moved toward DB.

### Files Touched In This Migration
- `src/llm_bawt/bots.py`
- `src/llm_bawt/bots.yaml`
- `src/llm_bawt/runtime_settings.py` (new)
- `src/llm_bawt/core/base.py`
- `src/llm_bawt/utils/history.py`
- `src/llm_bawt/service/core.py`
- `src/llm_bawt/service/routes/settings.py` (new)
- `src/llm_bawt/service/routes/__init__.py`
- `src/llm_bawt/service/schemas.py`
- `src/llm_bawt/service/scheduler.py`
- `src/llm_bawt/cli/app.py`
- `docs/BOT_SETTINGS_DB_MIGRATION_PLAN.md`

### Highest-Priority Next Tasks
1. Finish resolver wiring for remaining knobs:
   - generation (`temperature`, `top_p`, output limits in all paths)
   - search knobs
   - summarization/maintenance model selection + chunk/token knobs
2. Add API key/type allowlist validation for runtime settings at API boundary.
3. Add migration command from existing env + bots YAML into DB with dry-run report.
4. Decide/implement DB migration for bot identity fields (`system_prompt`, traits, flags):
   - proposed transition precedence: DB bot profile -> YAML fallback
5. Add tests:
   - resolver precedence
   - per-bot divergence in one process (`mira` vs `nova`)
   - scheduler interval resolution from runtime settings

### Quick Smoke Commands
```bash
uv run llm --settings-bootstrap
uv run llm --settings-list --settings-scope global
uv run llm --settings-list --settings-scope bot --bot mira
uv run llm --settings-set max_context_tokens 20000 --settings-scope bot --bot mira
uv run llm --settings-delete max_context_tokens --settings-scope bot --bot mira
uv run llm --bot-edit mira
```

### Known Operational Gotchas
- Running `llm` binary may use stale installed code; prefer `uv run llm` while iterating.
- Service route changes require service restart/reload to expose new endpoints.
- DB creds must be present (`LLM_BAWT_POSTGRES_PASSWORD`) or runtime settings store is unavailable.

## Progress Checkpoint (2026-02-13)

### Completed
- Added bot settings template overlay in bot loader:
  - top-level `bot_settings_template`
  - per-bot `bots.<slug>.settings`
  - effective settings hydration via deep-merge(template, bot override)
- Added runtime settings DB primitives in `src/llm_bawt/runtime_settings.py`:
  - `runtime_settings` table (`scope_type`, `scope_id`, `key`, `value_json`, `updated_at`)
  - unique index (`scope_type`, `scope_id`, `key`)
  - DB store + resolver with precedence:
    - request override -> bot DB -> global DB -> bot YAML settings -> config fallback
- Wired resolver into live request path (initial slice):
  - `BaseLLMBawt` now creates per-bot resolver
  - `HistoryManager` now resolves per-bot runtime settings for:
    - `history_duration_seconds`
    - `history_bridge_messages`
    - `summarization_compact_context`
    - `memory_protected_recent_turns`
    - `summarization_max_in_context`
  - `BaseLLMBawt` now resolves:
    - `max_context_tokens`
    - `max_output_tokens` (for context budget calculation)
    - `memory_n_results` (cold-start memory priming)
    - `memory_min_relevance` (cold-start retrieval)
  - `ServiceLLMBawt` now resolves:
    - `history_reload_ttl_seconds`
- Syntax validation passed on touched modules via `py_compile`.
- Added service runtime settings endpoints:
  - `GET /v1/settings` (list scope settings)
  - `PUT /v1/settings` (upsert single key)
  - `DELETE /v1/settings` (delete single key)
  - `POST /v1/settings/batch` (batch upsert)
- Added CLI runtime settings controls in `llm`:
  - `--settings-list`
  - `--settings-set KEY VALUE`
  - `--settings-delete KEY`
  - `--settings-scope bot|global`
  - `--settings-bootstrap` (seed DB runtime settings)
  - `--settings-bootstrap-overwrite` (replace existing keys during seed)
- Added first-pass bot editor command:
  - `--bot-edit <slug>`
  - Opens one-bot temp YAML and writes merged override to user `bots.yaml`.
  - Includes validation + diff preview + confirmation before write.
- Scheduler default-job intervals now resolve through runtime settings:
  - `profile_maintenance_interval_minutes` (global)
  - `history_summarization_interval_minutes` (per bot/global fallback)
- Wired generation parameters through resolver (per-bot temperature/top_p/max_tokens):
  - `BaseLLMBawt._get_generation_kwargs()` resolves `temperature`, `top_p`, `max_output_tokens`
  - CLI query path passes resolved kwargs to all `client.query()` calls
  - Tool loop (`ToolLoop`, `query_with_tools`) threads `generation_kwargs` to client calls
  - `OpenAIClient.stream_raw()` and `stream_with_tools()` accept kwargs overrides
  - Service streaming path resolves and passes gen kwargs to all stream calls
- Wired summarization settings through resolver in `HistorySummarizer`:
  - `summarization_session_gap_seconds` (per bot)
  - `summarization_min_messages` (per bot)
  - `history_duration_seconds` (per bot, for eligibility window)
  - Background service creates per-bot resolver for summarization jobs

### Next Immediate Tasks
- Add optional PATCH semantics and stronger key/type validation at API boundary.
- Wire remaining runtime knobs: `memory_dedup_similarity`, `memory_max_token_percent`.
- Wire model selection through resolver: `summarization_model`, `profile_maintenance_model`, `extraction_model`.
- Add migration/bootstrap command from env + bots.yaml into DB settings.

## Goal
Move subjective runtime behavior (context budgets, history/summarization behavior, memory retrieval limits, generation knobs) from mostly-global env settings to a layered configuration model with per-bot overrides.

Primary user intent:
- `mira` should keep far richer context/history.
- `nova` should remain utility-focused with lean context.

## Desired Resolution Order
For every tunable setting used at runtime:
1. Request-time override (API/CLI parameter)
2. Bot override (`bot.settings.<key>`)
3. Global template/default (DB-backed)
4. Env fallback (infra-safe default)

## Current State (as of this checkpoint)
- Bot YAML supports personality/capabilities plus `default_model`.
- Most context/memory/token settings are still global (`Config`).
- Background jobs are persisted in DB, but runtime tuning is still env-heavy.
- Added now: YAML-level bot settings template hydration pattern (template + per-bot overlay) in loader.

## Scope

### In Scope
- Global + per-bot settings stored in DB.
- Resolver used by request pipeline and maintenance jobs.
- CLI/API support to view/edit bot settings.
- Single-bot YAML edit UX in CLI.

### Out of Scope (for this phase)
- Secret management migration (API keys, DB passwords, auth tokens).
- Replacing infrastructure env vars (`POSTGRES_*`, service host/port, etc.).

---

## Phase 0: Canonical Settings Inventory

### Tasks
- [ ] Define canonical `SettingKey` registry for runtime tunables.
- [ ] Map existing env keys to canonical bot/global keys.
- [ ] Mark each key as one of: `infra`, `secret`, `runtime_tunable`.
- [ ] Add explicit docs table listing source of truth and fallback.

### Acceptance
- Every runtime-relevant setting in `Config` has a canonical key and ownership category.

---

## Phase 1: DB Schema for Settings

### Tasks
- [ ] Add SQLModel table `app_settings`:
  - `scope_type` (`global` | `bot`)
  - `scope_id` (`*` for global or bot slug)
  - `key`
  - `value_json`
  - `updated_at`
  - unique index on (`scope_type`, `scope_id`, `key`)
- [ ] Add data access layer for get/set/list/delete settings.
- [ ] Add migration/bootstrap step to create table.

### Acceptance
- Settings can be persisted and fetched for both global and bot scopes.

---

## Phase 2: Runtime Resolver

### Tasks
- [ ] Implement `SettingsResolver` service:
  - batched read for bot + global scope
  - in-memory cache with TTL
  - typed coercion + validation
- [ ] Add single API to resolve each setting with precedence:
  request override -> bot -> global -> env fallback.
- [ ] Instrument debug logging to show resolved source per key when verbose/debug is enabled.

### Acceptance
- Pipeline can ask for any tunable setting and receive stable typed values + source metadata.

---

## Phase 3: Wire Resolver Into Request Pipeline

### Tasks
- [ ] Replace direct reads of context/history/memory limits from global config in hot paths:
  - history assembly
  - summary inclusion limits
  - memory retrieval token budget and result limits
  - generation defaults (`max_context_tokens`, `max_output_tokens`, `temperature`, `top_p`)
- [ ] Ensure no shared mutable global config mutation between bots.
- [ ] Add tests showing `mira` and `nova` resolve distinct values in same process.

### Acceptance
- Same request path behaves differently per bot according to bot settings.

---

## Phase 4: Wire Resolver Into Maintenance Jobs

### Tasks
- [ ] Use resolver for summarization/profile job model + limits.
- [ ] Allow scheduled job config to optionally override resolved settings.
- [ ] Ensure fallback chain remains deterministic.

### Acceptance
- Maintenance behavior is consistent with per-bot policy and no longer hard-pinned to global env for tunables.

---

## Phase 5: CLI/API Editing UX

### Tasks
- [ ] Add API endpoints:
  - `GET /v1/settings?scope=global|bot&scope_id=...`
  - `PATCH /v1/settings/...`
  - `DELETE /v1/settings/...`
- [ ] Add CLI commands:
  - `llm settings list --global`
  - `llm settings list --bot mira`
  - `llm settings set --bot mira key value`
- [ ] Add editor workflow:
  - `llm bot edit <slug>` opens temp YAML for one bot
  - includes all supported settings (hydrated with template/defaults)
  - validates and persists on save/close

### Acceptance
- User can fully inspect and edit one bot config without hand-editing monolithic files.

---

## Phase 6: Transition + Backward Compatibility

### Tasks
- [ ] Add bootstrap importer from env + `bots.yaml` to DB settings.
- [ ] Keep env fallback to avoid breaking existing installs.
- [ ] Add clear startup log showing active source of config (DB/env).
- [ ] Add one-shot command:
  - `llm settings migrate --from-env --from-bots-yaml`

### Acceptance
- Existing deployments keep working; DB settings can be adopted incrementally.

---

## Proposed Canonical Per-Bot Tunables (Initial Set)

### Context + History
- `max_context_tokens`
- `max_output_tokens`
- `history_duration_seconds`
- `history_bridge_messages`

### Summarization
- `summarization_session_gap_seconds`
- `summarization_min_messages`
- `summarization_max_in_context`
- `summarization_compact_context`

### Memory Retrieval
- `memory_n_results`
- `memory_protected_recent_turns`
- `memory_min_relevance`
- `memory_max_token_percent`
- `memory_dedup_similarity`

### Generation
- `temperature`
- `top_p`

### Maintenance
- `maintenance_model`
- `profile_maintenance_model`

---

## YAML Overlay Pattern (Transitional)

Top-level template key:
- `bot_settings_template`

Per bot override key:
- `bots.<slug>.settings`

Behavior:
- Effective settings = deep-merge(`bot_settings_template`, `bots.<slug>.settings`)
- Unknown keys preserved for forward compatibility.

---

## Testing Plan

### Unit
- [ ] resolver precedence and typing
- [ ] template + override merge behavior
- [ ] invalid value rejection and fallback

### Integration
- [ ] same process, two bots, different context budgets
- [ ] scheduled summarization uses per-bot settings
- [ ] CLI editor round-trip writes valid config

### Regression
- [ ] env-only deployment still works with no DB settings rows
- [ ] old `bots.yaml` without `settings` still works

---

## Rollout Sequence
1. Phase 0-2 (inventory + schema + resolver)
2. Phase 3 (request pipeline)
3. Phase 4 (jobs)
4. Phase 5 (CLI/API editing)
5. Phase 6 (migration tooling + docs)

---

## Notes for Continuation
- Keep infra/secrets in env.
- Keep runtime knobs in DB with per-bot overlay.
- Do not mutate shared global config per request.
- Prefer resolver injection over ad-hoc `getattr(config, ...)` in runtime paths.
