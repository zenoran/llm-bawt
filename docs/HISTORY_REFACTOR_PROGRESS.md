# History Refactor — Progress Tracker

*Tracking document for the message history and context assembly refactoring.*
*Source design: [CONTEXT_AND_MEMORY_REDESIGN.md](CONTEXT_AND_MEMORY_REDESIGN.md)*

**Last updated:** 2026-02-12

---

## Status Overview

| Area | Status | Notes |
|------|--------|-------|
| Token budgeting | Done | Protected turns, summary priority, graceful degradation |
| Cold-start memory | Done | Injects 2-3 memories when history <= 3 messages |
| Tool result timestamps | Done | `[Tool Results @ HH:MM]` format |
| `_should_skip_history()` removal | Done | No longer in codebase |
| `TOOLS_SKIP_HISTORY` removal | Done | No longer in codebase |
| `recalled_history` DB column | Done | Column + migration exist |
| Debug turn log (Track C) | Done | Full tool call tracing in CLI + service |
| File-based `db_id` generation | Done | `uuid.uuid4()` on `add_message()` |
| Token budget logging (detailed) | Done | Percentages + remaining budget in debug log |
| Bot prompts (all memory actions) | Done | All 4 memory bots mention search/store/delete |
| `history(action='recall')` wiring | Done | `history_manager` threaded through executor/loop/streaming |
| Unified memory tool output | Done | Uses `build_memory_context_string()` with categorized sections |
| Memory timestamps in search | Done | `created_at`/`last_accessed` threaded through pipeline |
| Dual assembly path consolidation | Not started | `base.py` and `pipeline.py` both assemble — architectural debt |
| Proactive summarization scheduler | Done | Track F implemented: non-destructive summaries + scheduler job |

---

## Design Doc Tracking Corrections

Items marked incomplete in `CONTEXT_AND_MEMORY_REDESIGN.md` that are **actually done** in the codebase:

| Item | Doc Status | Actual | Location |
|------|-----------|--------|----------|
| Tool result timestamps | `[ ]` (Sprint 3 Track D) | Done | `base.py:352`, `pipeline.py:555` |
| Remove `_should_skip_history()` | `[ ]` (Sprint 3 Track D) | Done | Grep confirms no matches |
| Remove `TOOLS_SKIP_HISTORY` | `[ ]` (Sprint 3 Track D) | Done | Grep confirms no matches |
| `recalled_history` DB column | `[ ]` (Sprint 3 Track D) | Done | `postgresql.py:73`, `migrations.py:168-208` |
| Remove upfront memory injection | `[ ]` (Sprint 2 Track B) | Done | Only cold-start (<=3 msgs) remains |
| Read-only memory tool for non-tool bots | `[ ]` (Sprint 2 Track B) | Done | `base.py:402-413` |

**Action:** Update design doc checkboxes to `[x]` for these items.

---

## Known Bugs

### 1. `since_minutes` parameter is actually seconds
**File:** `utils/history.py:72`, `memory/postgresql.py:1935`
**Severity:** Low (works correctly, naming is misleading)
**Details:** Parameter named `since_minutes` but used as seconds in cutoff calculation.
**Fix:** Rename to `since_seconds` or add `* 60` conversion.
**Status:** Open

### 2. `db_id` not populated in file-based history
**File:** `utils/history.py:93`
**Severity:** Medium (blocks future recall for non-PostgreSQL users)
**Details:** PostgreSQL backend populates `Message.db_id`, file-based backend does not.
**Fix:** Either generate UUIDs in file backend or document as PostgreSQL-only feature.
**Status:** Open

### 3. Token budget logging lacks detail
**File:** `utils/history.py:209-215`
**Severity:** Low (debugging harder without per-message breakdown)
**Details:** Logs total counts but not which messages were dropped or per-message token costs.
**Status:** Open

---

## Architectural Debt

### Dual Assembly Paths
Two independent code paths build the same context messages:

| Path | File | Method | Used by |
|------|------|--------|---------|
| Legacy | `core/base.py:383-451` | `_build_context_messages()` | CLI directly |
| Pipeline | `core/pipeline.py:445-494` | `_stage_message_assembly()` | Pipeline-based flow |

**Risk:** Updates to one path may not reach the other, causing divergent behavior.
**Target:** Consolidate to pipeline-only when pipeline is fully adopted.

### Cold-Start Logic Duplicated
Same cold-start memory injection exists in both paths:
- `pipeline.py:367-380`
- `base.py:415-426`

**Fix:** Consolidate when dual paths are resolved.

---

## Sprint Progress Detail

### Sprint 2 Track A: Two-Layer History (7/7 done)

| # | Deliverable | Status | Notes |
|---|------------|--------|-------|
| 1 | `estimate_tokens()` utility | Done | `memory/summarization.py`, used in `utils/history.py` |
| 2 | Token budget from model context window | Done | `base.py` computes from `effective_context_window - effective_max_tokens` |
| 3 | Session-block-to-summary swapping | Done | `history.py` fills summaries first, then droppable newest-first |
| 4 | Graceful degradation (drop oldest) | Done | Oldest summaries dropped first |
| 5 | Protected recent turns | Done | `MEMORY_PROTECTED_RECENT_TURNS` config, default 3 |
| 6 | `db_id` field consistently used | Done | File-based history now generates UUIDs via `uuid.uuid4()` |
| 7 | Detailed token budget logging | Done | Percentages, remaining budget, per-category breakdown |

### Sprint 2 Track B: Memory On-Demand (5/5 done)

| # | Deliverable | Status | Notes |
|---|------------|--------|-------|
| 1 | Remove upfront memory injection | Done | Only cold-start path remains |
| 2 | Read-only memory tool for all memory bots | Done | `base.py:402-413` |
| 3 | Cold-start detection | Done | Both `pipeline.py` and `base.py` |
| 4 | Auto-extraction for all memory bots | Done | Gated on `use_memory`, not `uses_tools` |
| 5 | Bot prompts mention memory tool | Done | All 4 bots mention search/store/delete actions |

### Sprint 3 Track D: History Cleanup & Recall (5/6 done)

| # | Deliverable | Status | Notes |
|---|------------|--------|-------|
| 1 | Remove `_should_skip_history()` | Done | Already removed |
| 2 | Remove `TOOLS_SKIP_HISTORY` config | Done | Already removed |
| 3 | Tool result timestamps | Done | `[Tool Results @ HH:MM]` |
| 4 | `history(action='recall')` tool | Done | `_history_recall()` exists, `history_manager` wired through chain |
| 5 | `recalled_history` DB flag | Done | Column + migration exist |
| 6 | Prompt guidance for summaries | Open | Bot prompts don't mention summary-awareness |

### Sprint 3 Track E: Unified Memory Output (3/3 done)

| # | Deliverable | Status | Notes |
|---|------------|--------|-------|
| 1 | Use `build_memory_context_string()` in tool | Done | Executor uses structured context builder with categorization |
| 2 | Timestamps on memory search results | Done | `created_at`/`last_accessed` threaded through full pipeline |
| 3 | Formatted text (not raw JSON) | Done | Structured sections: core/concerns/preferences/background |

### Sprint 3 Track F: Proactive Summarization (7/7 done)

| # | Deliverable | Status | Notes |
|---|------------|--------|-------|
| 1 | `HISTORY_SUMMARIZATION` JobType | Done | Added to scheduler `JobType` enum |
| 2 | TaskType + factory | Done | Added `TaskType.HISTORY_SUMMARIZATION` + `create_history_summarization_task()` |
| 3 | Non-destructive summarization | Done | Summaries no longer move/delete source messages |
| 4 | Size-based session prioritization | Done | Sessions sorted by estimated token savings |
| 5 | Modify `HistorySummarizer` to keep originals | Done | Source rows marked `summarized=TRUE`, retained in `*_messages` |
| 6 | Seed default job in `init_default_jobs()` | Done | Default recurring summarization job added |
| 7 | Skip `recalled_history` in summarization | Done | Recalled rows excluded from summarization candidate scan |

---

## Iteration Log

### 2026-02-09 — Initial audit (Claude)
- Traced full message history flow: storage → load → token-budgeted filter → assembly → LLM
- Identified 3 bugs, 2 architectural debt items
- Found 6 items marked incomplete in design doc that are actually done
- Created this tracking document

### 2026-02-09 — Implementation pass (Claude)
- **A1:** File-based history now generates `db_id` via `uuid.uuid4()` on `add_message()`
- **A2:** Token budget logging enhanced with percentages, remaining budget, per-category breakdown
- **B1:** All 4 memory bots (nova, monika, mira, proto) updated to mention search/store/delete actions
- **D1:** `history_manager` wired through ToolExecutor → ToolLoop → query_with_tools → stream_with_tools, with all callers updated (base.py, pipeline.py, service/core.py, service/api.py)
- **E1:** Memory search metadata (`intent`, `stakes`, `emotional_charge`, `created_at`, `last_accessed`) threaded through postgresql.py → storage.py → client.py → executor.py; executor switched from `format_memories_for_result()` to `build_memory_context_string()`
- Sprint 2 Tracks A+B now complete; Track E complete; Track D at 5/6 (prompt guidance remaining)
- Updated both design doc and progress tracker

### 2026-02-12 — Track F implementation (Codex)
- Implemented non-destructive summarization in `memory/summarization.py` (source messages retained, marked `summarized=TRUE`)
- Added duplicate-avoidance guard for already summarized sessions and size-based prioritization by estimated token savings
- Added scheduler/task plumbing for proactive summarization:
  - `JobType.HISTORY_SUMMARIZATION`
  - `TaskType.HISTORY_SUMMARIZATION`
  - `create_history_summarization_task()`
  - default seeded job in `init_default_jobs()`
- Added MCP-accessible summary-recall helpers:
  - `get_messages_for_summary`
  - `mark_messages_recalled`
- Added tests for scheduler/task wiring and summarization helper behavior
