# Context Window & Memory Redesign Plan

## Problem Statement

The current system has two interrelated problems:

1. **Context window management is non-existent.** There's no token counting, no per-model context limits, and no budget enforcement. `MEMORY_MAX_TOKEN_PERCENT` (30%) exists in config but nothing reads it. A fast 30-minute conversation can silently overflow the context window.

2. **Memory injection is speculative and wastes context.** Non-tool bots get 10 pre-fetched memories stuffed into the system prompt on every query, whether relevant or not. Tool bots already prove that on-demand memory search works better.

The fix: per-model context/token configuration, proper token budgeting, and conversation-first context allocation with memory on-demand.

---

## Phase 1: Per-Model Context & Token Configuration

### Current State

All models share global config values:
- `LLAMA_CPP_N_CTX = 32768` — applies to every GGUF model
- `MAX_TOKENS = 4096` — output cap for every model
- `MAX_CONTEXT_TOKENS = 0` — unused (falls back to N_CTX - MAX_TOKENS)

This means a small 3B model and a 32B model get the same 32K context, even though the 3B model could handle 128K and the 32B model might OOM at 64K depending on VRAM.

### Proposed: models.yaml Per-Model Overrides

Add optional `context_window`, `max_tokens`, and `n_gpu_layers` fields to model definitions:

```yaml
models:
  qwen:
    type: gguf
    repo_id: bartowski/Qwen2.5-32B-Instruct-GGUF
    filename: Qwen2.5-32B-Instruct-Q4_K_M.gguf
    # --- NEW: per-model context/generation settings ---
    context_window: 32768       # n_ctx for GGUF, context hint for OpenAI
    max_tokens: 4096            # max output tokens for this model
    n_gpu_layers: -1            # GPU offload override

  dolphin:
    type: gguf
    repo_id: dphn/Dolphin3.0-Llama3.1-8B-GGUF
    filename: Dolphin3.0-Llama3.1-8B-Q8_0.gguf
    context_window: 65536       # 8B model, can afford larger context
    max_tokens: 4096

  gpt-5.2-chat-latest:
    type: openai
    model_id: gpt-5.2-chat-latest
    context_window: 128000      # API model's known limit
    max_tokens: 16384           # Can generate longer responses
```

### Resolution Order

When the system needs `context_window` or `max_tokens` for the active model:

1. **Model definition** (`models.yaml` per-model field) — highest priority
2. **Global config** (`LLAMA_CPP_N_CTX` / `MAX_TOKENS`) — fallback
3. **Hardcoded default** (32768 / 4096) — last resort

### Implementation

**Config changes (`config.py`):**
- Add `get_model_context_window(model_alias) -> int` method
- Add `get_model_max_tokens(model_alias) -> int` method
- These read from `defined_models[alias]` first, fall back to global config

**Client changes (`llama_cpp_client.py`):**
- Read `context_window` from model definition instead of global `LLAMA_CPP_N_CTX`
- Read `max_tokens` from model definition instead of global `MAX_TOKENS`
- Read `n_gpu_layers` from model definition instead of global `LLAMA_CPP_N_GPU_LAYERS`

**Client changes (`openai_client.py`):**
- Read `max_tokens` from model definition when available
- `context_window` used for budget calculations (not sent to API)

### Dynamic VRAM Detection & Context Window Sizing

The system runs on multiple machines with different GPUs (e.g., 32GB 5090, 16GB 5080). Context window size must be determined dynamically at app init based on available VRAM — not hardcoded per-model.

**At app startup:**
1. Detect available VRAM via `torch.cuda.get_device_properties()` or `nvidia-smi` parsing
2. For the active GGUF model, estimate weight size from the file (file size ≈ weight VRAM)
3. Calculate maximum safe context window:
   ```
   available_vram = detected_gpu_vram
   weight_vram = model_file_size  # good approximation for GGUF
   vram_for_kv = available_vram - weight_vram - safety_margin (e.g., 1-2GB)
   max_context = estimate_context_from_kv_budget(vram_for_kv, model_params)
   ```
4. Cap at the model's native maximum (e.g., Qwen2.5 = 128K) if VRAM allows more

**Reference: KV cache sizing**

KV cache VRAM scales linearly with context length. Rough per-token KV cost by model size:

| Model Params | KV bytes/token (FP16) | 32K ctx | 64K ctx | 128K ctx |
|-------------|----------------------|---------|---------|----------|
| 8B | ~0.5 KB | ~16MB | ~32MB | ~64MB |
| 24-27B | ~1.5 KB | ~1.5GB | ~3GB | ~6GB |
| 32B | ~2 KB | ~2GB | ~4GB | ~8GB |

**Example auto-sizing:**

| Machine | GPU VRAM | Model | Weights | Available for KV | Auto Context |
|---------|----------|-------|---------|-----------------|--------------|
| Desktop | 32GB (5090) | Qwen 32B Q4_K_M | ~18GB | ~12GB | 128K (capped at native max) |
| Desktop | 32GB (5090) | Dolphin 8B Q8_0 | ~8GB | ~22GB | 128K (capped at native max) |
| Laptop | 16GB (5080) | Qwen 32B Q4_K_M | ~18GB | ❌ won't fit | Error / partial offload |
| Laptop | 16GB (5080) | Dolphin 8B Q8_0 | ~8GB | ~6GB | ~64K |
| Laptop | 16GB (5080) | Gemma 27B Q4_K_M | ~15GB | won't fit well | 8K or partial offload |

**`models.yaml` override:** The per-model `context_window` field becomes an **optional cap**, not a requirement. If set, it limits the auto-detected value (useful for stability or testing). If unset, the system auto-sizes based on VRAM.

```yaml
models:
  qwen:
    type: gguf
    repo_id: bartowski/Qwen2.5-32B-Instruct-GGUF
    filename: Qwen2.5-32B-Instruct-Q4_K_M.gguf
    # context_window: 65536   # optional cap — omit to auto-size from VRAM
    max_tokens: 4096
    native_context_limit: 131072  # model's trained maximum (optional, for capping)
```

**Resolution order for context_window:**
1. Auto-detect from VRAM (primary)
2. Cap at `context_window` from models.yaml if set (user override / safety)
3. Cap at `native_context_limit` if set (model architecture limit)
4. Fall back to global `LLAMA_CPP_N_CTX` if VRAM detection fails
5. Hardcoded default (32768) as last resort

**For OpenAI-type models:** VRAM detection doesn't apply. Use `context_window` from models.yaml directly (API models have known limits). No auto-sizing needed.

---

## Phase 2: Two-Layer History & Model-Aware Context Assembly

### Core Principle: Summarization is Compression, Not Deletion

The current system treats summarization as destructive: originals get moved to `forgotten_messages`, replaced with a summary row. This means an 8K model triggers summarization that permanently loses detail a 128K model could've used.

**New principle:** All raw messages stay in the history buffer forever (in-memory + DB). Summaries exist alongside them as pre-computed compressed alternatives. Context assembly decides per-turn which representation to use based on the active model's context window.

### Two-Layer History Architecture

**Layer 1: Full History Buffer**
- ALL messages (user, assistant, system/tool results) persist in-memory and in the DB
- Never deleted by summarization
- `Message` objects get a `db_id` field referencing their PK in `{bot_id}_messages`
- Loaded from DB once at session startup, maintained in-memory thereafter
- Per-bot isolation unchanged (`nova_messages`, `mira_messages`, etc.)

**Layer 2: Summary Layer**
- Summaries are pre-computed compressed representations of session blocks
- Created by scheduler proactively (not triggered by context pressure)
- Stored with references to their source message IDs
- Used by context assembly as a swap-in replacement when raw messages don't fit

### Model-Aware Context Assembly (Per-Turn)

On each turn, when building context messages:

```
available = model.context_window - output_reserve - system_prompt_tokens

history_tokens = estimate_tokens(all_raw_history)
```

**Case 1: Everything fits** (e.g., 128K model, moderate history)
→ Send all raw messages. Summaries unused.

**Case 2: Doesn't fit, summaries available** (e.g., 8K model, long history)
→ Working from oldest to newest:
  - Swap oldest session blocks for their summary equivalents
  - Keep swapping until history fits the budget
  - Recent turns always stay raw (protected by `MEMORY_PROTECTED_RECENT_TURNS`)

**Case 3: Still doesn't fit even with summaries** (e.g., 8K model, very long history)
→ Natural progression: drop oldest summaries entirely from context
  - Priority: newest raw messages > recent summaries > older summaries
  - The dropped content still exists in-memory — just not sent to the LLM this turn
  - This is the correct behavior: conversation history gracefully degrades from detailed → summarized → absent as it ages, proportional to the model's capacity

**Assembly priority (highest to lowest):**
1. System prompt (always)
2. Last N raw conversation turns (protected, always included)
3. Recent raw messages (fill budget)
4. Older session summaries (substitute for raw when budget is tight)
5. Oldest summaries (dropped first when even summaries don't fit)

### Recall: Expanding a Summary Back to Raw

Because raw messages always stay in-memory, "recall" is trivial. When the bot is working from a summary and the user pushes for details:

- The bot calls `history(action='recall', summary_id='...')`
- Context assembly is told: "for this session block, use raw messages instead of the summary on the next turn"
- The raw messages get inserted into context with current timestamps and a `recalled_history` flag
- The `recalled_history` flag is stored in the DB so future maintenance can identify and filter these if they become noise
- No DB lookup needed — the messages are already in the in-memory buffer

**Triggering recall:** This is an explicit tool call, not automatic. The bot should be prompted to mention when it's working from a summary (e.g., "I have a summary of that conversation but not the details"). The user then explicitly asks to dig deeper. Same pattern as any other tool — the LLM decides when to use it, no keyword detection.

### History Cleanup (Broken Mechanisms to Remove)

1. **Remove `_should_skip_history()` entirely** — the trigger word list (`search`, `now`, `time`, etc.) is too broad and catches normal conversation. History should always be included.

2. **Remove `TOOLS_SKIP_HISTORY` config** — dead config that enables the broken behavior.

3. **Add timestamps to tool result history entries** — change `[Tool Results]` to `[Tool Results @ {time}]` when saving to history. The LLM can judge freshness on its own.

### Token Estimator

```python
def estimate_tokens(text: str) -> int:
    """Rough token estimate. ~4 chars per token for English."""
    return len(text) // 4
```

Simple and sufficient. Tiktoken would add a dependency for marginal accuracy improvement — chars/4 is within 10-15% for English text.

---

## Phase 3: Conversation-First Memory (Remove Upfront Injection)

### Current State: Two Paths

- **Path A (non-tool bots like Spark):** Memories pre-fetched and injected into system prompt
- **Path B (tool bots like Nova, Mira):** LLM calls `memory(action="search")` on demand

### Proposed: One Path — Always On-Demand

**Delete upfront memory injection.** All bots with `requires_memory: true` get the memory search tool. The two-path system collapses:

| Before | After |
|--------|-------|
| Non-tool bots: memories stuffed in system prompt | All bots: memory available via tool |
| Tool bots: LLM searches when it wants | Same |
| Non-tool bots: auto-extraction in post-process | All bots: auto-extraction in post-process |
| Tool bots: extraction only via explicit `memory(action="store")` | Same + auto-extraction also runs |

**What stays in the system prompt (always-on):**
- Bot personality (from `bots.yaml` system_prompt) — essential
- User profile summary (`## About the User`) — small, high-signal, always relevant
- Bot traits (`## Your Developed Traits`) — small, personality continuity
- Date/time — tiny

**What moves to on-demand:**
- Long-term memory search results — only when the LLM decides it needs them

### Pipeline Changes

In `_stage_memory_retrieval` (pipeline.py):
- **Before:** If non-tool bot, search memories and inject into prompt
- **After:** Skip entirely. Memory is accessed via tool calls during execution.

In `_stage_context_build` (pipeline.py):
- **Before:** Tools section is mutually exclusive with memory injection
- **After:** If `requires_memory`, always add a read-only `memory(action="search")` tool, even for bots that don't have `uses_tools: true`

### Cold-Start Heuristic

One exception to "never pre-inject": **cold starts.** When conversation history is empty or very short (< 3 messages), the bot has no conversational context. Pre-loading 2-3 high-importance core facts bridges the gap.

**Implementation:**
```python
# In _stage_memory_retrieval (repurposed)
if len(history_messages) < COLD_START_THRESHOLD:  # e.g., 3
    core_memories = memory_client.search(
        prompt, n_results=3, min_relevance=0.3
    )
    # Inject as small "reminder" section, not full memory block
```

This is small (~100-200 tokens) and only fires on session boundaries.

---

## Phase 4: Unify Memory Tool Output

### Current State

When tool bots call `memory(action="search")`, they get raw flat results:
```json
[{"id": "...", "content": "...", "relevance": 0.8, "importance": 0.7, "tags": [...]}]
```

No categorization, no formatting, no timestamps.

### Proposed

Reuse `context_builder.py` for tool results too:
- Categorize into core/concerns/preferences/background
- Add relative timestamps ("3 days ago", "2 months ago")
- Return structured text, not raw JSON

This gives tool bots the same quality of memory presentation that Path A non-tool bots used to get.

---

## Phase 6: Scheduler-Based Proactive Summarization

### Design: Summarize Proactively, Use Selectively

Summarization runs on a schedule independent of any model's context pressure. The goal is to have summaries **pre-computed and ready** so that context assembly can swap them in instantly when a smaller model needs them.

### What Triggers Summarization

**Scheduler job** (new `JobType.HISTORY_SUMMARIZATION`):
- Runs every N minutes (configurable, e.g., 15-30 min)
- Finds session blocks older than `SUMMARIZATION_SESSION_GAP_SECONDS` (default 1 hour)
- With at least `SUMMARIZATION_MIN_MESSAGES` messages (default 4)
- That don't already have a summary
- Generates summary via LLM (or heuristic fallback)
- Stores summary row linked to source message IDs

**Key difference from current system:** Summarization does NOT delete or move the original messages. It creates a summary row that references them. Both coexist.

### Size-Based Session Prioritization

When the scheduler runs, it prioritizes which sessions to summarize based on **how much context space they would save** if a model needed the compressed version:

```
savings = estimate_tokens(raw_session) - estimate_tokens(expected_summary)
```

Sessions that would save the most tokens get summarized first. Time is still used for session boundary detection, but the order of summarization work is driven by size impact.

### Per-Bot Isolation

Summarization runs per-bot (the scheduler already supports `bot_id` on jobs). Each bot's history is independent — summarizing Nova's history doesn't affect Mira's.

### Recalled History Handling

When the bot recalls detailed history behind a summary:
- Raw messages are re-inserted into context with current timestamps
- Each re-inserted message gets a `recalled_history = true` flag in the DB
- Future summarization passes skip `recalled_history` messages (they're duplicates of existing records)
- If recalled history accumulates and becomes noise, a maintenance job can clean it up

### Adding the Scheduler Job

Following the existing pattern in `scheduler.py`:
1. Add `HISTORY_SUMMARIZATION` to `JobType` enum
2. Add corresponding `TaskType` in `tasks.py`
3. Create `create_summarization_task()` factory
4. Add case in `_create_task_for_job()`
5. Seed a default job in `init_default_jobs()` with configurable interval
6. The task processor calls the existing `HistorySummarizer` but modified to be non-destructive

### App Memory Management

The in-memory history buffer grows indefinitely during a session. For practical purposes this is fine — even a very long session (thousands of messages) is a few MB in Python. If this ever becomes a concern:
- Evict oldest raw messages from the in-memory buffer (keep only summaries in memory for very old sessions)
- DB records remain — can be reloaded on demand
- This is a far-future concern, not part of this implementation

---

## Phase 5: Enhanced Debug Turn Log with Tool Call Tracing

### Current State

The debug turn log (`debug_turn.txt`, enabled via `--debug` or `LLM_BAWT_DEBUG_TURN_LOG`) captures:
- Request messages (system prompt + history + user prompt)
- Final response text
- JSON dump for machine parsing

**Missing:** All tool calls that happened during the turn are invisible. The tool loop tracks `tool_context` internally (tool names + results summary), but none of this reaches the debug log. You can't tell from the log that the LLM called `memory(action="search", query="recent conversations")` and got back 5 results — you just see the final answer.

### Proposed Enhancement

Add a `TOOL CALLS` section between `REQUEST MESSAGES` and `RESPONSE`:

```
────────────────────────────────────────
TOOL CALLS (3 calls across 2 iterations)
────────────────────────────────────────

[1] Tool: memory
    Action: search
    Parameters: {"query": "recent conversations", "n_results": 5}
    Result (342 chars):
    ────────────────────────────────────
    Found 3 memories:
    1. Nick prefers direct communication (relevance: 0.82)
    2. Nick was debugging memory extraction (relevance: 0.71)
    3. Nick is a software engineer in Ohio (relevance: 0.65)

[2] Tool: profile
    Action: get
    Parameters: {}
    Result (218 chars):
    ────────────────────────────────────
    name: Nick
    occupation: software engineer
    ...

[3] Tool: time
    Parameters: {}
    Result (35 chars):
    ────────────────────────────────────
    2026-02-07 14:52:32 EST
```

### Implementation

1. **Expand `tool_context` tracking in `ToolLoop`** — currently stores `{"tools_called": [...], "results": "..."}`. Change to store per-call detail:
   ```python
   {
       "tool": "memory",
       "action": "search",        # extracted from arguments
       "parameters": {"query": "...", "n_results": 5},
       "result": "Found 3 memories: ...",
       "iteration": 1,
   }
   ```

2. **Pass tool call details to `_write_debug_turn_log`** — add a `tool_calls` parameter alongside `context_messages`, `prompt`, `assistant_response`

3. **Format tool calls in the log** — human-readable section with tool name, parameters, and result for each call

4. **Include in JSON dump** — add `"tool_calls": [...]` array to the machine-parseable JSON section

### Phase 5 Checklist
- [ ] Expand `ToolLoop.tool_context` to store per-call details (tool name, full parameters, result text, iteration)
- [ ] Update `_write_debug_turn_log` signature to accept tool call data
- [ ] Add `TOOL CALLS` section to human-readable log output
- [ ] Add `tool_calls` array to JSON dump section
- [ ] Pass tool context from `query()` in `base.py` to the debug log writer
- [ ] Update `service/api.py` debug log writer to match

---

## Implementation Plan

### Sprint 1: Foundation (Sequential — blocks everything else)

**Owner:** Single agent
**Estimated scope:** Small
**Branch:** `feat/per-model-config`

All downstream work depends on having per-model context windows and VRAM detection in place.

#### Deliverables
- [ ] VRAM detection utility (`utils/vram.py`) — `torch.cuda` primary, `nvidia-smi` fallback, returns total/free VRAM in bytes
- [ ] Auto-sizing function: takes VRAM + model file size → max safe context window
- [ ] `models.yaml` schema additions: `max_tokens`, `n_gpu_layers`, `native_context_limit`, optional `context_window` cap
- [ ] Config methods: `get_model_context_window(alias)`, `get_model_max_tokens(alias)` with resolution order
- [ ] `llama_cpp_client.py` reads auto-sized context + per-model overrides
- [ ] `openai_client.py` reads per-model `max_tokens` and `context_window`
- [ ] `--status` output shows detected VRAM, auto-sized context, per-model info
- [ ] Startup logging of VRAM detection and context sizing decisions

#### Integration gate
- Team lead reviews `utils/vram.py` and resolution order logic
- Verify on both target machines (32GB 5090, 16GB 5080) if possible
- Merge to `main` — all Sprint 2 branches fork from here

---

### Sprint 2: Parallel Tracks (3 independent workstreams)

After Sprint 1 merges, these three tracks can be developed **concurrently** by separate agents. No dependencies between them.

#### Track A: Two-Layer History
**Owner:** Agent A
**Estimated scope:** Large
**Branch:** `feat/two-layer-history` (from `main` after Sprint 1)

Core history redesign. This is the biggest piece of work.

**Deliverables:**
- [ ] Add `db_id` field to `Message` model
- [ ] Update `PostgreSQLShortTermManager.get_messages()` to return message IDs
- [ ] Refactor `HistoryManager`: load once at startup, maintain in-memory, no per-turn DB reads
- [ ] `estimate_tokens()` utility function (`len(text) // 4`)
- [ ] Model-aware context assembly in `_build_context_messages()`:
  - Calculate available budget from model's context window
  - Session-block-to-summary swapping (oldest first)
  - Graceful degradation: drop oldest summaries when even summaries don't fit
  - Protect last N turns via `MEMORY_PROTECTED_RECENT_TURNS`
- [ ] Log token budget breakdown in verbose/debug mode

**Does NOT include:** recall tool, scheduler job, or summarization changes — those come in Sprint 3.

#### Track B: Memory On-Demand
**Owner:** Agent B
**Estimated scope:** Medium
**Branch:** `feat/memory-on-demand` (from `main` after Sprint 1)

Collapses the two-path memory system into one.

**Deliverables:**
- [ ] Remove upfront memory injection from `_stage_memory_retrieval` / `_build_context_messages()`
- [ ] Add read-only `memory(action="search")` tool for `requires_memory` bots without `uses_tools`
- [ ] Cold-start detection: inject 2-3 core memories when history < 3 messages
- [ ] Ensure auto-extraction runs for all memory-enabled bots (not just non-tool)
- [ ] Update bot system prompts to mention memory tool availability

#### Track C: Debug Turn Log Enhancement
**Owner:** Agent C (or any agent with spare capacity)
**Estimated scope:** Small
**Branch:** `feat/debug-turn-log` (from `main` after Sprint 1)

Completely independent, no overlap with A or B.

**Deliverables:**
- [ ] Expand `ToolLoop.tool_context`: store per-call details (tool name, full parameters, result text, iteration)
- [ ] Update `_write_debug_turn_log` signature to accept tool call data
- [ ] Add `TOOL CALLS` section to human-readable log output
- [ ] Add `tool_calls` array to JSON dump section
- [ ] Pass tool context from `query()` in `base.py` to debug log writer
- [ ] Update `service/api.py` debug log writer to match

#### Integration gate (Sprint 2)
- Each track opens a PR independently
- Team lead reviews each PR against `main`
- Merge order: **C first** (smallest, no conflicts), then **B**, then **A** (largest, most likely to touch shared files)
- If A and B have merge conflicts in `base.py` or `pipeline.py`, resolve during A's merge (A is the bigger change)

---

### Sprint 3: Dependent Features (Sequential after Sprint 2)

These features depend on Sprint 2 tracks being merged.

#### Track D: History Cleanup & Recall Tool
**Owner:** Agent A or D
**Estimated scope:** Medium
**Branch:** `feat/history-cleanup` (from `main` after Track A merges)
**Depends on:** Track A (two-layer history must exist)

**Deliverables:**
- [ ] **Remove `_should_skip_history()` from `base.py` and `pipeline.py`** — always include history
- [ ] Remove `TOOLS_SKIP_HISTORY` config setting
- [ ] Add timestamps to tool result history entries (`[Tool Results @ {time}]`)
- [ ] Add `history(action='recall', summary_id='...')` tool action
- [ ] Add `recalled_history` flag to DB schema
- [ ] Prompt guidance: bot mentions when working from summaries

**Why separate from Track A:** Track A is already large. The cleanup and recall tool are logically distinct and easier to review as a focused PR. Track A establishes the two-layer architecture; Track D adds the user-facing features on top.

#### Track E: Unified Memory Tool Output
**Owner:** Agent B or E
**Estimated scope:** Small
**Branch:** `feat/unified-memory-output` (from `main` after Track B merges)
**Depends on:** Track B (memory on-demand must be the only path)

**Deliverables:**
- [ ] Update memory tool search handler to use `build_memory_context_string()`
- [ ] Add relative timestamps to memory search results
- [ ] Return formatted text instead of raw JSON from memory tool

#### Track F: Proactive Summarization Scheduler
**Owner:** Agent A or F
**Estimated scope:** Medium
**Branch:** `feat/proactive-summarization` (from `main` after Track D merges)
**Depends on:** Track D (recall tool + `recalled_history` flag must exist)

**Deliverables:**
- [ ] Add `HISTORY_SUMMARIZATION` to `JobType` enum in `scheduler.py`
- [ ] Add corresponding `TaskType` and factory in `tasks.py`
- [ ] Non-destructive summarization: summary rows coexist with originals
- [ ] Size-based session prioritization (summarize biggest token savings first)
- [ ] Modify existing `HistorySummarizer` to stop deleting/moving originals
- [ ] Seed default summarization job in `init_default_jobs()`
- [ ] Skip `recalled_history` messages in summarization passes

#### Integration gate (Sprint 3)
- D, E, F each open PRs as they complete
- D and E have no conflicts (different subsystems) — can merge in either order
- F must merge after D
- Team lead does final integration review after all Sprint 3 PRs merge

---

### Timeline Visualization

```
Sprint 1 (foundation):
  [=== Per-model config + VRAM detection ===]
                                              ↓ merge to main
Sprint 2 (parallel):
  [======= Track A: Two-layer history =======]  ← largest
  [===== Track B: Memory on-demand =====]        ← medium
  [=== Track C: Debug log ===]                   ← smallest, merges first
                                              ↓ all merge to main

Sprint 3 (dependent):
  [==== Track D: Cleanup + recall ====]          ← after A
  [== Track E: Memory output ==]                 ← after B (parallel with D)
            [==== Track F: Summarization scheduler ====]  ← after D
                                              ↓ final integration review
```

### Risk Areas & Review Focus

| Area | Risk | Mitigation |
|------|------|------------|
| VRAM detection | May not work on all GPU configs (multi-GPU, CPU-only) | Graceful fallback chain; test on both machines |
| `Message.db_id` addition (Track A) | Touches a core data model — could break serialization, API | Ensure backward compat; `db_id` is optional/nullable |
| `_should_skip_history()` removal (Track D) | May reveal other context overflow issues | Token budgeting from Track A must be solid first |
| Non-destructive summarization (Track F) | DB storage grows without pruning | Acceptable for now; add archival/cleanup later if needed |
| Track A + B merge conflicts | Both touch `base.py`, `pipeline.py` | Merge B first (smaller), resolve A conflicts during A's merge |
