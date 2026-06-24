# Turn Integrity Design Doc

**Task:** TASK-306
**Status:** Draft for review (Nick)
**Author:** Caid
**Date:** 2026-06-24

---

## Governing invariant

> **A bot must never narrate a reality that did not happen.**
> Every claim of a side effect must be backed by a *confirmed* side effect,
> or the claim must not be made.

This document covers two gaps that are the **same** violation — a claim decoupled
from its effect:

| Section | The lie | Mechanism |
|---------|---------|-----------|
| **A. Persistence** | "I sent your question / approval request" when the row never committed | lie by *omission of failure* |
| **B. Finalization** | "I committed and pushed" when zero tools fired | lie by *fabrication of action* |

### Why this is not a durability ticket

The lost DB row is not the real damage. The real damage is the **behavioral
cascade**: an agent that does not get a truthful response back **runs off on its
own**. It assumes success, proceeds on a false premise, reports completion, and
compounds the fiction. Every later action is built on state that does not exist, and
the drift is unbounded.

So the fix is not "error handling." It is a **design-enforced break on autonomous
drift** — a hard structural point that forces the agent back into contact with
reality before it can wander. The agent may *end the turn* (that is acceptable), but
it ends it **honestly** ("I tried to send this and it did not go through; here is
why") instead of lying ("I sent the question").

### Priority order (applies to both sections)

1. **Durability is the floor** — the DB write is the source of truth.
2. **The *informed* failure is the point** — the agent must learn the truth.
3. **Honesty is the brake** — it is what stops the runaway.

> A retry the agent cannot observe is just a slower swallow.

### What is already sound

A fresh trace (2026-06-24, post-restart) confirms `record_request()` has **exactly
one call site** and there is **no duplicated or branched persistence path**. The
structure is correct. The problem is narrow and specific: the single correct write is
neither *guaranteed to run*, nor *confirmed*, nor *reported back to the agent*.

---

## Section A — Commit-confirmed persistence

### A.1 Current behavior

Persistence of approval requests (and the AskUserQuestion equivalent) happens
**inline inside the cancellable async SSE generator**.

The lone approval write — `src/llm_bawt/service/chat_streaming.py:2193-2216`:

```python
if req_id and store is not None and store.engine is not None:
    approval_id_holder[0] = req_id
    try:
        store.record_request(
            request_id=req_id,
            bot_id=bot_id,
            ...
        )
    except Exception as _persist_err:
        log.warning(
            "Failed to persist approval request id=%s: %s",
            req_id, _persist_err,
        )
```

Supporting facts:

- `store.record_request()` — `src/llm_bawt/approval_policies.py:371` — idempotent on
  `request_id` (safe to retry / replay).
- The bridge only **emits** the gate
  (`src/claude_code_bridge/bridge.py` `_emit_approval_required`, ~1813-1856) and
  `src/llm_bawt/agent_backends/agent_bridge.py:373` only **queues**
  `{"event": "approval_required"}` onto the shared chunk queue read at
  `chat_streaming.py:1705`. Neither persists. **Do not add a second write path here.**
- AskUserQuestion mirrors this shape via `upsert_awaiting`
  (`src/llm_bawt/service/chat_pending_questions.py`, TASK-269 path).

### A.2 Two independent defects

**Defect 1 — Unconfirmed (the write is swallowed).**
The `except` at `chat_streaming.py:2212-2216` logs a warning and the turn **continues
as if it persisted**. There is no commit gate. This is precisely "blindly assume it
happened," and it fails *even on the happy path* — a DB error is indistinguishable
from success to everything downstream.

**Defect 2 — Not guaranteed to run.**
Because the call lives inside the cancellable generator, if the stream is torn down
before that chunk is processed (client disconnect, SSE teardown, abort, timeout), the
write **never executes** — no row, no SSE, no Redis fanout, and `end_reason` defaults
to `"stop"` so the turn *looks like it ended normally*. The intended fallback "client
refresh hydrates from DB" then has nothing to hydrate from, because the floor was
never laid.

### A.3 Intended design (three layers)

1. **DB write is the durable floor** — must be *confirmed committed* before the turn
   moves on. The DB, not the SSE push, is the source of truth.
2. **SSE/Redis is best-effort delivery** on top of the floor. The bot does **not**
   block waiting for the client to acknowledge.
3. **Client refresh re-hydrates from DB** — if live delivery fails, the row is still
   there and the UI recovers on reload. Worst case degrades to "user sees it on
   refresh," never "it silently vanished."

### A.4 Requirements

1. **Persist at a guaranteed-reached point.** Move the write upstream of / outside the
   cancellable generator, or otherwise guarantee the chunk is processed before turn
   completion. It must not be skippable by stream teardown.
2. **Confirmed commit + bounded retry.** Verify the write committed; retry a bounded
   number of times on transient failure. No log-and-shrug.
3. **Fail-closed.** If the commit genuinely cannot be made after retries, the gated
   action does **not** proceed. (Turn ending here is acceptable.)
4. **Fail-INFORMED (the crux).** The structured failure reason — error class, what was
   being persisted, and the explicit fact *"this will NOT reach the user"* — is
   injected back into the agent's turn so the bot can (a) honestly inform the user
   instead of lying, and (b) reason about / investigate the cause (DB down? store
   engine null? schema/constraint error? pool exhaustion?). This requires the failure
   payload to carry **structure to reason about — not a boolean.**

### A.5 Proposed mechanism

```
                 gate decision (bridge)
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │  PERSIST (guaranteed-reached, confirmed) │
        │  record_request()  +  bounded retry      │
        └───────────────┬─────────────┬────────────┘
                committed?           failed after retries?
                        │                     │
                        ▼                     ▼
            ┌───────────────────┐   ┌────────────────────────┐
            │ best-effort fanout│   │ FAIL-CLOSED + INFORMED  │
            │ SSE + Redis       │   │ inject structured error │
            │ (may drop safely) │   │ into agent turn result; │
            │ refresh re-hydrate│   │ end turn honestly        │
            └───────────────────┘   └────────────────────────┘
```

**Placement.** The persist step must execute before the turn can complete and must not
be tied to the SSE consumer's liveness. Two viable shapes (decide in review):

- **(a) Upstream-of-generator:** the app persists the row when it first receives the
  bridge's `approval_required` event (before the SSE generator processes it), so the
  generator's job is reduced to *fanout* of an already-committed row.
- **(b) Confirm-before-complete:** keep the write where it is but make turn completion
  *depend* on it — the generator cannot reach a normal `end_reason` unless the commit
  was confirmed, and cancellation routes through a finalizer that still performs (or
  re-drives) the commit.

(a) is structurally cleaner — it removes the write from the cancellable path entirely,
which directly kills Defect 2. Recommended pending review.

**Confirmed commit.** `record_request()` already returns/raises on the insert; the
change is to (i) stop swallowing at the call site, (ii) wrap in a bounded retry for
transient errors, and (iii) return a definite committed / not-committed result to the
caller.

**Failure payload (agent-legible).** On terminal failure, the turn result carries a
structured object, e.g.:

```json
{
  "kind": "side_effect_failed",
  "effect": "approval_request_persist",
  "request_id": "...",
  "tool_name": "make rebuild-prod",
  "reachable_to_user": false,
  "error_class": "OperationalError",
  "detail": "connection pool exhausted",
  "retries": 3
}
```

This is what lets the bot say *"I could not record the approval gate for
`make rebuild-prod` — the database write failed (connection pool exhausted) after 3
tries, so you will not have seen a prompt"* instead of claiming it sent one.

### A.6 Turn-end semantics (decided)

The turn **may end** on persist failure — **no mid-turn round-trip back into the model
is required.** The hard requirement is only that the agent receives the structured
failure as its turn result, with enough information to inform the user truthfully and
to investigate. This is the design-enforced break: the agent does not get to invent a
world where the question was delivered.

### A.7 SSE ordering

Fanout (SSE + Redis) fires **after** the confirmed commit, best-effort. Client refresh
always reads from the now-guaranteed-populated DB. A dropped fanout is recoverable; a
dropped commit is not — hence commit first.

### A.8 Honest failure modes

- **DB transiently down → retries exhaust:** fail-closed, inform agent, end turn. User
  is told truthfully. (Better than a phantom approval the user never sees.)
- **`store` / `engine` is `None`** (misconfig): same path — this is currently a silent
  skip (`if ... store.engine is not None`) and should become an *informed* failure,
  not a no-op.
- **Fanout fails but commit succeeded:** acceptable; refresh recovers it. No agent-
  facing failure needed.

### A.9 Key files (Section A)

| File | Location | Role |
|------|----------|------|
| `service/chat_streaming.py` | ~1705 | shared chunk-queue read |
| `service/chat_streaming.py` | 2178-2250 | `approval_required` branch; lone `record_request` at **2196**; swallow at **2212-2216** |
| `approval_policies.py` | 371 | `record_request` (idempotent insert) |
| `service/chat_pending_questions.py` | — | `upsert_awaiting` (same treatment) |
| `agent_backends/agent_bridge.py` | 373 | event forwarding (no persistence — keep it that way) |
| `claude_code_bridge/bridge.py` | ~1813-1856 | `_emit_approval_required` |

---

## Section B — Turn finalization guard

### B.1 Current behavior

`_finalize_turn()` in `src/llm_bawt/service/turn_lifecycle.py:272-346` is the
**single funnel** where every turn type finalizes:

- streaming completion → `chat_streaming.py:1799`
- streaming abort → `chat_streaming.py:1771`
- bridge (OpenClaw / Claude Code) → `chat_streaming.py:436-449`

It receives two **independent** inputs and never compares them:

- `response_text: str` — what the bot **says** it did.
- `tool_call_details: list[dict]` — what it **actually** did.

For bridge bots, `tool_call_details` starts empty and is back-filled at
`turn_lifecycle.py:317` by `_extract_agent_backend_tool_calls()` (reads
`llm_bawt.client.get_tool_calls()` → `clients/agent_backend_client.py:228`). If
extraction returns nothing, the list stays `[]`.

The gap — `turn_lifecycle.py:317-346`:

```python
extracted_tool_calls = self._extract_agent_backend_tool_calls(llm_bawt=llm_bawt)
if extracted_tool_calls and not tool_call_details:
    tool_call_details.extend(extracted_tool_calls)
    ...
# <-- NOTHING here validates response_text against tool_call_details -->
llm_bawt.finalize_response(response_text, tool_context, ...)   # saves to history
self._update_turn_log(..., tool_calls=tool_call_details, ...)  # saves turn log
```

So `response_text = "I committed and pushed the fix"` with `tool_call_details = []` is
persisted to history (`service/core.py:495-524`) and the turn log
(`turn_logs.py:374-441`, writes `tool_calls_json`) as ground truth. This is the
"Snark claims work, emits no tool_use" bug.

### B.2 The guard

**Location: `_finalize_turn()` ONLY**, at ~line 333 — after the line-317 extraction
back-fill, before the line-334 save. This is the single funnel, so one check covers
every turn type. **Do not** scatter it into `_update_turn_log` (`turn_logs.py`) or
`core.py`; duplicating the policy across layers is the exact failure mode we are
fixing.

**Trigger condition:**

```
is_bridge_turn AND tool_call_details == []  AND  response_text is non-trivial
```

All state needed already exists at function entry — `response_text`,
`tool_call_details` (post-extraction), `llm_bawt`, `bot_id`, `turn_id`. **No new
plumbing.**

> Note the existing early-exit at `turn_lifecycle.py:301-302`: the function already
> returns *without persisting* when `response_text` is empty. The guard is the inverse
> case — **non-empty narration with empty tools.**

### B.3 Open decision — reject vs. flag

| Option | Behavior | Pro | Con |
|--------|----------|-----|-----|
| **Reject (fail-closed)** | Do not accept the narration as a normal completion; mark the turn failed / force the bot to actually act | Consistent with the governing invariant; hardest brake on drift | More disruptive; needs a retry/failed-turn path |
| **Flag-and-persist** | Save the turn but tag it suspect so the UI shows "claimed work, ran no tools" | Less disruptive; preserves the transcript; user sees the discrepancy | The lie still lands in history; relies on a human noticing the flag |

**Recommendation:** **reject / fail-closed**, consistent with Section A. The invariant
is "never narrate a reality that did not happen" — flag-and-persist still lets the
false claim into history. *This is the one item that needs Nick's call before
implementation.*

A possible middle path worth discussing: reject **and** retain the rejected text on
the turn record (not in conversational history) so it is auditable without being
treated as truth.

### B.4 What counts as "non-trivial narration"

To avoid false positives (a bot legitimately answering a pure-conversation turn with
no tools), the trigger should distinguish:

- pure conversational replies (no work claimed) — **allowed**, no tools expected;
- **work claims** — assertions of having performed a side effect.

Cheapest reliable signal is **the turn's own context**, not NLP on the text: a turn
that was dispatched as an agent/task turn (bridge backend, task-bound) is *expected* to
produce tool calls; an empty `tool_call_details` there is the anomaly. Resolve the
exact predicate in review — start with `is_bridge_turn AND task-bound` rather than
heuristic verb-matching on `response_text`, which is brittle.

### B.5 Key files (Section B)

| File | Location | Role |
|------|----------|------|
| `service/turn_lifecycle.py` | 272-346 | `_finalize_turn` — **guard goes here (~333)** |
| `service/turn_lifecycle.py` | 247-270 | `_extract_agent_backend_tool_calls` |
| `service/turn_lifecycle.py` | 317-322 | extraction back-fill |
| `clients/agent_backend_client.py` | 228 | `get_tool_calls()` |
| `service/chat_streaming.py` | 1621-1635, 1689-1692 | streaming tool-call accumulation |
| `service/core.py` | 495-524 | `finalize_response` (history save; no validation today) |
| `service/turn_logs.py` | 374-441 | `update_turn` (writes `tool_calls_json`; no validation today) |

---

## How A and B relate

Both are instances of the one invariant. A protects the **inbound** side effect (a row
the user must see); B protects the **outbound** claim (a result the user is told).
Together they close the loop: the system neither loses an effect it promised nor
asserts an effect it never produced. The shared root cause is a single chokepoint
trusted to be both *reached* and *truthful* with nothing validating either. The fix in
both cases is the same shape: **confirm the effect, and on failure inform the agent so
it stops instead of drifting.**

---

## Open decisions for review

1. **A.5 placement** — upstream-of-generator (a, recommended) vs. confirm-before-
   complete (b)?
2. **B.3** — reject/fail-closed (recommended) vs. flag-and-persist vs. the reject+retain
   middle path?
3. **B.4 predicate** — confirm `is_bridge_turn AND task-bound` as the trigger, or a
   different definition of "work claim"?
4. **Retry bounds** — concrete retry count / backoff for A.4(2)?

## Out of scope

- Implementation. This is a design doc; coding follows review.
- Pending-approval TTL/expiry (separate, deferred).
- The gold-lock badge bridge restart (separate operational item).
