# EPIC TASK-217 — Layer C browser E2E scenario specs (TASK-362)

DO NOT run browsers from the P0.5 harness. These are scripted scenario
definitions for later phases to drive via the Playwright MCP against the live
bawthub frontend. Each scenario names the phase that turns it GREEN.

Preconditions for every scenario:
- Use a disposable/test bot conversation, never a real operator thread.
- Never restart app/bridge/redis. Observe only.

## C1 — user turn id is a canonical UUID end-to-end   (GREEN in P1)
Steps:
1. Open a chat, send a user message "C1 ping".
2. Capture the optimistic message id rendered in the DOM (data-message-id).
3. After the server round-trip, capture the reconciled/persisted id.
Assert:
- The persisted id matches ^[0-9a-f]{8}-...-[0-9a-f]{12}$ (canonical uuid4).
- No `local-*` / `quick-user-*` id ever appears as the FINAL id on the row.

## C2 — tool-call activity joins the turn by trigger_message_id  (GREEN in P3)
Steps:
1. Send a message that triggers a tool call.
2. Observe the AgentActivityRow / tool-call cards attach to the correct turn.
Assert:
- Activity rows key off the real message UUID (turn_logs.trigger_message_id ==
  {bot}_messages.id), NOT an optimistic local key.
- No orphaned activity that needs re-keying (useOrphanedActivityReKey path is
  a no-op / removed).

## C3 — reload / history replay preserves ids & order  (GREEN in P1)
Steps:
1. Send 2 user + 2 assistant turns.
2. Hard-reload the page (history replay via /v1/history).
Assert:
- Every row's id after reload equals the id from before reload (stable UUIDs).
- Assistant reasoning/thinking hydrates by id without duplication.

## C4 — resume mid-stream has no synthetic-injection dupes  (GREEN in P3)
Steps:
1. Start a long streaming assistant turn.
2. Reload / resume while streaming (resumeScan path).
Assert:
- The resumed turn is the SAME row (same UUID), not a synthetic-injected clone.
- No duplicate assistant bubble; timestamps not re-reconciled after the fact.

## C5 — INSERT failure is visible, never silent  (GREEN in P1)
Steps:
1. Induce a persistence failure for a turn (test hook / oversized id in a
   disposable bot) — coordinate with Layer A A4.
Assert:
- The UI surfaces an error state for that turn (does NOT show a normal
  "sent/persisted" bubble for a row that never committed).
- Backend A4 assertion is GREEN (caller received the error signal).
