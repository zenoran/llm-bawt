# Correction: cache savings are NOT a property of persistence

**Date:** 2026-06-03
**Script:** `poc_cache_compare.py`

## What the original POC claimed
> "Massive cache-read amplification in turn 2 — only ~120 fresh-cache-creation tokens vs. 12,561 cached reads in turn 2 […] This is the cost story for the June 15 metering change."

## What actually happens

Same dialogue, two configurations, run back-to-back:

| Turn | Path A (persistent process) | Path B (cold spawn + `--resume`, ≈ SDK today) |
|---|---|---|
| 1 | elapsed=3.14s, cache_create=7303, **cache_read=12561** | elapsed=2.22s, cache_create=0, **cache_read=19864** |
| 2 | elapsed=1.56s, cache_create=163, **cache_read=19864** | elapsed=2.30s, cache_create=119, **cache_read=19864** |

`cache_read` in turn 2 is **identical** between the two paths: 19,864 tokens.

## Why

Anthropic's prompt cache is server-side and content-hashed. It is not bound to a process, connection, or session. If two requests send the same prefix bytes within the TTL window (default 5 min, 1 hour with `ephemeral_1h` markers — which the bundled CLI is using), they both hit the same cache. The cache doesn't know or care that Path A reused a process and Path B spawned a new one.

(You can even see this in Path B turn 1 — `cache_create=0` because the cache was already populated by Path A immediately before.)

## What this means for the migration

The migration is still worth doing, but for narrower reasons:

| Reason | Holds? | Magnitude |
|---|---|---|
| Lower per-turn latency (no respawn + skill load) | ✅ | ~0.5–1s per turn |
| Lower API cost via better cache hit rate | ❌ | spawn-per-turn-with-resume gets the same hit rate |
| Escape the June 15 Agent SDK metering | ❌ | `claude -p` is exactly the metered category |
| Cleaner process lifecycle / debuggability / proxy integration | ✅ | qualitative |
| Bridge owns abort / steering semantics directly | ✅ | qualitative |

The 2.8× total speedup in the first POC was real but mostly **one-time cold-start savings**, which the SDK can't avoid on first turn either. After the first turn per bot per bridge lifetime, the per-turn delta narrows to ~0.7s.

## So why still do it?

1. **Per-turn latency adds up.** 0.7s × N turns × M bots is real felt-snappiness.
2. **Process-lifecycle ownership** lets us implement the abort / MCP-context-injection / cost-table machinery on our terms rather than the SDK's.
3. **The proxy integration is cleaner** with stable long-lived processes (one ANTHROPIC_BASE_URL setting, one log stream per bot, easier demuxing).
4. **Stable cache prefix is easier to reason about** when you can see it across a single process's lifetime instead of reconstructed-from-disk each turn. (This is about *debuggability* of the cache, not the cache savings themselves.)
5. **It's a step toward owning the agent loop.** If we ever want to do bridge-side things the SDK doesn't expose (custom tool routing, mid-turn message injection from another bot, etc.), persistent processes are the foundation.

## Recalibrated motivation for the design doc

The original `README.md` and `01-migration-plan.md` lead with "the primary motivator is the June 15 billing split." That framing is wrong. Replace with:

> **Primary motivators:** per-turn latency reduction, cleaner ownership of process lifecycle and steering, easier proxy integration. **Not** an API-cost win — the cache benefit is server-side and the spawn-per-turn SDK path gets it via `--resume`. **Not** a billing-category change — stream-json is `-p`, which is in the new metered pool.

The June 15 change matters as *context* (it raises the value of any latency or efficiency improvement) but it is not solved by this migration.
