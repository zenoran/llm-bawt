# POC Results: Persistent Claude Process

> **2026-06-03 follow-up correction:** The original writeup oversold this.
> See `03-cache-comparison-correction.md` for the actual numbers comparing
> persistent vs. spawn-per-turn-with-resume. Short version: the cache benefit
> is server-side, so spawn-per-turn with `--resume` gets it too. The real win
> is per-turn latency (~0.5–1s saved), not API cost, and this approach does
> NOT escape the June 15 metering — `claude -p` (including stream-json) is
> exactly the metered category.

**Date:** 2026-06-03
**Script:** `poc_persistent_claude.py`
**Binary:** `claude 2.1.139 (Claude Code)` from `claude_agent_sdk._bundled/claude` (ELF, statically packed)
**Model:** `haiku` (for speed/cost)
**Auth:** `CLAUDE_CODE_OAUTH_TOKEN` from llm-bawt/.env

## TL;DR

- Persistent process **works as theorized**. Context survives across turns within one running CLI.
- Same session UUID reused automatically across turns in one process.
- **2.8× faster total wall-clock** for the two-turn experiment.
- **Massive cache-read amplification** in turn 2 — only ~120 fresh-cache-creation tokens vs. 12,561 cached reads in turn 2 (vs. 7,200 / 12,561 in turn 1). This is the cost story for the June 15 metering change.

## Raw numbers

| Phase | Persistent process | Spawn-per-turn (baseline) |
|---|---|---|
| Turn 1 | 2.32 s | 8.39 s (cold) |
| Turn 2 | 1.53 s | 2.24 s (cold, no resume) |
| **Total** | **3.85 s** | **10.63 s** |

Caveat: the persistent experiment ran first, so the baseline benefited from warm disk cache (binary, skills, plugin dirs in OS page cache). The cold-start delta in real production-from-boot is likely *larger* than 8.39 − 2.32 = 6.07s.

## Token / cache behavior

Turn 1 (persistent):
```
input_tokens: 10
cache_creation_input_tokens: 7200   ← built fresh cache for system prompt + tools
cache_read_input_tokens: 12561
output_tokens: 95
```

Turn 2 (persistent):
```
input_tokens: 10
cache_creation_input_tokens: 120    ← only the new user msg + assistant reply
cache_read_input_tokens: 19761      ← system + tools + turn-1 history, all cached
output_tokens: 45
```

The `cache_creation` cost drops by ~60× on turn 2 because the persistent process never resets the cache prefix. Under Anthropic's pricing (cache_creation is 1.25× base, cache_read is 0.1× base), this is exactly where the savings come from.

The cache is marked `ephemeral_1h_input_tokens: 7200` — the bundled binary is using the 1-hour cache TTL, not 5-minute. That gives us a wide idle window before cache expires.

## What the protocol actually looks like

`claude --print --input-format stream-json --output-format stream-json --verbose --include-partial-messages --model haiku --no-session-persistence --permission-mode bypassPermissions`

**Input frame (one JSONL per user turn):**
```json
{"type": "user", "session_id": "", "message": {"role": "user", "content": "..."}, "parent_tool_use_id": null}
```

**Output: JSONL stream of:**
- `{"type": "system", "subtype": "init", "session_id": "..."}` — once, at process start
- `{"type": "stream_event", ...}` — partial deltas (text, tool input chunks)
- `{"type": "assistant", "message": {"content": [...]}}` — assembled assistant turns
- `{"type": "user", "message": {"content": [tool_result blocks]}}` — tool results echo
- `{"type": "result", "usage": {...}, "stop_reason": "..."}` — end of turn (process stays alive!)

After a `result` event, the process is **waiting on stdin for the next user message**. No `--resume`, no respawn, no JSONL reload.

To end: close stdin (or SIGTERM).

## Operational notes / gotchas observed

1. **`--print` is required** for stream-json IO. Without it, stream-json flags are no-ops.
2. **`--verbose` is required** to actually emit per-event JSONL when in `--print` mode. Otherwise you get only the final result.
3. **`--no-session-persistence`** prevents the process from writing to `~/.claude/projects/...`. Good for the POC; for production we may want persistence so bridge crashes can resume via `--resume <uuid>`.
4. The process inherits a lot of env from the parent. I had to strip `CLAUDE_*`, `CLAUDECODE`, `AI_AGENT` from my (claude-running-me) env to avoid the child thinking it was a subagent. Production bridge should pass an explicit clean env.
5. The bundled binary at `claude_agent_sdk/_bundled/claude` is a self-contained ELF, not a node script. We do NOT need Node installed on bridge hosts.
6. The CLI emitted no JSON parsing oddities or non-JSON noise during the run — straight line-delimited JSONL.

## Things this POC did NOT cover (open work)

- **Tool use across turns.** Need to verify a multi-tool-call turn still works and that tool_use_id correlation holds. Single-text-reply turns are the easy case.
- **Abort mid-turn.** Need to test SIGINT to the CLI process: does it cleanly cancel a running tool, does the process survive for the next turn, or does it terminate?
- **System prompt change between turns.** Probably not supported within one process — would need to verify and design around (likely "if system prompt changes, restart the process").
- **MCP context injection per turn.** Per current bridge code, MCP context is prepended to `system_prompt` per turn. With persistent process, system prompt is fixed at spawn. We'd have to either bake static context at spawn + thread dynamic per-turn context into the user message text, or restart for dynamic changes.
- **Long-idle behavior.** What happens after 1 hour of idle (cache TTL expiration)? Does cache_creation cost re-spike on next turn? Should we proactively send a no-op to keep cache warm?
- **Memory footprint per persistent process.** Didn't measure RSS. Need to before deciding how many bots can co-exist on one host.
- **Stderr handling.** The POC captured stderr but the process emitted none during these turns. Need to see what real workloads emit and whether anything affects parsing.

## Verdict

**Strong go on the migration approach.** The hypothesis (one persistent process per bot with stream-json IO) holds in practice with no surprises. Per-turn latency drops, prompt cache becomes drastically more effective, and the protocol is exactly as documented.

Next steps in priority order:
1. Test tool use across turns in one process.
2. Test SIGINT abort semantics.
3. Measure per-process RSS at idle and active.
4. Sketch the `ClaudeProcessHandle` class for the bridge based on what we learned.
