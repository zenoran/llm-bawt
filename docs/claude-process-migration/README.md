# Claude Process Migration — Design Notes

**Status:** discussion / not approved
**Started:** 2026-06-03
**Author seed:** vex + Nick

## What this directory is

A scratchpad for designing the migration from `claude_agent_sdk.query()` (one-shot subprocess spawn per turn) to **long-lived `claude` processes per bot** (persistent conversations, prompts streamed in over the lifetime of the process).

**Primary motivators** (after correction in `03-cache-comparison-correction.md`):
1. Per-turn latency reduction — skill/plugin/CLAUDE.md load happens once, not every turn.
2. Cleaner ownership of process lifecycle and steering (abort, MCP context injection, cost table) on our terms rather than the SDK's.
3. Easier proxy integration — one `ANTHROPIC_BASE_URL`-pinned process per bot, one log stream per bot.

**Not** motivators despite an earlier draft claiming otherwise:
- API cost. The cache benefit is server-side and `claude -p --resume` gets it identically.
- Escaping the June 15 Agent SDK metering. `claude -p` (including stream-json mode) IS the metered category. The migration doesn't change billing class.

The June 15 change still matters as background — it raises the value of any latency / efficiency improvement — but it isn't *solved* by this migration.

Secondary motivator: the wire-level audit gap. The HTTP logging proxy at `echo:/home/nick/claude-proxy/proxy.py` (see `/tmp/PROXY_VS_SDK.md`) gives us full visibility, but only if our bridge isn't fighting it. Persistent processes + proxy + structured event publication = clean three-layer architecture.

## Files

**Read in this order — later docs correct earlier claims.**

| File | Purpose |
|---|---|
| **`HANDOFF.md`** ⭐ | **Start here when picking up the work in a new context.** TL;DR of what's settled, what's next, what NOT to redo, plus a ready-to-use boot prompt |
| `README.md` | this file — index + motivation |
| `01-migration-plan.md` | the main proposal: architecture, IO transport options, lifecycle, phased rollout |
| `02-poc-results.md` | first POC (persistent process, two-turn conversation). **Has a correction banner at top** — the cache-savings claim turned out to be wrong |
| `03-cache-comparison-correction.md` | the "cache benefit is server-side, not a property of persistence" correction. Reduces the migration's value prop from "lower API cost" to "lower latency + cleaner architecture" |
| `04-billing-and-token-scope.md` ⭐ | **The wire-comparison finding** + correction. TUI and `-p` send byte-identical bodies; the User-Agent suffix (sourced from `CLAUDE_CODE_ENTRYPOINT`) IS the billing-class classifier. Billing requires both token scope (eligibility) and entrypoint string (bucket). Has a correction banner at top — the original "scope alone decides" framing was incomplete |
| `05-vscode-extension-investigation.md` | ✅ done (2026-06-04). Result: the extension JS is the Agent SDK with stream-json IO, no signing code. Found the missing puzzle piece: it hardcodes `CLAUDE_CODE_ENTRYPOINT="claude-vscode"` in the spawn env (`:81940`), which the CLI propagates into `User-Agent` and the server routes billing on. See "Findings" + "Follow-up finding" sections |
| `poc_persistent_claude.py` | runnable POC: persistent process + stream-json IO + context retention across turns |
| `poc_cache_compare.py` | runnable comparison: persistent process vs. spawn-per-turn-with-resume. Shows cache behavior is server-side |

## Related material

- Current bridge: `src/claude_code_bridge/bridge.py` (uses `claude_agent_sdk`)
- Current bridge doc: `docs/CLAUDE_CODE_BRIDGE.md`
- Proxy POC: `/tmp/claude-proxy.py`, `/tmp/PROXY_VS_SDK.md` (also at `echo:~/claude-proxy/`)
- June 15 billing change context: see web search results in conversation history
