# Handoff: where we are and what's next

**Use this file as the entry point when continuing in a fresh context.**

## TL;DR of where we landed

1. **The migration to persistent `claude` processes is sound**, but the value prop is narrower than originally claimed:
   - ✅ Per-turn latency reduction (~0.5–1s saved per turn after first)
   - ✅ Cleaner process lifecycle / debug / proxy integration
   - ❌ Does NOT save API costs (cache benefit is server-side, same for spawn-per-turn-with-resume)
   - ❌ Does NOT escape the June 15 metering — `claude -p` (including stream-json) IS the metered category

2. **Billing classification uses TWO server-side signals.** *(Corrected 2026-06-04 from "scope alone" — see correction banner in `04-billing-and-token-scope.md`.)*
   - **Token scope** (from `Authorization:` bearer) gates eligibility. Setup-token = `user:inference` only = metered pool only. Full-scope token (from `claude auth login`) = eligible for either pool.
   - **`CLAUDE_CODE_ENTRYPOINT`** env var (propagated by CLI into `User-Agent` suffix) gates routing within eligibility. `cli` (TUI) and `claude-vscode` (extension, hardcoded at `extension.pretty.js:81940`) → subscription. `sdk-ts` / `sdk-py` / `sdk-cli` → metered Agent SDK pool.
   - Both signals required for subscription billing. Setup-token with `claude-vscode` entrypoint still hits metered (scope can't reach subscription). Full-scope token with `sdk-ts` entrypoint hits metered (entrypoint says "I'm SDK").
   - **Enforcement is honor-system at the wire level** — no cryptographic gate. The official wrappers cooperate by tagging themselves truthfully. Subverting it (setting `claude-vscode` on non-extension traffic) is misrepresenting one Anthropic product as another, trivially detectable, and we don't do it. We stay on metered with honest `sdk-*` tagging.

3. **`--remote-control` is closed.** Setup-tokens are rejected client-side and server-side. *Update 2026-06-04:* the suspected client-side request-signing requirement **is not present** in the official VS Code extension JS (verified via VSIX v2.1.162 grep — see `05-vscode-extension-investigation.md`). The gate is the OAuth scope, not a signature. Remote-control is closed by scope enforcement alone.

4. **The only theoretically-unbreakable + ToS-defensible path to subscription billing** is to run the actual `claude` TUI under a PTY with a user-acquired full-scope token, and inject prompts as keystrokes. Output parsing via `pyte` (in-memory VT100 emulator), not regex-stripping ANSI.

## Reading order for a fresh context

Read these in order; the later ones correct earlier claims, so don't skip:

1. **`README.md`** — index + current motivators
2. **`01-migration-plan.md`** — the proposal (skim §3 on IO transport options, the rest gives architecture)
3. **`02-poc-results.md`** — first POC. **Read the correction banner at the top** before trusting the body
4. **`03-cache-comparison-correction.md`** — the "I oversold the cache story" correction
5. **`04-billing-and-token-scope.md`** ⭐ — the wire-comparison finding + scope mechanics. **Most important single file in this dir**
6. **`05-vscode-extension-investigation.md`** — the next research step (planned, not yet done)
7. **`poc_persistent_claude.py`** and **`poc_cache_compare.py`** — runnable POC scripts. Both are self-contained, just point at the bundled `claude` binary

## What's next (in priority order)

### ~~1. VS Code extension investigation~~ ✅ done (2026-06-04)
Result: no extension-side magic. The extension JS is the Agent SDK; it spawns `claude` CLI with `--input-format stream-json --output-format stream-json --verbose`; reads OAuth token from keychain / credentials.json / env. **Zero signing code** (no `signRequest`, no `HMAC`, no `crypto.sign`, no `webcrypto`). Confirms H1 + H2 from the plan; kills H3 + H4 + H5. The architectural picture is complete. See "Findings" in `05-vscode-extension-investigation.md`.

### 1. Tool use across persistent-process turns (now top priority)
We POC'd a clean two-turn conversation. We did NOT POC:
- Tool use across multiple turns in one persistent process (does tool_use_id correlation hold?)
- SIGINT abort mid-turn (does the process survive for the next turn or does it terminate?)
- System-prompt changes between turns (probably not supported; need to verify the failure mode)
- Long-idle behavior (1-hour cache TTL expiration — does cache cost re-spike?)

### 2. (Eventually) Production migration
The plan in `01-migration-plan.md` §10 lays out: Phase 0 (validation) → Phase 1 (shadow mode) → Phase 2 (per-bot opt-in flag) → Phase 3 (default) → Phase 4 (SDK removal).

## What NOT to redo

- Don't re-investigate `--remote-control` — closed, documented, multiple layers of gate (scope, trusted-device, server-side cloud routing). The earlier suspected "signing requirement" was disproved by the VSIX grep.
- Don't re-investigate whether `-p` and TUI request *bodies* differ — they don't. The User-Agent suffix DOES differ and IS the billing classifier (corrected from earlier "one-word cosmetic difference" framing — see `04` correction banner).
- Don't believe earlier claims about cache savings being a property of persistence — they're server-side, both modes benefit equally.
- Don't believe the early framing of "primary motivator is June 15 billing" — that was wrong; the migration doesn't change billing class.
- Don't spoof `CLAUDE_CODE_ENTRYPOINT="claude-vscode"` to land in subscription billing. Technically works; cleanly ToS-violating; trivially detectable; not a stable foundation. Stay honest with `sdk-ts`/`sdk-py`.

## Useful artifacts left behind

- `/tmp/claude-proxy.py` — the logging proxy (also at `echo:~/claude-proxy/proxy.py`)
- `/tmp/PROXY_VS_SDK.md` — design note for the proxy vs SDK comparison
- `/tmp/rc-proxy-logs/` — captured `/v1/messages` traces from TUI vs `-p` comparison
- `/tmp/drive_tui.py` — PTY-based TUI driver used for the wire comparison
- `/tmp/diff_p_vs_tui.py` — the diff script for proxy logs

## Key files in the source we read

- `claude-code-source/src/bridge/trustedDevice.ts` — trusted-device enrollment (90d server-issued bearer token, gated on fresh `/login` < 10min, GrowthBook flag `tengu_sessions_elevated_auth_enforcement`)
- `claude-code-source/src/bridge/replBridge.ts` — the REPL bridge (what `--remote-control` powers)
- `claude-code-source/src/bridge/types.ts` — has the literal error string: *"Remote Control is only available with claude.ai subscriptions. Please use `/login` to sign in with your claude.ai account."*
- `claude-code-source/src/constants/oauth.ts` — defines `CLAUDE_AI_INFERENCE_SCOPE = 'user:inference'` and the full scope set
- `claude-code-source/src/utils/auth.ts` — token loading; line 1266 hardcodes setup-token scope to `['user:inference']`
- `claude-code-source/src/services/oauth/client.ts` — `shouldUseClaudeAIAuth(scopes)` check; `buildAuthUrl({ inferenceOnly: ... })`
- `claude-code-source/src/components/ConsoleOAuthFlow.tsx:203` — `expiresIn: 365 * 24 * 60 * 60` for setup-tokens (1-year expiry)

## Boot prompt for a fresh context

> Continuing work in `~/dev/llm-bawt/docs/claude-process-migration/`. Read `HANDOFF.md` first, then `04-billing-and-token-scope.md`, then `05-vscode-extension-investigation.md` (the Findings section). The architectural research is done — the VS Code extension is structurally identical to our POC (Agent SDK + stream-json + webview, no signing, no auth magic). The next concrete task is the persistent-process robustness POCs listed under "What's next" — specifically tool-use across turns, SIGINT abort behavior, and idle-cache TTL. Don't redo any of the things listed in HANDOFF.md's "What NOT to redo" section — they're already settled.
