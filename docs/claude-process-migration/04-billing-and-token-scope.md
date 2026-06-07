# Billing is determined by token scope AND entrypoint classifier

**Date:** 2026-06-03 (initial), 2026-06-04 (corrected after VSIX spike — see banner)
**Verified via:** logging proxy at `echo:~/claude-proxy/proxy.py` (also `/tmp/claude-proxy.py`), live wire capture comparing `claude -p` vs interactive TUI driven via PTY, source read of `claude-code-source/src/services/oauth/client.ts` and `src/utils/auth.ts`, plus VSIX v2.1.162 grep in `05-vscode-extension-investigation.md`.

> ⚠️ **Correction banner (2026-06-04):** The original headline below ("billing decided entirely by token scope") was wrong. The wire capture documented in this doc *did* show a one-word User-Agent suffix difference between TUI (`(external, cli)`) and `-p` (`(external, sdk-cli)`); we dismissed it as cosmetic. It isn't. That suffix is the `CLAUDE_CODE_ENTRYPOINT` env var propagated into the User-Agent, and **it is the server-side billing-class classifier.** See the new section "The actual classifier model" at the end of this doc, and the follow-up finding in `05-vscode-extension-investigation.md`. The corrected model: **token scope determines billing *eligibility*; the entrypoint classifier determines billing *bucket within eligibility*.** Both required.

## The original headline finding (partially wrong — see banner)

~~**There is no client-side marker on `/v1/messages` that distinguishes a TUI session from a `-p` session.**~~ There IS a marker — the User-Agent suffix derived from `CLAUDE_CODE_ENTRYPOINT`. Anthropic enforces the billing/feature split via a combination of OAuth token scope (eligibility, server-side) AND entrypoint string (bucket, also server-side, read off the User-Agent header).

This was important enough to upend our earlier mental model. The migration's previous "TUI = subscription, -p = metered" framing was wrong. The correct model is:

| OAuth scope | How obtained | What it can do | Billing class |
|---|---|---|---|
| `user:inference` only | `claude setup-token` (machine-friendly, 365d expiry, env var) | `/v1/messages` only | Metered Agent SDK credit pool (post June 15) |
| Full scope (`user:profile` + `user:inference` + `user:sessions:claude_code` + `user:mcp_servers` + `user:file_upload`) | `claude auth login` (interactive browser OAuth) | `/v1/messages` + remote-control bridge + sessions + MCP + uploads | Subscription billing |

Source evidence:
- `claude-code-source/src/constants/oauth.ts:33-58` — defines `CLAUDE_AI_INFERENCE_SCOPE` and `CLAUDE_AI_OAUTH_SCOPES` (the full set)
- `claude-code-source/src/utils/auth.ts:1261-1268` — `CLAUDE_CODE_OAUTH_TOKEN` env var is **hardcoded** to `scopes: ['user:inference']` regardless of the actual token contents
- `claude-code-source/src/components/ConsoleOAuthFlow.tsx:203` — `expiresIn: mode === 'setup-token' ? 365 * 24 * 60 * 60 : undefined`
- Debug log when running `--remote-control` with our setup token (literal text from CLI's stderr):
  > *"Remote Control requires a full-scope login token. Long-lived tokens (from `claude setup-token` or CLAUDE_CODE_OAUTH_TOKEN) are limited to inference-only for security reasons. Run `claude auth login` to use Remote Control."*

## The wire-level proof

Captured `/v1/messages` requests from both modes against the same OAuth token via the logging proxy. Diff:

**Identical across `-p` and TUI:**
- `Authorization: Bearer <same token>` — byte-for-byte
- Endpoint: `POST /v1/messages`
- `x-app: cli`
- `anthropic-version: 2023-06-01`
- `anthropic-dangerous-direct-browser-access: true`
- `X-Stainless-*` SDK fingerprint headers (both modes go through the same generated TS SDK underneath)
- Model, `max_tokens`, `temperature`, `stream`, `system` block structure
- `metadata.user_id` containing identical `device_id` + `account_uuid`

**Differs:**
- `User-Agent`: `claude-cli/2.1.139 (external, sdk-cli)` (for `-p`) vs `claude-cli/2.1.139 (external, cli)` (TUI) — literally one word ~~that we dismissed as cosmetic~~ — **this IS the billing-class classifier**; see "The actual classifier model" below
- `X-Claude-Code-Session-Id` — different per session (expected, not a mode marker)
- `anthropic-beta` — slightly different beta-flag set (none of them billing-relevant)

~~That's it. The TUI does NOT have a "this is a real human session" marker. Whatever's billed differently is decided by the token scope on the auth header, not the request shape.~~ Updated read: the TUI has a "real human session" marker — it's `(external, cli)` in the User-Agent, sourced from `CLAUDE_CODE_ENTRYPOINT="cli"` in the spawn env. The `-p` invocation gets `(external, sdk-cli)` from `CLAUDE_CODE_ENTRYPOINT="sdk-cli"`. The VS Code extension gets `(external, claude-vscode)` from `CLAUDE_CODE_ENTRYPOINT="claude-vscode"` (hardcoded in `extension.pretty.js:81940`, see `05-vscode-extension-investigation.md`). The server routes billing on which entrypoint string appears in the User-Agent.

## What this implies

1. **The persistent-process migration we POC'd does not change billing class.** Stream-json IO with `CLAUDE_CODE_ENTRYPOINT=sdk-ts` (the SDK default) is the metered category post-June-15. The migration's value is purely the previously-documented gains (latency, lifecycle, proxy integration), not cost. *Still correct after the entrypoint finding — we tag honestly as `sdk-ts`/`sdk-py`.*

2. ~~**The VS Code extension is not subscription-billed because of some "official-product entitlement."**~~ *Corrected 2026-06-04:* The VS Code extension IS subscription-billed because of an "official-product entitlement" — but the entitlement is the entrypoint string `claude-vscode` hardcoded into the extension's spawn-env (`extension.pretty.js:81940`), not the OAuth token. The full-scope token gates *eligibility*; the entrypoint string gates *routing within eligibility*. A subscriber's token with entrypoint `sdk-ts` still bills against the metered pool (we believe, untested directly).

3. ~~**A bridge running `-p` cannot be made subscription-billed without acquiring a full-scope token.**~~ *Corrected 2026-06-04:* A bridge cannot be made subscription-billed *honestly*. The technical path to subscription billing is `CLAUDE_CODE_ENTRYPOINT="claude-vscode"` + full-scope token. This is **misrepresenting llm-bawt as the official VS Code extension**, which is a ToS violation cleaner than the earlier "ToS-questionable" framing, and trivially server-side-detectable (no IDE telemetry RPCs, 24/7 traffic patterns, etc.). **Don't do this.**

## On `--remote-control`

We tried it. The CLI rejects it client-side with the warning quoted above. But even if we bypassed the client check:

1. **Server enforces scope.** `/v1/environments/bridge` endpoints check the OAuth token's scope claim server-side. Setup-tokens lack `user:sessions:claude_code` and get rejected at the server, not just the CLI.
2. **Routes through Anthropic cloud.** All bridge traffic goes through `sessionIngressUrl` on Anthropic infrastructure. There is no localhost bridge variant.
3. **Trusted-device enrollment is gated on fresh `/login` < 10min.** Setup-tokens have no `/login` event to enroll against.
4. **~~Likely additional signing requirement.~~** *Update 2026-06-04, from `05-vscode-extension-investigation.md` findings:* The VS Code extension JS bundle contains **zero** request-signing code — no `signRequest`, no `HMAC`, no `crypto.sign`, no `crypto.subtle.sign`, no `webcrypto`, no `createHmac`. The extension is structurally identical to our POC: spawn the CLI subprocess with `--input-format stream-json --output-format stream-json`, read OAuth token from keychain or credentials.json, pass via env. **There is no client-side signing in the JS layer of the Claude Code stack.** Combined with the wire capture above (TUI and `-p` send byte-identical requests, no extra signing headers), Nick's recalled signing requirement is almost certainly a misremembering. If it exists at all, it lives deeper than even the official VS Code extension touches (e.g. in the bundled binary's TLS layer), which means it's not a "missing piece" we'd add to our bridge — the extension doesn't add it either, and the extension still gets subscription billing when the user has a full-scope token. **The gate is the scope, not a signature.**

So: remote-control is closed. Not "annoying to access" — closed.

## Practical implications for the migration

- **Migration goes ahead** for the previously-documented reasons (latency, lifecycle ownership, proxy integration, debuggability). Billing class is unchanged — we're in the metered Agent SDK pool either way.
- **Do not invest more time** trying to get `-p`/persistent-process onto subscription billing via OAuth tokens. The split is enforced at token issuance and there's no client-side end-run.
- **The only theoretically-unbreakable + arguably-ToS-defensible path** to subscription billing is to run the actual `claude` TUI under a PTY with the user's interactively-acquired full-scope token, and inject user prompts as keystrokes (sendkeys). This is essentially what the VS Code extension does, just with a webview instead of tmux. Trade-off: output parsing is harder (ANSI/Ink in a TUI) — `pyte` (in-memory VT100 emulator) is the right tool, not regex-stripping ANSI.

## Things proven vs. things suspected

| Claim | Status |
|---|---|
| TUI and `-p` send byte-identical `/v1/messages` request *bodies* | ✅ Proven via proxy capture (modulo User-Agent suffix, see below) |
| User-Agent suffix differs by entrypoint string (`cli` / `sdk-cli` / `claude-vscode` / `sdk-ts` / `sdk-py`) | ✅ Proven via proxy capture (TUI vs `-p`) and VSIX grep (extension hardcoded at `:81940`) |
| User-Agent entrypoint suffix IS the server-side billing-class classifier | ⚠️ Inferred from Anthropic's policy language ("interactive use in terminal OR IDE stays on subscription") + the fact that it's the only request-shape signal that varies between billing classes. Not directly proved with a live billing-pool capture, but no other candidate signal exists |
| Setup-token has only `user:inference` scope | ✅ Proven in source (`auth.ts:1266`) |
| Setup-token cannot use `--remote-control` (client check) | ✅ Proven via live `--debug api` run |
| Server-side scope enforcement on bridge endpoints | ✅ Implied by source + warning text |
| Bridge traffic routes through Anthropic cloud | ✅ Proven in source (`bridgeApi.ts` endpoints) |
| Trusted-device enrollment requires fresh login < 10min | ✅ Proven in source (`trustedDevice.ts`) |
| Anthropic uses a request signature with a private key we can't replicate | ❌ **Disproven for the JS layer** (VSIX v2.1.162 grep, see `05-vscode-extension-investigation.md`). No signing code anywhere in the extension bundle. If a signature exists at all it's inside the bundled ELF binary's network/TLS layer, but the extension would have to call into it via the same subprocess we already spawn — and we see no extra headers on the wire. Treating this as "not present" for migration-planning purposes. |
| Full-scope token from `claude auth login` switches `/v1/messages` to subscription billing | ⚠️ Theoretically true given scope claims AND with a non-SDK entrypoint string, not directly tested |

---

## The actual classifier model (added 2026-06-04)

After the VSIX spike in `05-vscode-extension-investigation.md` and a web search confirming Anthropic's policy language ("interactive Claude Code in the terminal **or IDE** stays on subscription; Agent SDK / `claude -p` / GitHub Actions / third-party agents move to the metered credit pool"), the corrected billing model is:

**Two signals, server-side, both required for subscription billing:**

| Signal | What it is | Where it's set | What it gates |
|---|---|---|---|
| **OAuth token scope** | The scope claim in the JWT bearer token on `Authorization:` | Set at token issuance — `claude setup-token` → `user:inference` only; `claude auth login` → full scope set | **Eligibility.** Setup-token can only ever bill against metered. Full-scope token is *eligible* for either pool. |
| **`CLAUDE_CODE_ENTRYPOINT`** env var | Propagated by the CLI subprocess into the `User-Agent` suffix on outbound requests | Set in the spawn env by whoever spawns the CLI. `cli` for TUI, `claude-vscode` for the official extension (hardcoded), `sdk-ts` / `sdk-py` for the Agent SDKs, `sdk-cli` for `-p`. | **Bucket within eligibility.** `cli` and `claude-vscode` route to subscription. `sdk-*` route to metered. |

**Decision matrix:**

| Scope | Entrypoint | Result |
|---|---|---|
| Full-scope | `cli` (real TUI) | Subscription ✅ |
| Full-scope | `claude-vscode` (extension) | Subscription ✅ |
| Full-scope | `sdk-ts` / `sdk-py` (honest SDK use) | Metered pool |
| Full-scope | `sdk-cli` (`claude -p`) | Metered pool |
| Setup-token | any | Metered pool (scope can't reach subscription) |

**Why this matters for enforcement:** there's no cryptographic gate. The June 15 policy is enforced honor-system at the wire level. Anthropic distinguishes their wrappers from third-party callers by **giving their wrappers unique entrypoint strings their server routes on, then trusting client code to tag itself honestly.** The official wrappers cooperate (extension hardcodes `claude-vscode`; TUI sets `cli`; SDKs default to `sdk-ts`/`sdk-py`). The system works because subverting it requires actively misrepresenting which product is making the call — a much clearer ToS violation than ambiguous use of a personal token.

**Why we don't subvert it:**

1. **It's clean ToS violation** — misrepresenting llm-bawt as the official VS Code extension to dodge metered billing. Different category from "I'm overusing my seat."
2. **Trivially detectable** — Anthropic logs per-entrypoint traffic shapes. A "VS Code" client with 24/7 uptime, no IDE telemetry RPCs, no editor-mapping MCP traffic, multi-bot fan-out — won't look like any real VS Code session in their analytics. Detection probably happens automatically, not manually.
3. **Token logistics** — the full-scope token from `claude auth login` is short-lived (hours/days); re-acquisition needs a human at a browser; not a stable autonomous-bot foundation.
4. **It's brittle** — Anthropic can change the classifier scheme (e.g. add a signed nonce only the official extension knows how to produce) at any time. Today the gate is honor-system; tomorrow it can be cryptographic; we'd be one release away from being locked out anyway.

**What we do instead:** stay honest. Tag as `sdk-ts` (or `sdk-py` once we move the bridge there). Accept the metered Agent SDK pool. Use the June 15 credit. Plan budgets accordingly. The migration in `01-migration-plan.md` proceeds for its real reasons — latency, lifecycle, proxy integration — and the billing class is just what it is.
