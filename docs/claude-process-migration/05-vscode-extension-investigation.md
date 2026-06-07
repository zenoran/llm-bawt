# VS Code extension investigation — research plan + findings

**Date:** 2026-06-03 (plan), 2026-06-04 (findings)
**Status:** ✅ done — see Findings section at bottom
**Why this matters:** The VS Code extension is the only "official" wrapper we can read end-to-end. If there's a signing helper, custom auth handshake, or anything beyond `spawn claude-bundled-binary + render webview`, this is where it'd be visible — without having to reverse-engineer the bundled ELF binary directly.

> **TL;DR of the findings:** No extension-side signing. No extension-side auth magic. No xterm/PTY. The extension JS *is* the Agent SDK (`CLAUDE_CODE_ENTRYPOINT="sdk-ts"`), and spawns the bundled `cli.js` under `node`/`bun` with `--input-format stream-json --output-format stream-json --verbose` — structurally identical to our POC. Token comes from `~/.claude/.credentials.json` / OS keychain / env. **H1 + H2 confirmed. H4 + H5 killed.**
>
> **Implication:** there is no client-side signing anywhere in the JS layer of the Claude Code stack. Combined with the wire capture in `04-billing-and-token-scope.md` (TUI and `-p` send byte-identical requests, no extra signing headers in either), Nick's recalled "signature requirement" is almost certainly a misremembering — or, if it exists, it's deeper than anywhere a normal autonomous-bot wrapper would replicate (e.g. inside the bundled binary's TLS layer). Either way: not actionable, not a missing puzzle piece. The architectural picture is complete.

## What we know already

From `claude-code-source/src/services/mcp/client.ts:2148-2151` (a comment, not the extension's code, but it tells us how the extension is architected):

> *"This is necessary when another process (e.g. the VS Code extension host) has modified stored tokens (cleared auth, saved new OAuth tokens) and then asks the CLI subprocess to reconnect."*

So the extension:
1. Owns OAuth token storage (writes to keychain / Linux secret service)
2. Spawns the `claude` CLI binary as a **child subprocess**
3. Communicates with it bidirectionally (almost certainly stream-json over stdin/stdout)
4. Renders a webview that's either an xterm.js terminal showing the Ink TUI, or a custom React chat panel that parses stream-json events

The source code we have explicitly handles xterm.js as a host context (`PromptInputFooterLeftSide.tsx`, `ScrollKeybindingHandler.tsx`), confirming the extension uses xterm.js when in TUI-rendering mode.

What we DON'T know — and want to find out:
- Does the extension call any helper to **sign** outbound requests, or does it leave that to the bundled binary?
- Is there a special auth handshake the extension does that we'd need to replicate?
- Does the extension use `-p` stream-json (like our POC) or run the TUI in xterm.js (the user-types-keystrokes model)?
- Are there any environment variables, headers, or flags the extension sets that we don't?

## How to investigate (the plan)

### Step 1: Get the VSIX
The extension is named "Claude Code" by Anthropic on the VS Code Marketplace. Two ways:
```bash
# From the marketplace directly via curl:
EXT="anthropic.claude-code"
VER="$(curl -s "https://marketplace.visualstudio.com/items?itemName=${EXT}" | grep -oE 'Version[^<]*<[^>]*>[0-9.]+' | head -1 | grep -oE '[0-9.]+$')"
curl -L -o /tmp/claude-vsix.zip \
  "https://marketplace.visualstudio.com/_apis/public/gallery/publishers/anthropic/vsextensions/claude-code/${VER}/vspackage"

# Or via the vsce CLI / Open VSX mirror if marketplace gives trouble.
```

VSIX is just a renamed zip:
```bash
mkdir -p /tmp/claude-ext && unzip -q /tmp/claude-vsix.zip -d /tmp/claude-ext
ls /tmp/claude-ext/extension/
```

### Step 2: Find the entry point
`package.json` declares the entry. Typically:
```bash
jq '.main, .activationEvents' /tmp/claude-ext/extension/package.json
```
Look for `dist/extension.js` or similar.

### Step 3: Grep for the interesting things

The extension JS is probably minified but symbols should be readable. Targets:

| Pattern | What it tells us |
|---|---|
| `signRequest\|signHeader\|HMAC\|crypto\.sign\|jose\.sign` | Is there client-side request signing? |
| `claude auth login\|setup-token\|browser.*flow\|deviceCodeFlow` | What auth flow does it use? |
| `spawn.*claude\|execFile.*claude\|child_process.*claude` | Confirms subprocess model |
| `--input-format\|--output-format\|stream-json\|--ide` | What flags it passes |
| `x-trusted-device\|trusted_device` | Does the extension enroll trusted devices? |
| `bridgeApi\|sessionIngressUrl\|environments/bridge` | Does the extension use remote-control mechanisms? |
| `keytar\|secret-storage\|SecretStorage` | How it stashes the token |
| `oauth-2025-04-20\|anthropic-beta` | What beta headers it sends (does it match our captures?) |
| `xterm\.js\|TerminalRenderer\|pty\|node-pty` | TUI rendering mode |
| `webview\|postMessage\|onDidReceiveMessage` | Custom chat panel mode |

### Step 4: Read the auth path
Once we find the spawn-claude call site, trace backwards:
1. What env does it pass to the subprocess?
2. Does it use `CLAUDE_CODE_OAUTH_TOKEN` or rely on the keychain?
3. What CLI flags does it pass — is `--ide` set? `--print`? Stream-json?

### Step 5: Check the auth flow specifically
The login flow is most informative. The CLI source uses `buildAuthUrl` (in `oauth/client.ts`) with `inferenceOnly: mode === 'setup-token'`. The extension presumably uses the same with `inferenceOnly: false` (i.e., full scope). Confirming this in the extension code answers definitively whether the extension's "edge" is just "user did a browser login" or whether there's something else.

### Step 6: If we find a `signRequest`-style helper
Trace the key source. Three possibilities:
- **Hardcoded key in extension JS** — extractable, but they'd rotate it and we'd have an arms race
- **Reads from disk file Anthropic ships in the VSIX** — extractable but same problem
- **Calls into the bundled `claude` ELF binary** — closed door, we'd need binary RE

If it's path 3, the extension can't help us further; we'd be back to the binary.

## Hypotheses we want to confirm or kill

| Hypothesis | What confirms / kills it |
|---|---|
| H1: Extension is just "spawn `claude` with default flags, let the binary handle auth via keychain" — no extension-side magic | We see a clean `child_process.spawn(claudeBinary, [...defaultFlags], { env: { ...process.env } })` and no auth code in the extension JS |
| H2: Extension uses `-p --input-format stream-json` (same as our POC) and parses events into a webview | We see those flags + a JSONL parser + `postMessage` to a webview |
| H3: Extension runs the TUI in an xterm.js webview and forwards keystrokes | We see node-pty + xterm.js + a keystroke forwarder, no JSONL parser |
| H4: Extension has its own request-signing helper | We find a `signRequest` or `HMAC` or `crypto.subtle.sign` call site on the auth/API path |
| H5: Extension does device attestation / hardware-bound key | We find references to `webcrypto`, `SubtleCrypto.generateKey`, or a non-extractable key in keychain |

H1 + H2 (or H1 + H3) together would mean: "the extension is just our POC + a UI, the only secret sauce is the user's full-scope token in the keychain." That's the **most likely outcome** based on what we've seen so far.

H4 or H5 would mean: there's a real client-side cryptographic gate. Which would match Nick's recalled-but-not-yet-re-verified finding about a signature requirement.

## Time budget

This is a **20-minute spike**. If we don't have the answer to H1 vs H4 within 20 minutes of reading the extension JS, we should stop and reconsider. The extension is probably <500KB of JS; with grep and a small amount of code-reading, the key question (is there signing?) should resolve quickly.

## What to do after the investigation

If H1+H2/H3 confirmed (no extension-side magic):
- The "wrap claude in our own bridge" path is fully understood. The persistent-process migration we POC'd is structurally identical to what the extension does.
- The only path to subscription billing is a user-acquired full-scope token + sendkeys-to-TUI under PTY. Plan around `pyte` for output parsing.

If H4 confirmed (extension-side signing):
- That signing helper IS the gate. We document it as the closed door.
- For practical purposes, this just makes the "give up on subscription billing for autonomous bots" conclusion more firmly grounded.

If H5 confirmed (hardware attestation):
- Same as H4 but even more closed.

Either way, the migration plan in `01-migration-plan.md` doesn't change. This investigation is for **completing the picture and stopping further speculation**, not for unlocking a new path.

## Files to update after the investigation

- This file: add a "Findings" section
- `README.md`: link this file's outcome from the index
- `04-billing-and-token-scope.md`: replace the "⚠️ recalled" line in the proven-vs-suspected table with the actual finding

---

## Findings (2026-06-04)

### What we did

Pulled VSIX `anthropic.claude-code` **v2.1.162** from the VS Code Marketplace, unzipped, ran the targeted greps from the plan against `extension/extension.js` (a 2.1 MB minified bundle, 75 MB VSIX once the bundled `claude.exe` Windows binary and audio-capture component are included).

Artifacts (on the bridge host):
- `/tmp/claude-vsix-spike/claude.vsix` — raw VSIX (un-gzipped from marketplace response)
- `/tmp/claude-vsix-spike/ext/extension/extension.js` — original minified bundle
- `/tmp/claude-vsix-spike/ext/extension/extension.pretty.js` — re-newlined for grep (`s/[;{}]/&\n/g`)
- `/tmp/claude-vsix-spike/ext/extension/webview/index.js` — 4.6 MB React webview bundle (not deeply read; it's the UI, not the auth path)

### Verdict on each hypothesis

| Hypothesis | Status | Evidence |
|---|---|---|
| **H1** — extension is "spawn `claude` with default flags, no extension-side auth magic" | ✅ **Confirmed** | Subprocess spawn at `extension.pretty.js:42068` (`spawnLocalProcess`) and `:42179` (command resolution). Env is passed through from `process.env` (`:42090`). No `vscode.authentication` API usage. No `authentication` contribution in `package.json`. |
| **H2** — extension uses `-p --input-format stream-json` and parses events into a webview | ✅ **Confirmed** | `extension.pretty.js:42101`: `let l=["--output-format","stream-json","--verbose","--input-format","stream-json"]`. Subprocess output is consumed as an async iterator and dispatched to the webview as `io_message` events (`:64535`). |
| **H3** — extension runs the TUI in an xterm.js webview and forwards keystrokes | ❌ **Killed** | Zero matches for `node-pty`, `xterm.js`, `TerminalRenderer`, `pty.spawn`. The lone `xterm` hit at `:5813` is a TERM-env regex unrelated to rendering. |
| **H4** — extension has its own request-signing helper | ❌ **Killed** | Zero matches for `signRequest`, `signHeader`, `HMAC`, `crypto.sign`, `jose.sign`, `crypto.subtle.sign`, `createHmac`, `createSign`, `createPrivateKey`, `generateKey`, `JWK`, `signJwt`, `sign('RS\|ES\|HS...')`. |
| **H5** — extension does device attestation / hardware-bound key | ❌ **Killed** | Zero matches for `webcrypto`, `SubtleCrypto`. No `generateKey`. No keychain-stored non-extractable key references. |

### What the extension actually is

The 2.1 MB `extension.js` bundle is **the `@anthropic-ai/claude-code` Agent SDK** (TypeScript variant), embedded directly into the extension. Smoking gun:

```js
// extension.pretty.js:42176
if(!H.CLAUDE_CODE_ENTRYPOINT)H.CLAUDE_CODE_ENTRYPOINT="sdk-ts";
```

That env var (`CLAUDE_CODE_ENTRYPOINT=sdk-ts`) is set on every subprocess the extension launches. It's the same string the metering server uses to classify Agent SDK traffic. **The VS Code extension's runtime IS Agent SDK traffic, by Anthropic's own classification.**

The subprocess launch (`:42179`):

```js
let D0=QW0(O),                              // O = pathToClaudeCodeExecutable
    g=D0?O:K,                                // K = "node" or "bun"
    m=D0?[...x,...l]:[...x,O,...l],          // l = stream-json flag array
    c={command:g, args:m, cwd:N, env:H, signal:...};
```

— if the configured `pathToClaudeCodeExecutable` is itself an executable (compiled), spawn it directly; otherwise interpret it under `node`/`bun` with the same flag list. Either way, the spawned process is `claude`'s CLI, configured for stream-json IO on both directions, with `--verbose` to get full events.

The webview (`webview/index.js`, 4.6 MB React bundle) consumes those events via `postMessage` and renders the chat UI. No auth code in the webview — it can't see the OAuth token, it just renders what the subprocess emits.

### Where the OAuth token actually comes from

Standard CLI-style loader, no extension-specific path:

1. **macOS keychain** via shell-out to `security find-generic-password -a <user> -w -s claude.code-credentials` (`:27608` and `:28179`)
2. **`~/.claude/.credentials.json`** file (`:27764` defines `storagePath: <storageDir>/.credentials.json`; `:59750` reads it during session restore)
3. **Env vars** `ANTHROPIC_API_KEY` / `CLAUDE_CODE_OAUTH_TOKEN` as fallback (`:59753`)
4. On Linux/Windows, falls through to the credentials.json file

Whichever source wins gets passed into the subprocess env. **The extension does not call `vscode.authentication.getSession` or register any auth provider** — `package.json`'s `contributes` has no `authentication` key, and `extension.pretty.js` has no `registerAuthenticationProvider` / `authentication.getSession` against an Anthropic provider.

### `remoteControlState` references — what they actually are

`extension.pretty.js:64506`, `:65002`, `:65606` etc. reference a `remoteControlState` object — but this is **UI state**, not extension-side auth. The extension consumes `bridge_state` events from the subprocess and stores their `state: 'connected'|'disconnected'|'error'` on the channel object so the UI can show a status badge. The actual remote-control protocol runs inside the spawned CLI subprocess, gated by its own scope check — same gate that rejects our setup-token. The extension doesn't enable, sign, or unlock anything on the auth path.

The one CLI-side hint of remote-control control: `enableRemoteControl(z, V)` at `:42820` is a method on the SDK's `Query` class that sends a control message *to the running subprocess* to flip the flag at runtime. Still no flag passed at spawn time, still no auth bypass.

### What headers does the extension send?

The bundled Anthropic TypeScript SDK code (`:8656` onwards) declares the same OAuth scheme as the CLI:

```js
ez="oauth-2025-04-20"   // anthropic-beta value
fq="/v1/oauth/token"
Ql="urn:ietf:params:oauth:grant-type:jwt-bearer"
hl="refresh_token"
```

And the same `anthropic-beta` values we saw in `04-billing-and-token-scope.md`'s wire capture: `managed-agents-2026-04-01`, `files-api-2025-04-14`, `user-profiles-2026-03-24`. Nothing exotic. Nothing client-signed. The `Authorization: Bearer <token>` is the only credential on outbound `/v1/messages` requests, and the token's scope claim is the only thing that decides billing class — exactly as we proved at the wire level previously.

### Implications for the migration

**Architectural picture is now complete.** What the official VS Code extension does:

1. Owns OAuth token storage (delegates to OS keychain or credentials.json — same as CLI)
2. Spawns `claude` as a child subprocess with stream-json IO on both directions
3. Renders a React webview that consumes the JSONL event stream

That's **structurally identical** to our POC (`poc_persistent_claude.py`) — minus the React UI. The Agent SDK's "VS Code experience" is just a webview wrapped around the same thing we already have working. There is no hidden capability we'd unlock by using the extension instead.

**Therefore:**
- The persistent-process migration in `01-migration-plan.md` is structurally sound — it mirrors what Anthropic themselves do in their official VS Code wrapper.
- There is no realistic path to subscription billing through the extension model. The extension is subscription-billed only when the user has done `claude auth login` and the resulting full-scope token is on disk. We can't acquire that token autonomously.
- The "request-signing requirement" Nick recalled is **not present** in the extension JS. Either misremembered, or it lives inside the bundled `claude` ELF binary's TLS / network layer at a depth that even the official VS Code extension doesn't touch. Either way: not a door we can open, not a door the extension opens, not a missing piece in our understanding.

### Stop conditions met

- ✅ Answered H1 vs H4 within the 20-minute time budget
- ✅ All five hypotheses resolved
- ✅ No new mysteries surfaced — every grep that hit explained itself once read in context
- ✅ Nothing here changes the migration plan; it just confirms the plan is sound

---

## Follow-up finding (2026-06-04, same day): the entrypoint classifier

A web search for "is the VS Code extension included in the June 15 metering policy" surfaced Anthropic's official position: **interactive use of Claude Code in the terminal OR the IDE stays on the subscription**; only programmatic use (Agent SDK, `claude -p`, GitHub Actions, third-party agents) moves to the new metered credit pool. So the popular wisdom we'd assumed was actually correct — the extension IS subscription-billed.

That forced the question: *how* does the server distinguish IDE traffic from generic SDK traffic when (per our spike above) the extension uses the same stream-json IO that `-p` uses? Re-grepping the VSIX with the right keywords found it.

### The mechanism

`extension.pretty.js:81933-81940` — the helper that builds the subprocess env for every `claude` CLI spawn:

```js
function RV(z) {
  let V = _V0(_1("environmentVariables")),
      B = { ...process.env };
  if (z) B.PATH = z;
  B.MCP_CONNECTION_NONBLOCKING = "true";
  B.CLAUDE_CODE_ENABLE_TASKS = "0";
  for (let N of V) if (N.name) B[N.name] = N.value || "";
  return B.CLAUDE_CODE_ENTRYPOINT = "claude-vscode", B;   // <-- the classifier
}
```

That last line hardcodes `CLAUDE_CODE_ENTRYPOINT="claude-vscode"` as the **final write** to the env block — so user-configured environment variables can't override it. Every spawn of the CLI by the extension is tagged `claude-vscode`.

Compare to the other code paths:

| Spawn site | `CLAUDE_CODE_ENTRYPOINT` value | Where set |
|---|---|---|
| VS Code extension | `claude-vscode` | `extension.pretty.js:81940` (hardcoded, last write) |
| Agent SDK (TS) when used as library | `sdk-ts` | `extension.pretty.js:42176` and `:59840` (default-if-unset) |
| Agent SDK (Py) | `sdk-py` | (inferred from naming convention) |
| `claude -p` invoked from CLI | `sdk-cli` | (inferred from User-Agent capture in `04`) |
| Real TUI in terminal | `cli` | (inferred from User-Agent capture in `04`) |

The CLI subprocess reads `CLAUDE_CODE_ENTRYPOINT` and **propagates it into the `User-Agent` suffix** on `/v1/messages` — which is what we wire-captured in `04` and dismissed as "one word of cosmetic difference":

> `User-Agent: claude-cli/2.1.139 (external, sdk-cli)` (for `-p`)
> vs `claude-cli/2.1.139 (external, cli)` (TUI)

That one word IS the billing-class signal. The server-side billing router buckets requests by which entrypoint string appears in the User-Agent.

### What this changes in the picture

The earlier framing — "billing is determined entirely by OAuth token scope" — was incomplete. The actual model is **two signals working together:**

1. **OAuth token scope** determines what billing categories the token is *eligible* for. Setup-token (`user:inference` only) can only ever go to the metered pool. Full-scope token (from `claude auth login`) is eligible for either pool.
2. **`CLAUDE_CODE_ENTRYPOINT`** (propagated into `User-Agent`) determines which bucket within the eligible categories. `cli` and `claude-vscode` → subscription. `sdk-ts`/`sdk-py`/`sdk-cli` → metered Agent SDK pool.

Both signals are required for subscription billing. A full-scope token tagged `sdk-ts` still hits the metered pool (the entrypoint says "I'm SDK"). A setup-token tagged `claude-vscode` probably gets rejected or downgraded (the scope can't reach subscription regardless of what the entrypoint claims) — not tested, but consistent with the scope-enforcement pattern from `04`.

### What this means for enforcement

The June 15 policy enforcement is **honor-system at the wire level.** No cryptography. No signing. No hardware attestation. Anthropic distinguishes their own wrappers from third-party callers by giving each wrapper a unique entrypoint string that the server routes on. The official wrappers cooperate by tagging themselves truthfully (`cli`, `claude-vscode`). Third-party SDK consumers tag themselves with `sdk-*` and route to metered.

This is technically trivial to subvert: `env["CLAUDE_CODE_ENTRYPOINT"] = "claude-vscode"` on any spawn produces wire output indistinguishable from the official VS Code extension. Combined with a `claude auth login` full-scope token, third-party traffic would land in the subscription pool. But:

1. **It's misrepresenting one Anthropic product as another to evade billing.** Cleaner ToS violation than "use your full-scope token autonomously."
2. **It's trivially detectable.** Anthropic logs per-entrypoint traffic shapes server-side. A "VS Code" client with 24/7 uptime, no IDE telemetry RPCs, no editor-mapping MCP traffic, multi-bot fan-out patterns — will not look like any real VS Code session in their metrics.
3. **The token has to come from somewhere.** Acquiring a full-scope token requires interactive browser login on the bridge host; it expires; re-acquisition needs a human at a browser. Not a stable autonomous-bot foundation.

### Updated conclusion

The migration plan in `01-migration-plan.md` is unchanged. Our bridge stays on the metered Agent SDK pool (entrypoint `sdk-ts` or `sdk-py`, whichever truthfully describes the spawn site). The June 15 credit applies. We do not spoof the entrypoint.

The architectural picture is now genuinely complete: we know what classifies billing (entrypoint + scope), why the extension stays on subscription (it's honor-system and the extension is honest), how Anthropic enforces the new policy without breaking the extension (the extension *cooperates* with the policy), and what the only path to subscription billing for a third-party client would cost (ToS-violating impersonation, brittle token, trivial detection).
