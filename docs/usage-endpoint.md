# Subscription usage endpoint (`/v1/usage`)

Canonical, provider-pluggable view of **subscription/plan usage** (the
Claude `/usage`-style "5-hour limit / weekly" numbers) so the UI can show them
without anyone visiting claude.ai. Built to support multiple providers; only
**Claude** is implemented today (z.ai and OpenAI/codex are registered stubs).

- Code: `src/llm_bawt/service/usage/` (canonical model + adapter registry) and
  `src/llm_bawt/service/routes/usage.py` (the HTTP route).
- Lives in the **app** (not a bridge): the usage credential is independent of
  the bridge's inference token, so there's nothing to gain from RPC-ing the
  bridge, and we avoid a bridge restart.

## API

| Request | Returns |
|---|---|
| `GET /v1/usage` | All registered providers (the "across all subscriptions" view) |
| `GET /v1/usage?provider=claude` | One provider (one cached upstream call) |
| `GET /v1/usage?bot_id=<slug>` | Resolves the bot's provider, returns that provider's usage |
| `…&force=true` | Bypass the cache (rate-limited; use sparingly) |

Each provider returns a canonical `ProviderUsage`: `provider`, `display_name`,
`status`, `available`, `limits[]` (`{id,label,used_pct,resets_at,window}`),
and `raw` (the provider-native payload, kept for debugging / mapper-tuning).
Successful snapshots are cached `LLM_BAWT_USAGE_CACHE_TTL` seconds (default
120); on a 429 or fetch error the last-good snapshot is served `cached=true`.

## The credential model — READ THIS

There are **two different OAuth grants on the same Anthropic account**, and
they are NOT interchangeable:

| | Inference (the bridge) | Usage (this endpoint) |
|---|---|---|
| Credential | `sk-ant-oat0…` **setup-token** | `claudeAiOauth` **login bundle** |
| Minted by | `claude setup-token` (once) | interactive `claude login` (subscription) |
| Scope | `user:inference` only | …+ **`user:profile`** |
| Source | `.env` → `CLAUDE_CODE_OAUTH_TOKEN` | `~/.claude/.credentials.json` |
| Lifecycle | static, ~1yr, no refresh | 8h access + rotating refresh |

`/api/oauth/usage` **requires `user:profile`**. The bridge's setup-token does
not have it (and `setup-token` has no flag to request it), so the inference
token returns **403 `OAuth token does not meet scope requirement
user:profile`**. The usage endpoint therefore needs a *login bundle*, not the
bridge's token.

## How it's configured (default: a dedicated login the app owns)

The app uses its **own** Claude login bundle, separate from anything else, and
refreshes it itself. This never goes stale and never touches your interactive
Claude Code (TUI) login. Configured in `docker-compose.yml` on the `app`
service (the path sits under the already-RW `~/.config/llm-bawt` mount):

```yaml
environment:
  - CLAUDE_USAGE_CREDENTIALS_PATH=/root/.config/llm-bawt/claude-usage/.credentials.json
  - CLAUDE_USAGE_CREDENTIALS_MODE=owned
```

**One-time setup — mint an ISOLATED login (does not disturb your normal
`~/.claude` login):**

```bash
# On echo. HOME override gives this login its own credential dir so it doesn't
# overwrite your interactive ~/.claude login.
mkdir -p /tmp/usage-login
HOME=/tmp/usage-login claude login        # browser PKCE; paste code if prompted

# Move the resulting bundle to the path the app reads (under the RW config mount):
mkdir -p ~/.config/llm-bawt/claude-usage
cp /tmp/usage-login/.claude/.credentials.json ~/.config/llm-bawt/claude-usage/.credentials.json
rm -rf /tmp/usage-login
```

From here the app refreshes that bundle on its own (the refresh token is
long-lived and rotates in place; even after days idle the next fetch mints a
fresh access token). **Treat this grant as owned by the app** — don't reuse it
interactively elsewhere, or the two will rotate the refresh token out from
under each other.

After changing compose/env: `docker compose up -d --build app` (rebuilds the
`app` service only — safe; does NOT touch the bridges).

> Alternative — `mode=shared` (read-only reuse, no dedicated login): point
> `CLAUDE_USAGE_CREDENTIALS_PATH` at an existing login bundle you DON'T want the
> app to write (e.g. your TUI's `~/.claude/.credentials.json`, mounted `:ro`)
> and set `CLAUDE_USAGE_CREDENTIALS_MODE=shared`. The app then only *reads* it
> and never refreshes — so it relies on that bundle's real owner (your TUI) to
> keep the access token fresh, and reports `stale` if the owner is idle >8h.
> **Caveat:** mount the credential's *directory*, not the single file — the
> owner refreshes via atomic rename, and a single-file bind mount pins the old
> inode and reads stale forever (same class of bug as the codex `auth.json`
> EBUSY note). Use this only if you accept the idle-staleness; `owned` is the
> set-and-forget option.

## Verify

```bash
curl -s 'localhost:8642/v1/usage?provider=claude' | jq        # one provider
curl -s 'localhost:8642/v1/usage' | jq '.providers[]|{provider,status}'
```

A healthy Claude response has `status:"ok"`, `available:true`, and a populated
`limits[]`. The `raw` field carries the provider-native payload — the first
live `ok` response is the moment to confirm field names and tighten
`_to_canonical` in `adapters/claude.py`.

## Troubleshooting — "it mysteriously stopped working"

Always start with: `curl -s 'localhost:8642/v1/usage?provider=claude' | jq
'{status,error}'`. The `status` tells you which case you're in.

| `status` | Cause | Fix |
|---|---|---|
| `stale` | **owned mode:** the app's own refresh failed — the refresh token expired or was revoked (rare; e.g. you reused this login interactively elsewhere and rotated it away). **shared mode:** the bundle's real owner has been idle >8h. | owned: re-run the one-time isolated `claude login` (see "How it's configured"). shared: use Claude Code once to refresh it, or switch to `mode=owned`. |
| `unauthorized` + "No Claude usage credential found at …" | The mount is missing/empty or the path is wrong (file not present in the container). | Confirm the compose volume `${HOME}/.claude:/root/.claude-host:ro` exists and the host has `~/.claude/.credentials.json`. `docker exec llm-bawt-app ls -l /root/.claude-host/.credentials.json`. |
| `unauthorized` + "lacks user:profile scope" (HTTP 403) | The mounted credential is NOT a subscription login bundle (e.g. it's a setup-token, or a login that didn't grant `user:profile`). | Ensure the file is an interactive `claude login` bundle. Check scopes: it must include `user:profile`. Re-run `claude login` on the host if needed. |
| `rate_limited` | `/api/oauth/usage` returned 429 (it is aggressively rate-limited). | Nothing — the cache serves the last-good snapshot and backs off. Raise `LLM_BAWT_USAGE_CACHE_TTL` if it recurs; avoid `force=true`. |
| `error` | Network/upstream error reaching `api.anthropic.com`. | Check app egress + `error` detail; last-good snapshot is served if available. |
| `not_implemented` | A stub provider (z.ai, openai_chatgpt). | Expected until those adapters are implemented. |

Common "it broke" root causes, concretely:
- **(owned)** You **reused the dedicated usage login interactively** somewhere
  else → it rotated the refresh token out from under the app. Re-run the
  isolated `claude login` and replace the bundle.
- **(owned)** The dedicated bundle file is **missing/empty** after a host move
  or cleanup → re-run the one-time setup.
- **(shared)** You moved interactive Claude Code usage to **another machine**,
  so the shared bundle went idle/stale → use the owner once, or switch to
  `owned`.
- **(shared)** Someone **bind-mounted the single file** instead of the
  directory → token freezes after the first refresh. Mount the dir.
- The **bridge's** inference token is unrelated to this — don't "fix" usage by
  touching `CLAUDE_CODE_OAUTH_TOKEN`; that's the setup-token and it can't see
  usage by design.

## Env-var reference

| Var | Default | Purpose |
|---|---|---|
| `CLAUDE_USAGE_CREDENTIALS_PATH` | `~/.config/llm-bawt/claude-usage-credentials.json` | The login bundle file the Claude adapter reads. Set to the mounted host path for shared reuse. |
| `CLAUDE_USAGE_CREDENTIALS_MODE` | `shared` | `shared` = read-only (never refresh/write — safe to share with your TUI). `owned` = refresh + rewrite (dedicated credential only). |
| `LLM_BAWT_USAGE_CACHE_TTL` | `120` | Seconds to cache a successful snapshot per provider. |
