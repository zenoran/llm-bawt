# Subscription Usage Endpoint

`GET /v1/usage` exposes a provider-pluggable view of subscription or plan
usage. The route is in
[src/llm_bawt/service/routes/usage.py](../src/llm_bawt/service/routes/usage.py)
and the adapter registry is in
[src/llm_bawt/service/usage/](../src/llm_bawt/service/usage).

## API

| Request | Meaning |
|---|---|
| `GET /v1/usage` | All registered providers |
| `GET /v1/usage?provider=claude` | One provider |
| `GET /v1/usage?bot_id=<slug>` | Resolve provider from a bot's model |
| `...&force=true` | Bypass cache |

## Registered providers

The registry currently includes:

- `claude`
- `zai`
- `openai_chatgpt`
- `xai` (API-key only — no plan limits; used so Grok bots don't fall back to Claude)

Support level differs by adapter. The canonical response shape is the same for
all of them. API-key providers like `xai` return `status=ok` with an empty
`limits` list so the chat context popup shows turn tokens/cost only.

## Caching

Successful snapshots are cached for `LLM_BAWT_USAGE_CACHE_TTL` seconds
(default `120`). On refresh errors or upstream `429`s, the endpoint returns the
last good cached snapshot when it has one.

## ChatGPT / Codex usage

`openai_chatgpt` fetches current subscription limits from the same dedicated
endpoint used by the official Codex client:

`GET https://chatgpt.com/backend-api/wham/usage`

This is a non-inference request; it does not select a model or create a hidden
chat turn. A `401` triggers one bounded force-refresh through the app-owned
OAuth bundle, then one retry. Real Codex `/responses` calls also expose quota
headers; the claude-code bridge saves those passively in Redis as a fallback.
If the dedicated endpoint fails and only an old passive snapshot is available,
the adapter returns `status=usage_stale` with `cached=true`. This means the
quota values are old, not that the credential is broken. Credential expiry and
refresh-chain health remain authoritative under `GET /v1/providers/health`.

## Claude credential model

Claude usage is separate from Claude inference.

- The bridge inference token (`CLAUDE_CODE_OAUTH_TOKEN`) is not sufficient for
  `/api/oauth/usage`.
- The usage adapter needs a `claude login` style OAuth bundle with
  `user:profile` scope.

Credential handling is implemented in
[src/llm_bawt/service/usage/claude_oauth.py](../src/llm_bawt/service/usage/claude_oauth.py).

### Modes

- `shared`: read-only reuse of an existing login bundle; no refresh/write
- `owned`: the app refreshes and rewrites its own dedicated bundle

In the main compose stack, the app is configured for `owned` mode and uses:

`/root/.config/llm-bawt/claude-usage/.credentials.json`

### TASK-635: this is now THE Claude credential (single login)

The owned bundle is no longer usage-only — it is the deployment's ONE Claude
credential. The `claude` provider adapter's wizard login mints a full-scope
bundle (`user:inference` + `user:profile` + …); the app is the sole refresher
(serialized + a proactive lifespan loop that refreshes at `expiresAt − 20min`,
so it never lapses even when idle). Consumers are read-only:

- the usage adapter (same process),
- the claude-code bridge, via a read-only compose mount
  (`CLAUDE_CREDENTIALS_PATH`) with `GET /v1/providers/claude/token` as its
  stale-file/force fallback (the app refreshes on demand; optional
  `BRIDGE_CLAUDE_TOKEN_SECRET` guards the endpoint via `X-Bridge-Token`).

The bridge never refreshes — the old dual-login (`claude-sub` + `claude-usage`)
and its refresh-rotation race are gone.

## Relevant environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDE_CREDENTIALS_PATH` | unset | Preferred bundle-path override (TASK-635 name) |
| `CLAUDE_USAGE_CREDENTIALS_PATH` | `~/.config/llm-bawt/claude-usage-credentials.json` | Legacy bundle-path env (still honored) |
| `CLAUDE_USAGE_CREDENTIALS_MODE` | `shared` | `shared` or `owned` |
| `LLM_BAWT_USAGE_CACHE_TTL` | `120` | Cache TTL in seconds |
| `ZAI_API_KEY` | unset | Required for live z.ai usage |

## Quick check

```bash
curl -s 'http://localhost:8642/v1/usage?provider=claude' | jq
curl -s 'http://localhost:8642/v1/usage' | jq '.providers[] | {provider,status,available}'
```

## Common Claude failures

| Status | Meaning |
|---|---|
| `stale` | Shared credential exists but its access token expired |
| `usage_stale` | Cached quota values are shown because a live usage refresh failed; credential health is separate |
| `unauthorized` | Missing bundle or wrong scope |
| `rate_limited` | Upstream usage endpoint returned `429` |
| `error` | Network or unexpected upstream failure |
| `not_implemented` | Placeholder adapter rather than a live implementation |
