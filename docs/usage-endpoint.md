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

Support level differs by adapter. The canonical response shape is the same for
all of them.

## Caching

Successful snapshots are cached for `LLM_BAWT_USAGE_CACHE_TTL` seconds
(default `120`). On refresh errors or upstream `429`s, the endpoint returns the
last good cached snapshot when it has one.

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

## Relevant environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDE_USAGE_CREDENTIALS_PATH` | `~/.config/llm-bawt/claude-usage-credentials.json` | Claude usage bundle path |
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
| `unauthorized` | Missing bundle or wrong scope |
| `rate_limited` | Upstream usage endpoint returned `429` |
| `error` | Network or unexpected upstream failure |
| `not_implemented` | Placeholder adapter rather than a live implementation |
