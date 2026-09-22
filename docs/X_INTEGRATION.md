# X public-post search (TASK-900)

## Connect

In BawtHub provider accounts (also the setup wizard's Providers step), connect
**X (Twitter)** with an app-only Bearer Token from <https://console.x.com/>.
This is optional and separate from xAI/Grok; it cannot back the first bot.
Fund API credits and set a spending limit in X's console.

The existing `POST /v1/providers/x/connect/api-key` flow validates the token
against `GET https://api.x.com/2/usage/tweets`. A successful check proves usage
endpoint access, not search entitlement or sufficient credits. Failed validation
never replaces an existing credential.

Storage is the existing `CredentialStore`: global runtime-setting row
`provider_connection:x`, with Fernet-encrypted `secret_enc` containing `api_key`.
No environment fallback, schema migration, frontend storage, or token-broker
endpoint is added. Public descriptors never contain the token. The app's
existing encryption-key persistence requirements apply.

Use Reconnect to rotate; Disconnect deletes the row. Search resolves the token
fresh on every request. Provider health polling reads local credential state
only, without calling X or spending credits.

## Search

Agents use `x_search(query, max_results=10, start_time=None, end_time=None,
next_token=None)`. This intentionally does **not** join default `web_search`
fan-out: X calls are paid and must be explicit.

- One request/page, newest first, last seven days.
- `max_results`: integer 10–100 (X's upstream minimum is 10).
- Native query operators: `from:BlizzardCS`, `"retrieving character list"`,
  `-is:retweet`, etc.; query length 1–512.
- Optional ISO-8601 time bounds require a timezone, must be within seven days
  and at least 10 seconds ago. Start must precede end.
- Returns text, post ID, author ID, creation timestamp, original source URL,
  and next_token. Pagination is another explicit paid call; no retries or
  automatic pagination. User expansions are omitted to avoid extra billable
  user resources.
- No persistence of search datasets is added. Existing chat/tool-result
  retention still applies; this is not a bulk archive or training pipeline.

Missing connection, rejected token (401), credits (402), access denial (403),
rate limit (429), invalid query (400), transport failure and malformed response
have explicit error codes. Successful zero results are distinct from failure.
Upstream error bodies and transport exception text are not exposed, preventing
accidental token disclosure. There is no fallback to web search.

## Verification

Backend: `pytest -q tests/test_x_integration.py tests/test_provider_api_keys.py
 tests/test_setup_routes.py tests/test_mcp_catalog.py` (on one line).
Frontend: typecheck plus `tsx --test src/app/providers/ProviderConnections.test.tsx
 src/app/setup/api.test.ts src/app/setup/firstRun.test.ts` (on one line).

Live activation requires the app source reload via the authorized ops action;
then verify provider descriptor at `/v1/providers/x`, MCP `tools/list` contains
`x_search`, and a bounded live call through MCP. UI uses dev Vite/HMR.
