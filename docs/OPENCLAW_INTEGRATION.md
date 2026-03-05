# OpenClaw Integration Runbook

This runbook documents how llm-bawt integrates with OpenClaw for both chat runtime and model discovery.

## Scope

Use this when supporting:
- OpenClaw-backed bots (`agent_backend: openclaw`)
- `llm --add-model openclaw` discovery flow
- tool-call visibility in streaming responses
- session/channel mapping issues

## Runtime Path

### Previous path (legacy)
- `POST /v1/chat/completions`
- OpenAI-compatible shim
- effectively non-streaming for backend integration

### Current path (migrated)
- `POST /v1/responses`
- default `stream: true` with SSE events:
  - `response.created`
  - `response.output_item.added`
  - `response.output_text.delta`
  - `response.completed`
- non-stream fallback with `stream: false`

Auth and routing:
- `Authorization: Bearer <token>`
- OpenClaw headers are passed when configured (`x-openclaw-agent-id`, session/channel/account headers)

## Configuration

OpenClaw is configured through `Config` (pydantic-settings, `LLM_BAWT_` env prefix).

### Config fields
- `OPENCLAW_GATEWAY_URL`
- `OPENCLAW_GATEWAY_TOKEN`
- `OPENCLAW_AGENT_ID`
- `OPENCLAW_STREAM_ENABLED`
- `OPENCLAW_USE_SSH_FALLBACK`

### Recommended `.env` entries
- `LLM_BAWT_OPENCLAW_GATEWAY_URL=http://<host>:18789`
- `LLM_BAWT_OPENCLAW_GATEWAY_TOKEN=<token>`
- `LLM_BAWT_OPENCLAW_AGENT_ID=main`
- `LLM_BAWT_OPENCLAW_STREAM_ENABLED=true`
- `LLM_BAWT_OPENCLAW_USE_SSH_FALLBACK=false`

Backward-compatible bare env vars are still read during transition:
- `OPENCLAW_GATEWAY_URL`
- `OPENCLAW_GATEWAY_TOKEN`
- `OPENCLAW_AGENT_ID`
- `OPENCLAW_STREAM_ENABLED`
- `OPENCLAW_USE_SSH_FALLBACK`

## Transport Selection

Selection order:
1. Explicit bot config `transport` (`gateway_api` or `ssh`)
2. `OPENCLAW_USE_SSH_FALLBACK` / `LLM_BAWT_OPENCLAW_USE_SSH_FALLBACK`
3. default to `gateway_api`

Normal chat path should use gateway HTTP.
SSH is retained only as explicit emergency fallback.

## Discovery (`--add-model openclaw`)

Discovery mode:
- `OPENCLAW_DISCOVERY_MODE=api` (recommended)
- `OPENCLAW_DISCOVERY_MODE=hybrid` (API + SSH fallback for incomplete listings)

Discovery uses OpenClaw tooling endpoints and may optionally use SSH if enabled.

## Session and Alias Semantics

- Session key controls continuity and memory identity.
- Channel is metadata/routing context.
- Service may expose the virtual backend alias `openclaw` for backend bots.
- User aliases (for explicit model routing) should map cleanly to one session key.

## Tool Event Behavior

Streaming path surfaces tool lifecycle events during generation:
- `service.tool_call`
- `service.tool_result`

If upstream tool result payloads are not exposed by OpenClaw, llm-bawt stores a placeholder text instead of leaving unresolved pending state.

## Smoke Check

Use an OpenClaw-backed bot and run:

```bash
llm --bot <bot> "hello"
```

For direct service streaming verification:

```bash
curl -N http://localhost:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": null, "messages": [{"role": "user", "content": "hello"}], "stream": true, "bot_id": "<openclaw-bot>"}'
```

Expected:
- multiple `chat.completion.chunk` delta events (not one giant chunk)
- tool events appear during stream when tools are invoked
- final `[DONE]` arrives cleanly

## Troubleshooting

### Only `openclaw` appears in model list
- reload model catalog (`POST /v1/models/reload`) or restart service

### Session mismatch between clients
- verify bot/session mapping uses same OpenClaw session key

### Gateway auth failures
- verify token env and `Authorization` header path
- validate gateway URL and health endpoint reachability

### Missing detailed tool results
- inspect upstream OpenClaw session history payloads
- fallback placeholder is expected when upstream omits tool output
