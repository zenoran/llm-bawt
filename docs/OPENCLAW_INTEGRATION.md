# OpenClaw Integration

Current architecture and operational guide for the OpenClaw agent backend.

## Architecture

All traffic flows through the bridge's single persistent WebSocket connection:

```
Main App (OpenClawBackend)
  └─ Redis command stream ──> Bridge ──> WS chat.send ──> Gateway
                                                             │
Gateway ──> WS events ──> Bridge ──> Redis run stream ──> Main App
```

### Components

| Component | Container | Role |
|-----------|-----------|------|
| Main app | `llm-bawt-app` | Sends prompts, receives streamed responses |
| Bridge | `llm-bawt-openclaw-bridge` | Owns the WS connection, routes events via Redis |
| Redis | `llm-bawt-redis` | Command + event transport between app and bridge |
| Gateway | `vex@ubuntu` (10.0.0.97:18789) | OpenClaw agent runtime |

### Redis Streams

| Stream | Direction | Purpose |
|--------|-----------|---------|
| `openclaw:commands` | App → Bridge | Chat send commands |
| `openclaw:run:{request_id}` | Bridge → App | Per-run response events (deltas, tools, lifecycle) |
| `openclaw:events:{session}` | Bridge → App | All session events (for passive observation) |
| `openclaw:history` | Bridge → App | History persist commands |

### Why the bridge owns the WS

Tool events are **recipient-scoped** — only the WS connection that starts a run via `chat.send` receives `stream:"tool"` events. The bridge must be the sender so it can capture tool start/end events and relay them to the main app.

## Key Files

| File | Role |
|------|------|
| `src/openclaw_bridge/ws_client.py` | WS connection, challenge-response auth, `send_and_stream()` |
| `src/openclaw_bridge/bridge.py` | `SessionBridge` — command handling, run state, event routing |
| `src/openclaw_bridge/ingest.py` | `EventIngestPipeline` — wire format → `OpenClawEvent` |
| `src/openclaw_bridge/publisher.py` | `RedisPublisher` — publishes to Redis streams |
| `src/openclaw_bridge/subscriber.py` | `RedisSubscriber` — reads from Redis streams |
| `src/openclaw_bridge/store.py` | `EventStore` — Postgres persistence |
| `src/openclaw_bridge/events.py` | `OpenClawEvent`, `OpenClawEventKind` — event model |
| `src/openclaw_bridge/config.py` | Bridge config from env |
| `src/openclaw_bridge/__main__.py` | Bridge entrypoint |
| `src/llm_bawt/agent_backends/openclaw.py` | `OpenClawBackend` — main app's backend (Redis pipeline) |
| `src/llm_bawt/service/api.py` | Service startup — creates `RedisSubscriber`, registers with backend |
| `src/llm_bawt/cli/openclaw_handler.py` | `llm --add-model openclaw` discovery + bot creation wizard |

## Configuration

### Main app env vars

Set in `.env` (loaded by docker-compose):

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENCLAW_GATEWAY_TOKEN` | Bearer token for gateway auth | `sk-...` |
| `OPENCLAW_SESSION_KEY` | Default session key | `main` |
| `REDIS_URL` | Redis connection for bridge communication | `redis://redis:6379/0` |

### Bridge env vars

The bridge reads **unprefixed** env vars:

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENCLAW_WS_URL` | Gateway WebSocket URL | `ws://10.0.0.97:18789/v1/ws` |
| `OPENCLAW_GATEWAY_TOKEN` | Bearer token | `sk-...` |
| `POSTGRES_HOST` | Postgres host for event store | `10.0.2.32` |
| `POSTGRES_PORT` | Postgres port | `5432` |
| `POSTGRES_USER` | Postgres user | `askllm` |
| `POSTGRES_PASSWORD` | Postgres password | `askllm_mem0ry` |
| `POSTGRES_DATABASE` | Postgres database | `askllm` |

### Bot profile configuration

Each OpenClaw bot has:
- **Model alias** in `models.yaml` with `type: openclaw` and a `session_key`
- **Bot profile** in DB with `agent_backend: openclaw` and `agent_backend_config: {"session_key": "..."}`

Example model alias:
```yaml
openclaw-codebitch:
  type: openclaw
  model_id: openclaw:codebitch
  gateway_url: http://10.0.0.97:18789
  token_env: OPENCLAW_GATEWAY_TOKEN
  agent_id: codebitch
  session_key: agent:codebitch:main
  tool_support: none
```

## Request Flow (detailed)

1. **User sends message** → `BackgroundService.chat_completion_stream()`
2. Bot has `agent_backend: openclaw` → `AgentBackendClient` wraps `OpenClawBackend`
3. `OpenClawBackend.stream_raw()` is called:
   a. Creates a `RedisSubscriber` in a worker thread
   b. Publishes command to `openclaw:commands` with `session_key`, `message`, `request_id`
   c. Subscribes to `openclaw:run:{request_id}` for response events
4. **Bridge picks up command** from `openclaw:commands`:
   a. Calls `ws_client.send_and_stream(session_key, message)`
   b. WS sends `chat.send` RPC to gateway
   c. Gateway starts agent run, streams events back on WS
   d. Bridge's `EventIngestPipeline` parses each raw WS event → `OpenClawEvent`
   e. Bridge publishes each event to `openclaw:run:{request_id}`
   f. Bridge also stores events in Postgres and publishes to session stream
5. **Main app receives events** from the run stream:
   - `ASSISTANT_DELTA` → yields text delta string → SSE `chat.completion.chunk`
   - `TOOL_START` → yields `{"event": "tool_call", "name": ..., "arguments": ...}` → SSE `service.tool_call` with `status: "pending"`
   - `TOOL_END` → yields `{"event": "tool_call", ..., "result": ...}` → SSE `service.tool_call` with `status: "completed"`
   - `RUN_COMPLETED` → stream ends
6. **Finalization** → `_finalize_turn()` saves response to history, extracts tool context, writes turn log

## Tool Event Handling

### Wire format caveat

The gateway's tool end events often **do not include the actual tool result**. Instead they include:
- `meta` — a short summary string (e.g. the search query, file path)
- `isError` — boolean

The ingest pipeline falls back to `meta` when `result`/`output` are absent. The backend uses `"(completed)"` as a final fallback so the SSE handler always emits `status: "completed"` (never leaving the UI stuck on "Pending result...").

### Tool context persistence

After a run completes, `_finalize_turn()` synthesizes a `tool_context` string from the collected tool calls. This is saved as a system message in history so tool usage is visible on page refresh. Tool call details are also saved to the turn log and surfaced via the `/v1/tool-calls` API.

## Adding a new OpenClaw bot

### Interactive wizard

```bash
llm --add-model openclaw
```

This discovers available sessions from the gateway, lets you pick or enter a custom session key (e.g. `agent:codebitch:main`), creates a model alias, and optionally creates a bot profile with system prompt.

### Manual setup

1. Add model alias to `models.yaml`:
   ```yaml
   openclaw-mybot:
     type: openclaw
     session_key: agent:mybot:main
     # ... other fields
   ```

2. Create bot profile via API:
   ```bash
   curl -X PUT http://localhost:8642/v1/bots/mybot/profile \
     -H "Content-Type: application/json" \
     -d '{
       "name": "My Bot",
       "system_prompt": "You are a helpful assistant.",
       "default_model": "openclaw-mybot",
       "agent_backend": "openclaw",
       "agent_backend_config": {"session_key": "agent:mybot:main"}
     }'
   ```

## Running

Bridge and Redis start by default (no `--profile` needed):

```bash
docker compose up -d
```

Logs:
```bash
docker compose logs -f app
docker compose logs -f openclaw-bridge
```

## Smoke Check

```bash
# CLI test
llm --bot codebitch "hello"

# Streaming SSE test
curl -N http://localhost:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": null, "messages": [{"role": "user", "content": "hello"}], "stream": true, "bot_id": "codebitch"}'
```

Expected:
- Multiple `chat.completion.chunk` delta events (token-by-token streaming)
- `service.tool_call` events with `status: "pending"` then `status: "completed"` when tools are invoked
- Final `[DONE]` sentinel

## Troubleshooting

### No tool events visible
- Verify the bridge WS is the connection calling `chat.send` (check bridge logs)
- Passive WS observers don't receive tool events — this is by design

### "OpenClaw bridge subscriber not initialized"
- `REDIS_URL` not set or Redis not running
- Check `docker compose ps` — `llm-bawt-redis` should be healthy

### Tool results show "(completed)" instead of actual output
- The gateway doesn't include `result`/`output` in tool end events
- The `meta` field is used as fallback; if that's also absent, `"(completed)"` is shown
- This is a gateway limitation, not a bug

### Bot not appearing in `--list-bots`
- `--add-model` only creates model aliases. Use the wizard's bot creation step or create via API.
- Check that the bot profile has `agent_backend: openclaw` in the DB

### Session key format
- Gateway uses `agent:{agent_id}:{session_name}` format (e.g. `agent:codebitch:main`)
- The ingest pipeline normalizes these by extracting the middle segment for display

## Related Docs

- [openclaw-wire-format.md](openclaw-wire-format.md) — Gateway WS wire format reference
- [OPENCLAW_GATEWAY_WS_SPEC.md](OPENCLAW_GATEWAY_WS_SPEC.md) — Gateway `/v1/ws` protocol reference (challenge-response auth, RPC pattern, event streams)
