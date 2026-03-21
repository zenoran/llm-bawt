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
| `openclaw:commands` | App -> Bridge | Chat send commands |
| `openclaw:run:{request_id}` | Bridge -> App | Per-run response events (deltas, tools, lifecycle) |
| `openclaw:events:{session}` | Bridge -> App | All session events (for passive observation) |
| `openclaw:history` | Bridge -> App | History persist commands |

### Why the bridge owns the WS

Tool events are **recipient-scoped** -- only the WS connection that starts a run via `chat.send` receives `stream:"tool"` events. The bridge must be the sender so it can capture tool start/end events and relay them to the main app.

## Key Files

| File | Role |
|------|------|
| `src/openclaw_bridge/ws_client.py` | WS connection, challenge-response auth, `send_and_stream()` |
| `src/openclaw_bridge/bridge.py` | `SessionBridge` -- command handling, run state, event routing |
| `src/openclaw_bridge/ingest.py` | `EventIngestPipeline` -- wire format -> `OpenClawEvent` |
| `src/openclaw_bridge/publisher.py` | `RedisPublisher` -- publishes to Redis streams |
| `src/openclaw_bridge/subscriber.py` | `RedisSubscriber` -- reads from Redis streams |
| `src/openclaw_bridge/store.py` | `EventStore` -- Postgres persistence |
| `src/openclaw_bridge/events.py` | `OpenClawEvent`, `OpenClawEventKind` -- event model |
| `src/openclaw_bridge/config.py` | Bridge config from env |
| `src/openclaw_bridge/__main__.py` | Bridge entrypoint |
| `src/llm_bawt/agent_backends/openclaw.py` | `OpenClawBackend` -- main app's backend (Redis pipeline) |
| `src/llm_bawt/service/api.py` | Service startup -- creates `RedisSubscriber`, starts history drain |
| `src/llm_bawt/cli/openclaw_handler.py` | `llm --add-model openclaw` discovery + bot creation wizard |

## Bot Configuration

OpenClaw bots are configured entirely through **bot profiles** in the DB. There are no separate model definitions for OpenClaw -- the service registers a virtual `"openclaw"` model automatically from bots with `agent_backend: "openclaw"`.

Each bot profile needs:
- `agent_backend`: `"openclaw"`
- `agent_backend_config`: `{"session_key": "agent:<id>:main", "timeout_seconds": 600}`
- `default_model`: `null` (the virtual model is auto-registered)

### Session-to-bot mapping

The bridge needs to know which session keys map to which bots (for history persistence). At startup, it fetches this from the main app's `/v1/bots` API endpoint. The main app builds the mapping from bot profiles with `agent_backend == "openclaw"`.

## Environment Variables

### Main app

Set in `.env` (loaded by docker-compose):

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENCLAW_GATEWAY_TOKEN` | Bearer token for gateway auth | `sk-...` |
| `OPENCLAW_SESSION_KEY` | Default session key (fallback) | `main` |
| `REDIS_URL` | Redis connection for bridge communication | `redis://redis:6379/0` |

### Bridge

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENCLAW_WS_URL` | Gateway WebSocket URL | `ws://10.0.0.97:18789/v1/ws` |
| `OPENCLAW_GATEWAY_TOKEN` | Bearer token | `sk-...` |
| `LLM_BAWT_API_URL` | Main app API URL (for bot mapping) | `http://app:8642` |
| `POSTGRES_HOST` | Postgres host for event store | `10.0.2.32` |
| `POSTGRES_PORT` | Postgres port | `5432` |
| `POSTGRES_USER` | Postgres user | `askllm` |
| `POSTGRES_PASSWORD` | Postgres password | `askllm_mem0ry` |
| `POSTGRES_DATABASE` | Postgres database | `askllm` |

## Request Flow

1. **User sends message** via `llm -b vex "hello"` or API
2. CLI detects `agent_backend` on bot -> auto-enables service mode
3. Service resolves bot to virtual `"openclaw"` model via `_agent_backend_models` mapping
4. `AgentBackendClient` wraps `OpenClawBackend.stream_raw()`
5. `OpenClawBackend` reads `session_key` from `bot.agent_backend_config`:
   a. Publishes command to `openclaw:commands` with `session_key`, `message`, `request_id`
   b. Subscribes to `openclaw:run:{request_id}` for response events
6. **Bridge picks up command**:
   a. Calls `ws_client.send_and_stream(session_key, message)`
   b. Gateway starts agent run, streams events back on WS
   c. Bridge's `EventIngestPipeline` parses each raw WS event -> `OpenClawEvent`
   d. Bridge publishes events to `openclaw:run:{request_id}` and session stream
7. **Main app receives events**:
   - `ASSISTANT_DELTA` -> text delta -> SSE `chat.completion.chunk`
   - `TOOL_START` -> SSE `service.tool_call` with `status: "pending"`
   - `TOOL_END` -> SSE `service.tool_result`
   - `RUN_COMPLETED` -> stream ends
8. **Finalization** -> `_finalize_turn()` saves response to history, writes turn log

### History persistence

The bridge also handles history persistence for non-API runs (e.g., messages sent directly in the gateway UI):
- `ASSISTANT_DONE` events -> bridge publishes to `openclaw:history` stream
- Main app's `drain_history` task reads from the stream and calls `memory_client.add_message()`
- Bridge resolves `session_key -> bot_id` using the mapping fetched from `/v1/bots`

## Adding a New OpenClaw Bot

### Interactive wizard

```bash
llm --add-bot openclaw
```

This discovers available sessions from the gateway, lets you pick or enter a custom session key, and creates a bot profile directly.

### Manual setup

Create bot profile via API:

```bash
curl -X POST http://localhost:8642/v1/bots \
  -H "Content-Type: application/json" \
  -d '{
    "slug": "mybot",
    "bot_type": "agent",
    "name": "My Bot",
    "system_prompt": "You are a helpful assistant.",
    "agent_backend": "openclaw",
    "agent_backend_config": {
      "session_key": "agent:mybot:main",
      "timeout_seconds": 600
    }
  }'
```

Then restart the service (or call `/v1/admin/reload-bots`) to refresh bot state if needed. The service will still register the internal virtual model automatically for agent bots.

## Running

Bridge and Redis start by default:

```bash
docker compose up -d
```

The bridge depends on the main app (for bot mapping) and Redis. It will retry the WS connection to the gateway on failure.

Logs:
```bash
docker compose logs -f app
docker compose logs -f openclaw-bridge
```

## Smoke Check

```bash
# CLI test
llm -b vex "hello"

# Streaming SSE test
curl -N http://localhost:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hello"}], "stream": true, "bot_id": "vex"}'
```

Expected:
- Multiple `chat.completion.chunk` delta events (token-by-token streaming)
- `service.tool_call` / `service.tool_result` events when tools are invoked
- Final `[DONE]` sentinel

## Troubleshooting

### No tool events visible
- Verify the bridge WS is the connection calling `chat.send` (check bridge logs)
- Passive WS observers don't receive tool events -- this is by design

### "OpenClaw bridge subscriber not initialized"
- `REDIS_URL` not set or Redis not running
- Check `docker compose ps` -- `llm-bawt-redis` should be healthy

### Tool results show "(completed)" instead of actual output
- Set `agents.defaults.verboseDefault = "full"` in the gateway config
- Without `verbose=full`, the gateway strips `data.result` from tool end events

### Bot not appearing in `--list-bots`
- Check that the bot profile has `agent_backend: "openclaw"` in the DB
- The service auto-registers a virtual model for each openclaw bot at startup

### Bridge shows `bot_map={}`
- The bridge couldn't reach the main app's `/v1/bots` API
- Check `LLM_BAWT_API_URL` env var and that the main app is running
- Bridge depends on `app` service in docker-compose

### Session key format
- Gateway uses `agent:{agent_id}:{session_name}` format (e.g. `agent:vex:main`)
- The ingest pipeline normalizes these by extracting the middle segment

## Related Docs

- [openclaw-wire-format.md](openclaw-wire-format.md) -- Gateway WS wire format reference
- [OPENCLAW_GATEWAY_WS_SPEC.md](OPENCLAW_GATEWAY_WS_SPEC.md) -- Gateway `/v1/ws` protocol reference
