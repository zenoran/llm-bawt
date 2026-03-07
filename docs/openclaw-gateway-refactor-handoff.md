# OpenClaw Gateway Integration Refactor — COMPLETED

> **Status**: All phases completed as of 2026-03-06.
> This doc is kept for historical reference. See [OPENCLAW_INTEGRATION.md](OPENCLAW_INTEGRATION.md) for the current architecture.

## What Was Done

### Goal
Refactor llm-bawt's OpenClaw integration so the bridge's persistent WS connection is THE gateway integration point. The main app sends messages through the bridge (via Redis), and the bridge receives ALL events (assistant, lifecycle, tool, error, chat) on the same WS connection.

### Previous Architecture (removed)
- Main app sent HTTP POST to `/v1/responses` (SSE) directly to gateway
- Bridge was a passive WS observer — got no tool events
- Two separate connections doing half the job each
- SSH transport as emergency fallback

### Current Architecture
```
Main App (OpenClawBackend)
  └─ Redis command ──> Bridge ──> WS chat.send ──> Gateway
                                                      │
Gateway ──> WS events ──> Bridge ──> Redis run stream ──> Main App
```

Single WS connection through the bridge. Bridge owns every run, gets all streams including tool events.

### Phase 1: Bridge becomes the message sender (DONE)
- `ws_client.py` — `send_and_stream()` sends `chat.send` via WS, yields raw events filtered by `runId`
- `bridge.py` — `_handle_send_command()` reads from `openclaw:commands` Redis stream, routes through WS
- `publisher.py` — publishes per-run events to `openclaw:run:{request_id}` Redis stream
- `subscriber.py` — `subscribe_run()` reads per-run stream; `send_command()` publishes to commands stream

### Phase 2: Refactor OpenClawBackend (DONE)
- `openclaw.py` — `stream_raw()` publishes to Redis, subscribes to run stream, yields deltas and tool events
- HTTP `/v1/responses` path removed
- SSH transport removed
- `_fetch_session_tool_calls_gateway` removed — tool events come live on WS

### Phase 3: Cleanup (DONE)
- HTTP gateway references removed
- Bridge starts by default (no `--profile bridge` needed)
- Config consolidated — bridge reads unprefixed env vars
- Bot profile API extended with `agent_backend` and `agent_backend_config` fields
- `llm --add-model openclaw` wizard supports custom session keys and bot creation

### Tool Result Fix (2026-03-07)
- **Root cause**: Gateway tool end events don't include `result`/`output` — only `meta` (summary string) and `isError`
- **Fix**: Ingest falls back to `meta` field; backend uses `"(completed)"` as final fallback
- **Effect**: SSE handler now always sends `status: "completed"` for tool end events

## Key Files

| File | Role |
|------|------|
| `src/openclaw_bridge/ws_client.py` | WS connection + chat.send |
| `src/openclaw_bridge/bridge.py` | Event processing, run state, publishing |
| `src/openclaw_bridge/ingest.py` | Wire format → internal events |
| `src/openclaw_bridge/store.py` | Postgres persistence |
| `src/openclaw_bridge/publisher.py` | Redis event fanout |
| `src/openclaw_bridge/subscriber.py` | Redis event consumption |
| `src/openclaw_bridge/events.py` | Event model + serialization |
| `src/openclaw_bridge/config.py` | Bridge config from env |
| `src/openclaw_bridge/__main__.py` | Standalone bridge entrypoint |
| `src/llm_bawt/agent_backends/openclaw.py` | Main app's OpenClaw backend |
| `src/llm_bawt/service/api.py` | Service startup, subscriber init |

## Infrastructure

- **Gateway**: `vex@ubuntu` (10.0.0.97:18789), systemd user service
- **Postgres**: `postgres-pgvector` on unraid, host=10.0.2.32, DB=askllm
- **Redis**: `llm-bawt-redis` container, local docker-compose
- **Bridge**: `llm-bawt-openclaw-bridge` container, local docker-compose
