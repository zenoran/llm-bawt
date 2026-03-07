# OpenClaw Session Bridge — COMPLETED

> **Status**: Implemented as of 2026-03-06.
> See [OPENCLAW_INTEGRATION.md](OPENCLAW_INTEGRATION.md) for the current architecture.
> See [openclaw-gateway-refactor-handoff.md](openclaw-gateway-refactor-handoff.md) for implementation details.

This planning document described the refactor from request-coupled HTTP integration to session-subscribed WebSocket bridge. All phases have been completed:

- Session bridge with persistent WS connection
- Redis Streams for command/event transport
- Per-run response streaming
- Tool event capture (recipient-scoped via bridge-owned runs)
- Postgres event store
- Durable response persistence (client disconnect safe)
- `llm --add-model openclaw` discovery wizard with bot creation

The original planning content is preserved below for reference.

---

<details>
<summary>Original planning document (click to expand)</summary>

## Problem Statement

The current OpenClaw integration is **request-coupled**: llm-bawt issues a POST to `/v1/responses` per user turn, streams back that single response, and relies on the HTTP connection staying alive to persist the result. This has three fundamental flaws:

1. **Responses only arrive when llm-bawt initiates a request.** If OpenClaw does background work (heartbeat/cron/subagent completions), llm-bawt never sees it.
2. **HTTP timeouts lose responses.** OpenClaw tool execution can take 30+ seconds; client/proxy timeouts break the SSE stream, and even with the durable persistence fix (Commit 0), the recovery is fragile.
3. **No continuity across reconnects.** If the service restarts or the HTTP stream breaks, there's no replay of missed events.

## Target Architecture

llm-bawt becomes a **session subscriber** that maintains a persistent connection (WebSocket) to OpenClaw Gateway. All messages — user-initiated, system-triggered, subagent completions — flow through this connection and are durably persisted by llm-bawt regardless of whether any UI client is connected.

## Implementation

All components described in this plan were implemented in `src/openclaw_bridge/`:
- `OpenClawWsClient` → `ws_client.py`
- `OpenClawEvent` / `OpenClawEventKind` → `events.py`
- `EventIngestPipeline` → `ingest.py`
- `EventStore` → `store.py`
- `RedisPublisher` / `RedisSubscriber` → `publisher.py` / `subscriber.py`
- `SessionBridge` → `bridge.py`

The FanoutHub was replaced by Redis Streams, which provide the same broadcast + replay semantics with cross-container support.

</details>
