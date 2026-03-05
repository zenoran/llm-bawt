# OpenClaw Session Bridge — Refactor Plan

## Problem Statement

The current OpenClaw integration is **request-coupled**: llm-bawt issues a POST to `/v1/responses` per user turn, streams back that single response, and relies on the HTTP connection staying alive to persist the result. This has three fundamental flaws:

1. **Responses only arrive when llm-bawt initiates a request.** If OpenClaw does background work (heartbeat/cron/subagent completions), llm-bawt never sees it.
2. **HTTP timeouts lose responses.** OpenClaw tool execution can take 30+ seconds; client/proxy timeouts break the SSE stream, and even with the durable persistence fix (Commit 0), the recovery is fragile.
3. **No continuity across reconnects.** If the service restarts or the HTTP stream breaks, there's no replay of missed events.

## Target Architecture

llm-bawt becomes a **session subscriber** that maintains a persistent connection (WebSocket) to OpenClaw Gateway. All messages — user-initiated, system-triggered, subagent completions — flow through this connection and are durably persisted by llm-bawt regardless of whether any UI client is connected.

```
┌─────────┐    SSE     ┌──────────────────┐    WS (persistent)    ┌─────────────┐
│  unmute  │◄──────────►│    llm-bawt      │◄─────────────────────►│  OpenClaw    │
│  (UI)    │   /v1/chat │                  │  session subscribe    │  Gateway     │
└─────────┘            │  EventStore (PG) │                       └─────────────┘
                        └──────────────────┘
                         ▲ replay from DB
                         │ on UI reconnect
```

### Key Shifts

| Aspect | Current (request-coupled) | Target (session-subscribed) |
|--------|---------------------------|------------------------------|
| Who initiates | llm-bawt POSTs per user turn | WS bridge receives all session events |
| Stream lifetime | One HTTP request = one stream | Persistent WS per session |
| Tool events | Extracted post-hoc from `last_result` | Real-time via WS events |
| Background work | Invisible to llm-bawt | Flows into EventStore like any other event |
| Client disconnect | Response may be lost (mitigated by Commit 0) | Irrelevant — events land in DB regardless |
| Replay | Not possible | Since `event_id` / cursor from EventStore |
| User send path | Direct HTTP POST reading back SSE | WS `chat.send` + events arrive on subscription |

---

## Components

### A) `OpenClawWsClient` — new file: `src/llm_bawt/integrations/openclaw_ws.py`

Persistent WebSocket client using the `websockets` library (async).

**Responsibilities:**
- Authenticated connection to `ws://{gateway_host}:{gateway_port}/ws` (or `/v1/ws`)
- Auto-reconnect with exponential backoff + jitter (base 1s, max 60s, jitter ±30%)
- Ping/keepalive on the WS protocol level (websockets library handles this natively)
- Session subscription: send `{"type": "subscribe", "session_key": "..."}` after auth
- Re-subscribe after every reconnect
- Connection state callbacks for observability

**Interface:**
```python
class OpenClawWsClient:
    def __init__(self, config: OpenClawWsConfig) -> None: ...

    async def connect(self) -> None:
        """Establish WS, authenticate, subscribe. Auto-reconnects in background."""

    async def disconnect(self) -> None:
        """Graceful shutdown."""

    async def send_user_message(self, session_key: str, text: str, *, metadata: dict | None = None) -> str:
        """Send a user message into the session. Returns a message/run ID from gateway."""

    def on_event(self, callback: Callable[[OpenClawEvent], Awaitable[None]]) -> None:
        """Register the ingest callback. Called for every event on subscribed sessions."""

    @property
    def connected(self) -> bool: ...

    @property
    def subscribed_sessions(self) -> set[str]: ...
```

**Config** (added to `Config` class):
```python
# OpenClaw WebSocket Bridge
OPENCLAW_WS_URL: str = Field(default="", description="OpenClaw gateway WebSocket URL (e.g. ws://10.0.0.97:18789/ws). Empty = bridge disabled.")
OPENCLAW_WS_SESSIONS: str = Field(default="main", description="Comma-separated session keys to subscribe to")
OPENCLAW_WS_RECONNECT_MAX_DELAY: int = Field(default=60, description="Max reconnect delay in seconds")
OPENCLAW_WS_ENABLED: bool = Field(default=False, description="Enable WebSocket session bridge (Phase 1: shadow mode)")
```

**Dependencies:** Add `websockets>=13.0` to `pyproject.toml`.

**Important:** This is the *only* new external dependency. The rest is pure internal refactoring.

---

### B) `OpenClawEvent` — normalized event envelope

Defined in `src/llm_bawt/integrations/openclaw_events.py`:

```python
@dataclass
class OpenClawEvent:
    event_id: str               # Dedupe key — from gateway or synthetic UUID
    session_key: str
    run_id: str | None          # Links deltas to a run
    kind: OpenClawEventKind     # Enum: see below
    origin: str                 # "user" | "system" | "heartbeat" | "cron" | "subagent"
    text: str | None            # For text content events
    tool_name: str | None       # For tool events
    tool_arguments: dict | None
    tool_result: Any | None
    model: str | None           # Upstream model from gateway
    timestamp: datetime         # Gateway timestamp (or receive time if missing)
    raw: dict                   # Full original event for debugging
```

```python
class OpenClawEventKind(str, Enum):
    ASSISTANT_DELTA = "assistant_delta"   # Streaming text delta
    ASSISTANT_DONE = "assistant_done"     # Final assembled text
    TOOL_START = "tool_start"             # Tool invocation beginning
    TOOL_END = "tool_end"                 # Tool result returned
    USER_MESSAGE = "user_message"         # Echo/confirmation of sent message
    RUN_STARTED = "run_started"           # New run/response initiated
    RUN_COMPLETED = "run_completed"       # Response fully done
    SYSTEM_NOTE = "system_note"           # Heartbeat, cron, status
    ERROR = "error"
```

**Event classification rules**: The `EventIngestPipeline` (below) maps raw Gateway WS events to these kinds. The `origin` field is derived from event metadata — if the event's `run_id` was started by llm-bawt's `send_user_message()`, `origin = "user"`; otherwise inferred from gateway metadata (`heartbeat`, `cron`, `subagent`, `system`).

---

### C) `EventIngestPipeline` — `src/llm_bawt/integrations/openclaw_ingest.py`

Stateless mapper — receives raw WS JSON messages, returns `OpenClawEvent` objects.

```python
class EventIngestPipeline:
    def parse(self, raw: dict, session_key: str) -> OpenClawEvent | None:
        """Normalize a raw Gateway WS event into an OpenClawEvent.
        Returns None for events that should be silently dropped (e.g., pings)."""
```

Mapping from Gateway WS events (based on existing SSE event types from the `/v1/responses` integration):

| Gateway event type | OpenClawEventKind | Notes |
|---|---|---|
| `response.created` | `RUN_STARTED` | Captures run_id, model |
| `response.output_text.delta` | `ASSISTANT_DELTA` | `text` = delta string |
| `response.completed` | `RUN_COMPLETED` | Captures final usage |
| `response.output_item.added` (tool type) | `TOOL_START` | `tool_name`, `tool_arguments` |
| `response.output_item.added` (tool result type) | `TOOL_END` | `tool_name`, `tool_result` |
| `chat.message` / `message` | `USER_MESSAGE` or `ASSISTANT_DONE` | Based on role |
| `system.*` / `health.*` | `SYSTEM_NOTE` | |
| `error` | `ERROR` | |

This mapping will evolve as OpenClaw's WS event schema stabilizes. The raw JSON is always stored, so we can re-parse events later if the mapping changes.

---

### D) `EventStore` — `src/llm_bawt/integrations/openclaw_store.py`

Durable append-only event log in PostgreSQL. Uses SQLAlchemy/SQLModel (existing DB infra).

**Tables:**

```sql
-- Append-only event log (the source of truth for session history)
CREATE TABLE openclaw_events (
    id BIGSERIAL PRIMARY KEY,
    event_dedupe_key VARCHAR(255) NOT NULL UNIQUE,  -- From gateway event_id or SHA256 fallback
    session_key VARCHAR(255) NOT NULL,
    run_id VARCHAR(255),
    seq BIGINT,                                     -- Gateway monotonic sequence per session (nullable during transition)
    kind VARCHAR(50) NOT NULL,
    origin VARCHAR(50) NOT NULL DEFAULT 'unknown',
    text TEXT,
    tool_name VARCHAR(255),
    payload_json JSONB NOT NULL,                    -- Full raw event
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Indexes for common queries
    -- idx_openclaw_events_session_created ON (session_key, created_at)
    -- idx_openclaw_events_session_run ON (session_key, run_id)
    -- idx_openclaw_events_session_seq ON (session_key, seq)
    -- idx_openclaw_events_dedupe ON (event_dedupe_key) -- already UNIQUE
);

-- Lightweight cursor/state tracking per session
CREATE TABLE openclaw_session_state (
    session_key VARCHAR(255) PRIMARY KEY,
    last_event_id BIGINT,                           -- FK to openclaw_events.id
    last_cursor VARCHAR(255),                        -- Gateway cursor/sequence if provided
    ws_connected BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Optional: assembled run summaries for UI/status
CREATE TABLE openclaw_runs (
    run_id VARCHAR(255) PRIMARY KEY,
    session_key VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'running',  -- running | completed | error
    model VARCHAR(255),
    origin VARCHAR(50),
    full_text TEXT,                                  -- Assembled from deltas
    tool_calls_json JSONB,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**EventStore interface:**
```python
class EventStore:
    def __init__(self, engine: Engine) -> None: ...
    
    async def store(self, event: OpenClawEvent) -> bool:
        """Append event. Returns False if dedupe_key already exists (idempotent)."""
    
    async def get_events(
        self, session_key: str, *, since_id: int | None = None,
        since_ts: datetime | None = None, kinds: list[OpenClawEventKind] | None = None,
        limit: int = 100,
    ) -> list[OpenClawEvent]: ...
    
    async def get_session_cursor(self, session_key: str) -> int | None: ...
    async def update_session_cursor(self, session_key: str, event_id: int) -> None: ...
    
    async def assemble_run_text(self, run_id: str) -> str:
        """Reassemble full text from ASSISTANT_DELTA events for a run."""
    
    async def update_run(self, run_id: str, **fields) -> None: ...
```

**Migration:** Add to existing `src/llm_bawt/memory/migrations.py` system. Tables are created on first startup when `OPENCLAW_WS_ENABLED=true`.

---

### E) `SessionBridge` — orchestrator: `src/llm_bawt/integrations/openclaw_bridge.py`

The top-level component that ties everything together. Runs as a long-lived asyncio task alongside the FastAPI service.

```python
class SessionBridge:
    def __init__(
        self, ws_client: OpenClawWsClient, ingest: EventIngestPipeline,
        store: EventStore, fanout: FanoutHub, config: Config,
    ) -> None: ...

    async def start(self) -> None:
        """Start the bridge: connect WS, subscribe, begin consuming events."""

    async def stop(self) -> None:
        """Graceful shutdown."""

    async def send_user_message(self, session_key: str, text: str) -> str:
        """Send a user message via WS and return the run_id."""

    async def _on_event(self, event: OpenClawEvent) -> None:
        """Process a single event: store → assemble run → fanout."""
```

**Event processing flow:**
1. Raw WS message arrives
2. `EventIngestPipeline.parse()` → `OpenClawEvent`
3. `EventStore.store()` (idempotent, dedupe by event_id)
4. If `RUN_STARTED`: create `openclaw_runs` row
5. If `ASSISTANT_DELTA`: append to in-memory run buffer (for fanout), no extra DB write (deltas are stored as events)
6. If `RUN_COMPLETED`: assemble full text from deltas, update `openclaw_runs`, **call history save** (via existing `finalize_response` or direct insert into `{bot}_messages`)
7. If `TOOL_START`/`TOOL_END`: update run's tool_calls, fanout immediately
8. `FanoutHub.broadcast(event)` — push to connected UI clients

---

### F) `FanoutHub` — `src/llm_bawt/integrations/openclaw_fanout.py`

Broadcasts normalized events to connected SSE/streaming clients. Supports replay.

```python
class FanoutHub:
    def __init__(self, store: EventStore) -> None: ...
    
    def subscribe(self, session_key: str, *, since_event_id: int | None = None) -> AsyncIterator[OpenClawEvent]:
        """Subscribe to live events. If since_event_id provided, replays gap first."""
    
    def broadcast(self, event: OpenClawEvent) -> None:
        """Push to all subscribers of this session."""
    
    @property
    def subscriber_count(self) -> int: ...
```

**Implementation:** Uses `asyncio.Queue` per subscriber (same pattern as current `chunk_queue` in `_stream_to_queue`).

---

## Integration with Existing Service Layer

### User message send path (request-initiated)

The current `BackgroundService.chat_completion_stream()` flow changes:

**Before (request-coupled):**
```
User request → prepare_messages → client.stream_raw() → SSE chunks → accumulate → finalize_turn → history
```

**After (bridge mode):**
```
User request → bridge.send_user_message() → return run_id → subscribe to FanoutHub → stream events as SSE
                                                                    ↑
                              WS event arrives → ingest → store → fanout → UI
                                                                    ↓
                              RUN_COMPLETED → assemble text → finalize_response (history)
```

This means `BackgroundService` no longer directly calls `client.stream_raw()` for OpenClaw bots when bridge mode is active. Instead:

1. `chat_completion_stream()` detects the bot uses an OpenClaw agent backend with bridge mode enabled
2. Calls `bridge.send_user_message(session_key, user_prompt)` to inject the user message
3. Subscribes to `fanout_hub.subscribe(session_key, since_event_id=<current>)`
4. Forwards events from the subscription as SSE chunks to the UI client (same format as today):
   - `ASSISTANT_DELTA` → `chat.completion.chunk` with `delta.content`
   - `TOOL_START` → `service.tool_call` 
   - `TOOL_END` → `service.tool_result`
   - `RUN_COMPLETED` → `finish_reason: "stop"` + `[DONE]`
5. If the UI client disconnects, the subscription is dropped — but the WS bridge continues receiving events and persisting them regardless

### History integration

When `RUN_COMPLETED` fires, the `SessionBridge` needs to write the full assistant response to the bot's message history (the `{bot}_messages` table). Two approaches:

**Option A — Call existing `finalize_response`:**
- Requires a `ServiceLLMBawt` instance for the relevant bot
- Handles memory extraction, profile updates, etc.
- Pro: leverages all existing post-processing
- Con: heavier, requires bot context setup

**Option B — Direct history insert:**
- Insert directly into `{bot}_messages` table
- Skip memory extraction for bridge-originated events (or run it async)
- Pro: simpler, no coupling to LLMBawt lifecycle
- Con: loses memory/profile integration

**Recommendation: Option A for user-initiated runs, Option B for system/background runs.** User-initiated runs should get full memory extraction. Background events (heartbeat, cron, subagent) should be stored for visibility but don't need memory processing.

### Non-streaming / legacy fallback

The existing HTTP path (`/v1/responses` POST) remains available behind:
```python
OPENCLAW_MODE: str = Field(default="http", description="'ws_bridge' | 'http'. WS bridge for session-subscribed mode, HTTP for per-request mode.")
```

When `OPENCLAW_MODE=http`, the current `OpenClawBackend.stream_raw()` and `chat_full()` work exactly as they do today. No behavior change. This is the default until the WS bridge is validated.

---

## Files to Create

| File | Description |
|------|-------------|
| `src/llm_bawt/integrations/__init__.py` | Package init |
| `src/llm_bawt/integrations/openclaw_ws.py` | `OpenClawWsClient` — persistent WS with reconnect |
| `src/llm_bawt/integrations/openclaw_events.py` | `OpenClawEvent`, `OpenClawEventKind` — normalized event model |
| `src/llm_bawt/integrations/openclaw_ingest.py` | `EventIngestPipeline` — raw WS event → `OpenClawEvent` |
| `src/llm_bawt/integrations/openclaw_store.py` | `EventStore` — PostgreSQL append-only event log |
| `src/llm_bawt/integrations/openclaw_fanout.py` | `FanoutHub` — broadcasts events to subscribers with replay |
| `src/llm_bawt/integrations/openclaw_bridge.py` | `SessionBridge` — top-level orchestrator |

## Files to Modify

| File | Change |
|------|--------|
| `src/llm_bawt/utils/config.py` | Add `OPENCLAW_MODE`, `OPENCLAW_WS_URL`, `OPENCLAW_WS_SESSIONS`, `OPENCLAW_WS_ENABLED`, `OPENCLAW_WS_RECONNECT_MAX_DELAY` fields |
| `src/llm_bawt/service/api.py` | On startup, if `OPENCLAW_WS_ENABLED`, create and start `SessionBridge` as background task |
| `src/llm_bawt/service/background_service.py` | In `chat_completion_stream`, detect bridge mode and route through `FanoutHub` instead of `client.stream_raw()` |
| `src/llm_bawt/memory/migrations.py` | Add migration for `openclaw_events`, `openclaw_session_state`, `openclaw_runs` tables |
| `pyproject.toml` | Add `websockets>=13.0` dependency |

## Files NOT Modified (preserved as-is)

| File | Why |
|------|-----|
| `src/llm_bawt/agent_backends/openclaw.py` | Stays as HTTP fallback. `stream_raw()` still works for `OPENCLAW_MODE=http` |
| `src/llm_bawt/clients/agent_backend_client.py` | No changes — bridge bypasses this entirely |
| `src/llm_bawt/service/chat_stream_worker.py` | Still used for non-OpenClaw streaming |
| `tests/test_stream_persistence.py` | Existing tests still valid |
| `tests/test_openclaw.py` | Existing tests still valid (HTTP mode) |

---

## New Tests

### `tests/test_openclaw_bridge.py`

| Test | What it validates |
|------|-------------------|
| `test_event_ingest_maps_delta` | `EventIngestPipeline.parse()` maps `response.output_text.delta` → `ASSISTANT_DELTA` |
| `test_event_ingest_maps_tool_start` | Tool-type `output_item.added` → `TOOL_START` |
| `test_event_ingest_maps_run_completed` | `response.completed` → `RUN_COMPLETED` |
| `test_event_ingest_drops_pings` | Ping/keepalive events return `None` |
| `test_event_ingest_unknown_event_passthrough` | Unknown event types get `SYSTEM_NOTE` kind |
| `test_event_store_idempotent` | Storing same `event_dedupe_key` twice returns `False` second time |
| `test_event_store_replay` | `get_events(since_id=N)` returns only events after N |
| `test_event_store_assemble_run_text` | Deltas for a run reassemble into full text |
| `test_fanout_live_broadcast` | Subscriber receives events pushed via `broadcast()` |
| `test_fanout_replay_then_live` | Subscriber with `since_event_id` gets gap replay then live events |
| `test_ws_client_reconnect` | Mock WS disconnects, client reconnects and resubscribes |
| `test_bridge_user_send_to_history` | Full flow: send → events arrive → run completed → history saved |
| `test_bridge_background_event_persisted` | Event with no matching user send still stored in EventStore |
| `test_bridge_client_disconnect_no_data_loss` | UI disconnects mid-stream, run completes and persists |

### `tests/test_openclaw_ws.py`

| Test | What it validates |
|------|-------------------|
| `test_connect_and_subscribe` | WS handshake + subscribe message sent |
| `test_auth_header_sent` | Bearer token included in WS headers |
| `test_reconnect_backoff` | Exponential backoff with jitter on disconnect |
| `test_resubscribe_on_reconnect` | After reconnect, all sessions are resubscribed |
| `test_send_user_message` | Message sent as JSON via WS |
| `test_graceful_disconnect` | `disconnect()` sends close frame |

---

## Observability

### Metrics (logged, structured — use existing `get_service_logger`)

| Metric | Type | Description |
|--------|------|-------------|
| `openclaw_ws_connected` | gauge | 1 if WS is connected, 0 if not |
| `openclaw_ws_reconnect_count` | counter | Number of reconnects since startup |
| `openclaw_event_lag_ms` | histogram | Gateway timestamp vs ingest time |
| `openclaw_event_dedupe_hits` | counter | Events dropped due to duplicate key |
| `openclaw_events_stored` | counter | Total events stored in EventStore |
| `openclaw_fanout_subscribers` | gauge | Number of active FanoutHub subscribers |
| `openclaw_run_duration_ms` | histogram | Time from RUN_STARTED to RUN_COMPLETED |

### Logs

- Connection lifecycle: connect, disconnect, reconnect (with backoff delay)
- Subscription lifecycle: subscribe, unsubscribe, resubscribe
- Event processing: kind, session_key, run_id (at DEBUG level for deltas, INFO for run lifecycle)
- Gap replay decisions: cursor gap detected, backfill count
- Errors: WS errors, store failures, ingest parse failures

---

## Migration / Rollout Plan

### Phase 1 — Shadow Mode (`OPENCLAW_WS_ENABLED=true`, `OPENCLAW_MODE=http`)

- WS bridge runs alongside existing HTTP path
- Events are ingested and stored in `openclaw_events` table
- UI still renders from the HTTP SSE stream (current behavior)
- **Validation**: Compare EventStore contents with turn logs — they should contain the same responses
- **Duration**: ~1 week, or until we're confident the event stream is complete and reliable

### Phase 2 — Read Switch (`OPENCLAW_MODE=ws_bridge`)

- `chat_completion_stream` reads from `FanoutHub` for OpenClaw bots
- User messages sent via WS `send_user_message` (instead of HTTP POST)
- Background events now visible in UI
- HTTP path available as instant rollback (`OPENCLAW_MODE=http`)
- **Validation**: Full E2E — voice, chat, tool calls, client disconnect/reconnect, background events

### Phase 3 — Cleanup

- Remove `OPENCLAW_MODE=http` option (or keep as emergency escape hatch)
- Simplify `BackgroundService._stream_to_queue` — no more special-casing for OpenClaw streaming
- Consider if `OpenClawBackend.stream_raw()` HTTP path should be removed or kept for non-bridge use cases

---

## Execution Plan (Commits)

### Commit 0: Foundation — event model + store + migrations
- Create `src/llm_bawt/integrations/` package
- `openclaw_events.py` — `OpenClawEvent`, `OpenClawEventKind`
- `openclaw_store.py` — `EventStore` with PostgreSQL tables
- DB migration for `openclaw_events`, `openclaw_session_state`, `openclaw_runs`
- `tests/test_openclaw_bridge.py` — EventStore unit tests (idempotent store, replay, assemble)
- Config fields: `OPENCLAW_WS_ENABLED`, `OPENCLAW_MODE`

### Commit 1: WS client + ingest pipeline
- Add `websockets>=13.0` to `pyproject.toml`
- `openclaw_ws.py` — `OpenClawWsClient` with auth, reconnect, subscribe
- `openclaw_ingest.py` — `EventIngestPipeline` mapping
- Config fields: `OPENCLAW_WS_URL`, `OPENCLAW_WS_SESSIONS`, `OPENCLAW_WS_RECONNECT_MAX_DELAY`
- `tests/test_openclaw_ws.py` — WS client unit tests (mock websockets)
- `tests/test_openclaw_bridge.py` — Ingest pipeline mapping tests

### Commit 2: FanoutHub + SessionBridge
- `openclaw_fanout.py` — `FanoutHub` with subscribe, broadcast, replay
- `openclaw_bridge.py` — `SessionBridge` orchestrator
- Tests: fanout broadcast, replay, bridge flow (send → events → history)

### Commit 3: Service integration (shadow mode)
- `api.py` — Start `SessionBridge` on service startup when `OPENCLAW_WS_ENABLED=true`
- Events flow into EventStore silently alongside existing HTTP path
- No UI behavior change yet

### Commit 4: Service integration (bridge mode)
- `background_service.py` — new path in `chat_completion_stream` for bridge mode:
  - Detect OpenClaw bot + `OPENCLAW_MODE=ws_bridge`
  - Send via `bridge.send_user_message()`
  - Stream from `FanoutHub.subscribe()`
  - Map events to existing SSE format
- History integration: on `RUN_COMPLETED`, call `finalize_response` for user-initiated runs
- Tests: bridge mode E2E with mocked WS

### Commit 5: Docs + cleanup
- Update `docs/OPENCLAW_INTEGRATION.md`
- Add `docs/OPENCLAW_SESSION_BRIDGE.md` with operational guide
- Cleanup: remove any dead code paths

---

## Acceptance Criteria

1. **Shadow mode works**: `OPENCLAW_WS_ENABLED=true` + `OPENCLAW_MODE=http` — events are stored in `openclaw_events` table while UI still uses HTTP SSE. No behavior change for users.
2. **Bridge mode works**: `OPENCLAW_MODE=ws_bridge` — chat renders token-by-token from WS events. Tool events appear in real-time.
3. **Background events visible**: Events that OpenClaw generates without llm-bawt initiating a request (heartbeat, cron, subagent) appear in the EventStore and can be surfaced to UI.
4. **Client disconnect = no data loss**: Killing a voice/chat stream mid-response has zero effect — the WS bridge continues receiving and persisting.
5. **Reconnect with replay**: If WS drops and reconnects, the gap is backfilled and no events are lost or duplicated.
6. **Legacy fallback**: `OPENCLAW_MODE=http` works exactly as before for instant rollback.
7. **All existing tests pass**: No regressions in `test_stream_persistence.py`, `test_openclaw.py`, or any other test file.
8. **New tests pass**: `tests/test_openclaw_bridge.py`, `tests/test_openclaw_ws.py`

---

## Protocol Decisions

### 1. WS Endpoint
`ws://{host}:{port}/v1/ws` — follows the existing `/v1/` convention on the gateway. Config field `OPENCLAW_WS_URL` defaults to `ws://127.0.0.1:18789/v1/ws`.

### 2. Auth
Bearer token in HTTP upgrade headers — consistent with all other providers in this codebase:
```
GET /v1/ws HTTP/1.1
Authorization: Bearer {OPENCLAW_GATEWAY_TOKEN}
Upgrade: websocket
```
No separate auth message after connect. If the token is invalid, the gateway rejects the upgrade with 401.

### 3. Subscribe Protocol
After connect, send JSON:
```json
{"type": "subscribe", "session_keys": ["main"]}
```
Gateway acknowledges:
```json
{"type": "subscribed", "session_keys": ["main"]}
```
Multiple sessions on one connection supported. Unsubscribe:
```json
{"type": "unsubscribe", "session_keys": ["main"]}
```

### 4. Event Format over WS
Same event types as SSE (`response.created`, `response.output_text.delta`, `response.output_item.added`, `response.completed`) wrapped in a thin envelope:
```json
{
  "type": "event",
  "session_key": "main",
  "event_id": "evt_abc123",
  "event_type": "response.output_text.delta",
  "data": { ... same as SSE data field ... }
}
```
This reuses all existing `_iter_sse_events` parsing logic — the `event_type` + `data` fields are identical to what the SSE parser already handles. The envelope adds `session_key` and `event_id` for routing and deduplication.

### 5. Cursor / Replay
No gateway-side cursor replay initially. On reconnect, llm-bawt:
1. Checks `openclaw_session_state.last_event_id` for each subscribed session
2. Uses the existing `_fetch_session_tool_calls_gateway()` endpoint (`POST /tools/invoke` with `tool: "sessions_history"`) to backfill any gap
3. Deduplicates via `event_dedupe_key` in EventStore

This is sufficient because reconnect gaps should be short (seconds to low minutes). If gateway adds `since_event_id` support on subscribe later, we add it as an optimization.

### 6. Send Message via WS
User messages are sent over the WS connection:
```json
{"type": "chat.send", "session_key": "main", "text": "hello", "metadata": {"bot_id": "proto"}}
```
Gateway responds with a confirmation containing the `run_id`:
```json
{"type": "chat.sent", "session_key": "main", "run_id": "run_xyz", "message_id": "msg_abc"}
```
The actual response events then arrive on the subscription stream. HTTP POST to `/v1/responses` remains as fallback when `OPENCLAW_MODE=http`.

### 7. Multiple Sessions
One WS connection subscribes to multiple sessions via the `session_keys` array. Start with one (`"main"`), but the architecture supports subscribing to additional sessions at runtime (e.g., per-bot sessions).

### 8. Event Deduplication Key
Every WS event includes an `event_id` field in the envelope (e.g., `evt_abc123`). This is the dedupe key stored in `openclaw_events.event_dedupe_key`. If the gateway doesn't include one (e.g., for heartbeat events), `EventIngestPipeline` synthesizes a collision-resistant key:
```python
sha256(f"{session_key}:{event_type}:{canonical_json(data)}:{stream_seq}".encode()).hexdigest()
```
where `stream_seq` is a per-connection monotonically incrementing counter maintained by `OpenClawWsClient`. This avoids timestamp-based collisions and reordering issues.

### 9. Event Ordering
The envelope includes a `seq` field — monotonically increasing integer per session, assigned by the gateway. llm-bawt uses this as the primary ordering key within a session.

If `seq` is not available (e.g., gateway hasn't implemented it yet), the client falls back to:
1. Receive-time ordering (`OpenClawWsClient` stamps each message on arrival)
2. Stable tie-breaker via `stream_seq` (per-connection counter)
3. Treat as best-effort ordering — document that delta reassembly may be slightly off during the transition period

The `openclaw_events` table gets an additional column:
```sql
seq BIGINT,  -- Gateway-assigned monotonic sequence per session (nullable during transition)
```
With index: `idx_openclaw_events_session_seq ON (session_key, seq)`.

### 10. Idempotent Message Send
`chat.send` includes an optional `idempotency_key` to prevent duplicate sends during reconnect/retry:
```json
{"type": "chat.send", "session_key": "main", "text": "hello", "idempotency_key": "idem_abc123", "metadata": {"bot_id": "proto"}}
```
Gateway echoes it in `chat.sent`:
```json
{"type": "chat.sent", "session_key": "main", "run_id": "run_xyz", "message_id": "msg_abc", "idempotency_key": "idem_abc123"}
```
`OpenClawWsClient.send_user_message()` generates an idempotency key (UUID4) and caches `{key → run_id}` for a short TTL (60s). On reconnect, if `send_user_message()` is retried with the same key, the gateway returns the existing `run_id` instead of creating a duplicate run.

### 11. Backfill Limitations
**Important:** Backfill via `sessions_history` endpoint recovers **final transcript and tool call records**, not individual streaming deltas. After a reconnect gap:
- `openclaw_runs.full_text` will be populated correctly (from transcript)
- `openclaw_events` will contain a single `ASSISTANT_DONE` event per missed run (not per-delta `ASSISTANT_DELTA` events)
- UI behavior: if a user reconnects mid-run, they see the final assembled text appear at once rather than token-by-token replay. This is acceptable — the alternative (replaying cached deltas) adds significant complexity for marginal UX benefit.
- Tool calls are fully recovered (the transcript includes them)

### 12. Gateway Coordination
The `ws://.../v1/ws` endpoint is **new gateway work** — it does not exist today. This refactor requires coordinated implementation:

| Side | Work |
|------|------|
| **Gateway (OpenClaw)** | Implement `/v1/ws` endpoint: upgrade handler, subscribe/unsubscribe protocol, event envelope with `session_key` + `event_id` + `seq`, `chat.send` with idempotency, bridge to existing session event bus |
| **llm-bawt** | Everything in this plan — WS client, EventStore, FanoutHub, SessionBridge, service integration |

**Sequencing:** Gateway WS endpoint must be available before llm-bawt can test Commits 1+. Commit 0 (EventStore, event model, migrations) has no gateway dependency and can proceed immediately. The test suite uses mocked WS connections throughout, so llm-bawt development can proceed in parallel with gateway work — but E2E validation requires both sides.
