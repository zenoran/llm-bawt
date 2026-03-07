# OpenClaw Gateway WebSocket Wire Format

Wire format for the `/v1/ws` gateway stream.

## Outer Frame

Every WS message is a JSON object with a common envelope:

```json
{
  "type": "event",
  "event": "agent" | "chat" | "...",
  "payload": { ... },
  "seq": 123
}
```

Other `type` values (`ping`, `pong`, `heartbeat`, `chat.sent`, `req`, `res`)
are described in [Other Frame Types](#other-frame-types).

---

## event: "agent"

The authoritative stream for run lifecycle, assistant token deltas, tool
calls, and errors. The `payload` always has this shape:

```json
{
  "runId": "run-uuid-or-idem-key",
  "seq": 7,
  "stream": "lifecycle" | "assistant" | "tool" | "error",
  "ts": 1772850000000,
  "sessionKey": "main",
  "data": { ... }
}
```

### stream: "lifecycle"

Controls run boundaries.

**start** — a new run has begun:

```json
{ "data": { "phase": "start" } }
```

**end** — the run completed normally:

```json
{ "data": { "phase": "end", "startedAt": 200, "endedAt": 210 } }
```

**error** — the run errored out:

```json
{ "data": { "phase": "error", "error": "boom" } }
```

**fallback** — provider failover occurred:

```json
{ "data": { "phase": "fallback", "selectedProvider": "...", "activeProvider": "..." } }
```

### stream: "assistant"

Incremental token chunks from the model:

```json
{
  "runId": "r1",
  "seq": 2,
  "stream": "assistant",
  "ts": 1772850000100,
  "sessionKey": "main",
  "data": { "text": "partial token chunk..." }
}
```

### stream: "tool"

Tool use events. Each tool invocation produces a start/end pair.

**start** — tool is being called:

```json
{
  "runId": "r1",
  "seq": 4,
  "stream": "tool",
  "ts": 1772850000150,
  "sessionKey": "main",
  "data": {
    "phase": "start",
    "name": "web_search",
    "arguments": { "query": "..." }
  }
}
```

`data` fields for start:
- `phase`: `"start"` or `"calling"`
- `name` (or `tool`): tool function name
- `arguments` (or `args`, `input`): the tool input object

**end** — tool returned a result:

```json
{
  "runId": "r1",
  "seq": 5,
  "stream": "tool",
  "ts": 1772850000200,
  "sessionKey": "main",
  "data": {
    "phase": "end",
    "name": "web_search",
    "result": { "snippets": ["..."] }
  }
}
```

`data` fields for end:
- `phase`: `"end"`, `"result"`, or `"done"`
- `name` (or `tool`): tool function name
- `result` (or `output`): the tool's return value (any JSON) — **often absent**
- `meta`: short summary string (e.g. search query, file path) — used as fallback when `result` is absent
- `isError`: boolean, whether the tool errored

### stream: "error"

Agent-level error (distinct from lifecycle `phase:"error"`):

```json
{
  "runId": "r1",
  "seq": 6,
  "stream": "error",
  "ts": 1772850000300,
  "sessionKey": "main",
  "data": { "message": "something went wrong", "code": "RATE_LIMIT" }
}
```

---

## event: "chat"

UI-friendly synthesized chat stream. Provides assembled message objects
rather than raw token deltas.

### state: "delta"

Partial assembled message (streaming):

```json
{
  "runId": "r1",
  "sessionKey": "main",
  "seq": 2,
  "state": "delta",
  "message": {
    "role": "assistant",
    "content": [{ "type": "text", "text": "current assembled text" }],
    "timestamp": 1772850000150
  }
}
```

### state: "final"

Complete message when the run finishes:

```json
{
  "runId": "r1",
  "sessionKey": "main",
  "seq": 4,
  "state": "final",
  "message": {
    "role": "assistant",
    "content": [{ "type": "text", "text": "final text" }],
    "timestamp": 1772850000300
  }
}
```

`message` may be omitted for silent/heartbeat-style cases.

### state: "error"

```json
{
  "runId": "r1",
  "sessionKey": "main",
  "seq": 4,
  "state": "error",
  "errorMessage": "..."
}
```

---

## Other Frame Types

| `type` | Description | Ingested? |
|---|---|---|
| `ping` / `pong` / `heartbeat` | Keep-alive frames | Dropped |
| `chat.sent` | User message sent confirmation | Yes → `USER_MESSAGE` |
| `req` / `res` | Request/response control frames | Dropped |

---

## Bridge Ingest Mapping

How the bridge maps wire events to internal `OpenClawEventKind`:

| Wire path | Internal kind | Notes |
|---|---|---|
| `agent` → `lifecycle` → `phase:"start"` | `RUN_STARTED` | Creates run record |
| `agent` → `lifecycle` → `phase:"end"` | `RUN_COMPLETED` | Assembles full text + tool calls |
| `agent` → `lifecycle` → other phase | `SYSTEM_NOTE` | `fallback`, `error`, etc. |
| `agent` → `assistant` | `ASSISTANT_DELTA` | Reads `data.text` or `data.delta` |
| `agent` → `tool` → start/calling | `TOOL_START` | Captures name + arguments |
| `agent` → `tool` → end/result/done | `TOOL_END` | Captures name + result |
| `agent` → `error` | `ERROR` | |
| `chat` → `state:"final"` | `ASSISTANT_DONE` | Publishes to history |
| `chat` → `state:"delta"` | *(dropped)* | Redundant with agent assistant stream |
| `chat` → `state:"error"` | `SYSTEM_NOTE` *(gap)* | Falls through; `errorMessage` not extracted |
| `chat.sent` | `USER_MESSAGE` | Content filters applied |
