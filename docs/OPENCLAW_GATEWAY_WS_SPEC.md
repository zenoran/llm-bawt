# OpenClaw Gateway — `/v1/ws` WebSocket Protocol Reference

Documented from live testing against gateway `2026.2.26`, protocol v3.

---

## 1. Connection Lifecycle

### Upgrade

Plain WebSocket upgrade — no auth headers required at this stage.

```
GET /v1/ws HTTP/1.1
Host: gateway:18789
Upgrade: websocket
Connection: Upgrade
```

### Challenge-Response Handshake

Immediately after upgrade, the gateway sends a challenge:

```json
{
  "type": "event",
  "event": "connect.challenge",
  "payload": {
    "nonce": "4a4d2628-617d-4bb3-8f20-60f8ea75380d",
    "ts": 1772748905452
  }
}
```

Client must respond with a `connect` request including auth:

```json
{
  "type": "req",
  "id": "<uuid>",
  "method": "connect",
  "params": {
    "minProtocol": 3,
    "maxProtocol": 3,
    "client": {
      "id": "llm-bawt",
      "version": "0.1.0",
      "platform": "linux",
      "mode": "cli"
    },
    "role": "operator",
    "scopes": [
      "operator.read",
      "operator.write",
      "operator.admin",
      "chat.read",
      "chat.write",
      "session.read",
      "session.write"
    ],
    "auth": {
      "token": "<OPENCLAW_GATEWAY_TOKEN>"
    }
  }
}
```

**Required scopes:** `operator.write` is needed for `chat.send`, `operator.admin` for session operations.

Gateway responds:

```json
{
  "type": "res",
  "id": "<same-uuid>",
  "ok": true,
  "payload": {
    "type": "hello-ok",
    "protocol": 3,
    "server": {
      "version": "2026.2.26",
      "connId": "05a725c5-..."
    },
    "features": {
      "methods": ["chat.send", "chat.history", "chat.abort", "status", "..."],
      "events": ["agent", "chat", "health", "heartbeat", "..."]
    }
  }
}
```

### Keepalive

WebSocket-level ping/pong. Recommended: 20s ping interval, 10s timeout.

### Close

Standard WebSocket close frame.

---

## 2. Request/Response Pattern

All client-to-gateway calls use the same envelope:

### Request

```json
{
  "type": "req",
  "id": "<unique-request-id>",
  "method": "<method-name>",
  "params": { }
}
```

### Response

```json
{
  "type": "res",
  "id": "<same-request-id>",
  "ok": true,
  "payload": { }
}
```

On error:

```json
{
  "type": "res",
  "id": "<same-request-id>",
  "ok": false,
  "error": {
    "code": "INVALID_REQUEST",
    "message": "description of error"
  }
}
```

**Important:** Responses may arrive interleaved with `type: "event"` messages. Match responses by `id`, not by order.

---

## 3. Sending Messages — `chat.send`

### Request

```json
{
  "type": "req",
  "id": "<uuid>",
  "method": "chat.send",
  "params": {
    "sessionKey": "main",
    "message": "hello",
    "idempotencyKey": "idem_<uuid>"
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `sessionKey` | yes | Session to send to |
| `message` | yes | User message text |
| `idempotencyKey` | yes | Dedup key for retry safety |

### Response

```json
{
  "type": "res",
  "id": "<same-uuid>",
  "ok": true,
  "payload": {
    "runId": "idem_<...>"
  }
}
```

The `runId` is returned immediately. Streaming events for this run then flow as `agent` and `chat` events (Section 4).

---

## 4. Event Stream

After connecting, the gateway pushes events as they occur. No explicit subscribe is needed — events flow for all sessions visible to the authenticated operator.

### Event Envelope

```json
{
  "type": "event",
  "event": "<event-name>",
  "payload": { }
}
```

### Agent Events (`event: "agent"`)

These carry the streaming content for a run. The `payload.stream` field indicates the sub-type:

#### Lifecycle — Run Start

```json
{
  "type": "event",
  "event": "agent",
  "payload": {
    "runId": "idem_...",
    "stream": "lifecycle",
    "data": {
      "phase": "start",
      "startedAt": 1772749033513
    },
    "sessionKey": "agent:main:main",
    "seq": 1,
    "ts": 1772749033513
  }
}
```

#### Assistant Text Delta

```json
{
  "type": "event",
  "event": "agent",
  "payload": {
    "runId": "idem_...",
    "stream": "assistant",
    "data": {
      "text": "ok",
      "delta": "ok"
    },
    "sessionKey": "agent:main:main",
    "seq": 2,
    "ts": 1772749039486
  }
}
```

#### Lifecycle — Run End

```json
{
  "type": "event",
  "event": "agent",
  "payload": {
    "runId": "idem_...",
    "stream": "lifecycle",
    "data": {
      "phase": "end",
      "endedAt": 1772749039862
    },
    "sessionKey": "agent:main:main",
    "seq": 3,
    "ts": 1772749039862
  }
}
```

#### Lifecycle — Fallback Cleared

Emitted when the agent selects a model/provider:

```json
{
  "type": "event",
  "event": "agent",
  "payload": {
    "runId": "idem_...",
    "stream": "lifecycle",
    "data": {
      "phase": "fallback_cleared",
      "selectedProvider": "openai-codex",
      "selectedModel": "..."
    },
    "sessionKey": "agent:main:main"
  }
}
```

#### Error

```json
{
  "type": "event",
  "event": "agent",
  "payload": {
    "runId": "idem_...",
    "stream": "error",
    "data": {
      "reason": "seq gap",
      "expected": 1,
      "received": 4
    },
    "sessionKey": "agent:main:main",
    "ts": 1772749039879
  }
}
```

### Chat Events (`event: "chat"`)

Higher-level message events with assembled content:

#### Delta (streaming)

```json
{
  "type": "event",
  "event": "chat",
  "payload": {
    "runId": "idem_...",
    "sessionKey": "agent:main:main",
    "seq": 2,
    "state": "delta",
    "message": {
      "role": "assistant",
      "content": [{"type": "text", "text": "ok"}],
      "timestamp": "..."
    }
  }
}
```

#### Final (complete message)

```json
{
  "type": "event",
  "event": "chat",
  "payload": {
    "runId": "idem_...",
    "sessionKey": "agent:main:main",
    "seq": 3,
    "state": "final",
    "message": {
      "role": "assistant",
      "content": [{"type": "text", "text": "ok"}],
      "timestamp": "..."
    }
  }
}
```

### System Events

| Event | Description |
|-------|-------------|
| `health` | Periodic health check with channel/service status |
| `heartbeat` | Connection keepalive |
| `tick` | Periodic tick with timestamp |

These can be ignored for chat purposes.

---

## 5. Event Mapping for llm-bawt Ingest

| Gateway Event | Stream/State | llm-bawt EventKind |
|---------------|-------------|---------------------|
| `agent` | `stream: "lifecycle"`, `phase: "start"` | `RUN_STARTED` |
| `agent` | `stream: "assistant"` | `ASSISTANT_DELTA` |
| `agent` | `stream: "lifecycle"`, `phase: "end"` | `RUN_COMPLETED` |
| `agent` | `stream: "error"` | `ERROR` |
| `chat` | `state: "delta"` | `ASSISTANT_DELTA` (alternative to agent.assistant) |
| `chat` | `state: "final"` | `ASSISTANT_DONE` |

**Note:** Both `agent` (stream: assistant) and `chat` (state: delta) events carry the same text. Use `agent` events for streaming deltas and `chat` (state: final) for the assembled complete message.

---

## 6. Available Methods

Full list from gateway `2026.2.26` (connect payload `features.methods`):

**Chat:** `chat.send`, `chat.history`, `chat.abort`

**Sessions:** `sessions.list`, `sessions.delete`, `sessions.compact`, `sessions.patch`, `sessions.preview`, `sessions.reset`

**Agents:** `agents.list`, `agents.create`, `agents.update`, `agents.delete`, `agent`, `agent.identity.get`, `agent.wait`, `agents.files.list`, `agents.files.get`, `agents.files.set`

**Config:** `config.get`, `config.set`, `config.patch`, `config.apply`, `config.schema`

**TTS:** `tts.status`, `tts.providers`, `tts.enable`, `tts.disable`, `tts.convert`, `tts.setProvider`

**System:** `health`, `status`, `logs.tail`, `channels.status`, `channels.logout`, `usage.status`, `usage.cost`, `models.list`

**Other:** `cron.*`, `node.*`, `device.*`, `exec.*`, `skills.*`, `wizard.*`, `browser.request`, `secrets.reload`, `wake`, `send`, `voicewake.*`, `talk.*`

---

## 7. Quick Smoke Test

```python
import asyncio, json, uuid

async def smoke():
    import websockets

    url = "ws://127.0.0.1:18789/v1/ws"
    token = "YOUR_TOKEN"

    ws = await websockets.connect(url, ping_interval=None)

    # 1. Challenge
    challenge = json.loads(await ws.recv())
    assert challenge["event"] == "connect.challenge"

    # 2. Connect
    req_id = uuid.uuid4().hex
    await ws.send(json.dumps({
        "type": "req", "id": req_id, "method": "connect",
        "params": {
            "minProtocol": 3, "maxProtocol": 3,
            "client": {"id": "test", "version": "0.1.0", "platform": "linux", "mode": "cli"},
            "role": "operator",
            "scopes": ["operator.read", "operator.write", "operator.admin",
                       "chat.read", "chat.write", "session.read", "session.write"],
            "auth": {"token": token},
        },
    }))

    # Wait for connect response (skip interleaved events)
    while True:
        res = json.loads(await ws.recv())
        if res.get("type") == "res" and res.get("id") == req_id:
            assert res["ok"], f"Connect failed: {res.get('error')}"
            print("Connected:", res["payload"]["server"])
            break

    # 3. Send a message
    chat_id = uuid.uuid4().hex
    await ws.send(json.dumps({
        "type": "req", "id": chat_id, "method": "chat.send",
        "params": {
            "sessionKey": "main",
            "message": "hello from smoke test",
            "idempotencyKey": f"idem_{uuid.uuid4().hex}",
        },
    }))

    # 4. Collect events
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("type") == "res" and msg.get("id") == chat_id:
            print(f"chat.send ok={msg['ok']}, runId={msg.get('payload', {}).get('runId')}")
        elif msg.get("type") == "event" and msg.get("event") == "agent":
            p = msg["payload"]
            stream = p.get("stream")
            if stream == "assistant":
                print(p["data"].get("delta", ""), end="", flush=True)
            elif stream == "lifecycle" and p["data"].get("phase") == "end":
                print("\nRun completed.")
                break
            elif stream == "error":
                print(f"\nError: {p['data']}")
                break

    await ws.close()

asyncio.run(smoke())
```
