# Codex Bridge

The Codex bridge integrates OpenAI's Codex agent into llm-bawt using the
[OpenAI Codex SDK](https://github.com/openai/codex/tree/main/sdk/python).
It runs alongside the OpenClaw and Claude Code bridges and uses the same
Redis event protocol — the main app doesn't know which bridge handled a
request.

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │  llm-bawt-app (FastAPI)                     │
                        │  AgentBackendClient → CodexBackend          │
                        └──────────────┬──────────────────────────────┘
                                       │ Redis: openclaw:commands
                                       │ (fields: backend, session_key, message, ...)
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        Redis (shared stream)                             │
│                        openclaw:commands                                 │
│                                                                          │
│  ┌─────────────────────────┐  ┌──────────────────────────┐  ┌──────────┐ │
│  │ openclaw-bridge         │  │ claude-code-bridge       │  │ codex-   │ │
│  │ backend=openclaw        │  │ backend=claude-code      │  │ bridge   │ │
│  │ group: bridge           │  │ group: claude-code-bridge│  │ b=codex  │ │
│  └──────────┬──────────────┘  └──────────┬───────────────┘  │ group:   │ │
│             │                            │                  │ codex-   │ │
└─────────────┼────────────────────────────┼──────────────────┤ bridge   │─┘
              │                            │                  └────┬─────┘
              ▼                            ▼                       ▼
   OpenClaw Gateway (WS)      Claude Code Binary (stdio)   Codex Rust binary (stdio)
              │                            │                       │
              ▼                            ▼                       ▼
      Anthropic API                Anthropic API              OpenAI API
      (via gateway)                (OAuth subscription)       (ChatGPT-mode OAuth)
```

All three bridges publish events to `openclaw:run:{request_id}` in the
same `OpenClawEvent` format. The main app consumes them identically.

## How It Works

1. A bot with `agent_backend: "codex"` receives a chat request
2. `CodexBackend` sends a `chat.send` command to Redis with `backend: "codex"`
3. The codex bridge picks it up (filters by `backend` field)
4. Bridge calls into a long-lived `AsyncCodex` (one per container) which
   talks JSON-RPC 2.0 over stdio to the bundled `codex` Rust binary
5. The binary authenticates via the bind-mounted `~/.codex/auth.json`
   (ChatGPT-mode OAuth — your ChatGPT Plus/Pro/Team subscription)
6. SDK streams notifications: agentMessage deltas, commandExecution items,
   fileChange items, webSearch, mcpToolCall, etc.
7. Bridge translates them into `OpenClawEvent`s, mapping Codex item types
   to Claude tool names (Bash / Edit / Write / MultiEdit / WebSearch /
   Grep / WebFetch / Read) so the existing `ClaudeToolCallCard.tsx`
   renders them correctly. (Provider-aware UI mapping is TASK-212.)
8. Main app streams them as SSE to the frontend.

## Architectural decisions (locked)

1. **OAuth only — no API key.** `auth.json` from the host's `~/.codex/`
   is bind-mounted RW into the bridge at `/home/bridge/.codex/auth.json`.
   The Codex Rust binary self-refreshes the OAuth tokens and writes back
   to the same inode. `CODEX_API_KEY` and `OPENAI_API_KEY` env vars are
   scrubbed at startup.
2. **Single AsyncCodex per container.** Hosts multiple concurrent threads
   (one per session), bounded by `[agents] max_threads` in `~/.codex/config.toml`.
   Per-session ordering is enforced by the shared `SessionQueue` lock in
   `openclaw_bridge.session_queue`. One bridge container, not many — Docker
   `restart: unless-stopped` handles process death.
3. **Repo-managed local plugins are staged at startup.** The bridge mirrors
   `~/dev/agent-skills/codex/.codex-plugin/marketplace.json` into
   `~/.agents/plugins/marketplace.json` and materializes plugin directories
   under `~/plugins/` before the SDK starts. This makes the shared
   `agent-skills` repo available through Codex's home-local plugin contract
   even though the bridge cwd is the broader `~/dev`.
4. **Self-healing supervisor.** If the codex subprocess dies or auth fails
   mid-turn, the supervisor publishes ERROR + run_done, tears down
   `AsyncCodex`, and rebuilds it on the next request. Rebuild re-reads
   `auth.json` off disk, so `codex login` on the host self-heals the
   bridge with no `docker restart` needed.
5. **Reuses openclaw_bridge infrastructure unchanged.** Imports
   `RedisPublisher`, `OpenClawEvent`, `OpenClawEventKind`,
   `synthesize_event_id`, `SessionQueue`, `COMMANDS_STREAM` directly.
   Consumer group is `codex-bridge`; filters on `backend == "codex"`
   and ACKs others.
6. **Session continuity via thread_id.** The bot's
   `agent_backend_config.session_key` stores the Codex `thread_id`
   (managed by the bridge — not human-edited). On `chat.send` the
   bridge looks it up and calls `thread_resume()`. On model change →
   start fresh thread. On resume failures (`thread not found`,
   `Missing required parameter: input[N].encrypted_content`,
   `no rollout found`) → clear session and retry once. `/new` prefix
   in the user message clears the session.

## Authentication

The bridge uses your **ChatGPT subscription** via OAuth — not OpenAI API
billing.

### One-time provisioning (on host)

```bash
# Install codex CLI on the host (npm package — bundles the static-pie
# Rust binary the bridge depends on; pip's openai-codex-sdk wheel ships
# without it).
npm install -g @openai/codex

# Log in with ChatGPT mode — opens browser for OAuth
codex login

# Verify the bundle
ls -la ~/.codex/auth.json
python3 -c "
import json
d = json.load(open('/home/$USER/.codex/auth.json'))
tk = d.get('tokens', {})
print('auth_mode:', tk.get('auth_mode') or d.get('auth_mode'))
print('has_refresh:', bool(tk.get('refresh_token')))
"
```

`auth_mode` must be `chatgpt`. The bridge will hard-fail at startup
otherwise.

The compose service mounts `~/.codex/auth.json` **rw** so the codex
binary inside the container can refresh tokens on disk. The shared
inode means `codex login` on the host updates the same file the bridge
sees.

The compose service also bind-mounts the host's npm
`@openai/codex/node_modules/@openai/codex-linux-x64/vendor` directory
into the SDK's expected path (`<sdk>/vendor/x86_64-unknown-linux-musl/
codex/codex`). The static-pie musl binary works on any glibc/musl
Linux, so no in-container install is needed.

### Shared local skills

The bridge does not rely on repo-local `.agents/` discovery because its
working directory is `~/dev`, not a single repo root. Instead, on startup
it stages the Codex-specific mapping checked into `~/dev/agent-skills/codex`
into Codex's home-local layout:

- `~/dev/agent-skills/codex/.codex-plugin/marketplace.json`
  → `~/.agents/plugins/marketplace.json`
- `~/dev/agent-skills/codex/plugins/<plugin>`
  → `~/plugins/<plugin>`

Skill entries inside that plugin are re-linked to the live repo copy under
`~/dev/agent-skills/<skill>` and to built-in system skills under
`~/.codex/skills/.system/<skill>`. That avoids host-specific absolute paths
inside the repo mapping while still making the shared skills available to the
SDK in-container.

## Setup

### 1. Ensure auth and config exist on the host

```bash
ls ~/.codex/auth.json ~/.codex/config.toml
```

If `config.toml` doesn't exist or doesn't set `[agents] max_threads`,
add one to bound concurrent threads:

```toml
# ~/.codex/config.toml
model = "gpt-5.4"
model_reasoning_effort = "high"

[agents]
max_threads = 10
```

### 2. (Optional) Pin the model in `.env`

```bash
# .env
CODEX_MODEL=gpt-5.4
CODEX_BRIDGE_LOG_LEVEL=INFO
```

### 3. Build + start the bridge

```bash
docker compose up -d --build codex-bridge

# Verify health
curl -fsS http://localhost:8682/health
# {"redis": true}
```

### 4. Create a sample codex bot

```bash
curl -X POST http://localhost:8642/v1/bots \
  -H "Content-Type: application/json" \
  -d '{
    "slug": "codex",
    "name": "Codex",
    "color": "#0ea5e9",
    "bot_type": "agent",
    "agent_backend": "codex",
    "agent_backend_config": {"model": "gpt-5.4"},
    "system_prompt": "You are a coding agent named Codex. Be concise. Use tools to inspect, edit, and run code as needed."
  }'
```

The bot's `session_key` is managed by the bridge — leave it unset on
creation; the bridge populates it after `thread/started` fires on the
first turn.

## Configuration reference

Environment variables read by the bridge (via `docker-compose.yml`):

| Var | Default | Description |
|-----|---------|-------------|
| `CODEX_HOME` | `/home/bridge/.codex` | exported to the SDK; matches the auth.json mount |
| `CODEX_AUTH_PATH` | `/home/bridge/.codex/auth.json` | OAuth bundle path |
| `CODEX_MODEL` | `gpt-5.4` | default model when chat.send omits it |
| `CODEX_BACKEND_NAME` | `codex` | Redis stream `backend` filter |
| `CODEX_BRIDGE_LOG_LEVEL` | `INFO` | log level |
| `CODEX_BRIDGE_HEALTH_PORT` | `8682` | `/health` TCP port |
| `CODEX_BRIDGE_REQUEST_TIMEOUT` | `300` | per-call SDK timeout, seconds |
| `CODEX_BRIDGE_CWD` | `/home/bridge/dev` | codex thread cwd |
| `CODEX_LOCAL_PLUGINS_ENABLED` | `1` | stage repo-managed local plugins into `~/.agents` + `~/plugins` before SDK startup |
| `CODEX_LOCAL_PLUGINS_SRC` | `/home/bridge/dev/agent-skills/codex` | source repo containing the Codex-specific marketplace/plugin mapping |
| `CODEX_DEV_ROOT` | `/home/bridge/dev` | dev root used to resolve repo-managed skill directories |
| `CODEX_BIN` | `<bundled>` | optional explicit codex binary path |
| `LLM_BAWT_API_URL` | (set in compose) | for `/v1/bots` profile read/write |
| `REDIS_URL` | (set in compose) | shared Redis |

## Operational runbook

### Add a new codex bot
See "Setup → Create a sample codex bot" above. `slug` must be unique;
the bridge filters chat.send commands by `backend=codex` and looks up
the bot by `bot_id` to read/write `agent_backend_config.session_key`.

### Reset a stuck session
Two ways:

```bash
# REST: clears session_key in agent_backend_config
curl -X POST http://localhost:8642/v1/chat/session/reset \
  -H "Content-Type: application/json" \
  -d '{"bot_id": "codex"}'

# Or send "/new" as the next user message — bridge intercepts the
# prefix, clears session, and replies "Session reset" without an LLM call.
```

The bridge also recovers automatically when the persisted thread is
stale: a `thread not found` / encrypted_content / no-rollout error on
resume triggers a one-shot fresh-thread retry that the user never sees.

### Recover from OAuth failure (no docker restart)
The Codex Rust binary self-refreshes tokens. If the refresh token is
revoked or `auth.json` is corrupt:

```bash
# On the host where compose runs:
codex login           # writes a fresh ~/.codex/auth.json

# Bridge auto-heals on the next chat.send — no docker action needed.
# Optional: tail logs to confirm AsyncCodex was rebuilt:
docker logs -f --tail 50 llm-bawt-codex-bridge
```

If the bridge had already crashed (SIGTERM during shutdown), Docker's
`restart: unless-stopped` brings it back automatically.

### Recycle the bridge if the codex subprocess seems stuck
The supervisor should detect process death and rebuild on the next
request. If something pathological is wedging it (e.g. zombie
subprocess, full disk, stale lockfile):

```bash
docker restart llm-bawt-codex-bridge
docker logs -f --tail 100 llm-bawt-codex-bridge
```

### Free disk space
Codex writes a JSONL rollout per session under
`~/.codex/sessions/YYYY/MM/DD/`. The bridge runs a periodic cleanup
every 6 hours that deletes:

- `~/.codex/cache/`, `log/`, `tmp/`, `.tmp/`, `shell_snapshots/`
- `~/.codex/sessions/**/*.jsonl` older than 24 hours

Files preserved: `auth.json`, `config.toml`, `installation_id`, `version.json`.

If a bot's persisted thread had its rollout pruned, `thread_resume`
fails with "no rollout found" → the bridge's recoverable-error path
clears the session and starts fresh on the next turn.

### Log signals to watch for in production

| Log line | Meaning |
|----------|---------|
| `CodexBridge started (...)` | Bridge initialized; Redis ok; auth.json validated |
| `Codex OAuth ok: ... auth_mode=chatgpt ...` | Auth bundle accepted |
| `AsyncCodex started (...)` | Codex Rust binary spawned successfully |
| `Session persisted: <bot> -> <thread_id>` | First turn for a fresh bot — session_key written |
| `Session cleared: <bot> (had_session=true)` | Manual reset or recoverable error recovery |
| `chat.abort: session=<sk> detail=turn_interrupted,...` | Server-side interrupt landed |
| `Codex auth failure ...` | OAuth refresh failed — `codex login` on host required |
| `Codex app-server died ... — tearing down` | Subprocess crash; will rebuild on next request |
| `Recoverable session error ... — clearing session and retrying fresh` | Stale thread; retry will succeed |
| `Cache cleanup: removed N stale entries` | Periodic pruning ran |

## End-to-end smoke tests

> Run these from the host running compose. Replace `BOT=codex` with your
> bot's slug. Each curl uses streaming SSE; the chat UI is the better
> interactive surface.

### 1. First-turn round trip
```bash
curl -N -X POST http://localhost:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"codex","stream":true,"bot_id":"codex","messages":[
        {"role":"user","content":"List the files under /home/bridge/dev"}]}'
```
Expect: a Bash tool card (ls), then an assistant summary. Verify
`session_key` is now populated:
```bash
curl -s http://localhost:8642/v1/bots | python3 -c "import json,sys; \
  d=json.load(sys.stdin); \
  bot=next(b for b in d['data'] if b['slug']=='codex'); \
  print(bot['agent_backend_config'])"
```

### 2. Session resume
Send a follow-up that requires memory of the previous turn:
```bash
# Same model/bot, new request
... -d '{"model":"codex","stream":true,"bot_id":"codex","messages":[
        {"role":"user","content":"Which of those files is largest?"}]}'
```
Bridge logs should show `thread_resume` (not a fresh start). The reply
should reference the prior `ls` output.

### 3. Edit rendering
```bash
... '{"role":"user","content":"Add a single-line comment at the top of /home/bridge/dev/llm-bawt/README.md noting today is 2026-05-03"}'
```
Expect an Edit card with red/green diff in the chat UI.

### 4. chat.abort
Start a long task, then abort:
```bash
TURN_ID=$(curl -s -X POST http://localhost:8642/v1/chat/completions \
  ... '{"role":"user","content":"Run sleep 60 then list /tmp"}' \
  | grep -o 'turn_[a-f0-9]*' | head -1)
curl -X POST http://localhost:8642/v1/chat/abort \
  -H "Content-Type: application/json" -d "{\"turn_id\": \"$TURN_ID\"}"
```
Bridge log: `chat.abort: session=<sk> detail=turn_interrupted,...`.
Turn ends with status=interrupted.

### 5. session.reset
```bash
curl -X POST http://localhost:8642/v1/chat/session/reset \
  -H "Content-Type: application/json" -d '{"bot_id":"codex"}'
```
Verify `agent_backend_config.session_key` is empty; next chat.send starts
a fresh thread (visible in logs).

### 6. Resume failure recovery (manual)
Corrupt the bot's `session_key` to a bogus id, then send a chat:
```bash
# (use the bots admin UI or PATCH /v1/bots/codex/profile)
# Send a chat — bridge gets "thread not found", clears session, retries fresh, succeeds.
```
No ERROR event surfaces; the user just sees a normal response.

### 7. OAuth self-heal
```bash
# On host:
mv ~/.codex/auth.json ~/.codex/auth.json.bak
# Send a chat — gets ERROR "Codex OAuth failed — re-run codex login on echo"
mv ~/.codex/auth.json.bak ~/.codex/auth.json   # or: codex login
# Send another chat — succeeds (no docker restart)
```

### 8. Model change
```bash
curl -X PATCH http://localhost:8642/v1/bots/codex/profile \
  -H "Content-Type: application/json" \
  -d '{"agent_backend_config": {"model": "gpt-5.5"}}'
# Send a chat — bridge logs "Model changed", starts fresh thread
```

### 9. Concurrent sessions
Send two chats to two different codex bots simultaneously — both proceed
in parallel up to the `[agents] max_threads` cap. Same-bot concurrent
sends serialize via `SessionQueue.lock`.

### 10. Multimodal
```bash
B64=$(base64 -i /path/to/image.png)
curl -N -X POST http://localhost:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"codex\",\"stream\":true,\"bot_id\":\"codex\",\"messages\":[
        {\"role\":\"user\",\"content\":[
           {\"type\":\"text\",\"text\":\"What's in this image?\"},
           {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$B64\"}}
         ]}]}"
```
Bot describes the image. ASSISTANT_DONE includes `token_usage` with
`input_tokens > 0`.

## Known limitations (v1)

- **Reasoning surfaces** (`item/reasoning/*`) are suppressed — the chat
  UI has no card for them yet.
- **Plan / planUpdate** items are suppressed for the same reason.
- **Approval workflows** are disabled (`approval_policy=never`); the
  bridge runs in `sandbox=danger-full-access` and trusts the agent.
- **Provider-aware tool rendering** is interim. Codex tool events alias
  to Claude tool names so `ClaudeToolCallCard.tsx` renders them. The
  proper provider/tool dispatch system is TASK-212.
- **Cost telemetry** isn't exposed by Codex, so `total_cost_usd` is
  always `None` on the ASSISTANT_DONE token_usage payload.
- **Single bridge container.** No horizontal scaling — Docker restart
  handles process death; `[agents] max_threads` bounds concurrency.
