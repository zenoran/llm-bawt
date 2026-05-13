# Claude Code Bridge

The Claude Code bridge integrates Anthropic's Claude models into llm-bawt using the [Claude Agent SDK](https://docs.anthropic.com/en/docs/claude-code/sdk). It runs alongside the OpenClaw bridge and uses the same Redis event protocol — the main app doesn't know which bridge handled a request.

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │  llm-bawt-app (FastAPI)                     │
                        │  AgentBackendClient → ClaudeCodeBackend     │
                        └──────────────┬──────────────────────────────┘
                                       │ Redis: agent:commands
                                       │ (fields: backend, session_key, message, ...)
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        Redis (shared stream)                             │
│                        agent:commands                                 │
│                                                                          │
│  ┌─────────────────────────┐     ┌─────────────────────────────────────┐ │
│  │ openclaw-bridge         │     │ claude-code-bridge                  │ │
│  │ filters: backend=openclaw│    │ filters: backend=claude-code        │ │
│  │ consumer group: bridge  │     │ consumer group: claude-code-bridge  │ │
│  └──────────┬──────────────┘     └──────────┬──────────────────────────┘ │
└─────────────┼────────────────────────────────┼──────────────────────────┘
              │                                │
              ▼                                ▼
   OpenClaw Gateway (WS)            Claude Code Binary (stdio)
   on remote host                   bundled in Agent SDK
              │                                │
              ▼                                ▼
      Anthropic API                    Anthropic API
      (via gateway)                    (OAuth subscription)
```

Both bridges publish events to `agent:run:{request_id}` in the same `AgentEvent` format. The main app consumes them identically.

## How It Works

1. A bot with `agent_backend: "claude-code"` receives a chat request
2. `ClaudeCodeBackend` sends a `chat.send` command to Redis with `backend: "claude-code"`
3. The claude-code bridge picks it up (filters by `backend` field)
4. Bridge calls `claude_agent_sdk.query()` which spawns the Claude Code binary
5. The binary authenticates via `CLAUDE_CODE_OAUTH_TOKEN` (your Max/Pro subscription)
6. SDK streams events: text deltas, tool calls, tool results
7. Bridge translates them to `AgentEvent` format and publishes to Redis
8. Main app streams them as SSE to the frontend

## Authentication

The bridge uses your Claude subscription (Max/Pro) via OAuth — not API key billing.

```bash
# Generate token (one-time, on host):
claude setup-token

# Or extract from existing login:
python3 -c "import json; from pathlib import Path; print(json.loads((Path.home()/'.claude'/'.credentials.json').read_text())['claudeAiOauth']['accessToken'])"
```

Set `CLAUDE_CODE_OAUTH_TOKEN` in `.env`. The Agent SDK's bundled Claude binary reads this token.

## Setup

### 1. Add token to `.env`

```bash
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
```

### 2. Start the bridge

```bash
docker compose up -d claude-code-bridge
```

### 3. Create a bot

```bash
curl -X POST http://localhost:8642/v1/bots -H "Content-Type: application/json" -d '{
  "slug": "claude",
  "name": "Claude",
  "system_prompt": "You are Claude, a helpful AI assistant.",
  "agent_backend": "claude-code",
  "agent_backend_config": {"model": "opus[1m]", "timeout_seconds": 120},
  "requires_memory": false,
  "include_summaries": false
}'
```

### 4. Reload and test

```bash
curl -X POST http://localhost:8642/v1/admin/reload-bots
curl -X POST http://localhost:8642/v1/models/reload

curl http://localhost:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"claude","bot_id":"claude","user_id":"nick","stream":true,
       "messages":[{"role":"user","content":"hello"}]}'
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_CODE_OAUTH_TOKEN` | (required) | OAuth token from `claude setup-token` |
| `CLAUDE_CODE_MODEL` | `claude-sonnet-4-20250514` | Fallback model if bot config doesn't specify one |
| `CLAUDE_CODE_CWD` | `/app` | Working directory for Claude Code |
| `CLAUDE_CODE_ADD_DIRS` | | Optional extra directories to allow tool access to via Claude CLI `--add-dir` |
| `CLAUDE_CODE_BRIDGE_LOG_LEVEL` | `INFO` | Logging level |

### Bot Config (`agent_backend_config`)

| Key | Description |
|-----|-------------|
| `model` | Claude model alias: `default`, `opus[1m]`, `sonnet[1m]`, `haiku` |
| `session_key` | SDK session UUID (auto-managed by bridge — do not set manually) |
| `timeout_seconds` | Max wait for response (default: 120) |

### Available Models

| llm-bawt alias | SDK alias | Model |
|----------------|-----------|-------|
| `claude-sonnet` | `default` | Claude Sonnet 4.6 |
| `claude-sonnet-1m` | `sonnet[1m]` | Claude Sonnet 4.6 (1M context) |
| `claude-opus-1m` | `opus[1m]` | Claude Opus 4.6 (1M context) |
| `claude-haiku` | `haiku` | Claude Haiku 4.5 |

## Session Management

Each bot maintains a persistent Claude conversation. The SDK session UUID is stored in the bot's `agent_backend_config.session_key` in the database.

- **Conversation persists** across messages — Claude remembers context
- **Survives restarts** — session UUID is in PostgreSQL, not in-memory
- **`/new` prefix** resets the session — clears the UUID, next message creates fresh conversation
- **Model change** — if the bot's model changes, the bridge detects it and starts a new session
- **API reset** — `POST /v1/chat/session/reset` with `{"bot_id": "claude"}`

## Docker Setup

The bridge runs in a lightweight container (`python:3.12-slim`) shared with the OpenClaw bridge via `Dockerfile.bridge`. No GPU, no CUDA.

### Volumes

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `~/dev` | `/home/bridge/dev` | Project files — Claude can read/write code |
| `~/.config/claude-code-bridge` | `/home/bridge/.claude` | Claude Code config, settings, CLAUDE.md |
| `~/dev/agent-skills` | `/home/bridge/.claude/skills` | Shared skills repo mounted directly into Claude's skills directory |
| `~/.ssh` | `/home/bridge/.ssh` (ro) | SSH keys for git/remote access |
| `~/.config/claude-code-bridge/ssh_config` | `/etc/ssh/ssh_config` (ro) | SSH host config |

### Container Details

- Runs as non-root user `bridge` (required — `bypassPermissions` fails as root)
- Health check on port 8681: `curl http://localhost:8681/health`
- Depends on: `redis` (healthy), `app` (started)

## Skills

This deployment exposes shared skills by bind-mounting `~/dev/agent-skills` directly to `/home/bridge/.claude/skills`.

No per-skill symlinks are required for the bridge container. Edit skills on the host at `~/dev/agent-skills/` and the changes are visible immediately inside Claude.

`CLAUDE_CODE_ADD_DIRS` is not part of skill discovery here. If you set it, it is passed through as Claude CLI `--add-dir`, which only grants tool access to additional directories outside the main working tree.

## Switching a Bot from OpenClaw to Claude Code

```bash
curl -X PATCH http://localhost:8642/v1/bots/mybot/profile \
  -H "Content-Type: application/json" \
  -d '{
    "agent_backend": "claude-code",
    "agent_backend_config": {"model": "opus[1m]", "timeout_seconds": 120}
  }'

# Reload
curl -X POST http://localhost:8642/v1/admin/reload-bots
docker compose restart app
```

## Key Files

| File | Purpose |
|------|---------|
| `src/claude_code_bridge/bridge.py` | Core bridge — Redis listener, SDK event translation, session management |
| `src/claude_code_bridge/__main__.py` | Entrypoint with health check server |
| `src/llm_bawt/agent_backends/claude_code.py` | `ClaudeCodeBackend` — registered as `"claude-code"` |
| `src/llm_bawt/clients/agent_backend_client.py` | Extracts system prompt from messages, passes to backend |
| `Dockerfile.bridge` | Lightweight shared image for both bridges |
| `docker-compose.yml` | `claude-code-bridge` service definition |

## Differences from OpenClaw Bridge

| Aspect | OpenClaw | Claude Code |
|--------|----------|-------------|
| Upstream | WebSocket to gateway on remote host | Agent SDK spawns local binary |
| Auth | Ed25519 device identity + gateway token | OAuth subscription token |
| Session key | Explicit in config (e.g. `agent:byte:main`) | Auto-generated SDK UUID |
| System prompt | Pushed as SOUL.md file to agent | Passed inline per-request via Redis |
| History | Managed by gateway | Managed by Claude Code binary |
| Model selection | Gateway-side | Per-bot via `agent_backend_config.model` |
| Tools | Gateway's tool system | Claude Code's built-in tools (Read, Edit, Bash, etc.) |
