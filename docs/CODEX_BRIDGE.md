# Codex Bridge

The Codex bridge runs bots through the OpenAI Codex SDK. The service entrypoint
is [src/codex_bridge/__main__.py](../src/codex_bridge/__main__.py)
and the bridge logic is
[src/codex_bridge/bridge.py](../src/codex_bridge/bridge.py).

## What it does

- Consumes Redis `chat.send` commands for bots whose `agent_backend` is `codex`.
- Uses `~/.codex/auth.json` OAuth credentials, not API keys.
- Persists Codex thread IDs in `agent_backend_config.session_key` and tracks
  the model in `session_model`.
- Reuses a shared SDK transport and serializes turns per session.
- Emits provider-aware native Codex tool events instead of pretending they are
  Claude tool names.
- Stages repo-managed local plugins into the Codex home when enabled.

## Authentication model

The bridge is OAuth-only. On startup it:

- scrubs `OPENAI_API_KEY` and `CODEX_API_KEY` from the environment
- validates `CODEX_AUTH_PATH`
- points `CODEX_HOME` at the directory containing that auth file

If OAuth is broken, the recovery path is `codex login` on the host.

## Important environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CODEX_AUTH_PATH` | `/home/bridge/.codex/auth.json` | ChatGPT OAuth bundle |
| `CODEX_HOME` | `/home/bridge/.codex` | Codex home used by the SDK |
| `CODEX_MODEL` | `gpt-5.4` | Fallback model when the request omits one |
| `CODEX_BACKEND_NAME` | `codex` | Redis backend filter |
| `CODEX_BRIDGE_HEALTH_PORT` | `8682` | `/health` TCP listener |
| `CODEX_BRIDGE_LOG_LEVEL` | `INFO` | Bridge logging |
| `CODEX_BRIDGE_REQUEST_TIMEOUT` | `900` | Per-turn timeout |
| `CODEX_BRIDGE_CWD` | `/home/bridge/dev` | Codex working directory |
| `CODEX_BIN` | bundled | Override Codex binary path |
| `CODEX_LOCAL_PLUGINS_ENABLED` | `1` | Install repo-managed local plugins |
| `CODEX_LOCAL_PLUGINS_SRC` | `/home/bridge/dev/agent-skills/codex` | Local plugin source root |
| `CODEX_DEV_ROOT` | `/home/bridge/dev` | Repo root used for local plugin resolution |

## Session behavior

- `session_key` stores the Codex thread ID.
- `session_model` tracks which model that thread was created with.
- If the model changes, the bridge clears the old thread and starts a new one.
- `/new` and explicit reset RPCs clear the saved thread ID.

## Compose layout

The main compose stack mounts:

- `${HOME}/dev` at `/home/bridge/dev`
- `${HOME}/.codex/auth.json` at `/home/bridge/.codex/auth.json`
- `${HOME}/.codex/config.toml` at `/home/bridge/.codex/config.toml`
- `${HOME}/.codex/sessions` at `/home/bridge/.codex/sessions`

That matches the bridge's expected runtime layout.

## Related files

- [src/llm_bawt/agent_backends/codex.py](../src/llm_bawt/agent_backends/codex.py)
- [src/codex_bridge/local_plugins.py](../src/codex_bridge/local_plugins.py)
- [src/codex_bridge/transport.py](../src/codex_bridge/transport.py)
