# Claude Code Bridge

The Claude Code bridge runs agent-backed bots through the Claude Agent SDK.
The entrypoint is [src/claude_code_bridge/__main__.py](../src/claude_code_bridge/__main__.py)
and the bridge logic is
[src/claude_code_bridge/bridge.py](../src/claude_code_bridge/bridge.py).

## What it does

- Consumes Redis `chat.send` commands for bots whose `agent_backend` is `claude-code`.
- Spawns Claude Agent SDK turns and streams normalized `AgentEvent`s back to the app.
- Persists per-bot SDK session state in `agent_backend_config.session_key` and
  `session_model`.
- Supports `/new` and explicit reset RPCs by clearing the stored SDK session.
- Enforces approval-gated tool policies in the bridge before tool execution.
- Can proxy non-Anthropic providers through an Anthropic-compatible local proxy.

## Model resolution

The bridge does not have a meaningful global default model. The app resolves the
bot's `default_model` alias from the model catalog and sends the resulting
`model_id` with each request.

Common shapes:

- Plain Anthropic aliases for native Claude use.
- `openai_chatgpt/<model>` for ChatGPT-subscription routing through the proxy.
- `zai/<model>` for z.ai / GLM routing through the proxy.
- `xai/<model>` for xAI / Grok routing through the proxy (e.g. `xai/grok-4.5`).

If a bot has `agent_backend="claude-code"` but no valid `default_model`, the
turn is rejected.

## Authentication

The bridge requires a usable Claude OAuth token for the SDK path. Current lookup
order:

1. `~/.claude/.credentials.json` (`claudeAiOauth`, refreshed in place when needed)
2. `CLAUDE_CODE_OAUTH_TOKEN` as an env fallback

For provider-prefixed proxy models, the bridge also needs `~/.codex/auth.json`
because the local proxy reads ChatGPT OAuth from that file.

## Important environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDE_CODE_BRIDGE_LOG_LEVEL` | `INFO` | Bridge logging |
| `CLAUDE_CODE_BRIDGE_HEALTH_PORT` | `8681` | `/health` TCP listener |
| `CLAUDE_CODE_REQUEST_TIMEOUT` | `300` | Per-turn timeout |
| `CLAUDE_CODE_CWD` | `/app` | SDK working directory |
| `CLAUDE_CODE_PERMISSION_MODE` | `bypassPermissions` | SDK permission mode |
| `CLAUDE_CODE_ADD_DIRS` | unset | Extra Claude `--add-dir` paths |
| `CLAUDE_CODE_BACKEND_NAME` | `claude-code` | Redis backend filter |
| `XAI_API_KEY` | unset | xAI/Grok API key for `xai/<model>` routing (`LLM_BAWT_XAI_API_KEY` aliased) |
| `CLAUDE_CODE_BRIDGE_PROXY_DISABLED` | unset | Disable provider-routing proxy |
| `CLAUDE_CODE_BRIDGE_PROXY_PORT` | `0` | Pin proxy port instead of ephemeral |
| `CLAUDE_CODE_APPROVAL_BUNDLE_TTL` | `15` | Approval-policy bundle cache TTL |
| `CLAUDE_CODE_APPROVAL_FAIL_CLOSED` | unset | Fail closed if policy fetch breaks |

## SDK tool policy

The global runtime setting `claude_code_disallowed_tools` controls the base
`ClaudeAgentOptions.disallowed_tools` list. Its code default disables the
harness-only planning and worktree workflows:

```json
[
  "EnterPlanMode",
  "ExitPlanMode",
  "EnterWorktree",
  "ExitWorktree"
]
```

The app resolves this setting on every Claude Code dispatch and includes it in
the Redis command, so an operator edit takes effect on the **next turn** without
another restart. Proxy-routed turns always add `WebSearch` and `WebFetch`
because those Anthropic server-side tools cannot execute through a non-Anthropic
upstream. An explicit empty list enables every base SDK tool while retaining
those two proxy transport exclusions.

Manage it in BawtHub under **Tools → Settings → Defaults & Tuning → Global**, or
through `PUT /v1/settings` with a JSON array of tool-name strings. The policy
normalizer trims names, removes duplicates, and safely falls back to the defaults
if a malformed value reaches the bridge.

## Session behavior

- Session IDs are bridge-managed and persisted on the bot profile.
- Changing the resolved model causes the bridge to drop the old session and
  start a fresh one.
- `/new` clears the saved session before the next turn runs.
- Turns for the same session are serialized by `SessionQueue`.

## Docker compose expectations

The main compose stack mounts:

- `${HOME}/dev` at `/home/bridge/dev`
- `${HOME}/.config/claude-code-bridge` at `/home/bridge/.claude`
- `${HOME}/dev/agent-skills` at `/home/bridge/.claude/skills`
- `${HOME}/.codex/auth.json` at `/home/bridge/.codex/auth.json`

That is the layout the bridge code expects in production.

## Related files

- [src/llm_bawt/agent_backends/claude_code.py](../src/llm_bawt/agent_backends/claude_code.py)
- [src/claude_code_bridge/proxy/app.py](../src/claude_code_bridge/proxy/app.py)
- [docs/approval-policies.md](../docs/approval-policies.md)
- [docs/usage-endpoint.md](../docs/usage-endpoint.md)
