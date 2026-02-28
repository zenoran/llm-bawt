# OpenClaw Integration Runbook

This document explains how llm-bawt integrates with OpenClaw, including prerequisites, transport modes, model/session mapping, and support procedures.

## Scope

Use this runbook when supporting:
- `llm --add-model openclaw`
- `llm --list-models` OpenClaw entries
- `llm -b vex ...` / service-based OpenClaw chat routing
- tool-call events from OpenClaw-backed bots

## Mental Model

OpenClaw concepts that matter:
- **Session key**: conversation continuity + memory context identity.
- **Channel**: transport/source metadata (for example `nextcloud-talk`, `webchat`).
- **Agent ID**: OpenClaw agent namespace (usually `main`).

llm-bawt concepts that matter:
- **OpenClaw model alias**: local alias that maps to one OpenClaw `session_key`.
- **Virtual backend alias (`openclaw`)**: service-injected bridge model for bots using `agent_backend=openclaw`.

Rule of thumb:
- Session key controls continuity.
- Channel is metadata/routing context.

## Architecture

### Runtime chat path (API-first)

For normal chat requests, llm-bawt uses OpenClaw gateway API:
- `POST /v1/chat/completions`
- bearer token auth
- optional OpenClaw session/channel headers

Configured via bot `agent_backend_config` (DB profile) or model alias fields:
- `gateway_url`
- `token_env`
- `agent_id`
- `session_key`
- optional: `message_channel`, `account_id`

### Discovery path (`--add-model openclaw`)

Wizard discovery supports two modes:
- **API-only** (`OPENCLAW_DISCOVERY_MODE=api`)
  - uses OpenClaw HTTP tool invocation (`/tools/invoke`)
- **Hybrid** (`OPENCLAW_DISCOVERY_MODE=hybrid`)
  - API first, SSH fallback for richer `sessions.list` / `channels.status`

SSH in this integration is discovery-only fallback (not required for runtime chat).

## Prerequisites

## 1) OpenClaw gateway reachable

Required:
- OpenClaw gateway URL
- OpenClaw gateway token

Recommended in `~/.config/llm-bawt/.env`:
- `OPENCLAW_GATEWAY_URL=http://<host>:18789`
- `OPENCLAW_GATEWAY_TOKEN=<token>`
- `OPENCLAW_GATEWAY_TOKEN_ENV=OPENCLAW_GATEWAY_TOKEN`

## 2) Discovery mode selection

Set one:
- `OPENCLAW_DISCOVERY_MODE=api` (default/recommended)
- `OPENCLAW_DISCOVERY_MODE=hybrid` (if API discovery is incomplete)

Optional backward-compatible override:
- `OPENCLAW_USE_SSH_FALLBACK=true|false`

## 3) SSH fallback (only for hybrid discovery)

If using hybrid mode:
- `OPENCLAW_SSH_USER=vex` (or appropriate user)
- SSH access from llm-bawt host to OpenClaw host
- remote `openclaw` CLI available

## Alias Semantics

`--list-models` shows two OpenClaw categories:

1. **Local OpenClaw aliases** (real session mappings)
- Example: `openclaw-main`
- Contains explicit `session_key` such as `agent:main:main`
- Use these with `-m <alias>` when you want deterministic session targeting

2. **Virtual backend alias**
- Alias: `openclaw`
- Injected by service for bots configured with `agent_backend=openclaw`
- Not a user-created session alias

## Canonical Main Session Guidance

If you want one alias for TUI main session:
- Use one alias mapped to `agent:main:main` (recommended: `openclaw-main`)
- Avoid multiple aliases pointing to the same `session_key`

For `vex` parity with OpenClaw TUI main:
- pin backend config `session_key=agent:main:main`

## `--add-model openclaw` Behavior

Wizard flow:
1. Loads env defaults from configured env file and `~/.config/llm-bawt/.env`
2. Validates gateway health/auth
3. Discovers channels/sessions (API or hybrid)
4. Lets user subscribe to existing session or create new key
5. Saves alias to local config and syncs DB model definitions
6. Requests service model catalog reload

Duplicate prevention:
- When selecting an existing session key, wizard surfaces existing aliases for that key and defaults to reuse instead of creating duplicates.

## Service Catalog Consistency

Service model list can become stale after model changes unless refreshed.

Supported refresh route:
- `POST /v1/models/reload`

Reload behavior:
1. reset in-memory config from YAML
2. merge DB model definitions
3. rebuild service model catalog
4. clear stale cached instances as needed

## Tool Calls and Tool Results

OpenClaw-backed tool calls are surfaced in llm-bawt as events and persisted turn log entries.

Current behavior:
- tool call names + arguments are available
- tool result is populated when OpenClaw exposes it in retrievable history
- if OpenClaw API does not expose per-tool result payload, llm-bawt records:
  - `Result not exposed by OpenClaw API (see assistant response).`

Implication:
- "pending" should not appear for new events in this path
- exact raw tool output depends on OpenClaw API/tool history exposure

## Troubleshooting

## Symptom: only `openclaw` appears in service models

Cause:
- stale service model catalog

Fix:
1. call `POST /v1/models/reload`
2. if route unavailable, restart app/service

## Symptom: `--list-models` says no local models but service has many

Cause:
- YAML-only view without DB overlay

Fix:
- ensure current build includes DB merge in local model manager

## Symptom: Add-model shows only one session

Cause:
- API discovery restricted by OpenClaw policy/visibility

Fix:
- switch to hybrid discovery mode
- verify SSH fallback prerequisites

## Symptom: vex and TUI main are different conversations

Cause:
- mismatched session keys

Fix:
- set vex backend session key to `agent:main:main`
- verify headers/session mapping in runtime config

## Symptom: tool results not detailed

Cause:
- OpenClaw history payload missing `toolResult` content blocks

Fix:
- inspect OpenClaw `sessions_history` content blocks
- if absent, rely on assistant text summary or extend OpenClaw side to expose result blocks

## Support Checklist (for future agents)

1. Confirm env defaults in `~/.config/llm-bawt/.env`
2. Verify discovery mode (`api` vs `hybrid`)
3. Verify canonical OpenClaw alias and session_key mapping
4. Verify vex backend `session_key`
5. Reload service model catalog (`/v1/models/reload`)
6. Validate `--list-models` for local + service alignment
7. Validate one live chat turn and one tool-call event row

## Security Notes

- Do not store raw gateway token in bot profile JSON.
- Prefer `token_env` and secret in env file/secret manager.
- SSH fallback is optional and should be disabled when not needed.
