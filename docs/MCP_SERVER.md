# BawtHub MCP Server

The FastMCP instance and core memory/message tools are implemented in
[src/llm_bawt/mcp_server/server.py](../src/llm_bawt/mcp_server/server.py).
Focused registration modules keep tool families separate: task tools in
[`task_tools.py`](../src/llm_bawt/mcp_server/task_tools.py), durable/immediate
inter-bot tools in
[`inter_bot_tools.py`](../src/llm_bawt/mcp_server/inter_bot_tools.py), session
tools in [`session_tools.py`](../src/llm_bawt/mcp_server/session_tools.py),
profile tools in [`profile_tools.py`](../src/llm_bawt/mcp_server/profile_tools.py),
and self-handoff tools in
[`self_tools.py`](../src/llm_bawt/mcp_server/self_tools.py).

The server name is `bawthub`.

## Transport

- HTTP transport uses FastMCP streamable HTTP.
- `stdio` is also supported for local development.
- DNS rebinding protection is enabled; allowed hosts come from
  `LLM_BAWT_MCP_ALLOWED_HOSTS`.

Start it directly:

```bash
uv run python -m llm_bawt.mcp_server --transport http --host 0.0.0.0 --port 8001
```

## Tool groups

| Group | Examples |
|---|---|
| `memory_*` | `memory_store`, `memory_search`, `memory_search_all`, `memory_consolidate` |
| `messages_*` | `messages_get`, `messages_add`, `messages_search_all`, `messages_ignore_recent` |
| `context_*` | `context_get_recent` |
| `facts_*` | `facts_extract` |
| `system_*` | `system_stats`, `system_run_maintenance` |
| `bots_*` | `bots_list_available`, `bots_send_message`, `bots_delivery_get`, `bots_deliveries_list`, `bots_delivery_cancel` |
| `agent_context_*` | `agent_context_health`, `agent_context_compact`, `agent_context_reset` |
| `tasks_*` | `tasks_list`, `tasks_get`, `tasks_create`, `tasks_update`, `tasks_get_context` |
| `steps_*` | `steps_add`, `steps_update`, `steps_delete` |
| `projects_*` | `projects_list`, `projects_get`, `projects_create`, `projects_update`, `projects_delete`, `projects_get_context` |
| `activity_*` | `activity_get` |
| `self_*` | `self_recap`, `self_tail`, `self_fwd`, `self_system_prompt` |
| `profile` | unified user/bot profile tool |

## Notes on inter-bot tools

Default asynchronous `bots_send_message` is durable: it returns a stable delivery
ID, steers an active Claude Code turn in place when possible, and otherwise starts
one safe idle fallback turn. `delivery="when_idle"` deliberately skips steering.
`force=true` never authorizes concurrent agent turns. `session_policy` accepts
`continue`, `reset_retain_history`, or `reset_without_history`; reset policies wait
for idle and rotate non-destructively before delivery. Use `bots_delivery_get`,
`bots_deliveries_list`, and `bots_delivery_cancel` for lifecycle inspection.

`agent_context_health` exposes active-thread resident usage/headroom and backend
capabilities. `agent_context_compact` queues one idempotent Claude `/compact`
maintenance turn through the durable when-idle scheduler. `agent_context_reset`
performs an idle-only safe reset without deleting durable transcript rows. Full
accounting, capability, and overflow recovery contract:
[AGENT_CONTEXT_CONTROL.md](AGENT_CONTEXT_CONTROL.md).

## Notes on task tools

`tasks_*`, `steps_*`, `projects_*`, and `activity_*` are thin MCP wrappers over
the BawtHub task API. They call `LLM_BAWT_TASK_API_URL` plus `/api/tasks/...`
internally.

## Important environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BAWT_MCP_ALLOWED_HOSTS` | `127.0.0.1:*,localhost:*` | Host allowlist for HTTP clients |
| `LLM_BAWT_MCP_SERVER_VERBOSE` | `1` | Verbose logging |
| `LLM_BAWT_MCP_SERVER_DEBUG` | `0` | Debug logging |
| `LLM_BAWT_TASK_API_URL` | `http://echo.lan.zenoran.com` | Task API base URL for task-related tools |
| `LLM_BAWT_APP_BASE_URL` | `http://localhost:8642` | Used by some self tools |

## Related docs

- [docs/INTER_BOT_COMMUNICATION.md](../docs/INTER_BOT_COMMUNICATION.md)
- [docs/BACKGROUND_SCHEDULER.md](../docs/BACKGROUND_SCHEDULER.md)
