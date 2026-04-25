# llm-bawt MCP Server

The **llm-bawt MCP Server** is a [FastMCP](https://github.com/jlouis/fastmcp) HTTP service (port `8001`) that exposes the llm-bawt platform — bot memory, conversation history, fact extraction, inter-bot messaging, and the agent task system — as Model Context Protocol tools.

It runs in-process inside the `app` container alongside the FastAPI service, but is also reachable from any external MCP client (VSCode, Claude Desktop, custom Node/Python clients).

| Detail | Value |
|---|---|
| Server name | `llm-bawt` |
| Default endpoint | `http://localhost:8001/mcp` |
| Transport | `streamable-http` (default) or `stdio` |
| JSON-RPC | yes — standard MCP |
| Auth | none — relies on host allowlist |

---

## Tool Groups

All tools are namespaced by a prefix that identifies their domain. Use the prefix to discover related tools quickly.

| Prefix | Domain | Examples |
|---|---|---|
| `memory_*` | Bot memory CRUD, search, maintenance | `memory_store`, `memory_search`, `memory_consolidate` |
| `messages_*` | Conversation history CRUD, search, ignore/restore | `messages_add`, `messages_search_all`, `messages_ignore_recent` |
| `context_*` | Combined message + memory context | `context_get_recent` |
| `facts_*` | LLM-based fact extraction | `facts_extract` |
| `system_*` | Service-wide stats, maintenance jobs | `system_stats`, `system_run_maintenance` |
| `bots_*` | Bot discovery + inter-bot messaging | `bots_list_available`, `bots_send_message` |
| `tasks_*` | Agent task CRUD, briefing | `tasks_list`, `tasks_create`, `tasks_update`, `tasks_get_context` |
| `steps_*` | Task step lifecycle | `steps_add`, `steps_update` |
| `projects_*` | Agent project CRUD | `projects_list`, `projects_create`, `projects_delete` |
| `activity_*` | Agent activity log | `activity_get` |
| `profile` | User profile attributes (single router tool) | `profile` |

---

## Tool Reference

All tools are async and return JSON-serializable results. Parameters in **bold** are required; others have defaults.

### `memory_*` — Bot Memory

| Tool | Purpose |
|---|---|
| `memory_store(content, bot_id?, tags?, importance?, source_message_ids?)` | Store a new fact |
| `memory_search(query, bot_id?, n_results?, min_relevance?, tags?)` | Semantic search within one bot |
| `memory_search_all(query, n_results?, min_relevance?)` | Semantic search across **all bots** |
| `memory_search_source(source, query, n_results?, min_relevance?, tags?)` | Search a specific bot's memories (cross-bot read-only) |
| `memory_list_sources()` | Discover which bots have stored memories |
| `memory_list_recent(bot_id?, n?)` | List N most recent memories |
| `memory_list_high_importance(bot_id?, limit?, min_importance?)` | List memories sorted by importance |
| `memory_update(memory_id, bot_id?, content?, tags?, importance?)` | Update a memory |
| `memory_delete(memory_id, bot_id?)` | Delete a memory |
| `memory_supersede(old_memory_id, new_memory_id, bot_id?)` | Mark one memory as superseded by another |
| `memory_consolidate(bot_id?, dry_run?, similarity_threshold?)` | Merge near-duplicate memories |
| `memory_update_meaning(bot_id?, memory_id, intent?, stakes?, emotional_charge?, recurrence_keywords?, updated_tags?)` | Enrich a memory with intent/stakes/emotional metadata |
| `memory_regenerate_embeddings(bot_id?, batch_size?)` | Recompute vectors |
| `memory_delete_by_source_messages(bot_id?, message_ids?)` | Delete memories that came from given messages |

### `messages_*` — Conversation History

| Tool | Purpose |
|---|---|
| `messages_add(role, content, bot_id?, session_id?, message_id?, timestamp?)` | Append a message to history |
| `messages_get(bot_id?, since_seconds?, limit?)` | Fetch messages for context windows |
| `messages_get_by_id(bot_id?, message_id)` | Look up one message (prefix match supported) |
| `messages_get_for_summary(bot_id?, summary_id)` | Get raw messages a summary was built from |
| `messages_search_all(query, n_results?, role_filter?)` | Full-text search across **all bots** |
| `messages_preview_recent(bot_id?, count?)` | Preview most recent N messages |
| `messages_preview_since_minutes(bot_id?, minutes?)` | Preview messages from last N minutes |
| `messages_preview_ignored(bot_id?)` | Show currently soft-deleted messages |
| `messages_ignore_recent(bot_id?, count?)` | Soft-delete last N messages |
| `messages_ignore_since_minutes(bot_id?, minutes?)` | Soft-delete messages from last N minutes |
| `messages_ignore_by_id(bot_id?, message_id)` | Soft-delete a single message |
| `messages_restore_ignored(bot_id?)` | Restore all soft-deleted messages |
| `messages_mark_recalled(bot_id?, message_ids?)` | Mark messages as recalled from a summary |
| `messages_clear(bot_id?)` | Hard-delete all messages for a bot |
| `messages_remove_last_partial(bot_id?, role?)` | Drop a trailing partial message (e.g. aborted stream) |

### `context_*` — Combined Context

| Tool | Purpose |
|---|---|
| `context_get_recent(bot_id?, n_messages?, n_memories?, query?)` | Get recent messages + relevant memories in one call |

Returns:
```json
{
  "messages": [ { "role": "...", "content": "...", "created_at": "..." } ],
  "memories": [ { "id": "...", "content": "...", "importance": 0.8, "relevance": 0.95 } ]
}
```

### `facts_*` — Fact Extraction

| Tool | Purpose |
|---|---|
| `facts_extract(messages, bot_id, user_id, store?, use_llm?)` | LLM-based fact extraction from a list of messages |

`bot_id` and `user_id` are required. With `store=true`, extracted facts are written into the bot's memory.

### `system_*` — Service Stats & Maintenance

| Tool | Purpose |
|---|---|
| `system_stats(bot_id?)` | Memory/message counts and aggregates |
| `system_run_maintenance(bot_id?, run_consolidation?, run_recurrence_detection?, run_decay_pruning?, run_orphan_cleanup?, dry_run?)` | Run a full maintenance cycle |

### `bots_*` — Inter-Bot

| Tool | Purpose |
|---|---|
| `bots_list_available()` | List bots that can receive messages |
| `bots_send_message(target_bot_id, message, sender_bot_id?, max_tokens?, temperature?)` | Send a one-shot message to another bot and get its reply |

### `tasks_*` / `steps_*` / `projects_*` / `activity_*` — Agent Task System

These tools wrap the agent task REST API (`/api/agents/*` on the unmute frontend) so agents can manage work without thinking about HTTP.

| Tool | Purpose |
|---|---|
| `tasks_list(status?, project_id?, q?, limit?)` | List tasks with filters |
| `tasks_get(task_id)` | Full task with steps, deps, project |
| `tasks_create(title, description?, project_id?, priority?, status?, steps?, bot_id?)` | Create a new task |
| `tasks_update(task_id, status?, response?, model_id?, title?, description?, priority?, planned?, project_id?, agent_bot_id?, bot_id?)` | Update task fields |
| `tasks_get_context(task_id)` | Formatted briefing doc (task + steps + project context) |
| `steps_add(task_id, steps, bot_id?)` | Append steps to a task |
| `steps_update(task_id, step_id, status?, output?, bot_id?)` | Mark step running/completed/failed/skipped |
| `projects_list()` | List projects with task counts |
| `projects_get(project_id)` | Full project incl. tasks and context prompt |
| `projects_create(name, description?, color?, icon?, context_prompt?, agent_bot_id?, bot_id?)` | Create a project |
| `projects_update(project_id, name?, description?, color?, icon?, context_prompt?, agent_bot_id?, bot_id?)` | Update a project |
| `projects_delete(project_id, bot_id?)` | Delete a project (tasks become unassigned) |
| `activity_get(task_id?, project_id?, limit?)` | Recent activity log entries |

### `profile` — User Profile

| Tool | Purpose |
|---|---|
| `profile(action, entity_type?, entity_id?, category?, key?, value?)` | Action-router tool for user profile attribute CRUD |

---

## VSCode Integration

VSCode connects via the [Claude Dev / MCP-aware extension](https://marketplace.visualstudio.com/items?itemName=Anthropic.claude-dev) or any other MCP client.

### 1. Workspace settings

Edit `.vscode/settings.json`:

```jsonc
{
  "mcpServers": {
    "llm-bawt": {
      "command": "npx",
      "args": ["@anthropic-ai/mcp-http-client", "http://localhost:8001/mcp"]
    }
  }
}
```

If llm-bawt runs in Docker on the same host, replace `localhost` with `host.docker.internal` (macOS/Windows) or the container's published address.

### 2. User settings (global)

Same shape, in your user `settings.json`:

```jsonc
{
  "mcpServers": {
    "llm-bawt": {
      "command": "npx",
      "args": ["@anthropic-ai/mcp-http-client", "http://echo.lan.zenoran.com:8001/mcp"]
    }
  }
}
```

### 3. Verify

After reloading VSCode, the extension should list tools grouped by prefix (`memory_*`, `tasks_*`, …). Try `system_stats` with `bot_id: "default"` as a smoke test.

---

## Claude Desktop Integration

Edit `~/.config/Claude/claude_desktop_config.json` (Linux/macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "llm-bawt": {
      "command": "npx",
      "args": ["@anthropic-ai/mcp-http-client", "http://localhost:8001/mcp"]
    }
  }
}
```

Restart Claude Desktop.

---

## Direct HTTP Usage

Tools can also be invoked over plain HTTP JSON-RPC for scripts and CI:

```bash
curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "memory_search",
      "arguments": { "bot_id": "nova", "query": "kubernetes deployment", "n_results": 5 }
    }
  }'
```

To list all available tools:

```bash
curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

---

## Network Access Control

By default the MCP server only accepts requests from `127.0.0.1` and `localhost`. To allow LAN clients, set:

```bash
LLM_BAWT_MCP_ALLOWED_HOSTS="127.0.0.1:*,localhost:*,192.168.1.*:*,echo.lan.zenoran.com:*"
```

Format is comma-separated `host:port` patterns. Use `:*` to allow any port. Restart the `app` container after changing.

For remote/internet exposure, place the server behind an HTTPS reverse proxy (Nginx Proxy Manager, Traefik, etc.) and add an auth layer — the server itself does not authenticate.

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BAWT_MCP_ALLOWED_HOSTS` | `127.0.0.1:*,localhost:*` | DNS-rebinding allowlist |
| `LLM_BAWT_MCP_SERVER_VERBOSE` | `1` | Verbose logging |
| `LLM_BAWT_MCP_SERVER_DEBUG` | `0` | Debug logging |
| `LLM_BAWT_TASK_API_URL` | `http://echo.lan.zenoran.com` | Base URL for the agent task REST API (used by `tasks_*`, `projects_*`, `activity_*`) |

CLI flags for `python -m llm_bawt.mcp_server` (also installed as `llm-mcp-server`):

| Flag | Default | Purpose |
|---|---|---|
| `--transport` | `http` | `http` or `stdio` |
| `--host` | `0.0.0.0` | Bind host (HTTP only) |
| `--port` | `8001` | Bind port (HTTP only) |

---

## Usage Examples

### Search every bot's memory for context

```
memory_search_all(query="docker deployment", n_results=10, min_relevance=0.7)
```

### Pull recent messages + matching memories before answering

```
context_get_recent(bot_id="nova", n_messages=5, n_memories=10, query="error handling middleware")
```

### Extract and store facts from a conversation

```
facts_extract(
  bot_id="nova",
  user_id="nick",
  messages=[
    {"role": "user", "content": "I started using minikube for local k8s dev"},
    {"role": "assistant", "content": "Nice — minikube is a good fit..."}
  ],
  store=true
)
```

### Drive an agent task

```
tasks_get_context(task_id="TASK-42")              # read briefing
steps_update(task_id="TASK-42", step_id="...", status="RUNNING")
# ... do the work ...
steps_update(task_id="TASK-42", step_id="...", status="COMPLETED", output="...")
tasks_update(task_id="TASK-42", status="REVIEW", response="Done. Summary: ...")
```

### Bot-to-bot

```
bots_list_available()
bots_send_message(target_bot_id="mira", sender_bot_id="nova",
                  message="What do you remember about the homelab Redis setup?")
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `connection refused` from VSCode | server not listening | `curl http://localhost:8001/health` from the host; `docker compose ps` |
| `403 Forbidden` | host not in allowlist | add it to `LLM_BAWT_MCP_ALLOWED_HOSTS` and restart `app` |
| Tools missing in client | client cached old `tools/list` | reload VSCode / restart Claude Desktop |
| Tool returns `{"error":...,"status":...}` | underlying REST call failed (task tools) | check `LLM_BAWT_TASK_API_URL` and the unmute frontend |
| `memory_search` returns nothing | bot has no memories or threshold too high | call `system_stats` to check counts; lower `min_relevance` |

Server logs:

```bash
docker compose logs -f app | grep -i mcp
```

Health check:

```bash
curl http://localhost:8001/health
```

---

## Notes on the Rebrand

- The MCP server identity is now `llm-bawt` (was `llm-memory`). The internal Python package was renamed `llm_bawt.memory_server` → `llm_bawt.mcp_server` to reflect that it serves more than memory. The console entry point is `llm-mcp-server`.
- Tool **external names** were prefix-grouped (e.g. `store_memory` → `memory_store`). Internal Python function names are unchanged, so embedded callers (`MemoryClient`) continue to work without code changes; only the over-the-wire MCP names were updated and the `MemoryClient`'s server-mode JSON-RPC calls were retargeted to the new names in the same change.
- If you have external clients that called the old names, update them to the new prefixed names.
