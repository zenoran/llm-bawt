# Inter-Bot Communication

Inter-bot messaging lives in the MCP server, mainly through
`bots_send_message` and `bots_list_available`. The implementation is in
[src/llm_bawt/mcp_server/server.py](../src/llm_bawt/mcp_server/server.py).

## Current behavior

- Default mode is asynchronous dispatch. `bots_send_message(...)` returns as
  soon as the target turn has been queued.
- `wait_for_reply=true` switches to a blocking request/response mode.
- The target bot's own memory can be used during the turn
  (`augment_memory=True`).
- Inter-bot sends do not extract memory by default (`extract_memory=False`) to
  avoid cross-bot contamination.
- The sender is prepended into the message text unless `sender_bot_id` is
  `"unknown"`.

## In-turn gating

Before sending, the tool checks the target bot's live turn state via
`GET /v1/bots/{bot_id}/in-turn`.

- If the target is busy, the send is rejected with `in_turn=true`.
- `force=true` bypasses that check.
- Forced behavior depends on the target backend:
  - local-model style turns are cancelled/replaced
  - agent backends such as `claude-code` and `codex` run concurrently

## Main tools

| Tool | Purpose |
|---|---|
| `bots_list_available()` | List bot slugs and summary metadata |
| `bots_send_message(...)` | Deliver a one-shot message to another bot |
| `self_fwd(...)` | Handoff helper built on top of `bots_send_message` |

## Example

```python
bots_send_message(
    target_bot_id="byte",
    message="Review TASK-42 and summarize the blocker.",
    sender_bot_id="al",
    wait_for_reply=True,
)
```

## Practical notes

- Use async dispatch for delegation and long-running work.
- Use `wait_for_reply=true` only when the current turn genuinely needs the
  answer inline.
- If a send times out while waiting, do not blindly retry; the target turn may
  still be running.
