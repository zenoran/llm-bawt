# Approval-Gated Tool Policies

Approval policies are stored in the app and enforced in the Claude Code bridge.
The storage and API live in
[src/llm_bawt/approval_policies.py](../src/llm_bawt/approval_policies.py)
and
[src/llm_bawt/service/routes/approval_policies.py](../src/llm_bawt/service/routes/approval_policies.py).

## Current scope

- The policy bundle is service-owned and fetched over HTTP.
- The live gate is implemented in the Claude Code bridge's PreToolUse hook.
- The app persists approval requests and exposes resolve endpoints.
- Codex and OpenClaw are not currently enforcing this policy layer.

## Rule model

First match wins.

| Field | Meaning |
|---|---|
| `enabled` | Disabled rules are ignored |
| `backend_scope` | `*` or a backend name such as `claude-code` |
| `tool_name` | `*` or a specific tool name |
| `matcher_type` | `always`, `exact`, `prefix`, `contains`, `glob`, or `regex` |
| `pattern` | Matcher payload |
| `field` | Input field to inspect; blank uses tool defaults |
| `action` | `require_approval`, `allow`, or `deny` |
| `severity` | Operator-facing severity label |
| `category` | Operator-defined grouping |
| `approval_prompt` | Optional custom prompt shown to the user |
| `order` | Lower runs first |

## Main endpoints

| Method | Path |
|---|---|
| `GET` | `/v1/tool-approval-policies` |
| `POST` | `/v1/tool-approval-policies` |
| `GET` | `/v1/tool-approval-policies/{id}` |
| `PATCH` | `/v1/tool-approval-policies/{id}` |
| `DELETE` | `/v1/tool-approval-policies/{id}` |
| `POST` | `/v1/tool-approval-policies/seed-defaults` |
| `GET` | `/v1/tool-approval-policies/bundle` |
| `POST` | `/v1/admin/reload-tool-approval-policies` |
| `GET` | `/v1/tool-approval-requests` |
| `POST` | `/v1/chat/approvals/{request_id}/resolve` |

## Resolve flow

When a tool call is gated:

1. The bridge emits `approval_required`.
2. The app persists a `tool_approval_requests` row.
3. The UI or API resolves the request with `approve`, `deny`, `cancel`, or `respond`.
4. On approval, the app sends a one-shot `approval.grant` command back to the bridge.
5. The continuation turn can then re-issue the tool call.

## Claude bridge config

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDE_CODE_APPROVAL_BUNDLE_TTL` | `15` | Bundle refresh TTL in seconds |
| `CLAUDE_CODE_APPROVAL_FAIL_CLOSED` | unset | Fail closed if the bundle cannot be fetched |

## Default seeded rules

`seed-defaults` currently seeds broad high-risk patterns such as:

- recursive or forceful `rm`
- `sudo`
- force pushes
- destructive SQL (`DROP`, `TRUNCATE`)
- obvious system-destroying shell commands
- `curl|wget ... | sh`
