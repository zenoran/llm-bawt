# Approval-gated tool policies

Configurable approval gates for destructive agent tool actions. A policy can
**require approval**, **allow** (whitelist), or **deny** (hard-block) a tool call
before a bridge executes it — driven entirely by data, not hardcoded command
checks.

Source of truth lives in **llm-bawt**. Bridges fetch a compiled bundle, evaluate
it in their per-tool permission hook, and gate matching calls. BawtHub provides
the admin UI (`/tools/approval-policies`) which proxies to llm-bawt.

> Status (TASK-289…296): backend, bridge gate, app event plumbing, admin UI, and
> tests are implemented. **Activating the gate requires deploying the new bridge
> code (a bridge restart) — see "Rollout" below.** Until then the policies exist
> in the DB but no bridge enforces them.

---

## How it works

```
model wants to run Bash("rm -rf /etc")
   │
   ▼
claude-code bridge  can_use_tool hook
   │  evaluate(bundle, backend, tool, input)  ← pure engine, agent_bridge/approval.py
   │
   ├─ ALLOW            → run the tool (default when nothing matches)
   ├─ DENY             → refuse, tell the model it's blocked, end turn
   └─ REQUIRE_APPROVAL → is there a live grant for this exact call?
        ├─ yes → consume the one-shot grant, run the tool
        └─ no  → emit APPROVAL_REQUIRED, deny with a "pending" ack, END TURN
                    │
                    ▼
        app persists a tool_approval_requests row, streams an Approve/Deny
        card to the UI (SSE + unified event), marks the turn end_reason="approval"
                    │
            user clicks Approve
                    ▼
        POST /v1/chat/approvals/{id}/resolve
          ├─ records the decision (audit)
          ├─ sends approval.grant Redis command → bridge stores a one-shot grant
          └─ returns a continuation prompt the chat client dispatches as a new turn
                    │
                    ▼
        model re-issues the same Bash call → grant matches → ALLOWED once
```

This reuses the proven deferred-continuation model from `AskUserQuestion`
(TASK-269): the turn ends cleanly, no blocking Future, the decision returns as a
fresh continuation turn.

---

## Data model

`tool_approval_policies` (one rule):

| field | meaning |
|---|---|
| `enabled` | off = ignored |
| `backend_scope` | `*` any bridge, or `claude-code` / `codex` / `openclaw` |
| `tool_name` | `*` any tool, or `Bash`, `Write`, … (MCP-namespace-tail aware) |
| `matcher_type` | `always` / `exact` / `prefix` / `contains` / `glob` / `regex` |
| `pattern` | the matcher payload (ignored for `always`) |
| `field` | tool-input field to match (blank = per-tool default: `command` for Bash; `*` = whole input JSON) |
| `action` | `require_approval` / `allow` / `deny` |
| `severity` | `low` / `medium` / `high` / `critical` |
| `category` | free-text grouping (`filesystem`, `git`, …) |
| `approval_prompt` | optional human prompt shown on the card |
| `order` | lower = evaluated first; **first match wins** |
| `version` | bumped on each edit |

`tool_approval_requests` is the durable audit log: every gated call, the matching
policy, and how it resolved (`pending` → `approved` / `denied` / `expired`).

### First-match-wins ordering

Policies are sorted by `(order, id)`. Put a low-`order` `allow` rule **above** a
broad `require_approval` rule to carve safe exceptions:

```
order 10  allow            Bash prefix "git status"
order 100 require_approval Bash prefix "git"
```

`git status` is allowed; every other `git …` needs approval. Caution: a broad
`contains` deny like `rm -rf /` also matches `rm -rf /tmp/safe/x` — order and
specificity matter.

---

## API (llm-bawt, port 8642)

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/tool-approval-policies` | list rules |
| POST | `/v1/tool-approval-policies` | create |
| GET | `/v1/tool-approval-policies/{id}` | one rule |
| PATCH | `/v1/tool-approval-policies/{id}` | update (bumps version) |
| DELETE | `/v1/tool-approval-policies/{id}` | delete |
| POST | `/v1/tool-approval-policies/seed-defaults` | seed starter rules if empty |
| GET | `/v1/tool-approval-policies/bundle?etag=` | **compiled bundle bridges fetch** (304-style `{unchanged}` on etag match) |
| POST | `/v1/admin/reload-tool-approval-policies` | broadcast a cache-drop to every bridge |
| GET | `/v1/tool-approval-requests?status=&bot_id=&limit=` | audit list |
| POST | `/v1/chat/approvals/{id}/resolve` | approve/deny → grant + continuation prompt |

BawtHub reaches all of these through the existing `/api/chat/proxy/v1/*`
catch-all (no new Traefik prefixes).

---

## Bridge behaviour & config

The bridge fetches the bundle with a short TTL and drops its cache instantly on
an `approval:policies:reload` broadcast (so admin edits take effect without a
restart). Grants are one-shot, in-memory, with a TTL.

Env knobs (claude-code bridge):

| Var | Default | Meaning |
|---|---|---|
| `CLAUDE_CODE_APPROVAL_BUNDLE_TTL` | `15` | seconds between bundle refetches |
| `CLAUDE_CODE_APPROVAL_FAIL_CLOSED` | unset (fail-open) | `1`/`true` → deny **all** tools when the policy service is unreachable |

**Fail-open vs fail-closed.** Default is **fail-open**: if the bridge can't fetch
policies (app blip), tools run as normal. This favours availability — the app
being down already breaks the turn, and these are productivity gates, not auth.
Set fail-closed only if you'd rather halt all tool execution than risk an
ungated destructive command during an outage.

---

## Default policy set (`seed-defaults`)

Conservative, shell-destructive, **enabled** (a safety feature that ships
disabled protects nothing). All target `Bash`:

| pattern (regex/prefix) | severity | category |
|---|---|---|
| `rm` with a recursive/force flag | high | filesystem |
| `sudo ` (prefix) | high | privilege |
| `git push … --force/-f` | high | git |
| `DROP TABLE` / `DROP DATABASE` / `TRUNCATE` | critical | database |
| `mkfs` / `dd of=/dev/` / `> /dev/sd` / `chmod -R 777` / fork-bomb | critical | system |
| `curl\|wget … \| sh` | high | network |

Seeding is **not** automatic on startup — call `seed-defaults` (or the UI's
"Seed defaults" button) so activation is a deliberate operator action. Edit or
delete any rule afterward.

---

## Rollout

1. **Deploy the app** (new routes + event handling): `docker compose restart app`.
2. **Deploy the bridges** (the gate itself lives in `claude_code_bridge` /
   `codex_bridge`): restart the bridge containers. ⚠️ This kills any in-flight
   agent turn on that bridge — do it when idle.
3. Open `/tools/approval-policies`, click **Seed defaults** (or add rules).
4. Verify: have an agent run a gated command (e.g. `rm -rf` something harmless in
   a sandbox) → the turn should end with an Approve/Deny card; approving should
   let the re-issued command run once.

Until step 2, policies are stored but unenforced.

> **codex bridge:** the gate scaffolding mirrors claude-code, but the codex
> bridge's `can_use_tool` integration is not wired in this change — it still runs
> `approval_policy: "never"`. Claude-code is the enforced path today.

---

## Tests

- `tests/test_approval_engine.py` — 28 pure-engine cases (matchers, ordering,
  first-match-wins, grant-key stability, serialization). Runs anywhere.
- `tests/test_approval_bridge_gate.py` — bridge decision/grant/gate logic against
  injected bundles (skips if `claude_agent_sdk` absent).

Both run standalone (`python tests/test_*.py`) since the containers don't ship
pytest.
