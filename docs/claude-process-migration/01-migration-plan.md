# Migrating from Agent SDK to Persistent Claude Processes

**Status:** proposal / open questions throughout
**Date:** 2026-06-03

## 1. The shape of today

`ClaudeCodeBridge` (`src/claude_code_bridge/bridge.py`) receives `chat.send` commands from Redis and, for each turn, calls:

```python
async for event in claude_agent_sdk.query(prompt=..., options=ClaudeAgentOptions(resume=session_id, ...)):
    # translate to AgentEvent, publish to Redis
```

`query()` spawns a fresh `claude` CLI subprocess, pipes stream-json messages over stdin/stdout, and tears down when the iterator is exhausted (or when `msg_stream.aclose()` is called to abort). Session continuity across turns is achieved by passing `resume=session_id` so the new subprocess reloads conversation state from `~/.claude/projects/...`.

Costs of this model:
- Process startup, skill discovery, plugin load, CLAUDE.md hierarchical walk — paid every turn.
- Session JSON read from disk every turn (cheap, but not free).
- Tool schemas re-sent on every API call (~700KB+ in the test corpus).
- Each turn is a fresh process, so no in-memory state survives.

## 2. The proposed shape

One **persistent `claude` process per bot**, supervised by the bridge, with a stream-json IO channel kept open for the lifetime of the bot's "agent session." Prompts arrive over Redis and are written into the bot's process; events stream back out and become `AgentEvent`s the same way they do today.

```
Redis agent:commands
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  ClaudeCodeBridge                                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │ ClaudeProcessPool                                │   │
│  │  bot_foo → ClaudeProcessHandle (running)         │   │
│  │  bot_bar → ClaudeProcessHandle (running)         │   │
│  │  bot_baz → ClaudeProcessHandle (idle, warm)      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────┬───────┬───────┬───────────────────────────────┘
          │       │       │   stream-json over stdin/stdout
          ▼       ▼       ▼
     ┌──────┐┌──────┐┌──────┐
     │claude││claude││claude│  --input-format stream-json
     │ foo  ││ bar  ││ baz  │  --output-format stream-json
     └──┬───┘└──┬───┘└──┬───┘
        │       │       │   ANTHROPIC_BASE_URL=http://127.0.0.1:8765
        ▼       ▼       ▼
            claude-proxy (audit log)
                │
                ▼
        api.anthropic.com
```

### What we gain
1. **No per-turn process spawn cost.** Plugin / skill / CLAUDE.md load happens once per bot per process lifetime.
2. **Intrinsic session state.** No `--resume` reload per turn; the conversation IS the running process.
3. **Cheaper API bills under the new June-15 metering.** Fewer redundant tool-schema sends if we can keep cache hits high (long-lived process = stable cache prefix).
4. **Human attachability** (if we wrap in tmux): operator can `tmux attach -t claude-foo` and watch a bot work.
5. **Decoupled supervisor.** Bridge can restart without killing every bot mid-thought (if the process supervisor is external — see §4).

### What we lose / must rebuild
1. **`claude_agent_sdk` event type marshalling** — we'd be parsing JSONL directly.
2. **`total_cost_usd`** — not in the API; would compute from token counts via a price table.
3. **`stream.aclose()` abort semantics** — we'd send SIGTERM or close stdin instead.
4. **SDK-handled OAuth refresh** — already done outside the SDK in our bridge (`bridge.py:106-120`), so no change here.

## 3. The biggest open question: IO transport

Four serious options. None is obviously right.

### Option A: Direct `asyncio.subprocess` with stream-json
```python
proc = await asyncio.create_subprocess_exec(
    "claude", "--input-format", "stream-json", "--output-format", "stream-json",
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    env={...ANTHROPIC_BASE_URL=proxy...},
)
# Reader task pulls JSONL lines from proc.stdout, parses, emits AgentEvents
# Writer pushes user messages into proc.stdin when chat.send arrives
```

- **Pros:** simplest. One Python object per bot. Backpressure via asyncio. Easy to log every byte to a per-bot file.
- **Cons:** if the bridge crashes, every bot's claude process is orphaned and the kernel reaps them. Human attach requires `tail -f` of the log file, no live REPL.
- **Per-bot isolation:** trivial, one handle per bot.

### Option B: tmux + interactive TUI mode + `send-keys` / `pipe-pane`
```
tmux new-session -d -s claude-foo claude
tmux send-keys -t claude-foo "hello there" Enter
tmux pipe-pane -t claude-foo -o 'cat > /tmp/claude-foo.out'
```

- **Pros:** best human attach experience — `tmux attach -t claude-foo` and you're in a live REPL. Process survives bridge restart trivially.
- **Cons:** parsing TUI output is hell — ANSI escapes, redraws, status-line overwrites, scrollback eviction. We'd be doing OCR on a terminal. `send-keys` for prompts is OK for single-line, awkward for multi-line and large pasted contexts.
- **Per-bot isolation:** one tmux session per bot. Clean.

**Honest read:** this is the option that *sounds* right when you first think "tmux." It is almost certainly the wrong one. The TUI is a presentation layer; we want machine-readable events.

### Option C: tmux + stream-json + named fifos
```
mkfifo /run/claude-foo.in /run/claude-foo.out
tmux new-session -d -s claude-foo \
  'claude --input-format stream-json --output-format stream-json \
    < /run/claude-foo.in > /run/claude-foo.out'
# Bridge: open fifos, write/read JSONL
```

- **Pros:** keeps stream-json IO (parseable). Process supervisor is tmux (survives bridge restart). Human can `tmux attach` to see the JSONL flow live (ugly but functional).
- **Cons:** fifos have annoying semantics — open-for-write blocks until reader present, both ends must be opened in the right order to avoid deadlock. EOF behavior is finicky. Two extra files per bot to track.
- **Per-bot isolation:** tmux session + fifo pair per bot.

### Option D: systemd user units + stream-json via socket
```ini
# ~/.config/systemd/user/claude-bot@.service
[Service]
ExecStart=/usr/local/bin/claude --input-format stream-json --output-format stream-json
StandardInput=socket
StandardOutput=socket
Restart=on-failure

# ~/.config/systemd/user/claude-bot@.socket
[Socket]
ListenStream=%t/claude-bot-%i.sock
Accept=no
```

- **Pros:** proper supervisor with restart policy. Survives bridge restart. Standard ops tooling (`systemctl status claude-bot@foo`). Logs go to journald automatically.
- **Cons:** more infrastructure to set up and explain. Heavier than the problem really requires. Spawning bots dynamically requires `systemctl start claude-bot@foo` and waiting for socket readiness.
- **Per-bot isolation:** one instantiated unit per bot.

### Recommendation (tentative)

**Option A for v1**, with the bridge owning everything. Add **optional tmux wrapping (Option B but for supervision only, not IO)** in v2 if we miss human attach. Defer C and D unless an operational need surfaces.

Reasoning: A gets us 80% of the benefit (persistence, cheaper API, decoupled lifecycle within a process) with the least new infrastructure. The "bridge crashes → all bots die" risk is mitigated by the bridge being short — most of the complexity moves into the per-bot handle, which can be made robust. A systemd unit (or just `Restart=on-failure` on the bridge container) handles the supervisor question one level up.

If after running A in production for a few weeks we find ourselves wishing we could attach and watch a bot, add tmux as a *wrapper around the same asyncio.subprocess command*, not as the IO channel:

```python
cmd = ["tmux", "new-session", "-d", "-s", f"claude-{bot_id}",
       "claude", "--input-format", "stream-json", "--output-format", "stream-json"]
# but then... IO goes where? back to fifos. → Option C.
```

So the honest call: if we want both stream-json IO *and* tmux attach, we end up at C. If we only want stream-json IO, A is simpler. Tmux is not a free addition to A — it forces fifos.

## 4. Process lifecycle

### Cold start (bot's first message after bridge boot)
1. `chat.send` arrives for `bot_id`.
2. `ClaudeProcessPool.get_or_spawn(bot_id)` checks its map.
3. Spawn `claude` with `cwd=<bot's workspace>`, env (`ANTHROPIC_BASE_URL`, OAuth token, etc.), `--session-id <existing-or-new>` if we want determinism (open Q — see §7).
4. Wait for initial `system` JSONL message confirming readiness.
5. Push user message into stdin, stream events out as today.

### Warm reuse (subsequent turns)
1. `chat.send` arrives.
2. Handle exists and is healthy → just write the user message JSONL to stdin.
3. Stream events back.

### Idle eviction
- Track `last_used_at` per handle.
- Background task: every N minutes, kill processes idle > T minutes (configurable).
- Cold-start cost on next message is acceptable; alternative is unbounded RAM growth as bots multiply.

### Abort mid-turn
- Today: `msg_stream.aclose()` kills the SDK's subprocess.
- New: send a control message? The CLI doesn't have a documented "interrupt current turn" stream-json message AFAIK. Fallback: SIGINT to the process (the CLI handles Ctrl-C in interactive mode). If that doesn't cleanly cancel mid-tool-call, SIGTERM + respawn with `--resume`.
- **Open question:** does `claude --input-format stream-json` accept any in-band abort signal? Worth investigating before committing.

### Crash recovery
- If the process exits unexpectedly: log, mark handle dead, emit a synthetic `ASSISTANT_DONE` with `stop_reason="error"` so the UI doesn't hang, respawn lazily on next message with `--resume <last_known_session>`.

### Bridge restart
- Today: in-flight turn dies with the bridge.
- With Option A: same — bots all die when bridge dies, respawn cold on next message.
- With Option B/C/D: bots survive bridge restart, bridge reattaches on boot. This is the actual reason to consider B/C/D.

## 5. Input channel: how user prompts get to the process

Path: `app → Redis agent:commands → bridge → ClaudeProcessHandle → proc.stdin`.

The stream-json input format expects newline-delimited JSON like:
```json
{"type": "user", "message": {"role": "user", "content": "hello"}}
```

Pseudocode in the bridge:
```python
async def on_chat_send(cmd: ChatSendCommand):
    handle = await pool.get_or_spawn(cmd.bot_id, session_key=cmd.session_key)
    await handle.send_user_message(cmd.message_text)
    # handle's reader task is already publishing AgentEvents to Redis
```

`handle.send_user_message`:
```python
async def send_user_message(self, text: str):
    msg = {"type": "user", "message": {"role": "user", "content": text}}
    line = json.dumps(msg) + "\n"
    self.proc.stdin.write(line.encode())
    await self.proc.stdin.drain()
```

A single in-flight turn per bot is enforced by the existing `SessionQueue` (`agent_bridge/session_queue.py`) — we already serialize turns per session, so no new lock required.

## 6. Output channel: parsing stream-json back into `AgentEvent`s

The SDK currently translates SDK types → `AgentEvent`. Without the SDK we translate JSONL → `AgentEvent` directly. The format is documented (`docs/CLAUDE_CODE_BRIDGE.md` and the SDK source) and stable. Reader task per handle:

```python
async def _read_loop(self):
    async for line in self.proc.stdout:
        msg = json.loads(line)
        await self._dispatch(msg)

async def _dispatch(self, msg: dict):
    t = msg.get("type")
    if t == "assistant":
        # extract text deltas, tool_use blocks → emit ASSISTANT_DELTA / TOOL_START
    elif t == "user":
        # tool_result blocks → emit TOOL_END
    elif t == "result":
        # final usage + stop reason → emit ASSISTANT_DONE
    elif t == "system":
        # session_id, model → persist to bot profile
```

This is a roughly-line-for-line port of `bridge.py:650-818`. The events are the same shape; only the source changes.

## 7. Steering operations to preserve

| Operation | Today (SDK) | Proposed |
|---|---|---|
| Resume session | `ClaudeAgentOptions(resume=session_id)` per query | Pass `--session-id` once at spawn; persistence is the process |
| MCP context injection | Prepend to `system_prompt` in options | Either (a) bake into bot's CLAUDE.md / settings, (b) prepend to each user message, or (c) send as a `system` JSONL message at process start. **Open question.** |
| Abort mid-turn | `msg_stream.aclose()` | SIGINT to process; fall back to SIGTERM + respawn |
| Per-bot model override | `options.model="claude-sonnet-4-..."` | `claude --model <id>` at spawn |
| Per-bot working directory | `options.cwd=...` | `cwd` kwarg on `create_subprocess_exec` |
| Cost reporting | `ResultMessage.total_cost_usd` | Compute from token counts × per-model price table |

## 8. Per-bot isolation

Today the SDK gets a `--session-id` argument and stores conversation under `~/.claude/projects/-<encoded-cwd>/<session-id>.jsonl`. Several bots sharing a cwd → several session files in the same project dir. Should be fine.

To be safe and clean:
- Each bot gets its own `cwd` (e.g. `/var/lib/llm-bawt/bot-workspaces/<bot_id>/`).
- Each bot's process has its own env (no cross-bot env leaks).
- If we ever want bot-specific skills / CLAUDE.md, the per-bot cwd is the hook.

Parallel conversations: handles are independent asyncio coroutines; no shared lock. The only contention is the proxy (one HTTP server, fine) and the Redis publish pipeline (already concurrent).

## 9. The proxy fits cleanly into this

Set `ANTHROPIC_BASE_URL=http://127.0.0.1:8765` in each bot's process env. Per `PROXY_VS_SDK.md` §5, demux options:
- **Port-per-bot:** one proxy listener per bot. Cheap if proxy is in-process.
- **Header tagging:** bridge sets `X-Bawt-Bot-Id` somehow (CLI doesn't have a documented way to inject custom outbound headers — possibly via `--settings` JSON? **Open question**).
- **One proxy, demux by source port / connection:** bridge knows which bot's pid owns which outbound connection, joins after the fact.

The simplest: one proxy, log everything to a single log dir, post-hoc tag by inspecting the request `user-agent` or `metadata.user_id` field (the CLI sends something we can match).

## 10. Phased rollout

### Phase 0: validation (no migration yet)
- Run the existing SDK bridge with `ANTHROPIC_BASE_URL` pointed at the proxy.
- Build the JSONL → AgentEvent translator in a separate module, run it against proxy logs offline, verify byte-for-byte parity with the events the SDK emitted for the same turn.
- Outcome: confidence that we can fully reconstruct AgentEvents from raw stream-json.

### Phase 1: shadow mode (one bot)
- Pick a low-traffic bot. Add a `ClaudeProcessHandle` for it.
- Run *both* the SDK path and the new persistent-process path in parallel for the same chat.send.
- Publish AgentEvents only from the SDK path; log the new path's events to disk.
- Diff. Fix discrepancies.

### Phase 2: cutover (per-bot opt-in)
- Add `agent_backend_config.use_persistent_process: true` flag.
- For bots with the flag set, route via `ClaudeProcessPool`. Others continue via SDK.
- Watch for regressions, especially in abort semantics and session resume after process restart.

### Phase 3: deprecate SDK path
- Once stable, flip the default. Keep SDK code path for one release as a fallback.

### Phase 4: cleanup
- Remove `claude_agent_sdk` dependency.
- Remove SDK-specific branches in `bridge.py`.

## 11. Open questions / things to investigate before committing

1. **Does `claude --input-format stream-json` accept an in-band cancel signal?** Or do we always need SIGINT? Probably worth a quick local experiment.
2. **Can the CLI inject custom outbound HTTP headers** so the proxy can tag per-bot? Check `--settings` JSON schema.
3. **How does `--session-id` interact with `--resume` over a persistent process's lifetime?** If the process owns the session for hours, is there any divergence vs. the JSONL on disk?
4. **What's the per-bot RAM footprint** of a long-lived `claude` process at steady state? If it's 500MB×20 bots, we have a sizing problem.
5. **Skill/plugin reload:** if a skill changes on disk, does the running process pick it up, or do we need to recycle? (Likely recycle.)
6. **Cost table source.** Where do we get authoritative per-model pricing? Anthropic publishes a table; we'd need to keep it current. Or scrape from `total_cost_usd` while the SDK is still in the loop and back-derive prices.
7. **MCP context injection mechanism.** Probably the messiest single thing to redesign. The SDK has a clean `options.system_prompt` hook; the CLI's equivalent is `--append-system-prompt` (per turn) or settings.json. With a persistent process, we set it once at spawn and it's baked in — but per-turn context (like the active task ID) can't be baked in. May need to prepend it to each user message text instead, which is uglier.

## 12. Non-goals

- Replacing the proxy. The proxy is complementary, not a competitor to the SDK or to persistent processes.
- Moving to direct API calls. The whole point is to keep using `claude` as the agent loop; we're just changing how we invoke it.
- Migrating the OpenClaw bridge. This proposal is Claude Code only.

## TL;DR for the impatient reviewer

- Move from `query()`-per-turn to one long-lived `claude` process per bot.
- IO over stream-json on stdin/stdout, parsed in the bridge into `AgentEvent`s.
- Use `asyncio.subprocess` directly. Skip tmux for v1. Revisit if we want human attach.
- Big preservation work: abort semantics, MCP context injection per turn, cost computation.
- Proxy fits in front of every process for audit. Already designed.
- Phase via shadow mode → opt-in flag → default → SDK removal.
