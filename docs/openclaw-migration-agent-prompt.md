# OpenClaw Integration Migration Prompt (for coding agent)

You are implementing a migration in llm-bawt from the legacy OpenClaw integration (non-streaming POST to /v1/chat/completions + SSH fallback) to the new Gateway HTTP/OpenResponses path with SSE streaming.

**This migration also fixes a cross-cutting bug: streaming responses are not persisted if the client disconnects.** The fix is general (benefits all backends, not just OpenClaw) and should be done first.

## Context

### Current state (what exists)
- src/llm_bawt/agent_backends/openclaw.py (~720 lines) — Gateway API transport via POST /v1/chat/completions (OpenAI-compatible, **non-streaming**) + SSH legacy transport
- src/llm_bawt/clients/agent_backend_client.py — wraps OpenClawBackend as LLMClient with SUPPORTS_STREAMING = False
- Tool calls extracted from: (a) chat completions message.tool_calls, (b) /tools/invoke sessions_history endpoint
- Config via bare os.getenv("OPENCLAW_GATEWAY_URL") etc. — **not** through the Config pydantic-settings class
- No tests for any OpenClaw code path
- Registered via entry point in pyproject.toml: openclaw = "llm_bawt.agent_backends.openclaw:OpenClawBackend"

### Critical bug: response persistence depends on client connection
In src/llm_bawt/service/background_service.py, the response finalization logic (history save, turn log, memory extraction) only runs if:
1. The full stream completes without exception
2. cancel_event is not set (no new request came in)
3. loop.call_soon_threadsafe doesn't raise RuntimeError (event loop still alive)

**Failure modes (all backends, not just OpenClaw):**
- Client disconnects mid-voice-stream → thread hits RuntimeError in call_soon_threadsafe → falls into except block → finalize_response() never runs → response lost
- New request arrives while slow backend (OpenClaw, vLLM cold start) is still generating → cancel_event set → finalize skipped → response lost
- OpenClaw takes 30+ seconds doing tool work → client times out → same failure

**Additionally, finalization logic is duplicated between the non-streaming path (_do_query, ~line 860) and streaming path (_stream_to_queue, ~line 1322) with copy-pasted code for:**
- finalize_response() call
- _update_turn_log() call
- _write_debug_turn_log() call
- AgentBackendClient tool-call extraction (duplicated at ~line 828 and ~line 1292)

### Target integration
- OpenResponses API: POST /v1/responses
- Streaming via SSE (stream: true) with events:
  - response.created
  - response.output_item.added
  - response.output_text.delta
  - response.completed
- Gateway tools: POST /tools/invoke (for session tool history, not direct execution)

### Gateway connectivity (verified healthy)
- http://127.0.0.1:18789
- http://10.0.0.97:18789

## High-level objective
1. **Fix durable response persistence** — responses are always persisted regardless of client state (Commit 0, all backends)
2. **Migrate OpenClaw to OpenResponses** — streaming via /v1/responses SSE
3. **Surface tool events in real-time** — during stream, not after
4. **No implicit SSH/legacy fallback** in normal chat path

## Files to modify

| File | Change |
|------|--------|
| src/llm_bawt/service/background_service.py | **Commit 0**: Extract shared _finalize_turn() helper; make persistence durable in streaming path; fix mixed str/dict chunk accumulation |
| src/llm_bawt/agent_backends/openclaw.py | Replace _run_gateway_api() to call /v1/responses; add stream_raw() SSE method; keep SSH behind explicit flag |
| src/llm_bawt/clients/agent_backend_client.py | Set SUPPORTS_STREAMING = True; implement stream_raw() delegating to backend's new SSE method |
| src/llm_bawt/utils/config.py | Add OpenClaw config fields to the Config class (see section 7 below) |
| tests/test_openclaw.py | New test file (see section 9 below) |
| tests/test_stream_persistence.py | New test file for durable persistence (see section 9) |
| docs/OPENCLAW_INTEGRATION.md | Update with new path, config, smoke check |

## Requirements

### 0. Durable response persistence (Commit 0 — do this first, benefits all backends)

**Problem:** The streaming path in _stream_to_queue() has 3 failure modes where finalize_response() is skipped and the LLM response is never saved to history. The non-streaming path has a similar issue on cancellation. The finalization logic is also copy-pasted between both paths.

**Solution: extract a shared _finalize_turn() method and make it always run.**

#### 0a. Extract _finalize_turn() helper

Create a single method on the service class that consolidates all post-response work:

    def _finalize_turn(
        self,
        *,
        llm_bawt,
        turn_id: str,
        response_text: str,
        tool_context: str,
        tool_call_details: list[dict],
        prepared_messages: list,
        user_prompt: str,
        model: str,
        bot_id: str,
        user_id: str,
        elapsed_ms: float,
        stream: bool,
    ) -> None:

This method does (in order):
1. Adapter output cleaning (if adapter exists)
2. Agent backend tool-call extraction (if client is AgentBackendClient) — **one place, not two**
3. llm_bawt.finalize_response(response_text, tool_context) — history save
4. self._update_turn_log(turn_id=turn_id, status="ok", ...)
5. _write_debug_turn_log(...) if debug enabled

Both _do_query() (non-streaming) and _stream_to_queue() (streaming) call this same method. Delete the duplicated inline code from both.

#### 0b. Make streaming persistence client-disconnect-safe

In _stream_to_queue(), restructure the stream consumption loop:

    for chunk in stream_iter:
        # Always accumulate text (even if cancelled/disconnected)
        if isinstance(chunk, str):
            full_response_holder[0] += chunk

        # Best-effort send to SSE consumer — skip silently if loop is gone
        try:
            loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
        except RuntimeError:
            pass  # event loop closed, client disconnected — that's fine

    # Always finalize if we have any response
    if full_response_holder[0]:
        self._finalize_turn(...)

Key changes:
- **Remove the `not cancel_event.is_set()` guard** from finalize. Always persist.
- **Wrap call_soon_threadsafe in try/except RuntimeError** so client disconnect doesn't abort the thread.
- **Guard str accumulation with isinstance** so dict tool-call events don't crash the concatenation.
- **Move finalize into a finally block** (or unconditional post-loop) so it runs even on partial responses.

#### 0c. What this fixes
- Voice streaming: kill the stream mid-response → response still saved to history
- OpenClaw slow tools: client times out → response still saved when it eventually completes
- New request cancels old: partial response from old request still persisted (better than nothing)
- Any backend, any client — universal fix

### 1. Replace gateway transport to use /v1/responses
- Replace the current _run_gateway_api() method (which POSTs to /v1/chat/completions) with a new method that POSTs to /v1/responses.
- Request body should include: input (user message), model (if applicable), stream flag, and any session/agent context.
- Non-streaming: parse the final JSON response body.
- Streaming: return an Iterator[str] of text deltas (see section 2).

### 2. Add SSE streaming parser
- Parse the SSE event stream from /v1/responses with stream=true:
  - Handle event: and data: lines per SSE spec (RFC 8895)
  - Tolerate partial chunks, empty lines, and data: [DONE] sentinel
  - Do **not** implement automatic reconnect — if the stream breaks, let it fail and log the termination reason (the service layer already handles keepalive/error propagation to the frontend)
- Emit text deltas as plain str chunks from stream_raw(), matching the contract used by OpenAIClient.stream_raw() and LlamaCppClient.stream_raw()

### 3. Map OpenResponses events to existing SSE protocol
The service already emits these SSE event types to the frontend — no new event types needed:

| OpenResponses SSE event | Action in stream_raw() |
|-------------------------|------------------------|
| response.created | Log response ID; record stream start time for first-token latency |
| response.output_item.added | If tool-type item: yield dict with event=tool_call, name, arguments (service layer already handles service.tool_call SSE emission) |
| response.output_text.delta | Yield the delta string directly (becomes chat.completion.chunk in service layer) |
| response.completed | Log completion; do not yield (service layer sends [DONE] sentinel) |

**No new internal event types** (assistant_started etc.) are needed — the existing EventCollector categories (EventCategory.LLM) and the SSE chunk protocol already cover this.

### 4. Non-stream fallback
- When OPENCLAW_STREAM_ENABLED is false (or stream=False in request), POST to /v1/responses with stream: false and return the full text response.
- Default to stream=true.

### 5. SSH removal from normal path
- Keep _run_ssh() and related methods but gate behind OPENCLAW_USE_SSH_FALLBACK=true (already exists as env var).
- Default: SSH is **not** used. _resolve_transport() should return gateway_api unless SSH is explicitly enabled.
- Do not delete SSH code yet (emergency fallback).

### 6. Mixed chunk type handling in service layer
The streaming chunk consumer in _stream_to_queue() currently does:

    full_response_holder[0] += chunk  # assumes str

When stream_raw() yields a mix of str deltas and dict tool events, this crashes. Fix:
- Guard accumulation: only concatenate str chunks
- The dict chunks (tool events) are already handled by the SSE consumer — they just need to pass through the queue without crashing the accumulator
- This is resolved by the isinstance guard in section 0b

### 7. Config fields — add to Config class
Add these to src/llm_bawt/utils/config.py in the Config class, following existing conventions (LLM_BAWT_ prefix auto-applied by pydantic-settings):

    # OpenClaw Gateway
    OPENCLAW_GATEWAY_URL: str = Field(default="http://127.0.0.1:18789", description="OpenClaw gateway base URL")
    OPENCLAW_GATEWAY_TOKEN: str = Field(default="", description="Bearer token for OpenClaw gateway")
    OPENCLAW_AGENT_ID: str = Field(default="main", description="OpenClaw agent namespace")
    OPENCLAW_STREAM_ENABLED: bool = Field(default=True, description="Enable SSE streaming for OpenClaw responses")
    OPENCLAW_USE_SSH_FALLBACK: bool = Field(default=False, description="Allow SSH transport as emergency fallback")

Then update openclaw.py to read from the injected Config instance rather than bare os.getenv() calls, with backward-compatible fallback to env vars for the transition period.

**Auth header**: Authorization: Bearer {token} — consistent with all other providers in this codebase.

### 8. Structured logging
Use logging.getLogger(__name__) (existing pattern). Log:
- **Request start**: response ID, agent ID, stream mode
- **First-token latency**: time from request to first response.output_text.delta
- **Total latency**: time from request to response.completed
- **Stream termination**: reason (completed, error, client disconnect)
- **Tool events**: tool name and timing when response.output_item.added with tool type is received

### 9. Tests

#### tests/test_stream_persistence.py (durable persistence)
| Test | What it validates |
|------|-------------------|
| test_finalize_turn_saves_history | _finalize_turn() calls finalize_response() and _update_turn_log() |
| test_finalize_turn_extracts_agent_backend_tools | _finalize_turn() extracts tool calls when client is AgentBackendClient |
| test_stream_persists_on_client_disconnect | Simulate RuntimeError from call_soon_threadsafe; verify finalize still runs |
| test_stream_persists_on_cancellation | Set cancel_event mid-stream; verify response is still persisted |
| test_stream_accumulates_only_str_chunks | Mixed str/dict stream; verify full_response_holder only contains text |

#### tests/test_openclaw.py (OpenClaw SSE migration)
| Test | What it validates |
|------|-------------------|
| test_sse_parser_basic_events | Parse well-formed SSE lines into (event_type, data) tuples |
| test_sse_parser_partial_chunks | Handle split-across-chunk SSE data correctly |
| test_sse_parser_done_sentinel | data: [DONE] terminates the stream |
| test_stream_raw_yields_text_deltas | Mock HTTP response -> stream_raw() yields text strings |
| test_stream_raw_emits_tool_call_dicts | Tool-type output items yield dict events |
| test_non_stream_response | stream=False -> returns complete text |
| test_gateway_auth_header | Request includes Authorization: Bearer {token} |
| test_ssh_fallback_disabled_by_default | _resolve_transport() returns gateway_api when OPENCLAW_USE_SSH_FALLBACK is not set |

Run with: make docker-exec CMD="uv run pytest tests/test_openclaw.py tests/test_stream_persistence.py -v"

### 10. End-to-end verification (run after each commit)

After each commit, run real E2E tests inside Docker to verify behavior. The service runs on port 8642.

**Always use `--bot proto` for testing** (nova is also fine; never use mira).

#### After Commit 0 (durable persistence):

Step 1 — Verify unit tests pass:

    make docker-exec CMD="uv run pytest tests/test_stream_persistence.py -v"

Step 2 — Rebuild and start the service:

    make run

Step 3 — Send a streaming chat request and verify it persists:

    # Stream a response (should render token by token)
    curl -N http://localhost:8642/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": null, "messages": [{"role": "user", "content": "say hello in 3 words"}], "stream": true, "bot_id": "proto"}'

    # Check history — the assistant response should be there
    curl -s http://localhost:8642/v1/history?bot_id=proto | python3 -m json.tool | tail -20

    # Check turn log — should show status "ok"
    curl -s http://localhost:8642/v1/turn-logs?bot_id=proto | python3 -m json.tool | head -30

Step 4 — **Critical test: kill the stream mid-response and verify persistence.**

    # Start a longer streaming response, then Ctrl-C after a few chunks
    curl -N http://localhost:8642/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": null, "messages": [{"role": "user", "content": "write a detailed paragraph about the history of computing"}], "stream": true, "bot_id": "proto"}'
    # Press Ctrl-C after seeing a few tokens

    # Wait 2-3 seconds for the backend thread to finish, then check history
    sleep 3
    curl -s http://localhost:8642/v1/history?bot_id=proto | python3 -m json.tool | tail -30

    # The response should be saved (possibly the full response, not just up to where you killed it)
    # The turn log should show status "ok", not missing
    curl -s http://localhost:8642/v1/turn-logs?bot_id=proto | python3 -m json.tool | head -30

Step 5 — Clean up test history:

    curl -X DELETE http://localhost:8642/v1/history?bot_id=proto

#### After Commits 1-3 (OpenClaw migration):

Prerequisites: OpenClaw gateway must be running at the configured URL with a valid token.

Step 1 — Verify OpenClaw health:

    make docker-exec CMD="llm --status"

Step 2 — Test non-streaming:

    curl -s http://localhost:8642/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": null, "messages": [{"role": "user", "content": "hello"}], "stream": false, "bot_id": "<openclaw-bot>"}' | python3 -m json.tool

Step 3 — Test streaming with token-by-token output:

    curl -N http://localhost:8642/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": null, "messages": [{"role": "user", "content": "explain what you can do"}], "stream": true, "bot_id": "<openclaw-bot>"}'

    # Verify: should see multiple "chat.completion.chunk" events with small deltas
    # NOT a single giant chunk (which would mean fake streaming)

Step 4 — Test tool events appear mid-stream (if the prompt triggers tool use):

    curl -N http://localhost:8642/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": null, "messages": [{"role": "user", "content": "search for the latest news about AI"}], "stream": true, "bot_id": "<openclaw-bot>"}'

    # Verify: should see "service.tool_call" events BEFORE the final text chunks
    # NOT all tool events bunched at the end after text is done

Step 5 — Verify persistence after disconnect (same pattern as Commit 0 Step 4):

    # Start request, Ctrl-C after a few tokens, wait, check history
    curl -N http://localhost:8642/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": null, "messages": [{"role": "user", "content": "write a long explanation of your capabilities"}], "stream": true, "bot_id": "<openclaw-bot>"}'
    # Ctrl-C
    sleep 5
    curl -s http://localhost:8642/v1/history?bot_id=<openclaw-bot> | python3 -m json.tool | tail -30

#### After Commit 4 (all tests):

    make docker-exec CMD="uv run pytest -v"

    # All tests must pass, no regressions

### 11. Preserve existing contracts
- AgentBackendClient.query() must still work (non-streaming path)
- chat_full() / OpenClawResult dataclass must still be populated for history/tool-call tracking
- get_tool_calls() and get_upstream_model() on the client must still work
- Entry point registration in pyproject.toml unchanged
- Bot config schema (agent_backend, agent_backend_config) unchanged

## Deliverables
- Code changes with clear commit(s)
- Updated docs/OPENCLAW_INTEGRATION.md:
  - Old path (POST /v1/chat/completions, non-streaming) vs new path (POST /v1/responses, SSE)
  - Required config (env vars or .env entries)
  - Smoke check: make docker-exec CMD="llm --bot <bot> 'hello'" with an OpenClaw-backed bot
- Brief "what changed" summary with file list

## Acceptance criteria
- **Durable persistence**: killing a voice/chat stream mid-response does NOT lose the response — it is saved to history
- **Durable persistence**: slow OpenClaw tool execution completes and persists even if client disconnects
- **No duplicate code**: finalization logic exists in exactly one place (_finalize_turn)
- Chat renders assistant text token-by-token via SSE when using an OpenClaw-backed bot
- Tool lifecycle events (service.tool_call, service.tool_result) are emitted during execution, not only at end
- Non-stream mode still works when OPENCLAW_STREAM_ENABLED=false
- SSH transport is not used unless OPENCLAW_USE_SSH_FALLBACK=true
- uv run pytest tests/test_openclaw.py tests/test_stream_persistence.py passes
- uv run pytest (full suite) passes with no regressions

## Execution style
1. **Commit 0: Durable response persistence** (do first — standalone, benefits everything)
   - Extract _finalize_turn() helper, deduplicate from both paths
   - Make streaming loop resilient to client disconnect
   - Add tests/test_stream_persistence.py
2. Commit 1: Config fields + SSE parser
3. Commit 2: openclaw.py gateway transport replacement + stream_raw()
4. Commit 3: agent_backend_client.py streaming enablement
5. Commit 4: tests/test_openclaw.py
6. Commit 5: Doc updates

Keep changes minimal but complete; avoid broad unrelated refactors.
