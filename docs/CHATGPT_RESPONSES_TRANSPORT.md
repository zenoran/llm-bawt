# ChatGPT Responses transport (TASK-864)

## Scope

The Claude proxy's `openai_chatgpt/gpt-6-astra` route uses a dedicated
WebSocket Responses client with Responses Lite payloads. Other ChatGPT models,
OpenAI API-key routes, and other providers retain their current transports.

Protocol reference: public `openai/codex`, tag `rust-v0.153.4`:
`codex-rs/core/src/client.rs`, `codex-api/src/common.rs`,
`codex-api/src/endpoint/responses_websocket.rs`, and
`codex-api/src/sse/responses.rs`.

- Upgrade `/backend-api/codex/responses` using the existing broker bearer and
  account/session headers; beta `responses_websockets=2026-02-06`.
- Send flattened `response.create`, with Lite enabled by
  `x-openai-internal-codex-responses-lite: true` and corresponding request metadata.
- Move tools to an `additional_tools` developer input item and instructions to a
  developer message. Stable IDs derive from conversation and visible content.
  Disable parallel tool calls; use `reasoning.context=all_turns`; strip image
  detail fields. Existing explicit reasoning settings win; Astra defaults low.
- Capture `x-codex-turn-state` from upgrade/HTTP headers or `response.metadata`;
  replay in reconnect headers and WS create metadata.
- Explicit HTTP 426 upgrade rejection selects SSE with the same Lite payload.
  Authentication, quota, malformed request and server errors are not hidden by
  fallback. The existing output-aware retry state machine remains authoritative.
- A valid terminal ends consumption immediately. EOF without terminal errors;
  incomplete responses retain the translator's max-token semantics. No transport
  layer silently retries generation. Never replay after forwarded tool calls.

## Ownership and bounds

The adapter owns the client and closes it at proxy shutdown. Exclusive leases
are keyed by endpoint, account, hashed bearer, durable conversation, bridge
request/turn, and model. Parallel requests receive separate sockets; anonymous
calls cannot reuse a lease. Token rotation cannot reuse an old authenticated socket.

At most 32 active leases and 32 idle entries are retained. Idle entries expire
after 60 seconds; connect/pool waits are bounded at 15 seconds, sends at 60,
and upstream-event inactivity at 300 (Codex's default, not the abandoned
Astra-only 60-second breaker). Cancellation and incomplete/error responses close
the socket; scoped sticky routing can survive in an idle entry for safe retry.

Full history is sent each hop. `previous_response_id`/delta optimization is
intentionally not used: Claude SDK histories can branch and omit upstream
reasoning/output items. No cross-turn warm pool, Codex tool namespaces,
attestation, or claim of complete Codex runtime parity.

## Verification receipts (2026-09-07 ET)

- 147 hermetic tests passed across the new transport suite and existing retry,
  concurrency, Claude proxy, and passthrough suites. Scoped Ruff and diff checks
  passed.
- Isolated live raw transport: Astra produced an `echo_check(value=ping)` tool
  call at 5.40s; on the reused WS connection the supplied tool result produced
  `pong` at 8.24s total. Both hops emitted `response.completed`.
- Isolated live full adapter: Anthropic request -> WS/Lite -> Anthropic tool_use
  at 3.20s; supplied tool_result -> `pong` / end_turn at 4.95s total. Usage was
  captured for both hops. No real tool execution or bot session was involved.
- Raw local smoke logs: `/tmp/task864-smoke.log` and
  `/tmp/task864-adapter-smoke.log`. Durable summary lives in TASK-864.

Nick activated the client by restarting the Claude bridge at
2026-09-08T00:54:52Z. Normal Loopy chat confirmed WebSocket transport, connection
reuse on the next tool hop, and 98.5% prompt-cache hits on that follow-up.
These checks establish activation and protocol compatibility, not proof that
every historical stall is eliminated. Long-running reliability remains unverified.
This change does not repair pre-existing orphaned UI turns.

## Activation

No services were restarted while implementing. The source-mounted Claude bridge
must reload to activate this client for normal chat. Only restart
`claude-code-bridge` with Nick's explicit approval for that run; do not restart
other services. No dependency/image rebuild is required (`websockets` is already
a declared dependency). Rollback is a source revert of TASK-864 plus the same
approved bridge reload; model profiles and credentials are unchanged.
