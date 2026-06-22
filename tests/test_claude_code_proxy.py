from __future__ import annotations

import json
from types import SimpleNamespace

from claude_code_bridge.proxy.adapters.openai_chatgpt import (
    OpenAIChatGPTAdapter,
    _prompt_cache_key,
)
from claude_code_bridge.proxy.stream import responses_to_anthropic_sse
from claude_code_bridge.proxy.translate import anthropic_to_responses


def _sse_payloads(frames: list[bytes]) -> list[dict]:
    payloads: list[dict] = []
    for frame in frames:
        text = frame.decode("utf-8")
        data_line = next(line for line in text.splitlines() if line.startswith("data: "))
        payloads.append(json.loads(data_line[6:]))
    return payloads


def test_translate_moves_leading_datetime_out_of_instructions() -> None:
    body = {
        "model": "openai_chatgpt/gpt-5.4",
        "system": (
            "Current date/time: Saturday, June 20, 2026 2:16 PM EDT\n\n"
            "Keep tools byte stable."
        ),
        "messages": [{"role": "user", "content": "Fix the proxy"}],
        "stream": True,
    }

    payload = anthropic_to_responses(body, "gpt-5.4")

    assert payload["instructions"] == "Keep tools byte stable."
    assert payload["input"][0] == {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Current date/time: Saturday, June 20, 2026"},
            {"type": "input_text", "text": "Fix the proxy"},
        ],
    }


def test_prompt_cache_key_uses_stable_opening_only() -> None:
    base_payload = {
        "instructions": "Stable instructions",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Opening user turn"}]},
            {"role": "assistant", "content": "Earlier reply"},
            {"role": "user", "content": [{"type": "input_text", "text": "Later user turn"}]},
        ],
    }

    changed_later_history = {
        **base_payload,
        "input": [
            base_payload["input"][0],
            {"role": "assistant", "content": "Different reply"},
            {"role": "user", "content": [{"type": "input_text", "text": "Totally different later turn"}]},
        ],
    }
    changed_opening = {
        **base_payload,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Different opening"}]},
            *base_payload["input"][1:],
        ],
    }

    assert _prompt_cache_key(base_payload) == _prompt_cache_key(changed_later_history)
    assert _prompt_cache_key(base_payload) != _prompt_cache_key(changed_opening)


async def _collect_stream_frames(events: list[object]) -> list[bytes]:
    async def gen():
        for event in events:
            yield event

    frames: list[bytes] = []
    async for frame in responses_to_anthropic_sse(gen(), anthropic_model="openai_chatgpt/gpt-5.4"):
        frames.append(frame)
    return frames


def test_stream_reports_cached_and_uncached_input_tokens() -> None:
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="hello"),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=1400,
                    output_tokens=55,
                    input_tokens_details=SimpleNamespace(cached_tokens=1200),
                ),
                status="stop",
            ),
        ),
    ]

    frames = __import__("asyncio").run(_collect_stream_frames(events))
    payloads = _sse_payloads(frames)

    message_start = payloads[0]["message"]
    message_delta = next(p for p in payloads if p["type"] == "message_delta")

    assert message_start["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    assert message_delta["usage"] == {
        "input_tokens": 200,
        "output_tokens": 55,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 1200,
    }
    assert message_delta["delta"]["stop_reason"] == "end_turn"


def test_stream_message_delta_carries_bridge_fallback_input_usage() -> None:
    events = [
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=155730,
                    output_tokens=488,
                    input_tokens_details=SimpleNamespace(cached_tokens=154624),
                ),
                status="stop",
            ),
        ),
    ]

    frames = __import__("asyncio").run(_collect_stream_frames(events))
    payloads = _sse_payloads(frames)
    message_delta = next(p for p in payloads if p["type"] == "message_delta")

    assert message_delta["usage"] == {
        "input_tokens": 1106,
        "output_tokens": 488,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 154624,
    }
    assert (
        message_delta["usage"]["input_tokens"]
        + message_delta["usage"]["cache_read_input_tokens"]
    ) == 155730
    assert message_delta["delta"]["stop_reason"] == "end_turn"


def test_codex_usage_headers_parse_to_snapshot() -> None:
    """Proxy side: x-codex-* response headers -> canonical Redis snapshot."""
    from claude_code_bridge.proxy.usage_capture import _snapshot_from_headers

    headers = {
        "x-codex-plan-type": "plus",
        "x-codex-active-limit": "premium",
        "x-codex-primary-used-percent": "43",
        "x-codex-primary-window-minutes": "300",
        "x-codex-primary-reset-at": "1782093029",
        "x-codex-primary-reset-after-seconds": "12303",
        "x-codex-secondary-used-percent": "85",
        "x-codex-secondary-window-minutes": "10080",
        "x-codex-secondary-reset-at": "1782335997",
        "x-codex-secondary-reset-after-seconds": "255271",
        "x-codex-credits-has-credits": "False",
        "x-codex-credits-unlimited": "False",
    }
    snap = _snapshot_from_headers(headers)
    assert snap is not None
    assert snap["plan_type"] == "plus"
    assert snap["primary"] == {
        "used_percent": 43,
        "window_minutes": 300,
        "reset_at": 1782093029,
        "reset_after_seconds": 12303,
    }
    assert snap["secondary"]["used_percent"] == 85
    assert snap["secondary"]["window_minutes"] == 10080
    assert snap["credits"]["has_credits"] is False
    # No codex signal at all -> nothing published.
    assert _snapshot_from_headers({"content-type": "text/event-stream"}) is None


def test_openai_chatgpt_usage_adapter_maps_snapshot_to_canonical() -> None:
    """App side: Redis snapshot -> canonical limit windows + tiered name."""
    from llm_bawt.service.usage.adapters.openai_chatgpt import (
        OpenAIChatGPTUsageAdapter,
        _display_name,
        _limit_from_window,
    )

    primary = {"used_percent": 10, "window_minutes": 300, "reset_at": 1782093029}
    secondary = {"used_percent": 80, "window_minutes": 10080, "reset_at": 1782335997}

    p = _limit_from_window(primary, "session_5h", "5-hour limit", "5h")
    s = _limit_from_window(secondary, "weekly_all", "Weekly · all models", "7d")
    assert (p.id, p.label, p.window, p.used_pct, p.resets_at, p.active) == (
        "session_5h", "5-hour limit", "5h", 10.0, 1782093029, True,
    )
    assert (s.id, s.window, s.used_pct, s.resets_at) == (
        "weekly_all", "7d", 80.0, 1782335997,
    )
    # A window with no used_percent yields no limit (not a zero).
    assert _limit_from_window({"window_minutes": 300}, "session_5h", "x", "5h") is None
    assert _display_name("plus") == "ChatGPT · codex · Plus"
    assert _display_name(None) == "ChatGPT · codex"
    assert OpenAIChatGPTUsageAdapter.provider == "openai_chatgpt"


def test_claude_usage_percent_not_rescaled_to_fraction() -> None:
    """Claude `limits[].percent` is provider-native 0-100.

    Regression: a weekly window that just reset reports ``percent: 1`` (1%).
    The old ``0 <= v <= 1`` heuristic read that as a 0-1 fraction and showed
    100%. `percent` must never be rescaled; only the legacy top-level
    `utilization` fallback can be a 0-1 ratio (and even then 1 stays 1).
    """
    from llm_bawt.service.usage.adapters.claude import _parse_pct, _from_limit_item

    # The authoritative `percent` path: 0-100, never rescaled.
    assert _parse_pct(1, allow_fraction=False) == 1.0      # the bug: was 100.0
    assert _parse_pct(0, allow_fraction=False) == 0.0
    assert _parse_pct(0.5, allow_fraction=False) == 0.5    # genuine 0.5%
    assert _parse_pct(50, allow_fraction=False) == 50.0
    assert _parse_pct(100, allow_fraction=False) == 100.0
    assert _parse_pct(None, allow_fraction=False) is None

    # The `utilization` fallback keeps ratio rescaling, but the literal 1 is
    # 1%, not 100% (strict open interval).
    assert _parse_pct(0.35) == 35.0
    assert _parse_pct(0.92) == 92.0
    assert _parse_pct(1) == 1.0
    assert _parse_pct(0) == 0.0

    # End-to-end: a just-reset weekly window maps to used_pct=1, not 100.
    weekly = _from_limit_item(
        {"kind": "weekly_all", "percent": 1, "resets_at": 1782335997,
         "severity": "normal", "is_active": True},
        0,
    )
    assert weekly is not None
    assert weekly.id == "weekly_all"
    assert weekly.used_pct == 1.0
    assert weekly.resets_at == 1782335997


def test_prepare_request_sets_prompt_cache_key_and_preserves_store_false() -> None:
    adapter = OpenAIChatGPTAdapter()
    responses_body = {
        "model": "gpt-5.4",
        "instructions": "Stable instructions",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "Opening user turn"}]}],
        "store": True,
        "stream": False,
    }

    prepared = adapter.prepare_request(responses_body)

    assert prepared["store"] is False
    assert prepared["stream"] is True
    assert prepared["prompt_cache_key"] == _prompt_cache_key(prepared)
    assert prepared["prompt_cache_key"]


def test_prepare_request_accepts_gpt_5_4_effort_values(monkeypatch) -> None:
    adapter = OpenAIChatGPTAdapter()
    for effort in ("none", "low", "medium", "high", "xhigh"):
        monkeypatch.setenv("OPENAI_CHATGPT_REASONING_EFFORT", effort)
        prepared = adapter.prepare_request({"input": [], "instructions": "stable"})
        assert prepared["reasoning"] == {"effort": effort, "summary": "auto"}


def test_prepare_request_rejects_unsupported_minimal_effort(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_CHATGPT_REASONING_EFFORT", "minimal")
    prepared = OpenAIChatGPTAdapter().prepare_request(
        {"input": [], "instructions": "stable"}
    )
    assert prepared["reasoning"] == {"effort": "high", "summary": "auto"}


# ── cache_diag ──────────────────────────────────────────────────────────────
from claude_code_bridge.proxy import cache_diag  # noqa: E402


def _diag_body(*input_items: dict, instructions: str = "Stable instructions") -> dict:
    return {"instructions": instructions, "tools": [], "input": list(input_items)}


def test_cache_diag_is_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("PROXY_CACHE_DIAG", raising=False)
    cache_diag._state.clear()
    # Should not raise and should not record anything.
    cache_diag.record(_diag_body({"role": "user", "content": "hi"}))
    assert cache_diag._state == {}


def test_cache_diag_reports_healthy_when_only_tail_grows(monkeypatch, caplog) -> None:
    monkeypatch.setenv("PROXY_CACHE_DIAG", "1")
    cache_diag._state.clear()
    with caplog.at_level("INFO", logger="claude_code_bridge.proxy.cache_diag"):
        # Turn 1: stable opening.
        cache_diag.record(_diag_body(
            {"role": "user", "content": "Opening"},
            {"role": "assistant", "content": "Reply 1"},
        ))
        # Turn 2: same prefix, one new item appended → expected marginal turn.
        cache_diag.record(_diag_body(
            {"role": "user", "content": "Opening"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Next question"},
        ))
    last = [r for r in caplog.records if "turn=2" in r.getMessage()]
    assert last, "expected a turn=2 diag line"
    msg = last[-1].getMessage()
    assert "healthy" in msg
    assert "BREAKPOINT" not in msg


def test_cache_diag_reports_breakpoint_on_volatile_mid_history(monkeypatch, caplog) -> None:
    monkeypatch.setenv("PROXY_CACHE_DIAG", "1")
    cache_diag._state.clear()
    with caplog.at_level("INFO", logger="claude_code_bridge.proxy.cache_diag"):
        # Turn 1: a tool result with a fixed value.
        cache_diag.record(_diag_body(
            {"role": "user", "content": "Opening"},
            {"type": "function_call_output", "call_id": "tu_1", "output": "result-A"},
        ))
        # Turn 2: same call_id, but the tool returned fresh content this turn.
        cache_diag.record(_diag_body(
            {"role": "user", "content": "Opening"},
            {"type": "function_call_output", "call_id": "tu_1", "output": "result-B"},
        ))
    last = [r for r in caplog.records if "turn=2" in r.getMessage()]
    assert last, "expected a turn=2 diag line"
    msg = last[-1].getMessage()
    assert "BREAKPOINT=input[1]:function_call_output" in msg


# ── reasoning → thinking + keepalive (TASK-270 stall fix) ──────────────────

def _evt(etype: str, **kw: object) -> SimpleNamespace:
    return SimpleNamespace(type=etype, **kw)


def test_stream_translates_reasoning_to_thinking_blocks() -> None:
    """High-effort reasoning must surface as Anthropic thinking blocks (with a
    signature) instead of being silently dropped — the codex/byte stall."""
    events = [
        _evt("response.created"),
        _evt("response.output_item.added",
             item=SimpleNamespace(type="reasoning", id="rs_1")),
        _evt("response.reasoning_summary_part.added", summary_index=0,
             part=SimpleNamespace(type="summary_text", text="")),
        _evt("response.reasoning_summary_text.delta", delta="Thinking hard. "),
        _evt("response.reasoning_summary_part.added", summary_index=1,
             part=SimpleNamespace(type="summary_text", text="")),
        _evt("response.reasoning_summary_text.delta", delta="Step two."),
        _evt("response.output_item.done",
             item=SimpleNamespace(type="reasoning", id="rs_1",
                                  encrypted_content="ENC-XYZ")),
        _evt("response.output_item.added",
             item=SimpleNamespace(type="message", id="m1")),
        _evt("response.output_text.delta", delta="Answer."),
        _evt("response.output_item.done",
             item=SimpleNamespace(type="message", id="m1")),
        _evt("response.completed",
             response=SimpleNamespace(status="completed", usage=None)),
    ]
    payloads = _sse_payloads(__import__("asyncio").run(_collect_stream_frames(events)))
    types_seq = [p["type"] for p in payloads]

    # thinking block opened, streamed, signed, closed — before the text block.
    starts = [p for p in payloads if p["type"] == "content_block_start"]
    assert starts[0]["content_block"]["type"] == "thinking"
    assert starts[1]["content_block"]["type"] == "text"

    thinking_text = "".join(
        p["delta"]["thinking"] for p in payloads
        if p["type"] == "content_block_delta"
        and p["delta"].get("type") == "thinking_delta"
    )
    assert thinking_text == "Thinking hard. \n\nStep two."

    sig = next(p for p in payloads if p["type"] == "content_block_delta"
               and p["delta"].get("type") == "signature_delta")
    assert sig["delta"]["signature"] == "ENC-XYZ"

    # ping keepalive emitted during the pre-content lifecycle.
    assert "ping" in types_seq

    # blocks never overlap (Anthropic requires one open block at a time).
    depth = 0
    for t in types_seq:
        depth += t == "content_block_start"
        depth -= t == "content_block_stop"
        assert depth in (0, 1)
    assert depth == 0


def test_stream_reasoning_without_encrypted_content_still_signs() -> None:
    """A provider that omits encrypted_content still yields a non-empty
    signature so the SDK's thinking block parses."""
    events = [
        _evt("response.output_item.added",
             item=SimpleNamespace(type="reasoning", id="rs_9")),
        _evt("response.reasoning_text.delta", delta="raw cot"),
        _evt("response.output_item.done",
             item=SimpleNamespace(type="reasoning", id="rs_9",
                                  encrypted_content=None)),
        _evt("response.completed",
             response=SimpleNamespace(status="completed", usage=None)),
    ]
    payloads = _sse_payloads(__import__("asyncio").run(_collect_stream_frames(events)))
    sig = next(p for p in payloads if p["type"] == "content_block_delta"
               and p["delta"].get("type") == "signature_delta")
    assert sig["delta"]["signature"]  # non-empty sentinel


def test_stream_refusal_maps_to_refusal_stop_reason() -> None:
    events = [
        _evt("response.content_part.added",
             part=SimpleNamespace(type="refusal")),
        _evt("response.refusal.delta", delta="I can't help with that."),
        _evt("response.completed",
             response=SimpleNamespace(status="completed", usage=None)),
    ]
    payloads = _sse_payloads(__import__("asyncio").run(_collect_stream_frames(events)))
    message_delta = next(p for p in payloads if p["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "refusal"
    text = "".join(p["delta"]["text"] for p in payloads
                   if p["type"] == "content_block_delta"
                   and p["delta"].get("type") == "text_delta")
    assert text == "I can't help with that."


def test_stream_incomplete_maps_to_max_tokens() -> None:
    events = [
        _evt("response.output_text.delta", delta="partial"),
        _evt("response.incomplete",
             response=SimpleNamespace(
                 incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                 usage=None)),
    ]
    payloads = _sse_payloads(__import__("asyncio").run(_collect_stream_frames(events)))
    message_delta = next(p for p in payloads if p["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "max_tokens"


def test_stream_error_closes_open_block_then_errors() -> None:
    events = [
        _evt("response.output_item.added",
             item=SimpleNamespace(type="reasoning", id="r1")),
        _evt("response.reasoning_summary_text.delta", delta="hmm"),
        _evt("response.error",
             error=SimpleNamespace(message="boom", code="server_error")),
    ]
    payloads = _sse_payloads(__import__("asyncio").run(_collect_stream_frames(events)))
    types_seq = [p["type"] for p in payloads]
    assert types_seq[-1] == "error"
    assert "content_block_stop" in types_seq      # dangling block was closed
    assert "message_stop" not in types_seq        # error short-circuits finalize


def test_heartbeat_injects_pings_without_extending_stream() -> None:
    import asyncio

    from claude_code_bridge.proxy.heartbeat import with_heartbeat

    async def slow():
        yield b"A"
        await asyncio.sleep(0.25)
        yield b"B"

    async def run():
        out = []
        async for chunk in with_heartbeat(slow(), interval=0.05):
            out.append(chunk)
        return out

    out = asyncio.run(run())
    reals = [c for c in out if c in (b"A", b"B")]
    pings = [c for c in out if b"ping" in c]
    assert reals == [b"A", b"B"]
    assert len(pings) >= 2
    assert out[-1] == b"B"  # stream ends on a real frame, never a trailing ping
