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
