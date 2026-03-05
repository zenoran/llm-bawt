"""Unit tests for OpenClaw responses API + SSE migration."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from llm_bawt.agent_backends.openclaw import OpenClawBackend


class _FakeStreamingResponse:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks
        self._idx = 0

    def read(self, n: int = -1) -> bytes:
        if self._idx >= len(self._chunks):
            return b""
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _make_backend() -> OpenClawBackend:
    backend = OpenClawBackend()
    backend._config = SimpleNamespace(
        OPENCLAW_GATEWAY_URL="http://127.0.0.1:18789",
        OPENCLAW_GATEWAY_TOKEN="test-token",
        OPENCLAW_AGENT_ID="main",
        OPENCLAW_STREAM_ENABLED=True,
        OPENCLAW_USE_SSH_FALLBACK=False,
    )
    return backend


def test_sse_parser_basic_events() -> None:
    backend = _make_backend()
    chunks = [
        b"event: response.created\n",
        b'data: {"type":"response.created","response":{"id":"r1"}}\n\n',
        b"event: response.output_text.delta\n",
        b'data: {"type":"response.output_text.delta","delta":"hello"}\n\n',
    ]

    events = list(backend._iter_sse_events(iter(chunks)))

    assert len(events) == 2
    assert events[0][0] == "response.created"
    assert json.loads(events[0][1])["type"] == "response.created"
    assert events[1][0] == "response.output_text.delta"
    assert json.loads(events[1][1])["delta"] == "hello"


def test_sse_parser_partial_chunks() -> None:
    backend = _make_backend()
    chunks = [
        b"event: response.output",
        b"_text.delta\n",
        b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"hel",
        b"lo\"}\n\n",
    ]

    events = list(backend._iter_sse_events(iter(chunks)))

    assert len(events) == 1
    assert events[0][0] == "response.output_text.delta"
    assert json.loads(events[0][1])["delta"] == "hello"


def test_sse_parser_done_sentinel(monkeypatch) -> None:
    backend = _make_backend()
    sse = b"".join(
        [
            b"event: response.output_text.delta\n",
            b'data: {"type":"response.output_text.delta","delta":"a"}\n\n',
            b"data: [DONE]\n\n",
            b"event: response.output_text.delta\n",
            b'data: {"type":"response.output_text.delta","delta":"b"}\n\n',
        ]
    )

    monkeypatch.setattr(
        backend,
        "_request_gateway_responses",
        lambda prompt, config, stream: _FakeStreamingResponse([sse]),
    )

    chunks = list(backend.stream_raw("hello", {}))

    assert chunks == ["a"]


def test_stream_raw_yields_text_deltas(monkeypatch) -> None:
    backend = _make_backend()
    sse = b"".join(
        [
            b"event: response.created\n",
            b'data: {"type":"response.created","response":{"id":"r1","model":"openclaw:main"}}\n\n',
            b"event: response.output_text.delta\n",
            b'data: {"type":"response.output_text.delta","delta":"hel"}\n\n',
            b"event: response.output_text.delta\n",
            b'data: {"type":"response.output_text.delta","delta":"lo"}\n\n',
            b"event: response.completed\n",
            b'data: {"type":"response.completed","response":{"id":"r1","model":"openclaw:main"}}\n\n',
            b"data: [DONE]\n\n",
        ]
    )

    monkeypatch.setattr(
        backend,
        "_request_gateway_responses",
        lambda prompt, config, stream: _FakeStreamingResponse([sse]),
    )

    out = list(backend.stream_raw("hello", {}))

    assert out == ["hel", "lo"]
    result = backend.get_last_stream_result()
    assert result is not None
    assert result.text == "hello"


def test_stream_raw_emits_tool_call_dicts(monkeypatch) -> None:
    backend = _make_backend()
    sse = b"".join(
        [
            b"event: response.output_item.added\n",
            b'data: {"type":"response.output_item.added","item":{"type":"function_call","id":"call_1","name":"search_web","arguments":{"query":"ai"}}}\n\n',
            b"event: response.output_text.delta\n",
            b'data: {"type":"response.output_text.delta","delta":"done"}\n\n',
            b"event: response.completed\n",
            b'data: {"type":"response.completed","response":{"id":"r2"}}\n\n',
            b"data: [DONE]\n\n",
        ]
    )

    monkeypatch.setattr(
        backend,
        "_request_gateway_responses",
        lambda prompt, config, stream: _FakeStreamingResponse([sse]),
    )

    out = list(backend.stream_raw("hello", {}))

    assert isinstance(out[0], dict)
    assert out[0]["event"] == "tool_call"
    assert out[0]["name"] == "search_web"
    assert out[1] == "done"


def test_non_stream_response(monkeypatch) -> None:
    backend = _make_backend()

    async def _fake_json(prompt: str, config: dict) -> dict:
        return {
            "model": "openclaw:main",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "hello from non-stream"}
                    ],
                }
            ],
        }

    monkeypatch.setattr(backend, "_run_gateway_response_json", _fake_json)

    result = asyncio.run(backend.chat_full("hello", {}))

    assert result.text == "hello from non-stream"
    assert result.model == "openclaw:main"


def test_gateway_auth_header(monkeypatch) -> None:
    backend = _make_backend()
    captured = {}

    class _Resp:
        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_urlopen(req, timeout=0):
        captured["headers"] = dict(req.headers)
        return _Resp()

    from llm_bawt.agent_backends import openclaw as openclaw_module

    monkeypatch.setattr(openclaw_module.urlrequest, "urlopen", _fake_urlopen)

    with backend._request_gateway_responses("hello", {}, stream=False):
        pass

    assert captured["headers"]["Authorization"] == "Bearer test-token"


def test_ssh_fallback_disabled_by_default(monkeypatch) -> None:
    backend = _make_backend()
    monkeypatch.delenv("OPENCLAW_USE_SSH_FALLBACK", raising=False)

    assert backend._resolve_transport({}) == "gateway_api"


def test_stream_timeout_recovery_persists_response(monkeypatch) -> None:
    """When the SSE stream times out but OpenClaw finished, recover via non-stream."""
    backend = _make_backend()
    import socket

    # Simulate: SSE stream sends response.created then times out during tool work
    sse_initial = b"".join([
        b"event: response.created\n",
        b'data: {"type":"response.created","response":{"id":"r-timeout","model":"openclaw:main"}}\n\n',
    ])

    class _TimingOutResponse:
        """Fake HTTP response that yields initial SSE then raises socket.timeout."""
        def __init__(self):
            self._sent_initial = False

        def read(self, n: int = -1) -> bytes:
            if not self._sent_initial:
                self._sent_initial = True
                return sse_initial
            raise socket.timeout("timed out")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        backend,
        "_request_gateway_responses",
        lambda prompt, config, stream: _TimingOutResponse(),
    )

    # Mock the non-stream recovery to return a completed response
    async def _fake_json(prompt: str, config: dict) -> dict:
        return {
            "model": "openclaw:main",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "recovered response from OpenClaw"}
                    ],
                }
            ],
        }

    monkeypatch.setattr(backend, "_run_gateway_response_json", _fake_json)

    out = list(backend.stream_raw("fix the thing", {}))

    # Should have recovered the response text
    assert len(out) == 1
    assert out[0] == "recovered response from OpenClaw"

    # The last stream result should reflect recovery
    result = backend.get_last_stream_result()
    assert result is not None
    assert result.text == "recovered response from OpenClaw"
    assert result.raw["termination_reason"] == "recovered"


def test_stream_timeout_no_response_id_raises(monkeypatch) -> None:
    """When SSE stream times out before getting a response_id, it should raise."""
    backend = _make_backend()
    import socket

    class _ImmediateTimeout:
        def read(self, n: int = -1) -> bytes:
            raise socket.timeout("timed out")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        backend,
        "_request_gateway_responses",
        lambda prompt, config, stream: _ImmediateTimeout(),
    )

    import pytest
    with pytest.raises(socket.timeout):
        list(backend.stream_raw("hello", {}))
