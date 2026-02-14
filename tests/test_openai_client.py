"""Targeted tests for OpenAI client payload behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_bawt.clients.base import LLMClient
from llm_bawt.clients.openai_client import OpenAIClient
from llm_bawt.models.message import Message


@dataclass
class _ConfigStub:
    PLAIN_OUTPUT: bool = True
    NO_STREAM: bool = False
    MAX_OUTPUT_TOKENS: int = 4096
    TEMPERATURE: float = 0.8
    TOP_P: float = 0.95
    VERBOSE: bool = False
    LLAMA_CPP_N_CTX: int = 8192
    SYSTEM_MESSAGE: str = ""


class _DummyClient(LLMClient):
    def query(self, messages: list[Message], plaintext_output: bool = False, **kwargs: Any) -> str:
        return ""

    def get_styling(self) -> tuple[str | None, str]:
        return None, "green"


def test_effective_context_window_treats_grok_as_large_context() -> None:
    client = _DummyClient(
        "grok-4-fast-non-reasoning",
        _ConfigStub(),
        model_definition={"type": "grok"},
    )
    assert client.effective_context_window == 128000


def test_query_respects_max_tokens_and_temperature_kwargs() -> None:
    client = OpenAIClient(
        model="grok-4-fast-non-reasoning",
        config=_ConfigStub(),
        base_url="http://localhost:9999",
        api_key="dummy",
        model_definition={"type": "grok"},
    )
    captured: dict[str, Any] = {}

    def _fake_chat_create(payload: dict[str, Any]):
        captured.update(payload)

        class _Usage:
            prompt_tokens = 1
            completion_tokens = 1
            total_tokens = 2

        class _Msg:
            content = "ok"

        class _Choice:
            message = _Msg()

        class _Completion:
            choices = [_Choice()]
            usage = _Usage()

        return _Completion()

    client._chat_create_with_fallback = _fake_chat_create  # type: ignore[method-assign]
    _ = client.query(
        [Message(role="user", content="hello")],
        stream=False,
        max_tokens=123,
        temperature=0.2,
        top_p=0.7,
    )
    assert captured["max_completion_tokens"] == 123
    assert captured["temperature"] == 0.2
    assert captured["top_p"] == 0.7


def test_query_with_tools_respects_max_tokens_kwargs() -> None:
    client = OpenAIClient(
        model="grok-4-fast-non-reasoning",
        config=_ConfigStub(),
        base_url="http://localhost:9999",
        api_key="dummy",
        model_definition={"type": "grok"},
    )
    captured: dict[str, Any] = {}

    def _fake_chat_create(payload: dict[str, Any]):
        captured.update(payload)

        class _Msg:
            content = "ok"
            tool_calls = []

        class _Choice:
            message = _Msg()

        class _Completion:
            choices = [_Choice()]

        return _Completion()

    client._chat_create_with_fallback = _fake_chat_create  # type: ignore[method-assign]
    _ = client.query_with_tools(
        [Message(role="user", content="hello")],
        tools_schema=[],
        max_tokens=222,
    )
    assert captured["max_completion_tokens"] == 222
