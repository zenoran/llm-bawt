"""LLM client shell for agent backends (e.g. OpenClaw).

This wraps an :class:`AgentBackend` as a standard :class:`LLMClient` so that
the entire pipeline (history, memory, extraction, summarization, tool-call
SSE events) flows through the normal ``ServiceLLMBawt`` path.

The model alias for these virtual models is the backend name itself
(e.g. ``openclaw``).  A synthetic model definition with
``type: "agent_backend"`` is injected into ``config.defined_models`` at
service startup for each bot that declares ``agent_backend``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterator, List

from ..utils.config import Config
from ..models.message import Message
from .base import LLMClient

logger = logging.getLogger(__name__)


class AgentBackendClient(LLMClient):
    """LLMClient that delegates to an external agent backend.

    ``query()`` runs the backend's ``chat()`` coroutine synchronously
    (via ``asyncio.run`` or the running loop's executor) and returns the
    text response.  Structured metadata — tool calls, model info, usage —
    is stashed in ``last_result`` for the caller to inspect after the
    query completes.
    """

    SUPPORTS_STREAMING = False

    def __init__(
        self,
        backend_name: str,
        config: Config,
        bot_config: dict[str, Any] | None = None,
        model_definition: dict | None = None,
    ):
        model_definition = model_definition or {
            "type": "agent_backend",
            "backend": backend_name,
        }
        super().__init__(
            model=backend_name,
            config=config,
            model_definition=model_definition,
        )
        self._backend_name = backend_name
        self._bot_config: dict[str, Any] = bot_config or {}
        self._backend = self._load_backend(backend_name)

        # Populated after each query()
        self.last_result: Any | None = None

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

    def query(
        self,
        messages: List[Message],
        plaintext_output: bool = False,
        **kwargs: Any,
    ) -> str:
        """Send the last user message to the agent backend."""
        # Extract the last user message as the prompt
        prompt = ""
        for msg in reversed(messages):
            if msg.role == "user":
                prompt = msg.content or ""
                break

        if not prompt:
            logger.warning("AgentBackendClient.query() called with no user message")
            return ""

        # Run the async backend call synchronously.
        # If there's already a running loop we schedule via run_in_executor.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(
                    asyncio.run, self._chat_full(prompt)
                ).result()
        else:
            result = asyncio.run(self._chat_full(prompt))

        self.last_result = result
        return result.text if hasattr(result, "text") else str(result)

    def get_styling(self) -> tuple[str | None, str]:
        return self.bot_name, "cyan"

    def stream_raw(self, messages: List[Message], **kwargs: Any) -> Iterator[str]:
        """Agent backends don't natively stream; yield the full response."""
        response = self.query(messages, plaintext_output=True, **kwargs)
        if response:
            yield response

    @property
    def effective_context_window(self) -> int:
        """Agent backends manage their own context windows."""
        return 128_000

    @property
    def effective_max_tokens(self) -> int:
        return 16_384

    # ------------------------------------------------------------------
    # Tool call helpers (used by background_service streaming path)
    # ------------------------------------------------------------------

    def get_tool_calls(self) -> list[dict[str, Any]]:
        """Return tool calls from the last query as dicts for SSE broadcast.

        Each dict has ``name``, ``arguments``, and ``display_name``.
        """
        result = self.last_result
        if result is None:
            return []

        tool_calls = getattr(result, "tool_calls", [])
        return [
            {
                "name": tc.name,
                "display_name": getattr(tc, "display_name", tc.name),
                "arguments": tc.arguments,
                "result": getattr(tc, "result", None),
            }
            for tc in tool_calls
        ]

    def get_upstream_model(self) -> str:
        """Return the actual model used by the upstream agent (e.g. gpt-5.3-codex)."""
        result = self.last_result
        if result is None:
            return self._backend_name
        return getattr(result, "model", "") or self._backend_name

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _chat_full(self, prompt: str) -> Any:
        """Call the backend's ``chat_full`` (or fall back to ``chat``)."""
        if hasattr(self._backend, "chat_full"):
            return await self._backend.chat_full(prompt, self._bot_config)
        # Fallback for backends that only implement chat()
        text = await self._backend.chat(prompt, self._bot_config)
        # Wrap in a minimal result-like object
        from types import SimpleNamespace

        return SimpleNamespace(text=text, tool_calls=[], model="", duration_ms=0)

    @staticmethod
    def _load_backend(name: str):
        """Load the agent backend by name."""
        from ..agent_backends import get_backend

        backend = get_backend(name)
        if backend is None:
            raise ValueError(f"Agent backend '{name}' not found")
        return backend
