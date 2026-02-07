from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCallRequest:
    """Unified representation of a tool call."""
    name: str
    arguments: dict
    raw_text: str | None = None
    tool_call_id: str | None = None


class ToolFormatHandler(ABC):
    """Abstract handler for different tool calling formats."""

    @abstractmethod
    def get_system_prompt(self, tools: list) -> str:
        """Generate system prompt instructions for this format."""
        raise NotImplementedError

    @abstractmethod
    def get_stop_sequences(self) -> list[str]:
        """Return stop sequences for this format."""
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, response: str) -> tuple[list[ToolCallRequest], str]:
        """Parse tool calls from response. Returns (calls, remaining_text)."""
        raise NotImplementedError

    @abstractmethod
    def format_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: str | None = None,
        error: str | None = None,
    ) -> Any:
        """Format tool result for injection back to LLM."""
        raise NotImplementedError

    @abstractmethod
    def sanitize_response(self, response: str) -> str:
        """Remove any unparsed tool markers from final response."""
        raise NotImplementedError
