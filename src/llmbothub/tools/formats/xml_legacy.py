from __future__ import annotations

import re
from typing import Any

from .base import ToolCallRequest, ToolFormatHandler


class LegacyXMLFormatHandler(ToolFormatHandler):
    """Legacy <tool_call> XML-like tool calling handler."""

    def get_system_prompt(self, tools: list) -> str:
        from llmbothub.tools.definitions import get_tools_prompt

        return get_tools_prompt(tools=tools)

    def get_stop_sequences(self) -> list[str]:
        return ["</tool_call>"]

    def parse_response(self, response: str) -> tuple[list[ToolCallRequest], str]:
        from llmbothub.tools.parser import parse_tool_calls

        tool_calls, remaining = parse_tool_calls(response)
        calls = [
            ToolCallRequest(name=call.name, arguments=call.arguments, raw_text=call.raw_text)
            for call in tool_calls
        ]
        return calls, remaining

    def format_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: str | None = None,
        error: str | None = None,
    ) -> str:
        from llmbothub.tools.parser import format_tool_result
        if isinstance(result, str) and "<tool_result" in result:
            return result
        return format_tool_result(tool_name, result, error=error)

    def sanitize_response(self, response: str) -> str:
        if not response:
            return ""
        cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<function_call>.*?</function_call>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()
