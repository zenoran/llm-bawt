from __future__ import annotations

from enum import Enum

from .base import ToolCallRequest, ToolFormatHandler


class ToolFormat(Enum):
    REACT = "react"
    NATIVE_OPENAI = "native"
    XML = "xml"


def get_format_handler(tool_format: ToolFormat | str) -> ToolFormatHandler:
    """Return a handler instance for the requested tool format."""
    if isinstance(tool_format, str):
        normalized = tool_format.strip().lower()
        tool_format = ToolFormat(normalized)

    if tool_format == ToolFormat.REACT:
        from .react import ReActFormatHandler

        return ReActFormatHandler()
    if tool_format == ToolFormat.NATIVE_OPENAI:
        from .native_openai import NativeOpenAIFormatHandler

        return NativeOpenAIFormatHandler()
    if tool_format == ToolFormat.XML:
        from .xml_legacy import LegacyXMLFormatHandler

        return LegacyXMLFormatHandler()

    raise ValueError(f"Unsupported tool format: {tool_format}")


__all__ = [
    "ToolCallRequest",
    "ToolFormatHandler",
    "ToolFormat",
    "get_format_handler",
]
