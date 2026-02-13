"""Home Assistant MCP client.

Provides a small synchronous wrapper around an MCP server that exposes
smart-home tools via JSON-RPC over HTTP.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from ...utils.config import Config

logger = logging.getLogger(__name__)


class HomeAssistantMCPClient:
    """Typed client for Home Assistant MCP tool calls."""

    def __init__(self, config: Config):
        self._base_url = (getattr(config, "HA_MCP_URL", "") or "").rstrip("/")
        self._timeout = float(max(1, int(getattr(config, "HA_MCP_TIMEOUT", 10) or 10)))
        self._auth_token = (getattr(config, "HA_MCP_AUTH_TOKEN", "") or "").strip()  # Reserved for future authenticated transports.

    @property
    def available(self) -> bool:
        return bool(self._base_url)

    def status(self) -> str:
        return self._call_text_tool("status", {})

    def query(self, pattern: str | None = None, domain: str | None = None) -> str:
        args: dict[str, Any] = {}
        if pattern:
            args["pattern"] = pattern
        if domain:
            args["domain"] = domain
        return self._call_text_tool("query", args)

    def get(self, entity: str) -> str:
        return self._call_text_tool("get", {"entity": entity})

    def set(self, entity: str, state: str, brightness: int | None = None) -> str:
        args: dict[str, Any] = {"entity": entity, "state": state}
        if brightness is not None:
            args["brightness"] = brightness
        return self._call_text_tool("set", args)

    def scene(self, name: str) -> str:
        return self._call_text_tool("scene", {"name": name})

    def _call_text_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = self._call_tool(name=name, arguments=arguments)
        text = self._extract_text(result)
        if text:
            return text
        return json.dumps(result, default=str)

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self._base_url:
            raise RuntimeError("HA MCP URL is not configured")
        sse_url = f"{self._base_url}/sse"
        try:
            return self._run_async(self._call_tool_async(sse_url, name, arguments))
        except Exception as e:
            raise ConnectionError(f"HA MCP request failed: {e}") from e

    async def _call_tool_async(self, sse_url: str, name: str, arguments: dict[str, Any]) -> Any:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        async with sse_client(sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(name, arguments)

    @staticmethod
    def _run_async(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: Any = None
        error: BaseException | None = None

        def _runner():
            nonlocal result, error
            try:
                result = asyncio.run(coro)
            except BaseException as exc:  # pragma: no cover - defensive
                error = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if error is not None:
            raise error
        return result

    @staticmethod
    def _extract_text(result: Any) -> str:
        if hasattr(result, "content"):
            parts: list[str] = []
            for item in getattr(result, "content", []) or []:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            return "\n".join(p for p in parts if p).strip()

        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text" and isinstance(item.get("text"), str):
                            parts.append(item["text"])
                        elif item.get("type") == "json":
                            parts.append(json.dumps(item.get("json"), default=str))
                return "\n".join(p for p in parts if p).strip()

            structured = result.get("structuredContent")
            if isinstance(structured, dict) and "result" in structured:
                embedded = structured["result"]
                if isinstance(embedded, str):
                    return embedded.strip()

        if isinstance(result, str):
            return result.strip()
        return ""
