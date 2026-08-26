"""State-aware liveness watchdog for Claude Agent SDK turns.

The SDK emits a tool-use message before executing a tool and its tool-result
message only after execution finishes. A fixed per-message timeout therefore
cannot distinguish a wedged CLI from a legitimate long-running tool. This
module tracks that SDK-visible lifecycle and gives active tools a deadline
based on their declared timeout while preserving the normal idle limit.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from claude_agent_sdk.types import (
    AssistantMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

HealthCallback = Callable[[dict[str, Any]], None]

logger = logging.getLogger("claude_code_bridge.bridge")

DEFAULT_HEALTH_INTERVAL_SECONDS = 30.0
DEFAULT_TOOL_GRACE_SECONDS = 30.0
MAX_TOOL_TIMEOUT_SECONDS = 600.0


class TurnWatchdogTimeout(TimeoutError):
    """A phase-aware timeout from :class:`TurnWatchdog`."""

    def __init__(self, message: str, *, phase: str) -> None:
        super().__init__(message)
        self.phase = phase

    @property
    def retry_fresh_session(self) -> bool:
        """Only an idle SDK can safely use the existing fresh-session retry."""
        return self.phase == "sdk_idle"


@dataclass(frozen=True)
class ActiveTool:
    tool_use_id: str
    name: str
    started_at: float
    timeout_seconds: float


class TurnWatchdog:
    """Wait for SDK messages without misclassifying active tools as hangs."""

    def __init__(
        self,
        idle_timeout: float,
        *,
        health_interval: float = DEFAULT_HEALTH_INTERVAL_SECONDS,
        tool_grace: float = DEFAULT_TOOL_GRACE_SECONDS,
        max_tool_timeout: float = MAX_TOOL_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        on_health: HealthCallback | None = None,
    ) -> None:
        if idle_timeout <= 0:
            raise ValueError("idle_timeout must be positive")
        if health_interval <= 0:
            raise ValueError("health_interval must be positive")
        if tool_grace < 0:
            raise ValueError("tool_grace cannot be negative")
        if max_tool_timeout <= 0:
            raise ValueError("max_tool_timeout must be positive")
        self._idle_timeout = float(idle_timeout)
        self._health_interval = float(health_interval)
        self._tool_grace = float(tool_grace)
        self._max_tool_timeout = float(max_tool_timeout)
        self._clock = clock
        self._on_health = on_health
        self._active_tools: dict[str, ActiveTool] = {}

    @property
    def active_tools(self) -> tuple[ActiveTool, ...]:
        return tuple(self._active_tools.values())

    def observe(self, message: Any) -> None:
        """Update tool state from one SDK message."""
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if not isinstance(block, ToolUseBlock):
                    continue
                self._active_tools[block.id] = ActiveTool(
                    tool_use_id=block.id,
                    name=block.name,
                    started_at=self._clock(),
                    timeout_seconds=self._tool_timeout(block.name, block.input),
                )
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    self._active_tools.pop(block.tool_use_id, None)

    async def receive_next(
        self,
        message_stream: AsyncIterator[Any],
        *,
        request_id: str,
        session_key: str,
    ) -> Any:
        """Return the next SDK message or raise when the current phase wedges.

        One persistent ``__anext__`` task is shielded across periodic health
        checks. This is important: repeatedly timing out ``__anext__`` directly
        would cancel the SDK stream read and could corrupt the iterator.
        """
        receive_task = asyncio.create_task(message_stream.__anext__())
        wait_started_at = self._clock()
        try:
            while True:
                now = self._clock()
                deadline, phase = self._deadline(wait_started_at)
                remaining = deadline - now
                if remaining <= 0:
                    raise TurnWatchdogTimeout(
                        self._timeout_message(phase),
                        phase=phase,
                    )

                done, _ = await asyncio.wait(
                    {receive_task},
                    timeout=min(self._health_interval, remaining),
                )
                if receive_task in done:
                    message = receive_task.result()
                    self.observe(message)
                    return message

                self._log_health(
                    request_id=request_id,
                    session_key=session_key,
                    phase=phase,
                    remaining=max(0.0, deadline - self._clock()),
                )
        finally:
            if not receive_task.done():
                receive_task.cancel()
                await asyncio.gather(receive_task, return_exceptions=True)

    def _deadline(self, wait_started_at: float) -> tuple[float, str]:
        if not self._active_tools:
            return wait_started_at + self._idle_timeout, "sdk_idle"
        deadlines = [
            tool.started_at + tool.timeout_seconds + self._tool_grace
            for tool in self._active_tools.values()
        ]
        return max(deadlines), "tool_running"

    def _timeout_message(self, phase: str) -> str:
        if phase == "sdk_idle":
            return f"No SDK messages for {self._idle_timeout}s — CLI may be hung"
        tools = ", ".join(
            f"{tool.name}({tool.tool_use_id})" for tool in self._active_tools.values()
        )
        return (
            "No SDK tool result before the active-tool deadline "
            f"(tools: {tools or 'unknown'})"
        )

    def _log_health(
        self,
        *,
        request_id: str,
        session_key: str,
        phase: str,
        remaining: float,
    ) -> None:
        if phase == "tool_running":
            details = ",".join(
                f"{tool.name}:{self._clock() - tool.started_at:.0f}s/"
                f"{tool.timeout_seconds:.0f}s"
                for tool in self._active_tools.values()
            )
        else:
            details = "none"
        logger.info(
            "Turn health: request_id=%s session=%s phase=%s "
            "active_tools=%s deadline_in=%.1fs",
            request_id,
            session_key,
            phase,
            details,
            remaining,
        )
        if self._on_health is not None:
            try:
                self._on_health(
                    {
                        "phase": phase,
                        "active_tools": [
                            {
                                "tool_use_id": tool.tool_use_id,
                                "name": tool.name,
                                "elapsed_seconds": round(
                                    self._clock() - tool.started_at, 1
                                ),
                                "timeout_seconds": tool.timeout_seconds,
                            }
                            for tool in self._active_tools.values()
                        ],
                        "deadline_in_seconds": round(remaining, 1),
                    }
                )
            except Exception:
                logger.warning(
                    "Failed to publish turn health: request_id=%s session=%s",
                    request_id,
                    session_key,
                    exc_info=True,
                )

    def _tool_timeout(self, tool_name: str, tool_input: dict[str, Any]) -> float:
        if tool_name != "Bash":
            return self._idle_timeout
        raw_timeout = tool_input.get("timeout")
        if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)):
            return self._idle_timeout
        timeout_seconds = float(raw_timeout) / 1000.0
        if timeout_seconds <= 0:
            return self._idle_timeout
        return min(timeout_seconds, self._max_tool_timeout)
