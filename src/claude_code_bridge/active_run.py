"""Live Claude SDK run control for mid-turn user steering."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClaudeActiveRun:
    """Own the control plane for one live ``ClaudeSDKClient`` run.

    Steering is an SDK interrupt followed by a new query on the same client.
    The CLI emits one ``error_during_execution`` ResultMessage for the
    interrupted request before it starts the replacement query; the single
    bridge response consumer must drain that boundary rather than finalizing
    the llm-bawt turn on it.
    """

    client: Any
    request_id: str
    _steer_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _expected_interrupt_results: int = 0

    async def steer(self, message: str) -> None:
        """Interrupt current work and inject ``message`` into the same SDK run."""
        text = message.strip()
        if not text:
            raise ValueError("Steering message must not be empty")

        async with self._steer_lock:
            self._expected_interrupt_results += 1
            interrupted = False
            try:
                await self.client.interrupt()
                interrupted = True
                await self.client.query(text)
            except BaseException:
                # If the replacement query was not accepted, do not hide the
                # interrupted result as though a continuation were coming. Once
                # interrupt succeeded the old task is already dead, so tear down
                # the client rather than leaving _handle_send parked forever.
                self._expected_interrupt_results = max(
                    0, self._expected_interrupt_results - 1
                )
                if interrupted:
                    try:
                        await self.client.disconnect()
                    except BaseException:
                        pass
                raise

    def consume_replaced_result(self, result_message: Any) -> bool:
        """Consume one terminal boundary superseded by a steering message.

        Normally the SDK labels it ``error_during_execution``. If the original
        request finishes in the narrow race between our counter increment and
        the interrupt control request, its success result is still the replaced
        boundary; the already-accepted correction owns the next result.
        """
        if self._expected_interrupt_results <= 0:
            return False
        self._expected_interrupt_results -= 1
        return True

    async def disconnect(self) -> None:
        """Preserve SessionQueue's duck-typed abort-handle contract."""
        await self.client.disconnect()
