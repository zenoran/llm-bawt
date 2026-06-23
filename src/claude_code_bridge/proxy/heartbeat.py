"""SSE keepalive wrapper.

Wraps any byte-emitting async generator and injects Anthropic ``ping`` frames
whenever the upstream goes quiet for longer than ``interval`` seconds. This is
the universal anti-stall: it covers BOTH proxy paths —

  * the OpenAI translate path (``stream.responses_to_anthropic_sse``), which can
    have a gap between ``response.created`` and the first reasoning token, and
  * the Z.AI Anthropic passthrough, where the upstream itself may pause (z.ai's
    Anthropic surface does not always emit the periodic pings api.anthropic.com
    does).

Native api.anthropic.com sprinkles ``ping`` events through long turns precisely
so intermediaries and clients don't treat an idle SSE connection as dead. The
Claude Agent SDK / bridge watchdog (``asyncio.wait_for`` per event) is fed by
these frames, and any HTTP intermediary's idle timeout is reset too.

``ping`` is a no-op control frame in the Anthropic SSE contract — the SDK reads
and discards it — so injecting extra ones is always safe.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)

_PING_FRAME = b'event: ping\ndata: {"type":"ping"}\n\n'

# Default cadence. Comfortably under common 30–60s idle-timeout thresholds and
# the bridge's per-event watchdog, while staying quiet enough to be invisible.
DEFAULT_INTERVAL = 10.0


async def with_heartbeat(
    source: AsyncIterator[bytes],
    interval: float = DEFAULT_INTERVAL,
) -> AsyncIterator[bytes]:
    """Yield from ``source``, emitting a ping frame every ``interval`` seconds
    of silence. Real frames reset the timer; the stream ends exactly when the
    source is exhausted (pings never extend it)."""

    if interval <= 0:
        async for chunk in source:
            yield chunk
        return

    source_iter = source.__aiter__()
    pending: asyncio.Task | None = asyncio.ensure_future(source_iter.__anext__())
    # Track whether the last yielded chunk ended at an SSE event boundary
    # (``\n\n``).  Injecting a ping frame mid-event would corrupt the SSE
    # parse on the receiving end — e.g. z.ai sometimes delivers a partial
    # ``data:`` line in one TCP segment and completes it in the next.
    at_boundary = True
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(asyncio.shield(pending), timeout=interval)
            except asyncio.TimeoutError:
                # Upstream quiet — emit a keepalive and keep awaiting the SAME
                # pending pull (shield kept it alive across the timeout).
                # Only inject when we're at an SSE event boundary to avoid
                # splitting a partial event.
                if at_boundary:
                    yield _PING_FRAME
                continue
            except StopAsyncIteration:
                pending = None
                break

            # Got a real frame; schedule the next pull and forward this one.
            pending = asyncio.ensure_future(source_iter.__anext__())
            yield chunk
            at_boundary = chunk.endswith(b"\n\n")
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await pending
            except (asyncio.CancelledError, StopAsyncIteration, Exception):  # noqa: BLE001
                pass
        # Best-effort close of the underlying async generator so the upstream
        # HTTP connection / subprocess is released promptly on early exit.
        aclose = getattr(source_iter, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001
                pass
