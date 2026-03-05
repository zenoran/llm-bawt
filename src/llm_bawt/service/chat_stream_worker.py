"""Helpers for chat stream worker thread operations."""

from __future__ import annotations

import asyncio
import threading
from typing import Any


def append_text_chunk(full_response_holder: list[str], chunk: Any) -> None:
    """Accumulate only text chunks into the response buffer."""
    if isinstance(chunk, str):
        full_response_holder[0] += chunk


def put_queue_item_threadsafe(loop: asyncio.AbstractEventLoop, chunk_queue: asyncio.Queue, item: Any) -> bool:
    """Best-effort queue delivery from worker thread to event loop.

    Returns False when the loop is already closed/disconnected.
    """
    try:
        loop.call_soon_threadsafe(chunk_queue.put_nowait, item)
        return True
    except RuntimeError:
        return False


def consume_stream_chunks(
    stream_iter,
    *,
    cancel_event: threading.Event,
    loop: asyncio.AbstractEventLoop,
    chunk_queue: asyncio.Queue,
    full_response_holder: list[str],
) -> bool:
    """Consume stream chunks without losing text when cancelled/disconnected.

    Returns True when cancellation was observed at least once.
    """
    cancelled = False
    for chunk in stream_iter:
        append_text_chunk(full_response_holder, chunk)
        put_queue_item_threadsafe(loop, chunk_queue, chunk)
        if cancel_event.is_set():
            cancelled = True
    return cancelled
