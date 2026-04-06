"""OpenClaw agent backend.

All traffic flows through the bridge's persistent WebSocket connection:
  Main App → Redis command → Bridge → WS chat.send → Gateway
  Gateway → WS events → Bridge → Redis run stream → Main App
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from .base import AgentBackend
from ..shared.output_sanitizer import strip_tool_protocol_leakage
from ..utils.config import Config

logger = logging.getLogger(__name__)

# Module-level subscriber singleton — set by service startup
_subscriber = None


def set_openclaw_subscriber(subscriber) -> None:
    """Set the Redis subscriber for all OpenClawBackend instances."""
    global _subscriber
    _subscriber = subscriber


def get_openclaw_subscriber():
    """Get the current Redis subscriber."""
    return _subscriber


# ---------------------------------------------------------------------------
# Structured result returned by chat_full()
# ---------------------------------------------------------------------------

@dataclass
class OpenClawToolCall:
    """One tool invocation extracted from an OpenClaw response."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    display_name: str = ""  # friendly name for UI (e.g. first word of exec command)
    result: Any | None = None

    def __post_init__(self):
        if not self.display_name:
            self.display_name = _friendly_tool_name(self.name, self.arguments)


@dataclass
class OpenClawResult:
    """Full structured result from an OpenClaw agent run."""

    text: str
    model: str = ""
    provider: str = ""
    duration_ms: int = 0
    session_id: str = ""
    tool_calls: list[OpenClawToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def _friendly_tool_name(name: str, arguments: dict[str, Any]) -> str:
    if name == "exec":
        cmd = arguments.get("command", "")
        if cmd:
            first_word = cmd.split()[0] if cmd.split() else name
            first_word = first_word.rsplit("/", 1)[-1]
            return first_word
    return name


class OpenClawBackend(AgentBackend):
    """Agent backend that sends messages through the bridge's WebSocket.

    Configuration keys (``agent_backend_config`` in bot DB profile):
        session_key: OpenClaw session key (default: "main")
        timeout_seconds: Max wait for response (default: 600)
    """

    name = "openclaw"

    def __init__(self) -> None:
        self._config = Config()
        # Per-request state is stored in threading.local to avoid races
        # between concurrent stream_raw() calls (e.g. different bots).
        self._thread_local = threading.local()
        self._active_request_id: str | None = None

    def _resolve_session_key(self, config: dict) -> str:
        explicit = str(config.get("session_key") or "").strip()
        if explicit:
            return explicit
        return os.getenv("OPENCLAW_SESSION_KEY", "main")

    def stream_raw(self, prompt: str, config: dict, attachments: list | None = None) -> Iterator[str | dict[str, Any]]:
        """Stream OpenClaw response deltas and tool events via the bridge.

        Sends a command to the bridge via Redis, then subscribes to the
        per-run response stream. Yields text deltas and tool event dicts.
        """
        subscriber = get_openclaw_subscriber()
        if not subscriber:
            raise RuntimeError(
                "OpenClaw bridge subscriber not initialized. "
                "Ensure REDIS_URL is set and the bridge is running."
            )

        session_key = self._resolve_session_key(config)
        timeout = int(config.get("timeout_seconds", 600))
        request_id = f"req_{uuid.uuid4().hex}"
        self._active_request_id = request_id

        request_started = time.time()
        first_delta_at: float | None = None
        tool_calls: list[OpenClawToolCall] = []
        text_parts: list[str] = []
        upstream_model: str = ""
        termination_reason = "completed"

        import queue as queue_mod

        result_queue: queue_mod.Queue[str | dict | None] = queue_mod.Queue()
        error_holder: list[Exception] = []

        try:
            # Schedule async work on the running event loop from this sync context.
            # We use a separate thread with its own Redis client to avoid
            # event loop cross-contamination.
            import threading
            import redis.asyncio as aioredis

            def _run_in_thread():
                async def _worker():
                    from openclaw_bridge.events import OpenClawEventKind
                    # Create a fresh async Redis client for this thread's loop
                    redis_url = subscriber._redis.connection_pool.connection_kwargs.get(
                        "url", None
                    )
                    if redis_url is None:
                        # Reconstruct URL from connection kwargs
                        ck = subscriber._redis.connection_pool.connection_kwargs
                        host = ck.get("host", "localhost")
                        port = ck.get("port", 6379)
                        db = ck.get("db", 0)
                        redis_url = f"redis://{host}:{port}/{db}"

                    from openclaw_bridge.subscriber import RedisSubscriber
                    local_sub = RedisSubscriber(redis_url)
                    await local_sub.connect()
                    try:
                        await local_sub.send_command(
                            session_key=session_key,
                            message=prompt,
                            request_id=request_id,
                            attachments=attachments or [],
                            system_prompt=config.get("system_prompt"),
                            model=config.get("model"),
                            backend=self.name,
                        )
                        logger.info(
                            "%s request via bridge: session=%s request_id=%s",
                            self.name,
                            session_key, request_id,
                        )

                        nonlocal first_delta_at
                        async for event in local_sub.subscribe_run(
                            request_id, timeout_s=timeout,
                        ):
                            if event.kind == OpenClawEventKind.ASSISTANT_DELTA:
                                delta = event.text or ""
                                if delta:
                                    if first_delta_at is None:
                                        first_delta_at = time.time()
                                        logger.info(
                                            "%s first-token latency: %.1fms request_id=%s",
                                        self.name,
                                            (first_delta_at - request_started) * 1000,
                                            request_id,
                                        )
                                    text_parts.append(delta)
                                    result_queue.put(delta)

                            elif event.kind == OpenClawEventKind.ASSISTANT_DONE:
                                # Capture actual upstream model if provided
                                if event.model:
                                    nonlocal upstream_model
                                    upstream_model = event.model
                                    result_queue.put({
                                        "event": "metadata",
                                        "upstream_model": upstream_model,
                                    })
                                # ASSISTANT_DONE carries the complete response
                                # text.  Yield any portion not already streamed
                                # as deltas — this is critical for tool-heavy
                                # turns where the final synthesis text only
                                # appears in the done event, not as individual
                                # ASSISTANT_DELTA events.
                                done_text = event.text or ""
                                if done_text:
                                    accumulated = "".join(text_parts)
                                    if done_text.startswith(accumulated):
                                        extra = done_text[len(accumulated):]
                                        if extra.strip():
                                            text_parts.append(extra)
                                            result_queue.put(extra)
                                            logger.info(
                                                "ASSISTANT_DONE: supplemented %d chars "
                                                "(accumulated=%d, done=%d) request_id=%s",
                                                len(extra), len(accumulated),
                                                len(done_text), request_id,
                                            )
                                    elif len(done_text) > len(accumulated):
                                        # Deltas don't prefix-match done text;
                                        # yield the full done text so the
                                        # response is at least complete for
                                        # persistence.  May duplicate some
                                        # content in the SSE stream, but the
                                        # DB record will be correct on reload.
                                        logger.warning(
                                            "ASSISTANT_DONE: prefix mismatch, "
                                            "yielding full done_text (done=%d, "
                                            "accumulated=%d) request_id=%s",
                                            len(done_text), len(accumulated),
                                            request_id,
                                        )
                                        text_parts.clear()
                                        text_parts.append(done_text)
                                        result_queue.put(
                                            done_text[len(accumulated):]
                                            if accumulated
                                            else done_text
                                        )

                            elif event.kind == OpenClawEventKind.TOOL_START:
                                tc = OpenClawToolCall(
                                    name=event.tool_name or "unknown",
                                    arguments=event.tool_arguments or {},
                                )
                                tool_calls.append(tc)
                                logger.info("%s tool event: %s", self.name, tc.display_name)
                                result_queue.put({
                                    "event": "tool_call",
                                    "name": tc.display_name,
                                    "arguments": tc.arguments,
                                })

                            elif event.kind == OpenClawEventKind.TOOL_END:
                                if tool_calls:
                                    # Gateway may not include actual result —
                                    # use meta/summary or a placeholder so the
                                    # SSE handler always sees a non-None result.
                                    result = event.tool_result or "(completed)"
                                    tool_calls[-1].result = result
                                    result_queue.put({
                                        "event": "tool_result",
                                        "name": tool_calls[-1].display_name,
                                        "result": result,
                                    })

                            elif event.kind == OpenClawEventKind.ERROR:
                                raise RuntimeError(f"{self.name} error: {event.text}")
                    finally:
                        await local_sub.close()

                try:
                    asyncio.run(_worker())
                except Exception as e:
                    error_holder.append(e)
                finally:
                    result_queue.put(None)  # sentinel

            t = threading.Thread(target=_run_in_thread, daemon=True)
            t.start()

            # Drain the queue, yielding items to the caller
            while True:
                try:
                    item = result_queue.get(timeout=2.0)
                except queue_mod.Empty:
                    if not t.is_alive():
                        break
                    continue
                if item is None:
                    break
                yield item

            t.join(timeout=5.0)

            if error_holder:
                termination_reason = "error"
                raise error_holder[0]

        except Exception as exc:
            termination_reason = "error"
            raise exc
        finally:
            end_time = time.time()
            logger.info(
                "%s stream termination: reason=%s total_latency_ms=%.1f request_id=%s",
                self.name,
                termination_reason,
                (end_time - request_started) * 1000,
                request_id,
            )
            full_text = "".join(text_parts)
            result = OpenClawResult(
                text=full_text,
                model=upstream_model or self.name,
                provider=self.name,
                duration_ms=int((end_time - request_started) * 1000),
                tool_calls=tool_calls,
                raw={"request_id": request_id, "termination_reason": termination_reason},
            )
            self._thread_local.last_stream_result = result

    def get_last_stream_result(self) -> OpenClawResult | None:
        return getattr(self._thread_local, "last_stream_result", None)

    # ----- public API -----------------------------------------------------

    async def chat(
        self,
        prompt: str,
        config: dict,
        stream: bool = False,
    ) -> str:
        """Return just the text response (backward compat)."""
        result = await self.chat_full(prompt, config)
        return result.text

    async def chat_full(
        self,
        prompt: str,
        config: dict,
    ) -> OpenClawResult:
        """Run the agent and return a structured result with tool call data."""
        # Collect all stream results
        text_parts: list[str] = []
        tool_calls: list[OpenClawToolCall] = []

        for item in self.stream_raw(prompt, config):
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if item.get("event") == "tool_call":
                    if item.get("result") is not None and tool_calls:
                        # TOOL_END: update last tool call with result
                        tool_calls[-1].result = item["result"]
                    else:
                        # TOOL_START: new tool call
                        tool_calls.append(OpenClawToolCall(
                            name=item.get("name", "unknown"),
                            arguments=item.get("arguments", {}),
                        ))

        result = self._last_stream_result
        if result:
            return result

        return OpenClawResult(
            text="".join(text_parts),
            tool_calls=tool_calls,
            provider="openclaw-bridge",
        )

    async def health_check(self, config: dict) -> bool:
        """Check if bridge subscriber is connected."""
        subscriber = get_openclaw_subscriber()
        return subscriber is not None and subscriber.connected
