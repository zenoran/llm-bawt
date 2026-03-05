"""OpenClaw agent backend.

Primary transport is OpenClaw's official gateway API (OpenAI-compatible
``/v1/chat/completions``). SSH CLI execution is retained as an explicit
legacy fallback transport.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import logging
import os
import shlex
import time
from urllib import request as urlrequest
from urllib import error as urlerror
from dataclasses import dataclass, field
from typing import Any, Iterator

from .base import AgentBackend
from ..shared.output_sanitizer import strip_tool_protocol_leakage
from ..utils.config import Config

logger = logging.getLogger(__name__)


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
    """Derive a human-friendly display name for a tool call.

    For ``exec`` tools the first word of the command is used (e.g.
    ``find``, ``grep``, ``curl``).  For everything else the tool name
    itself is already descriptive enough.
    """
    if name == "exec":
        cmd = arguments.get("command", "")
        if cmd:
            first_word = cmd.split()[0] if cmd.split() else name
            # Strip leading path (e.g. /usr/bin/find → find)
            first_word = first_word.rsplit("/", 1)[-1]
            return first_word
    return name


def _clean_tool_call_leakage(text: str) -> str:
    """Strip raw tool-call protocol text that OpenClaw sometimes leaks.

    When the gateway returns an intermediate step, the content contains the
    model's internal reasoning plus raw JSON like::

        to=multi_tool_use.parallel 派奖中commentary...
        {"tool_uses":[{"recipient_name":"functions.exec",...}]}

    This strips the protocol noise, leaving only the readable portion
    (if any).
    """
    lower = text.lower()
    if (
        "tool_uses" not in lower
        and "to=multi_tool_use." not in lower
        and "to=functions." not in lower
    ):
        return text

    cleaned = strip_tool_protocol_leakage(text)

    if cleaned:
        logger.debug("Cleaned tool-call leakage from response (%d -> %d chars)", len(text), len(cleaned))
        return cleaned

    # If stripping left nothing readable, return a placeholder so the
    # response isn't empty (which would raise an error downstream).
    logger.warning("OpenClaw response was entirely tool-call protocol; returning placeholder")
    return "(OpenClaw is executing tools — response pending)"


def _extract_tool_calls(data: dict) -> list[OpenClawToolCall]:
    """Walk the OpenClaw session transcript embedded in the JSON response.

    The ``--json`` output doesn't include per-call details directly, but the
    session transcript (``result.transcript``) entries with
    ``type: "toolCall"`` in assistant ``content`` blocks do.  When those are
    absent, fall back to tool names listed in the systemPromptReport (tools
    that *exist*, not tools that were *called*) — which is less precise.

    In practice we fetch transcript data separately via
    ``sessions_history`` when available, but the ``--json`` response from
    ``openclaw agent`` already embeds a minimal transcript when the run
    includes tool use.
    """
    calls: list[OpenClawToolCall] = []

    # Primary source: inline transcript entries
    transcript = data.get("result", {}).get("transcript", [])
    for entry in transcript:
        msg = entry.get("message", {})
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "toolCall":
                calls.append(OpenClawToolCall(
                    name=block.get("name", "unknown"),
                    arguments=block.get("arguments", {}),
                ))

    return calls


class OpenClawBackend(AgentBackend):
    """Agent backend that delegates to an OpenClaw instance over SSH.

    Configuration keys (``agent_backend_config`` in bot DB profile):
        transport: ``gateway_api`` (default) or ``ssh``

        Gateway API mode (recommended):
            gateway_url: OpenClaw gateway base URL (default: from ``host``)
            token: OpenClaw gateway bearer token
            model: model id (default: "gpt-5.3-codex")
            timeout_seconds: HTTP timeout (default: 300)

        Legacy SSH mode:
            host: SSH host IP or hostname (required)
            user: SSH username (required)
            agent_id: OpenClaw agent id (default: "main")
            timeout_seconds: Max wait for agent response (default: 300)
            session_id: Optional persistent session id
            thinking: Optional thinking level (off|minimal|low|medium|high)
    """

    name = "openclaw"

    def __init__(self) -> None:
        self._config = Config()
        self._last_stream_result: OpenClawResult | None = None

    def _bool_from_any(self, value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        raw = str(value).strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        return default

    def _resolve_ssh_fallback_enabled(self, config: dict) -> bool:
        if "use_ssh_fallback" in config:
            return self._bool_from_any(config.get("use_ssh_fallback"), default=False)
        if "OPENCLAW_USE_SSH_FALLBACK" in os.environ:
            return self._bool_from_any(os.getenv("OPENCLAW_USE_SSH_FALLBACK"), default=False)
        return bool(self._config.OPENCLAW_USE_SSH_FALLBACK)

    def _resolve_transport(self, config: dict) -> str:
        transport = str(config.get("transport") or "gateway_api").strip().lower()
        if transport in ("api", "gateway", "gateway_api", "openai_api"):
            return "gateway_api"
        if transport in ("ssh", "legacy_ssh"):
            return "ssh"
        if self._resolve_ssh_fallback_enabled(config):
            return "ssh"
        return "gateway_api"

    def _resolve_gateway_base_url(self, config: dict) -> str:
        explicit = str(config.get("gateway_url") or "").strip()
        if explicit:
            return explicit.rstrip("/")
        host = str(config.get("host") or "").strip()
        if host:
            port = int(config.get("gateway_port", 18789))
            return f"http://{host}:{port}"
        env_url = os.getenv("OPENCLAW_GATEWAY_URL", "").strip() or os.getenv("LLM_BAWT_OPENCLAW_GATEWAY_URL", "").strip()
        if env_url:
            return env_url.rstrip("/")
        return str(self._config.OPENCLAW_GATEWAY_URL or "").strip().rstrip("/")

    def _resolve_gateway_token(self, config: dict) -> str:
        token_env_name = str(config.get("token_env") or "").strip()
        env_token = os.getenv(token_env_name, "").strip() if token_env_name else ""
        return (
            str(config.get("token") or "").strip()
            or str(config.get("gateway_token") or "").strip()
            or env_token
            or str(self._config.OPENCLAW_GATEWAY_TOKEN or "").strip()
            or os.getenv("OPENCLAW_GATEWAY_TOKEN", "").strip()
            or os.getenv("LLM_BAWT_OPENCLAW_GATEWAY_TOKEN", "").strip()
        )

    def _resolve_gateway_model(self, config: dict) -> str:
        explicit_model = str(config.get("model") or config.get("model_id") or "").strip()
        if explicit_model:
            return explicit_model
        agent_id = self._resolve_agent_id(config)
        return f"openclaw:{agent_id}"

    def _resolve_agent_id(self, config: dict) -> str:
        explicit = str(config.get("agent_id") or "").strip()
        if explicit:
            return explicit
        env_value = os.getenv("OPENCLAW_AGENT_ID", "").strip() or os.getenv("LLM_BAWT_OPENCLAW_AGENT_ID", "").strip()
        if env_value:
            return env_value
        return str(self._config.OPENCLAW_AGENT_ID or "main").strip() or "main"

    def _resolve_stream_enabled(self, config: dict) -> bool:
        if "stream_enabled" in config:
            return self._bool_from_any(config.get("stream_enabled"), default=True)
        if "OPENCLAW_STREAM_ENABLED" in os.environ:
            return self._bool_from_any(os.getenv("OPENCLAW_STREAM_ENABLED"), default=True)
        return bool(self._config.OPENCLAW_STREAM_ENABLED)

    def _resolve_gateway_headers(self, config: dict) -> dict[str, str]:
        headers: dict[str, str] = {}
        session_key = str(config.get("session_key") or "").strip()
        if session_key:
            headers["x-openclaw-session-key"] = session_key

        agent_id = self._resolve_agent_id(config)
        if agent_id:
            headers["x-openclaw-agent-id"] = agent_id

        message_channel = str(config.get("message_channel") or "").strip()
        if message_channel:
            headers["x-openclaw-message-channel"] = message_channel

        account_id = str(config.get("account_id") or "").strip()
        if account_id:
            headers["x-openclaw-account-id"] = account_id

        return headers

    def _request_gateway_responses(
        self,
        prompt: str,
        config: dict,
        *,
        stream: bool,
    ) -> dict | Any:
        base_url = self._resolve_gateway_base_url(config)
        token = self._resolve_gateway_token(config)
        timeout = int(config.get("timeout_seconds", 300))
        model = self._resolve_gateway_model(config)
        agent_id = self._resolve_agent_id(config)
        extra_headers = self._resolve_gateway_headers(config)

        if not base_url:
            raise ValueError(
                "OpenClaw gateway API transport requires 'gateway_url' "
                "(or 'host' so gateway_url can be derived)."
            )
        if not token:
            raise ValueError(
                "OpenClaw gateway API transport requires 'token' "
                "(or env OPENCLAW_GATEWAY_TOKEN / LLM_BAWT_OPENCLAW_GATEWAY_TOKEN)."
            )

        url = f"{base_url}/v1/responses"
        payload: dict[str, Any] = {
            "model": model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "stream": stream,
        }

        if agent_id:
            payload["metadata"] = {"agent_id": agent_id}

        user = str(config.get("user") or "").strip()
        if user:
            payload["user"] = user

        data = json.dumps(payload).encode("utf-8")
        req_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            **extra_headers,
        }
        req = urlrequest.Request(
            url,
            data=data,
            method="POST",
            headers=req_headers,
        )

        logger.info("OpenClaw request start: agent_id=%s stream=%s timeout=%ds", agent_id, stream, timeout)

        try:
            resp = urlrequest.urlopen(req, timeout=timeout)
            if stream:
                # For SSE streaming, extend the per-read socket timeout.
                # OpenClaw may go silent for minutes during tool execution;
                # the initial connection timeout (above) is for the HTTP
                # round-trip, but once connected we need patience.
                stream_read_timeout = max(timeout * 3, 300)
                try:
                    resp.fp.raw._sock.settimeout(stream_read_timeout)
                    logger.debug(
                        "SSE stream connected, per-read timeout extended to %ds",
                        stream_read_timeout,
                    )
                except (AttributeError, OSError) as sock_err:
                    logger.debug("Could not extend SSE socket timeout: %s", sock_err)
            return resp
        except urlerror.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(f"OpenClaw API HTTP {e.code}: {body[:400]}")
        except urlerror.URLError as e:
            raise RuntimeError(f"OpenClaw API unavailable: {e}")

    async def _run_gateway_response_json(
        self,
        prompt: str,
        config: dict,
    ) -> dict:
        def _do_request() -> dict:
            with self._request_gateway_responses(prompt, config, stream=False) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"OpenClaw API returned invalid JSON: {e}")

        return await asyncio.to_thread(_do_request)

    def _iter_sse_events(self, byte_chunks: Iterator[bytes]) -> Iterator[tuple[str, str]]:
        """Parse SSE events from byte chunks (supports partial lines)."""
        decoder = codecs.getincrementaldecoder("utf-8")()
        buffer = ""
        event_name = "message"
        data_parts: list[str] = []

        def _flush_event() -> tuple[str, str] | None:
            nonlocal event_name, data_parts
            if not data_parts:
                event_name = "message"
                return None
            payload = "\n".join(data_parts)
            out = (event_name, payload)
            event_name = "message"
            data_parts = []
            return out

        for chunk in byte_chunks:
            if not chunk:
                continue
            buffer += decoder.decode(chunk)
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.rstrip("\r")
                if not line:
                    flushed = _flush_event()
                    if flushed is not None:
                        yield flushed
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line[6:].strip() or "message"
                    continue
                if line.startswith("data:"):
                    data_parts.append(line[5:].lstrip())

        buffer += decoder.decode(b"", final=True)
        if buffer.strip():
            # Treat trailing fragment as final data line.
            if buffer.startswith("data:"):
                data_parts.append(buffer[5:].lstrip())
        flushed = _flush_event()
        if flushed is not None:
            yield flushed

    def _extract_response_text(self, data: dict[str, Any]) -> str:
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        texts: list[str] = []
        for item in data.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []) or []:
                if not isinstance(content, dict):
                    continue
                ctype = str(content.get("type") or "").lower()
                if ctype in {"output_text", "text"}:
                    text_value = content.get("text") or content.get("value")
                    if isinstance(text_value, str) and text_value:
                        texts.append(text_value)
        return "".join(texts)

    def _extract_tool_call_from_item(self, item: dict[str, Any]) -> OpenClawToolCall | None:
        item_type = str(item.get("type") or item.get("item_type") or "").lower()
        if item_type not in {"function_call", "tool_call"}:
            return None

        name = str(item.get("name") or item.get("tool_name") or "unknown")
        raw_args = item.get("arguments") or item.get("input") or {}
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                args = parsed if isinstance(parsed, dict) else {"raw": raw_args}
            except Exception:
                args = {"raw": raw_args}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {"raw": raw_args}

        return OpenClawToolCall(name=name, arguments=args)

    def _extract_tool_result_from_item(self, item: dict[str, Any]) -> tuple[str, Any] | None:
        item_type = str(item.get("type") or item.get("item_type") or "").lower()
        if item_type not in {"function_call_output", "tool_result", "tool_call_result"}:
            return None
        call_id = str(item.get("call_id") or item.get("tool_call_id") or item.get("id") or "").strip()
        result = item.get("output")
        if result is None:
            result = item.get("result")
        return call_id, result

    def stream_raw(self, prompt: str, config: dict) -> Iterator[str | dict[str, Any]]:
        """Stream OpenClaw response deltas and tool events via SSE."""
        if not self._resolve_stream_enabled(config):
            # Fall back to single full text chunk in non-stream mode.
            data = asyncio.run(self._run_gateway_response_json(prompt, config))
            text = self._clean_and_validate_non_stream_text(data)
            self._last_stream_result = OpenClawResult(
                text=text,
                model=str(data.get("model") or self._resolve_gateway_model(config)),
                provider="openclaw-gateway",
                raw=data,
            )
            if text:
                yield text
            return

        request_started = time.time()
        first_delta_at: float | None = None
        tool_calls: list[OpenClawToolCall] = []
        tool_calls_by_id: dict[str, OpenClawToolCall] = {}
        text_parts: list[str] = []
        response_id = ""
        upstream_model = self._resolve_gateway_model(config)
        termination_reason = "completed"

        try:
            with self._request_gateway_responses(prompt, config, stream=True) as resp:
                def _iter_chunks() -> Iterator[bytes]:
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        yield chunk

                for event_name, raw_data in self._iter_sse_events(_iter_chunks()):
                    if raw_data == "[DONE]":
                        termination_reason = "completed"
                        break
                    try:
                        payload = json.loads(raw_data)
                    except json.JSONDecodeError:
                        logger.debug("OpenClaw SSE non-JSON payload for event=%s", event_name)
                        continue

                    event_type = str(payload.get("type") or event_name or "message")
                    if event_type == "response.created":
                        response_id = str(payload.get("response", {}).get("id") or payload.get("id") or "")
                        upstream_model = str(payload.get("response", {}).get("model") or upstream_model)
                        logger.info(
                            "OpenClaw stream created: response_id=%s agent_id=%s",
                            response_id or "?",
                            self._resolve_agent_id(config),
                        )
                        continue

                    if event_type == "response.output_text.delta":
                        delta = str(payload.get("delta") or "")
                        if delta:
                            if first_delta_at is None:
                                first_delta_at = time.time()
                                logger.info(
                                    "OpenClaw first-token latency: %.1fms response_id=%s",
                                    (first_delta_at - request_started) * 1000,
                                    response_id or "?",
                                )
                            text_parts.append(delta)
                            yield delta
                        continue

                    if event_type == "response.output_item.added":
                        item = payload.get("item", {})
                        if isinstance(item, dict):
                            tool_call = self._extract_tool_call_from_item(item)
                            if tool_call:
                                call_id = str(item.get("call_id") or item.get("id") or "").strip()
                                if call_id:
                                    tool_calls_by_id[call_id] = tool_call
                                tool_calls.append(tool_call)
                                logger.info("OpenClaw tool event: %s", tool_call.display_name)
                                yield {
                                    "event": "tool_call",
                                    "name": tool_call.display_name,
                                    "arguments": tool_call.arguments,
                                }
                                continue

                            tool_result = self._extract_tool_result_from_item(item)
                            if tool_result:
                                call_id, result_payload = tool_result
                                if call_id and call_id in tool_calls_by_id:
                                    tool_calls_by_id[call_id].result = result_payload
                                    tool_name = tool_calls_by_id[call_id].display_name
                                else:
                                    tool_name = str(item.get("name") or "unknown")
                                yield {
                                    "event": "tool_result",
                                    "name": tool_name,
                                    "result": result_payload,
                                }
                        continue

                    if event_type == "response.completed":
                        response = payload.get("response", {})
                        if isinstance(response, dict):
                            upstream_model = str(response.get("model") or upstream_model)
                        termination_reason = "completed"
                        break
        except Exception as exc:
            termination_reason = "error"
            # If we got a response_id but no text (tool work timeout), try
            # a synchronous non-stream fetch as last resort — OpenClaw may
            # have finished by the time we retry.
            if not text_parts and response_id:
                logger.warning(
                    "SSE stream failed with no text (response_id=%s); "
                    "attempting non-stream recovery...",
                    response_id,
                )
                try:
                    data = asyncio.run(
                        self._run_gateway_response_json(prompt, config)
                    )
                    recovered_text = self._extract_response_text(data)
                    if recovered_text:
                        recovered_text = _clean_tool_call_leakage(recovered_text)
                        text_parts.append(recovered_text)
                        termination_reason = "recovered"
                        logger.info(
                            "Recovery succeeded: %d chars from non-stream fallback",
                            len(recovered_text),
                        )
                        # Yield the recovered text so it flows to the consumer
                        yield recovered_text
                except Exception as recovery_err:
                    logger.warning(
                        "Non-stream recovery also failed: %s", recovery_err
                    )
            if termination_reason != "recovered":
                raise exc
        finally:
            end_time = time.time()
            logger.info(
                "OpenClaw stream termination: reason=%s total_latency_ms=%.1f response_id=%s",
                termination_reason,
                (end_time - request_started) * 1000,
                response_id or "?",
            )
            full_text = "".join(text_parts)
            self._last_stream_result = OpenClawResult(
                text=full_text,
                model=upstream_model,
                provider="openclaw-gateway",
                duration_ms=int((end_time - request_started) * 1000),
                tool_calls=tool_calls,
                raw={"response_id": response_id, "termination_reason": termination_reason},
            )

    def _clean_and_validate_non_stream_text(self, data: dict[str, Any]) -> str:
        response_text = self._extract_response_text(data)
        if not response_text:
            raise RuntimeError("OpenClaw API returned empty message content")
        return _clean_tool_call_leakage(response_text)

    def get_last_stream_result(self) -> OpenClawResult | None:
        return self._last_stream_result

    async def _fetch_session_tool_calls_gateway(
        self,
        config: dict,
        request_started_ms: int,
        limit: int = 8,
    ) -> list[OpenClawToolCall]:
        """Fetch recent tool calls using OpenClaw's HTTP tools endpoint.

        This relies on ``sessions_history`` via ``POST /tools/invoke`` which
        returns structured transcript entries including ``toolCall`` blocks.
        """
        base_url = self._resolve_gateway_base_url(config)
        token = self._resolve_gateway_token(config)
        session_key = str(config.get("session_key") or "").strip()
        if not base_url or not token or not session_key:
            return []

        timeout = int(config.get("timeout_seconds", 300))
        url = f"{base_url}/tools/invoke"
        payload = {
            "tool": "sessions_history",
            "args": {
                "sessionKey": session_key,
                "limit": max(3, int(limit)),
                "includeTools": True,
            },
        }
        req = urlrequest.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )

        def _do_request() -> dict:
            with urlrequest.urlopen(req, timeout=min(timeout, 20)) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))

        try:
            data = await asyncio.to_thread(_do_request)
        except Exception:
            return []

        details = data.get("result", {}).get("details", {})
        messages = details.get("messages", []) if isinstance(details, dict) else []
        calls: list[OpenClawToolCall] = []
        call_by_id: dict[str, OpenClawToolCall] = {}

        def _extract_tool_result_content(msg: dict[str, Any]) -> Any:
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts: list[str] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        text_parts.append(block.get("text", ""))
                if text_parts:
                    merged = "\n".join(part for part in text_parts if part)
                    return merged if len(merged) <= 4000 else (merged[:4000] + "\n...(truncated)...")
            # Fallback: preserve structure but cap size for logs/UI.
            raw = json.dumps(msg, ensure_ascii=False, default=str)
            return raw if len(raw) <= 2000 else (raw[:2000] + "...(truncated)...")

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            ts = int(msg.get("timestamp") or 0)
            if ts and ts < (request_started_ms - 5000):
                continue
            role = str(msg.get("role") or "").lower()
            if role == "assistant":
                for block in msg.get("content", []):
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "toolCall":
                        call = OpenClawToolCall(
                            name=block.get("name", "unknown"),
                            arguments=block.get("arguments", {}),
                        )
                        calls.append(call)
                        block_id = str(block.get("id") or "").strip()
                        if block_id:
                            call_by_id[block_id] = call
                    elif btype in ("toolResult", "tool_result"):
                        target_id = str(
                            block.get("toolCallId")
                            or block.get("tool_call_id")
                            or block.get("id")
                            or ""
                        ).strip()
                        result_payload = block.get("result")
                        if result_payload is None:
                            result_payload = block.get("content")
                        if target_id and target_id in call_by_id:
                            call_by_id[target_id].result = result_payload
                        elif calls:
                            calls[-1].result = result_payload
            elif role in ("toolresult", "tool_result", "tool"):
                target_id = str(msg.get("toolCallId") or msg.get("tool_call_id") or "").strip()
                result_payload = _extract_tool_result_content(msg)
                if target_id and target_id in call_by_id:
                    call_by_id[target_id].result = result_payload
                elif calls:
                    calls[-1].result = result_payload
        return calls

    # ----- low-level SSH runner -------------------------------------------

    async def _run_ssh(
        self,
        prompt: str,
        config: dict,
    ) -> dict:
        """Execute the ``openclaw agent`` CLI over SSH and return raw JSON."""
        host = config.get("host")
        user = config.get("user")
        if not host or not user:
            raise ValueError(
                "OpenClaw agent_backend_config requires 'host' and 'user' "
                "(set via bot DB profile, e.g. host: 10.0.0.97, user: vex)"
            )
        agent_id = config.get("agent_id", "main")
        timeout = int(config.get("timeout_seconds", 300))
        session_id = config.get("session_id")
        thinking = config.get("thinking")

        remote_parts = [
            "openclaw", "agent",
            "--agent", shlex.quote(agent_id),
            "--message", shlex.quote(prompt),
            "--json",
            "--timeout", str(timeout),
        ]
        if session_id:
            remote_parts.extend(["--session-id", shlex.quote(str(session_id))])
        if thinking:
            remote_parts.extend(["--thinking", shlex.quote(str(thinking))])

        remote_cmd = " ".join(remote_parts)
        cmd = ["ssh", f"{user}@{host}", remote_cmd]

        logger.debug("OpenClaw SSH: %s@%s → openclaw agent --agent %s", user, host, agent_id)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout + 30,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"OpenClaw agent timed out after {timeout + 30}s")

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"OpenClaw agent failed (exit {proc.returncode}): {err}")

        raw = stdout.decode(errors="replace").strip()
        if not raw:
            raise RuntimeError("OpenClaw agent returned empty response")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"OpenClaw returned invalid JSON: {e}")

        status = data.get("status")
        if status != "ok":
            summary = data.get("summary", "unknown error")
            raise RuntimeError(f"OpenClaw returned status '{status}': {summary}")

        return data

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
        request_started_ms = int(time.time() * 1000)
        transport = self._resolve_transport(config)
        if transport == "ssh":
            data = await self._run_ssh(prompt, config)
        else:
            data = await self._run_gateway_response_json(prompt, config)

        # OpenClaw gateway responses API
        if "output" in data or "output_text" in data:
            response_text = self._clean_and_validate_non_stream_text(data)
            usage = data.get("usage", {}) or {}
            upstream_model = str(data.get("model") or self._resolve_gateway_model(config))

            tool_calls: list[OpenClawToolCall] = []
            for item in data.get("output", []) or []:
                if not isinstance(item, dict):
                    continue
                tool_call = self._extract_tool_call_from_item(item)
                if tool_call:
                    tool_calls.append(tool_call)

            result = OpenClawResult(
                text=response_text,
                model=upstream_model,
                provider="openclaw-gateway",
                duration_ms=0,
                session_id="",
                tool_calls=tool_calls,
                usage=usage,
                raw=data,
            )

            # OpenClaw's chat-completions shim currently doesn't always expose
            # tool_calls directly. Pull recent calls from sessions_history when
            # a stable session key is configured.
            if not result.tool_calls and config.get("session_key"):
                result.tool_calls = await self._fetch_session_tool_calls_gateway(
                    config=config,
                    request_started_ms=request_started_ms,
                    limit=int(config.get("tool_history_limit", 8)),
                )

            logger.debug(
                "OpenClaw API response: %d chars, model=%s, tools=%d",
                len(response_text),
                result.model or "?",
                len(result.tool_calls),
            )
            return result

        # Legacy OpenClaw CLI JSON format (SSH transport)
        payloads = data.get("result", {}).get("payloads", [])
        texts = [p["text"] for p in payloads if p.get("text")]
        if not texts:
            raise RuntimeError("OpenClaw returned no text payloads")

        response_text = "\n\n".join(texts)

        meta = data.get("result", {}).get("meta", {})
        agent_meta = meta.get("agentMeta", {})
        tool_calls = _extract_tool_calls(data)

        result = OpenClawResult(
            text=response_text,
            model=agent_meta.get("model", ""),
            provider=agent_meta.get("provider", ""),
            duration_ms=meta.get("durationMs", 0),
            session_id=agent_meta.get("sessionId", ""),
            tool_calls=tool_calls,
            usage=agent_meta.get("usage", {}),
            raw=data,
        )

        logger.debug(
            "OpenClaw response: %d chars, model=%s, duration=%dms, tools=%d",
            len(response_text),
            result.model or "?",
            result.duration_ms,
            len(tool_calls),
        )

        return result

    async def fetch_session_tool_calls(
        self,
        config: dict,
        session_id: str,
        limit: int = 20,
    ) -> list[OpenClawToolCall]:
        """Fetch recent tool calls from an OpenClaw session transcript.

        Uses ``openclaw sessions_history --json`` to read the JSONL
        transcript and extract ``toolCall`` blocks.
        """
        _ = session_id  # maintained for compatibility
        return await self._fetch_session_tool_calls_gateway(
            config=config,
            request_started_ms=0,
            limit=limit,
        )

    async def health_check(self, config: dict) -> bool:
        if self._resolve_transport(config) == "gateway_api":
            base_url = self._resolve_gateway_base_url(config)
            token = self._resolve_gateway_token(config)
            if not base_url or not token:
                return False
            req = urlrequest.Request(
                f"{base_url}/health",
                method="GET",
                headers={"Authorization": f"Bearer {token}"},
            )
            try:
                with urlrequest.urlopen(req, timeout=10) as resp:
                    return int(getattr(resp, "status", 0)) < 500
            except Exception:
                return False

        host = config.get("host")
        user = config.get("user")
        if not host or not user:
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                "ssh", f"{user}@{host}",
                "openclaw", "health", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            data = json.loads(stdout.decode(errors="replace"))
            return data.get("ok", False)
        except Exception:
            return False
