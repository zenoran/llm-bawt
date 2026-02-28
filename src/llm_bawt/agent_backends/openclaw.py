"""OpenClaw agent backend.

Primary transport is OpenClaw's official gateway API (OpenAI-compatible
``/v1/chat/completions``). SSH CLI execution is retained as an explicit
legacy fallback transport.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import time
from urllib import request as urlrequest
from urllib import error as urlerror
from dataclasses import dataclass, field
from typing import Any

from .base import AgentBackend

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

    def _resolve_transport(self, config: dict) -> str:
        transport = str(config.get("transport") or "gateway_api").strip().lower()
        if transport in ("api", "gateway", "gateway_api", "openai_api"):
            return "gateway_api"
        if transport in ("ssh", "legacy_ssh"):
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
        return ""

    def _resolve_gateway_token(self, config: dict) -> str:
        token_env_name = str(config.get("token_env") or "").strip()
        env_token = os.getenv(token_env_name, "").strip() if token_env_name else ""
        return (
            str(config.get("token") or "").strip()
            or str(config.get("gateway_token") or "").strip()
            or env_token
            or os.getenv("OPENCLAW_GATEWAY_TOKEN", "").strip()
            or os.getenv("LLM_BAWT_OPENCLAW_GATEWAY_TOKEN", "").strip()
        )

    def _resolve_gateway_model(self, config: dict) -> str:
        explicit_model = str(config.get("model") or config.get("model_id") or "").strip()
        if explicit_model:
            return explicit_model
        agent_id = str(config.get("agent_id") or "main").strip() or "main"
        return f"openclaw:{agent_id}"

    def _resolve_gateway_headers(self, config: dict) -> dict[str, str]:
        headers: dict[str, str] = {}
        session_key = str(config.get("session_key") or "").strip()
        if session_key:
            headers["x-openclaw-session-key"] = session_key

        agent_id = str(config.get("agent_id") or "").strip()
        if agent_id:
            headers["x-openclaw-agent-id"] = agent_id

        message_channel = str(config.get("message_channel") or "").strip()
        if message_channel:
            headers["x-openclaw-message-channel"] = message_channel

        account_id = str(config.get("account_id") or "").strip()
        if account_id:
            headers["x-openclaw-account-id"] = account_id

        return headers

    async def _run_gateway_api(
        self,
        prompt: str,
        config: dict,
    ) -> dict:
        base_url = self._resolve_gateway_base_url(config)
        token = self._resolve_gateway_token(config)
        timeout = int(config.get("timeout_seconds", 300))
        model = self._resolve_gateway_model(config)
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

        url = f"{base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
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

        logger.debug("OpenClaw API: POST %s (model=%s)", url, model)

        def _do_request() -> dict:
            try:
                with urlrequest.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            except urlerror.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                raise RuntimeError(
                    f"OpenClaw API HTTP {e.code}: {body[:400]}"
                )
            except urlerror.URLError as e:
                raise RuntimeError(f"OpenClaw API unavailable: {e}")

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"OpenClaw API returned invalid JSON: {e}")
            return parsed

        return await asyncio.to_thread(_do_request)

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
            data = await self._run_gateway_api(prompt, config)

        # OpenClaw gateway API (OpenAI-compatible)
        if "choices" in data:
            choices = data.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
            response_text = str(message.get("content") or "")
            if not response_text:
                raise RuntimeError("OpenClaw API returned empty message content")

            usage = data.get("usage", {}) or {}
            upstream_model = str(data.get("model") or config.get("model") or "")

            tool_calls: list[OpenClawToolCall] = []
            for tc in message.get("tool_calls", []) or []:
                if not isinstance(tc, dict):
                    continue
                func = tc.get("function", {})
                name = str(func.get("name") or "unknown")
                raw_args = func.get("arguments")
                args: dict[str, Any]
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                        if not isinstance(args, dict):
                            args = {"raw": raw_args}
                    except Exception:
                        args = {"raw": raw_args}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}
                tool_calls.append(OpenClawToolCall(name=name, arguments=args))

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
                len(tool_calls),
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
