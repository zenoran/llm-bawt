"""ChatGPT Responses WS/Lite transport, based on Codex rust-v0.153.4.

Full-history response.create requests deliberately omit previous_response_id:
Claude SDK histories can branch and do not retain all upstream output items.
Only successful sockets are reusable; no transport-layer generation retries.
The adapter's existing output-aware retry policy owns all replay decisions.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit
from uuid import NAMESPACE_OID, uuid5

import httpx
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

logger = logging.getLogger(__name__)
LITE_HEADER = "x-openai-internal-codex-responses-lite"
TURN_HEADER = "x-codex-turn-state"
WS_BETA = "responses_websockets=2026-02-06"
TERMINALS = {"response.completed", "response.incomplete"}


def lite_request(body: dict, scope: str) -> dict:
    """Move tools/instructions into Lite input, without changing caller history."""
    body = copy.deepcopy(body)
    namespace = uuid5(NAMESPACE_OID, scope)
    tools = body.pop("tools", [])
    instructions = body.pop("instructions", "")
    prefix = [{
        "type": "additional_tools", "role": "developer", "tools": tools,
        "id": "at_" + str(uuid5(namespace, json.dumps(tools, separators=(",", ":")))),
    }]
    if instructions:
        prefix.append({
            "type": "message", "role": "developer",
            "id": "msg_" + str(uuid5(namespace, instructions)),
            "content": [{"type": "input_text", "text": instructions}],
        })
    body["input"] = prefix + body.get("input", [])
    # Codex strips detail from both message images and tool-result images.
    for item in body["input"]:
        for field in ("content", "output"):
            content = item.get(field)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "input_image":
                        part.pop("detail", None)
    body["parallel_tool_calls"] = False
    body.setdefault("reasoning", {})["context"] = "all_turns"
    body.setdefault("include", [])
    if "reasoning.encrypted_content" not in body["include"]:
        body["include"].append("reasoning.encrypted_content")
    return body


def _event_object(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _event_object(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_event_object(v) for v in value]
    return value


@dataclass
class _Session:
    socket: object | None = None
    turn_state: str | None = None
    http_only: bool = False
    timer: asyncio.TimerHandle | None = None


class ChatGPTResponsesTransport:
    """Bounded, exclusive socket leases scoped to account/conversation/turn/model.

    Parallel requests never share a socket. Idle leases expire automatically;
    cancellation or an unfinished response discards its lease immediately.
    """

    def __init__(self, *, idle_timeout: float = 300, keepalive: float = 60,
                 max_connections: int = 32, connector=connect):
        self.idle_timeout = idle_timeout
        self.keepalive = keepalive
        self.max_connections = max_connections
        self.connector = connector
        self._idle: dict[tuple, _Session] = {}
        self._slots = asyncio.Semaphore(max_connections)
        self._active: set[ChatGPTStream] = set()
        self._cleanup: set[asyncio.Task] = set()
        self._closed = False

    def _expire(self, key, session):
        if self._idle.get(key) is session:
            self._idle.pop(key)
            task = asyncio.create_task(self._disconnect(session))
            self._cleanup.add(task)
            task.add_done_callback(self._cleanup.discard)

    async def _disconnect(self, session):
        if session.timer:
            session.timer.cancel()
        if session.socket is not None:
            await session.socket.close()
            session.socket = None

    async def open(self, *, body, headers, bearer, base_url, context, http_client):
        if self._closed:
            raise RuntimeError("ChatGPT transport is closed")
        await asyncio.wait_for(self._slots.acquire(), timeout=15)
        # Never cache anonymous callers or use prompt hashes as security identity.
        scoped = bool(context and context.conversation_id and context.request_id)
        key = (base_url, headers.get("chatgpt-account-id"),
               hashlib.sha256(bearer.encode()).hexdigest(),
               context.conversation_id if scoped else None,
               context.request_id if scoped else None, body["model"])
        session = self._idle.pop(key, _Session()) if scoped else _Session()
        if session.timer:
            session.timer.cancel()
        stream = ChatGPTStream(self, key if scoped else None, session)
        self._active.add(stream)
        try:
            request = lite_request(body, headers["session_id"])
            wire_headers = {**headers, "Authorization": f"Bearer {bearer}",
                            "OpenAI-Beta": WS_BETA, LITE_HEADER: "true"}
            if session.turn_state:
                wire_headers[TURN_HEADER] = session.turn_state
            if session.socket is not None:
                logger.info("chatgpt_transport transport=websocket model=%s reused=true", body["model"])
            if not session.http_only and session.socket is None:
                url = urlsplit(base_url.rstrip("/") + "/responses")
                ws_url = urlunsplit(url._replace(scheme="wss" if url.scheme == "https" else "ws"))
                try:
                    session.socket = await self.connector(
                        ws_url, additional_headers=wire_headers,
                        open_timeout=15, close_timeout=2, max_size=16 * 1024 * 1024,
                        ping_interval=20, ping_timeout=20,
                    )
                    response = session.socket.response
                    session.turn_state = session.turn_state or response.headers.get(TURN_HEADER)
                    logger.info("chatgpt_transport transport=websocket model=%s reused=false", body["model"])
                except InvalidStatus as exc:
                    # Codex's explicit fallback contract is HTTP 426. Never
                    # hide authentication, quota, bad payload or server errors.
                    response = exc.response
                    if response.status_code != 426:
                        raise httpx.HTTPStatusError(
                            f"WebSocket upgrade HTTP {response.status_code}",
                            request=httpx.Request("GET", ws_url),
                            response=httpx.Response(response.status_code, headers=response.headers),
                        ) from exc
                    session.http_only = True
                    logger.warning("chatgpt_transport transport=sse fallback=upgrade_426 model=%s", body["model"])
            if session.http_only:
                http_headers = {LITE_HEADER: "true"}
                if session.turn_state:
                    http_headers[TURN_HEADER] = session.turn_state
                # Lite adds fields not yet represented by the OpenAI SDK schema.
                from openai import AsyncStream
                from openai.types.responses import ResponseStreamEvent

                stream.http = await http_client.post(
                    "/responses", body=request, cast_to=object, stream=True,
                    stream_cls=AsyncStream[ResponseStreamEvent],
                    options={"headers": http_headers},
                )
                response = getattr(stream.http, "response", None)
                if response is not None:
                    session.turn_state = session.turn_state or response.headers.get(TURN_HEADER)
                stream.response = response
                stream.http_iterator = stream.http.__aiter__()
            else:
                metadata = request.setdefault("client_metadata", {})
                metadata["ws_request_header_x_openai_internal_codex_responses_lite"] = "true"
                if session.turn_state:
                    metadata["x-codex-turn-state"] = session.turn_state
                await asyncio.wait_for(session.socket.send(json.dumps({"type": "response.create", **request})), 60)
                stream.response = session.socket.response
            return stream
        except ConnectionClosed as exc:
            await stream.close()
            raise ConnectionError("ChatGPT WebSocket closed before response.create completed") from exc
        except BaseException:
            await stream.close()
            raise

    async def release(self, stream):
        self._active.discard(stream)
        try:
            if not stream.complete:
                # Keep the turn's routing token across a safe outer retry, but
                # never reuse a socket containing an unfinished generation.
                await self._disconnect(stream.session)
            if (stream.key and not self._closed
                    and stream.key not in self._idle
                    and len(self._idle) < self.max_connections):
                self._idle[stream.key] = stream.session
                stream.session.timer = asyncio.get_running_loop().call_later(
                    self.keepalive, self._expire, stream.key, stream.session,
                )
            else:
                await self._disconnect(stream.session)
        finally:
            self._slots.release()

    async def close(self):
        self._closed = True
        await asyncio.gather(*(s.close() for s in list(self._active)))
        idle, self._idle = self._idle, {}
        await asyncio.gather(*(self._disconnect(s) for s in idle.values()))
        if self._cleanup:
            await asyncio.gather(*self._cleanup)


class ChatGPTStream:
    def __init__(self, owner, key, session):
        self.owner, self.key, self.session = owner, key, session
        self.http = self.http_iterator = self.response = None
        self.complete = self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed or self.complete:
            raise StopAsyncIteration
        try:
            if self.http is not None:
                event = await asyncio.wait_for(anext(self.http_iterator), self.owner.idle_timeout)
            else:
                raw = await asyncio.wait_for(self.session.socket.recv(), self.owner.idle_timeout)
                payload = json.loads(raw)
                if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
                    raise ValueError("Invalid Responses WebSocket event")
                event = _event_object(payload)
            kind = getattr(event, "type", "")
            if kind == "response.metadata" and not self.session.turn_state:
                headers = getattr(event, "headers", None)
                if isinstance(headers, SimpleNamespace):
                    headers = vars(headers)
                if isinstance(headers, dict):
                    for name, value in headers.items():
                        if name.lower() == TURN_HEADER and isinstance(value, str):
                            self.session.turn_state = value
                            break
            if kind in TERMINALS:
                self.complete = True
            return event
        except (ConnectionClosed, StopAsyncIteration) as exc:
            raise ConnectionError("stream closed before response.completed") from exc
        except asyncio.TimeoutError as exc:
            raise TimeoutError("idle timeout waiting for ChatGPT Responses event") from exc

    async def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.http is not None:
                await self.http.close()
        finally:
            await self.owner.release(self)

    aclose = close
