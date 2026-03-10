"""Tests proving concurrent agent tool calls stream in parallel without cross-talk.

The bridge uses a single WebSocket to the OpenClaw gateway. When multiple
agents run simultaneously, events are routed by runId to per-run asyncio
queues. These tests prove:

1. Interleaved tool events from two agents are correctly isolated
2. One agent erroring doesn't disrupt the other
3. Same-tool calls from different agents get their own results
"""

from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, patch

from openclaw_bridge.ws_client import OpenClawWsClient, OpenClawWsConfig


# ---------------------------------------------------------------------------
# Fake gateway WebSocket
# ---------------------------------------------------------------------------


class FakeGatewayWS:
    """Mock WebSocket simulating the OpenClaw gateway wire protocol.

    Supports the two-phase connection:
      1. Handshake via explicit recv() calls (challenge + connect response)
      2. Event streaming via async-for iteration (receive loop)
    """

    def __init__(self):
        self._handshake_queue: asyncio.Queue[str] = asyncio.Queue()
        self._stream_queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self._sent: list[dict] = []
        self._on_send = None

    async def recv(self) -> str:
        return await asyncio.wait_for(self._handshake_queue.get(), timeout=5)

    async def send(self, data: str) -> None:
        msg = json.loads(data)
        self._sent.append(msg)
        if self._on_send:
            await self._on_send(msg)

    async def close(self) -> None:
        pass

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        msg = await self._stream_queue.get()
        if msg is None:
            raise StopAsyncIteration
        return json.dumps(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_event(run_id: str, stream: str, data: dict) -> dict:
    """Build a gateway agent event in wire format."""
    return {
        "type": "event",
        "event": "agent",
        "payload": {"runId": run_id, "stream": stream, "data": data},
    }


async def _setup_client() -> tuple[OpenClawWsClient, FakeGatewayWS]:
    """Connect an OpenClawWsClient through a FakeGatewayWS.

    The fake auto-responds to connect and chat.send requests.
    Run IDs are deterministic: "run_{sessionKey}".
    """
    ws = FakeGatewayWS()

    # Pre-load the challenge nonce for the first recv()
    await ws._handshake_queue.put(json.dumps({
        "event": "connect.challenge",
        "payload": {"nonce": uuid.uuid4().hex},
    }))

    async def on_send(msg: dict) -> None:
        if msg.get("type") == "req" and msg.get("method") == "connect":
            # Connect response goes through handshake recv()
            await ws._handshake_queue.put(json.dumps({
                "type": "res", "id": msg["id"], "ok": True, "payload": {},
            }))
        elif msg.get("type") == "req" and msg.get("method") == "chat.send":
            # chat.send response goes through stream (receive loop routes it)
            session = msg["params"]["sessionKey"]
            await ws._stream_queue.put({
                "type": "res", "id": msg["id"], "ok": True,
                "payload": {"runId": f"run_{session}"},
            })

    ws._on_send = on_send

    client = OpenClawWsClient(OpenClawWsConfig(url="ws://fake:18789", token="t"))
    with patch("openclaw_bridge.ws_client.websockets") as mock_lib:
        mock_lib.connect = AsyncMock(return_value=ws)
        await client.connect()

    assert client.connected
    return client, ws


async def _wait_for_queues(client: OpenClawWsClient, n: int) -> None:
    """Poll until the client has n per-run queues (consumers are ready)."""
    for _ in range(200):
        if len(client._run_queues) >= n:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"Expected {n} run queues, got {len(client._run_queues)}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParallelToolStreams:

    def test_interleaved_tool_events_isolated(self):
        """Two agents with interleaved tool calls → correctly isolated streams.

        Events are injected in order: A-tool-start, B-tool-start, A-tool-end,
        B-delta, A-delta, B-tool-end, A-lifecycle-end, B-lifecycle-end.

        The single receive loop routes all 8 events by runId. Each consumer
        must see exactly its own 4 events with zero cross-contamination.
        """

        async def _run():
            client, ws = await _setup_client()
            events_a, events_b = [], []

            async def consume(session, collector):
                async for evt in client.send_and_stream(session, f"hi {session}", timeout=10):
                    collector.append(evt)

            task_a = asyncio.create_task(consume("agent_a", events_a))
            task_b = asyncio.create_task(consume("agent_b", events_b))
            await _wait_for_queues(client, 2)

            for evt in [
                _agent_event("run_agent_a", "tool",      {"phase": "start", "name": "search_web", "arguments": {"q": "cats"}}),
                _agent_event("run_agent_b", "tool",      {"phase": "start", "name": "read_file",  "arguments": {"path": "/x"}}),
                _agent_event("run_agent_a", "tool",      {"phase": "end",   "name": "search_web", "result": "found 42 cats"}),
                _agent_event("run_agent_b", "assistant", {"delta": "Reading..."}),
                _agent_event("run_agent_a", "assistant", {"delta": "Found cats!"}),
                _agent_event("run_agent_b", "tool",      {"phase": "end",   "name": "read_file",  "result": "contents"}),
                _agent_event("run_agent_a", "lifecycle", {"phase": "end"}),
                _agent_event("run_agent_b", "lifecycle", {"phase": "end"}),
            ]:
                await ws._stream_queue.put(evt)

            await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=10)

            # Correct count
            assert len(events_a) == 4
            assert len(events_b) == 4

            # Zero cross-contamination
            assert all(e["payload"]["runId"] == "run_agent_a" for e in events_a)
            assert all(e["payload"]["runId"] == "run_agent_b" for e in events_b)

            # Tool events routed to the right consumer
            a_tools = [e for e in events_a if e["payload"]["stream"] == "tool"]
            b_tools = [e for e in events_b if e["payload"]["stream"] == "tool"]
            assert a_tools[0]["payload"]["data"]["name"] == "search_web"
            assert a_tools[1]["payload"]["data"]["result"] == "found 42 cats"
            assert b_tools[0]["payload"]["data"]["name"] == "read_file"
            assert b_tools[1]["payload"]["data"]["result"] == "contents"

            await client.disconnect()

        asyncio.run(_run())

    def test_agent_error_doesnt_affect_other(self):
        """Agent A errors mid-tool; Agent B completes normally."""

        async def _run():
            client, ws = await _setup_client()
            events_a, events_b = [], []

            async def consume(session, collector):
                async for evt in client.send_and_stream(session, f"hi {session}", timeout=10):
                    collector.append(evt)

            task_a = asyncio.create_task(consume("agent_a", events_a))
            task_b = asyncio.create_task(consume("agent_b", events_b))
            await _wait_for_queues(client, 2)

            for evt in [
                _agent_event("run_agent_a", "tool",      {"phase": "start", "name": "risky_op", "arguments": {}}),
                _agent_event("run_agent_b", "tool",      {"phase": "start", "name": "safe_op",  "arguments": {}}),
                _agent_event("run_agent_a", "lifecycle", {"phase": "error"}),  # A crashes
                _agent_event("run_agent_b", "tool",      {"phase": "end",   "name": "safe_op", "result": "ok"}),
                _agent_event("run_agent_b", "assistant", {"delta": "Done!"}),
                _agent_event("run_agent_b", "lifecycle", {"phase": "end"}),
            ]:
                await ws._stream_queue.put(evt)

            await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=10)

            # A stopped after tool_start + lifecycle error
            assert len(events_a) == 2
            assert events_a[-1]["payload"]["data"]["phase"] == "error"

            # B completed fully, unaffected
            assert len(events_b) == 4
            b_tools = [e for e in events_b if e["payload"]["stream"] == "tool"]
            assert len(b_tools) == 2
            assert b_tools[1]["payload"]["data"]["result"] == "ok"

            await client.disconnect()

        asyncio.run(_run())

    def test_same_tool_different_agents(self):
        """Two agents call the same tool; each gets its own result."""

        async def _run():
            client, ws = await _setup_client()
            events_a, events_b = [], []

            async def consume(session, collector):
                async for evt in client.send_and_stream(session, f"hi {session}", timeout=10):
                    collector.append(evt)

            task_a = asyncio.create_task(consume("agent_a", events_a))
            task_b = asyncio.create_task(consume("agent_b", events_b))
            await _wait_for_queues(client, 2)

            for evt in [
                _agent_event("run_agent_a", "tool",      {"phase": "start", "name": "search_web", "arguments": {"q": "dogs"}}),
                _agent_event("run_agent_b", "tool",      {"phase": "start", "name": "search_web", "arguments": {"q": "cats"}}),
                _agent_event("run_agent_b", "tool",      {"phase": "end",   "name": "search_web", "result": "found cats"}),
                _agent_event("run_agent_a", "tool",      {"phase": "end",   "name": "search_web", "result": "found dogs"}),
                _agent_event("run_agent_a", "lifecycle", {"phase": "end"}),
                _agent_event("run_agent_b", "lifecycle", {"phase": "end"}),
            ]:
                await ws._stream_queue.put(evt)

            await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=10)

            a_result = [e for e in events_a if e["payload"]["data"].get("result")][0]
            b_result = [e for e in events_b if e["payload"]["data"].get("result")][0]
            assert a_result["payload"]["data"]["result"] == "found dogs"
            assert b_result["payload"]["data"]["result"] == "found cats"

            await client.disconnect()

        asyncio.run(_run())

    def test_three_concurrent_agents(self):
        """Scales to 3+ simultaneous agents without degradation."""

        async def _run():
            client, ws = await _setup_client()
            collectors = {"a": [], "b": [], "c": []}

            async def consume(agent, collector):
                async for evt in client.send_and_stream(f"agent_{agent}", f"hi", timeout=10):
                    collector.append(evt)

            tasks = [
                asyncio.create_task(consume(k, v))
                for k, v in collectors.items()
            ]
            await _wait_for_queues(client, 3)

            # Round-robin tool events across 3 agents
            for evt in [
                _agent_event("run_agent_a", "tool", {"phase": "start", "name": "t1", "arguments": {}}),
                _agent_event("run_agent_b", "tool", {"phase": "start", "name": "t2", "arguments": {}}),
                _agent_event("run_agent_c", "tool", {"phase": "start", "name": "t3", "arguments": {}}),
                _agent_event("run_agent_a", "tool", {"phase": "end",   "name": "t1", "result": "r1"}),
                _agent_event("run_agent_b", "tool", {"phase": "end",   "name": "t2", "result": "r2"}),
                _agent_event("run_agent_c", "tool", {"phase": "end",   "name": "t3", "result": "r3"}),
                _agent_event("run_agent_a", "lifecycle", {"phase": "end"}),
                _agent_event("run_agent_b", "lifecycle", {"phase": "end"}),
                _agent_event("run_agent_c", "lifecycle", {"phase": "end"}),
            ]:
                await ws._stream_queue.put(evt)

            await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)

            for key, events in collectors.items():
                run_id = f"run_agent_{key}"
                assert len(events) == 3, f"agent_{key} got {len(events)} events"
                assert all(e["payload"]["runId"] == run_id for e in events)

            await client.disconnect()

        asyncio.run(_run())
