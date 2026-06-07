#!/usr/bin/env python3
"""
POC: persistent `claude` subprocess with stream-json bidirectional IO.

Goal: verify that we can spawn ONE `claude` process and send multiple user
messages to it over its lifetime, with the conversation state preserved
intrinsically (no --resume / no JSONL reload per turn).

What the test does:
  Turn 1: send "My favorite number is 73. Acknowledge briefly."
  Turn 2: send "What number did I just tell you? Respond with just the number."

If Turn 2's answer is "73", the process kept context across turns within
one running process — which is the core hypothesis of the migration.

It also times the spawn-once vs. two-shot-spawn paths so we have a real number
for how much we save by keeping the process alive.

Usage:
  python3 poc_persistent_claude.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# --- config -----------------------------------------------------------------

CLAUDE_BIN = (
    Path.home()
    / ".local/share/pipx/venvs/llm-bawt/lib/python3.13/"
    / "site-packages/claude_agent_sdk/_bundled/claude"
)
MODEL = "haiku"  # cheap + fast for a POC
# Use a scratch cwd so we don't trip over the bridge's project history.
SCRATCH_CWD = Path("/tmp/claude-poc-cwd")


def load_oauth_from_env_file() -> str | None:
    """Read CLAUDE_CODE_OAUTH_TOKEN from llm-bawt/.env."""
    env_file = Path.home() / "dev/llm-bawt/.env"
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        if line.startswith("CLAUDE_CODE_OAUTH_TOKEN="):
            return line.split("=", 1)[1].strip() or None
    return None


def build_env() -> dict[str, str]:
    """Env for the claude subprocess. Prefer OAuth token from .env."""
    env = dict(os.environ)
    # Strip my-parent's CLAUDE_* leakage so the child gets a clean slate
    for k in list(env.keys()):
        if k.startswith("CLAUDE_") or k.startswith("CLAUDECODE") or k.startswith("AI_AGENT"):
            del env[k]
    token = load_oauth_from_env_file()
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
        env.pop("ANTHROPIC_API_KEY", None)
    return env


# --- the protocol -----------------------------------------------------------

def encode_user_message(text: str) -> bytes:
    """Frame a user prompt as the JSONL line claude expects."""
    msg = {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": text},
        "parent_tool_use_id": None,
    }
    return (json.dumps(msg) + "\n").encode()


# --- the actual POC ---------------------------------------------------------

class PersistentClaude:
    """Wraps a single long-lived `claude --input-format stream-json` process."""

    def __init__(self, model: str = MODEL, cwd: Path = SCRATCH_CWD):
        self.model = model
        self.cwd = cwd
        self.proc: asyncio.subprocess.Process | None = None
        # Reader keeps pulling lines and dropping them into per-turn queues.
        self._current_queue: asyncio.Queue[dict[str, Any] | None] | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self.stderr_lines: list[str] = []

    async def start(self):
        self.cwd.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(CLAUDE_BIN),
            "--print",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",  # required to emit stream events with --print
            "--include-partial-messages",
            "--model", self.model,
            "--no-session-persistence",  # POC: don't pollute on-disk session store
            "--permission-mode", "bypassPermissions",
        ]
        print(f"[spawn] {' '.join(cmd)}")
        env = build_env()
        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())
        print(f"[spawn] pid={self.proc.pid}")

    async def _read_stdout(self):
        assert self.proc and self.proc.stdout
        try:
            async for raw in self.proc.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # Non-JSON line — log and skip.
                    print(f"[stdout/non-json] {line[:120]}", file=sys.stderr)
                    continue
                if self._current_queue is not None:
                    await self._current_queue.put(msg)
        except Exception as e:
            print(f"[reader] crashed: {e}", file=sys.stderr)
        finally:
            if self._current_queue is not None:
                await self._current_queue.put(None)

    async def _read_stderr(self):
        assert self.proc and self.proc.stderr
        async for raw in self.proc.stderr:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                self.stderr_lines.append(line)
                print(f"[stderr] {line}", file=sys.stderr)

    async def send_turn(self, text: str) -> tuple[list[dict[str, Any]], float]:
        """Send one user message; collect all events through the 'result' msg.

        Returns (events, elapsed_seconds).
        """
        assert self.proc and self.proc.stdin
        self._current_queue = asyncio.Queue()
        t0 = time.perf_counter()
        self.proc.stdin.write(encode_user_message(text))
        await self.proc.stdin.drain()
        events: list[dict[str, Any]] = []
        while True:
            msg = await self._current_queue.get()
            if msg is None:
                # reader EOF — process died mid-turn
                break
            events.append(msg)
            if msg.get("type") == "result":
                break
        elapsed = time.perf_counter() - t0
        self._current_queue = None
        return events, elapsed

    async def close(self):
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.close()
                await self.proc.stdin.wait_closed()
            except Exception:
                pass
        if self.proc:
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
        if self._reader_task:
            self._reader_task.cancel()
        if self._stderr_task:
            self._stderr_task.cancel()


def extract_assistant_text(events: list[dict[str, Any]]) -> str:
    """Pull all assistant text content from a turn's event list."""
    parts: list[str] = []
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        msg = ev.get("message") or {}
        for block in msg.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
    return "".join(parts).strip()


def extract_session_id(events: list[dict[str, Any]]) -> str | None:
    for ev in events:
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            return ev.get("session_id")
    return None


def extract_usage(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for ev in events:
        if ev.get("type") == "result":
            return ev.get("usage") or {}
    return None


def short(text: str, n: int = 200) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


# --- the two experiments ----------------------------------------------------

async def experiment_persistent():
    print("\n" + "=" * 72)
    print("EXPERIMENT 1: persistent process, two turns")
    print("=" * 72)
    pc = PersistentClaude()
    await pc.start()
    try:
        # Turn 1
        events1, t1 = await pc.send_turn(
            "My favorite number is 73. Acknowledge with just 'ok'."
        )
        text1 = extract_assistant_text(events1)
        sid1 = extract_session_id(events1)
        usage1 = extract_usage(events1)
        print(f"\n[turn 1] elapsed={t1:.2f}s  session_id={sid1}")
        print(f"[turn 1] assistant: {short(text1)}")
        print(f"[turn 1] usage: {usage1}")
        print(f"[turn 1] proc alive? returncode={pc.proc.returncode}")

        if pc.proc.returncode is not None:
            print("\n[!] process died after turn 1 — persistence FAILS")
            return False

        # Turn 2 — does context carry?
        events2, t2 = await pc.send_turn(
            "What number did I just tell you? Respond with just the digits."
        )
        text2 = extract_assistant_text(events2)
        sid2 = extract_session_id(events2)
        usage2 = extract_usage(events2)
        print(f"\n[turn 2] elapsed={t2:.2f}s  session_id={sid2}")
        print(f"[turn 2] assistant: {short(text2)}")
        print(f"[turn 2] usage: {usage2}")
        print(f"[turn 2] proc alive? returncode={pc.proc.returncode}")

        context_kept = "73" in text2
        print(f"\n[verdict] context kept across turns? {context_kept}")
        print(f"[verdict] same session_id reused? {sid1 == sid2}")
        print(f"[verdict] turn-1 time {t1:.2f}s  turn-2 time {t2:.2f}s")
        return context_kept
    finally:
        await pc.close()


async def experiment_spawn_per_turn():
    """Baseline: spawn a fresh process for each turn, no --resume."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: spawn-per-turn baseline (no resume, two cold spawns)")
    print("=" * 72)
    total = 0.0
    for i, text in enumerate([
        "My favorite number is 73. Acknowledge with just 'ok'.",
        "What number did I just tell you? Respond with just the digits.",
    ], start=1):
        pc = PersistentClaude()
        await pc.start()
        try:
            events, elapsed = await pc.send_turn(text)
            total += elapsed
            ans = extract_assistant_text(events)
            print(f"[cold turn {i}] elapsed={elapsed:.2f}s  ans={short(ans)}")
        finally:
            await pc.close()
    print(f"\n[baseline total] {total:.2f}s for two cold spawns")
    return total


async def main():
    if not CLAUDE_BIN.exists():
        print(f"ERROR: claude binary not at {CLAUDE_BIN}")
        sys.exit(1)
    persistent_ok = await experiment_persistent()
    baseline_total = await experiment_spawn_per_turn()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Persistent process kept context across two turns: {persistent_ok}")
    print(f"Spawn-per-turn total time (2 turns): {baseline_total:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
