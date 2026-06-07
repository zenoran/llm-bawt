#!/usr/bin/env python3
"""
Compare cache token counts for two paths over the SAME two-turn dialogue:
  A) one persistent process, two user messages
  B) two cold spawns (each with --resume) — closer to the SDK's actual behavior

The question: is the cache_read benefit a property of persistence, or
purely server-side (so spawn-per-turn gets it too)?
"""

from __future__ import annotations
import asyncio, json, os, sys, time, uuid
from pathlib import Path

CLAUDE_BIN = Path("/home/bridge/.local/share/pipx/venvs/llm-bawt/lib/python3.13/site-packages/claude_agent_sdk/_bundled/claude")
MODEL = "haiku"
CWD = Path("/tmp/claude-poc-cwd")

def env() -> dict[str, str]:
    e = dict(os.environ)
    for k in list(e.keys()):
        if k.startswith(("CLAUDE_", "CLAUDECODE", "AI_AGENT")):
            del e[k]
    for line in (Path.home()/"dev/llm-bawt/.env").read_text().splitlines():
        if line.startswith("CLAUDE_CODE_OAUTH_TOKEN="):
            tok = line.split("=",1)[1].strip()
            if tok:
                e["CLAUDE_CODE_OAUTH_TOKEN"] = tok
                e.pop("ANTHROPIC_API_KEY", None)
    return e

def user_msg(text: str) -> bytes:
    return (json.dumps({
        "type":"user","session_id":"",
        "message":{"role":"user","content":text},
        "parent_tool_use_id":None,
    })+"\n").encode()

async def spawn(extra_args: list[str]) -> asyncio.subprocess.Process:
    CWD.mkdir(parents=True, exist_ok=True)
    cmd = [str(CLAUDE_BIN),
        "--print","--input-format","stream-json","--output-format","stream-json",
        "--verbose","--model",MODEL,
        "--no-session-persistence" if "--session-id" not in extra_args else None,
        "--permission-mode","bypassPermissions",
        *extra_args]
    cmd = [c for c in cmd if c is not None]
    return await asyncio.create_subprocess_exec(
        *cmd, cwd=str(CWD), env=env(),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )

async def one_turn(proc, text: str) -> tuple[dict, float, str]:
    t0 = time.perf_counter()
    proc.stdin.write(user_msg(text))
    await proc.stdin.drain()
    usage: dict = {}
    ans = ""
    sid = ""
    while True:
        line = await proc.stdout.readline()
        if not line: break
        try: msg = json.loads(line)
        except: continue
        if msg.get("type")=="system" and msg.get("subtype")=="init":
            sid = msg.get("session_id","")
        if msg.get("type")=="assistant":
            for b in msg.get("message",{}).get("content",[]):
                if isinstance(b,dict) and b.get("type")=="text":
                    ans += b.get("text","")
        if msg.get("type")=="result":
            usage = msg.get("usage") or {}
            break
    return usage, time.perf_counter()-t0, sid, ans

async def close(proc):
    try:
        proc.stdin.close()
        await asyncio.wait_for(proc.wait(), timeout=5)
    except Exception:
        proc.kill(); await proc.wait()

def show(label, usage, dt):
    print(f"  {label}: elapsed={dt:.2f}s "
          f"cache_create={usage.get('cache_creation_input_tokens',0)} "
          f"cache_read={usage.get('cache_read_input_tokens',0)} "
          f"new_input={usage.get('input_tokens',0)} "
          f"output={usage.get('output_tokens',0)}")

async def path_a_persistent():
    print("\nPATH A: persistent process, two turns")
    p = await spawn([])
    try:
        u1,d1,sid1,a1 = await one_turn(p, "My favorite number is 73. Say only 'ok'.")
        show("A1", u1, d1)
        u2,d2,sid2,a2 = await one_turn(p, "What number did I tell you? Just the digits.")
        show("A2", u2, d2)
        print(f"  same session? {sid1==sid2}  answer2={a2!r}")
    finally:
        await close(p)

async def path_b_spawn_with_resume():
    """Cold-spawn each turn with --session-id <fixed> so turn-2 has the same on-disk session as if SDK had used --resume."""
    print("\nPATH B: cold spawn per turn with shared --session-id (≈ SDK's --resume path)")
    sid = str(uuid.uuid4())
    # Turn 1 with persistence so the on-disk session exists for turn 2
    p1 = await spawn(["--session-id", sid])
    try:
        u1,d1,_,a1 = await one_turn(p1, "My favorite number is 73. Say only 'ok'.")
        show("B1", u1, d1)
    finally:
        await close(p1)
    # Turn 2 resumes via --resume
    p2 = await spawn(["--resume", sid])
    try:
        u2,d2,_,a2 = await one_turn(p2, "What number did I tell you? Just the digits.")
        show("B2", u2, d2)
        print(f"  answer2={a2!r}")
    finally:
        await close(p2)

async def main():
    await path_a_persistent()
    await path_b_spawn_with_resume()
    print("\nIf B2's cache_read ≈ A2's cache_read, the cache benefit is server-side,")
    print("not a property of persistence — and the migration's cost-savings story shrinks.")

if __name__ == "__main__":
    asyncio.run(main())
