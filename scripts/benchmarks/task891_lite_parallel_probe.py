"""TASK-891 probe: does the Astra (gpt-6-astra) WS/Lite route support
parallel tool calls? Flips exactly ONE variable through the real adapter.

Isolated: own conversation identity, no bot session, no production mutation.
"""
import asyncio, json, os, sys, time, uuid

sys.path.insert(0, "/app/src")

from claude_code_bridge.proxy import chatgpt_transport
from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
from claude_code_bridge.proxy.request_context import ProxyRequestContext

MODEL = "gpt-6-astra"

TOOLS = [
    {"name": "get_weather", "description": "Get current weather for a city.",
     "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                      "required": ["city"]}},
    {"name": "get_time", "description": "Get current local time for a city.",
     "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                      "required": ["city"]}},
]

PROMPT = ("Call get_weather for Paris and get_time for Tokyo. "
          "These are independent. Issue BOTH tool calls now, in this one turn. "
          "Do not ask questions. Do not call them one at a time.")


def body():
    return {"model": MODEL, "max_tokens": 1024, "tools": TOOLS,
            "messages": [{"role": "user", "content": PROMPT}]}


async def run(label, force_parallel):
    orig = chatgpt_transport.lite_request
    if force_parallel:
        def patched(b, scope):
            out = orig(b, scope)
            out["parallel_tool_calls"] = True
            return out
        chatgpt_transport.lite_request = patched
    adapter = OpenAIChatGPTAdapter()
    ctx = ProxyRequestContext(
        request_id=f"task891-{uuid.uuid4().hex[:12]}",
        provider="openai_chatgpt", bot_id="task891-probe",
        conversation_id=uuid.uuid4().hex,   # isolated cache/lease scope
    )
    tool_calls, stop_reason, err = [], None, None
    t0 = time.perf_counter()
    try:
        async for chunk in adapter.call(body(), MODEL, ctx):
            for line in chunk.decode("utf-8", "replace").splitlines():
                if not line.startswith("data: "):
                    continue
                try:
                    ev = json.loads(line[6:])
                except Exception:
                    continue
                if ev.get("type") == "content_block_start":
                    cb = ev.get("content_block") or {}
                    if cb.get("type") == "tool_use":
                        tool_calls.append(cb.get("name"))
                if ev.get("type") == "message_delta":
                    stop_reason = (ev.get("delta") or {}).get("stop_reason", stop_reason)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    finally:
        await adapter.close()
        chatgpt_transport.lite_request = orig
    dt = time.perf_counter() - t0
    print(json.dumps({"config": label, "parallel_tool_calls": force_parallel,
                      "tool_calls": tool_calls, "n_tool_calls": len(tool_calls),
                      "stop_reason": stop_reason, "elapsed_s": round(dt, 2),
                      "error": err}, indent=2), flush=True)
    return len(tool_calls), err


async def main():
    reps = int(os.environ.get("REPS", "2"))
    results = {}
    for label, flag in (("baseline_false", False), ("forced_true", True)):
        counts = []
        for i in range(reps):
            n, err = await run(f"{label}#{i+1}", flag)
            counts.append(n if err is None else -1)
            await asyncio.sleep(1.0)
        results[label] = counts
    print("SUMMARY " + json.dumps(results), flush=True)

asyncio.run(main())
