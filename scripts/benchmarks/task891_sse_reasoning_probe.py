"""TASK-891: can the ordinary SSE path keep Lite's reasoning-continuity fields
(reasoning.context=all_turns + include reasoning.encrypted_content) while ALSO
getting parallel tool calls? If yes, option A loses nothing."""
import asyncio, json, sys, time, uuid
sys.path.insert(0, "/app/src")
from claude_code_bridge.proxy.adapters.base import ProviderAdapter
from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
from claude_code_bridge.proxy.request_context import ProxyRequestContext

MODEL = "gpt-6-astra"
TOOLS = [{"name": n, "description": f"Return the {n} fact. Independent.",
          "input_schema": {"type": "object", "properties": {}, "required": []}}
         for n in ("read_config", "read_schema", "read_log")]
PROMPT = ("Call read_config, read_schema and read_log. They are independent. "
          "Issue all three tool calls in this one turn.")

class P(OpenAIChatGPTAdapter):
    mode = "plain"
    async def open_stream(self, **kw):
        return await ProviderAdapter.open_stream(self, **kw)   # force ordinary SSE
    def prepare_request(self, rb, context=None):
        rb = super().prepare_request(rb, context)
        if self.mode == "reasoning":
            rb.setdefault("reasoning", {})["context"] = "all_turns"
            inc = rb.setdefault("include", [])
            if "reasoning.encrypted_content" not in inc:
                inc.append("reasoning.encrypted_content")
        return rb

async def run(mode, i):
    ad = P(); ad.mode = mode
    ctx = ProxyRequestContext(request_id=f"t891r-{uuid.uuid4().hex[:12]}",
                              provider="openai_chatgpt", bot_id="task891-probe",
                              conversation_id=uuid.uuid4().hex)
    calls, stop, err, thinking = [], None, None, 0
    t0 = time.perf_counter()
    try:
        async for chunk in ad.call({"model": MODEL, "max_tokens": 1024, "tools": TOOLS,
                                    "messages": [{"role": "user", "content": PROMPT}]},
                                   MODEL, ctx):
            for line in chunk.decode("utf-8", "replace").splitlines():
                if not line.startswith("data: "): continue
                try: ev = json.loads(line[6:])
                except Exception: continue
                if ev.get("type") == "content_block_start":
                    cb = ev.get("content_block") or {}
                    if cb.get("type") == "tool_use": calls.append(cb.get("name"))
                    if cb.get("type") == "thinking": thinking += 1
                if ev.get("type") == "message_delta":
                    stop = (ev.get("delta") or {}).get("stop_reason", stop)
    except Exception as exc:
        err = f"{type(exc).__name__}: {str(exc)[:260]}"
    finally:
        await ad.close()
    print(json.dumps({"mode": mode, "run": i, "n_tools": len(calls), "tools": calls,
                      "thinking_blocks": thinking, "stop": stop,
                      "s": round(time.perf_counter()-t0, 2), "error": err}), flush=True)

async def main():
    for mode in ("plain", "reasoning"):
        for i in range(1, 4):
            await run(mode, i); await asyncio.sleep(1)
asyncio.run(main())
