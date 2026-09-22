"""TASK-891 benchmark: Lite/WS (production) vs ordinary supervised SSE for gpt-6-astra.

Matched fan-out tool workload driven to completion through the REAL adapter.
Measures model hops, tool calls per hop, wall time, and token/cache usage.
Isolated conversation ids; no bot session, no prod mutation.
"""
import asyncio, json, os, statistics, sys, time, uuid

sys.path.insert(0, "/app/src")

from claude_code_bridge.proxy.adapters.base import ProviderAdapter
from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
from claude_code_bridge.proxy.request_context import ProxyRequestContext

MODEL = "gpt-6-astra"
FANOUT = int(os.getenv("FANOUT", "4"))

FACTS = {
    "read_config":   "max_workers=17",
    "read_schema":   "tables=58",
    "read_log":      "errors=3",
    "read_manifest": "version=0.1.33",
    "read_index":    "shards=9",
    "read_quota":    "limit=250",
}
NAMES = list(FACTS)[:FANOUT]
TOOLS = [
    {"name": n, "description": f"Return the {n.split('_')[1]} fact. Independent of all other tools.",
     "input_schema": {"type": "object", "properties": {}, "required": []}}
    for n in NAMES
]
PROMPT = (
    "Gather these independent facts by calling every one of these tools exactly once: "
    + ", ".join(NAMES)
    + ". They have no dependencies on each other. When you have all of them, reply with a "
      "single line listing each value separated by spaces, and nothing else."
)


class ForcedAdapter(OpenAIChatGPTAdapter):
    """force='lite' keeps production routing; force='sse' takes the ordinary path."""
    force = "lite"
    captured_body = None

    async def open_stream(self, **kw):
        if self.force == "sse":
            return await ProviderAdapter.open_stream(self, **kw)
        return await super().open_stream(**kw)

    def prepare_request(self, responses_body, context=None):
        rb = super().prepare_request(responses_body, context)
        if ForcedAdapter.captured_body is None:
            ForcedAdapter.captured_body = {self.force: {k: v for k, v in rb.items() if k != "input"}}
        return rb


async def one_hop(adapter, ctx, messages):
    """Run one model turn; return (blocks, stop_reason, usage, elapsed)."""
    body = {"model": MODEL, "max_tokens": 2048, "tools": TOOLS, "messages": messages}
    blocks, stop, usage = [], None, {}
    cur = None
    t0 = time.perf_counter()
    async for chunk in adapter.call(body, MODEL, ctx):
        for line in chunk.decode("utf-8", "replace").splitlines():
            if not line.startswith("data: "):
                continue
            try:
                ev = json.loads(line[6:])
            except Exception:
                continue
            t = ev.get("type")
            if t == "content_block_start":
                cur = dict(ev.get("content_block") or {})
                if cur.get("type") == "tool_use":
                    cur["_args"] = ""
                blocks.append(cur)
            elif t == "content_block_delta":
                d = ev.get("delta") or {}
                if cur is None:
                    continue
                if d.get("type") == "input_json_delta":
                    cur["_args"] += d.get("partial_json") or ""
                elif d.get("type") == "text_delta":
                    cur["text"] = (cur.get("text") or "") + (d.get("text") or "")
            elif t == "message_delta":
                stop = (ev.get("delta") or {}).get("stop_reason", stop)
                if ev.get("usage"):
                    usage = ev["usage"]
    return blocks, stop, usage, time.perf_counter() - t0


async def run_rep(force, rep):
    ad = ForcedAdapter()
    ad.force = force
    ctx_id = uuid.uuid4().hex
    messages = [{"role": "user", "content": PROMPT}]
    hops, total_in, total_cached, total_out = [], 0, 0, 0
    err, final = None, ""
    t0 = time.perf_counter()
    try:
        for hop in range(12):
            ctx = ProxyRequestContext(
                request_id=f"t891b-{uuid.uuid4().hex[:12]}",
                provider="openai_chatgpt", bot_id="task891-bench",
                conversation_id=ctx_id,
            )
            blocks, stop, usage, el = await one_hop(ad, ctx, messages)
            tools = [b for b in blocks if b.get("type") == "tool_use"]
            ti = int(usage.get("input_tokens") or 0)
            tc = int(usage.get("cache_read_input_tokens") or 0)
            to = int(usage.get("output_tokens") or 0)
            total_in += ti + tc; total_cached += tc; total_out += to
            hops.append({"hop": hop + 1, "s": round(el, 2), "tools": [t["name"] for t in tools],
                         "n_tools": len(tools), "in": ti, "cached": tc, "out": to, "stop": stop})
            final = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            if not tools:
                break
            asst = [{"type": "text", "text": b.get("text", "")} for b in blocks
                    if b.get("type") == "text" and b.get("text")]
            asst += [{"type": "tool_use", "id": t["id"], "name": t["name"],
                      "input": json.loads(t["_args"] or "{}")} for t in tools]
            messages = messages + [{"role": "assistant", "content": asst}]
            messages = messages + [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": t["id"],
                 "content": FACTS.get(t["name"], "unknown")} for t in tools]}]
    except Exception as exc:
        err = f"{type(exc).__name__}: {str(exc)[:240]}"
    finally:
        await ad.close()
    got = sum(1 for v in (FACTS[n] for n in NAMES) if v in final)
    rec = {"config": force, "rep": rep, "total_s": round(time.perf_counter() - t0, 2),
           "model_hops": len(hops), "max_tools_per_hop": max([h["n_tools"] for h in hops] or [0]),
           "total_tool_calls": sum(h["n_tools"] for h in hops),
           "in_tok": total_in, "cached_tok": total_cached, "out_tok": total_out,
           "cache_pct": round(100.0 * total_cached / total_in, 1) if total_in else 0.0,
           "facts_correct": f"{got}/{len(NAMES)}", "error": err, "hops": hops}
    print(json.dumps({k: v for k, v in rec.items() if k != "hops"}), flush=True)
    return rec


async def main():
    reps = int(os.getenv("REPS", "5"))
    results = []
    for r in range(1, reps + 1):
        for force in ("lite", "sse"):        # interleaved, controls for drift
            results.append(await run_rep(force, r))
            await asyncio.sleep(2)
    print("\n===== SUMMARY =====", flush=True)
    for force in ("lite", "sse"):
        rs = [x for x in results if x["config"] == force and not x["error"]]
        if not rs:
            print(f"{force}: ALL FAILED"); continue
        med = lambda k: round(statistics.median([x[k] for x in rs]), 2)
        rng = lambda k: (min(x[k] for x in rs), max(x[k] for x in rs))
        print(json.dumps({
            "config": force, "n_ok": len(rs), "n_err": sum(1 for x in results
                if x["config"] == force and x["error"]),
            "median_total_s": med("total_s"), "range_total_s": rng("total_s"),
            "median_hops": med("model_hops"), "range_hops": rng("model_hops"),
            "max_tools_per_hop": rng("max_tools_per_hop"),
            "median_in_tok": med("in_tok"), "median_cache_pct": med("cache_pct"),
            "median_out_tok": med("out_tok"),
            "correct": [x["facts_correct"] for x in rs],
        }), flush=True)
    with open("/tmp/task891/bench_results.json", "w") as f:
        json.dump({"fanout": FANOUT, "results": results,
                   "bodies": ForcedAdapter.captured_body}, f, indent=2)
    print("raw -> /tmp/task891/bench_results.json", flush=True)

asyncio.run(main())
