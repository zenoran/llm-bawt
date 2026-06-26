"""Integration test: multi-bot concurrent streaming.

Hits the LIVE service at localhost:8642 to prove two OpenClaw bots
can stream concurrently.  Before the fix, the single-worker
_llm_executor serializes all LLM/agent calls — the second bot won't
start until the first finishes.

Run:
    uv run python -m pytest tests/test_concurrent_bot_streaming.py -v -s

Expected results:
    BEFORE fix (old code on server): FAILS — wall time ≈ sum of both
        individual latencies (serial execution).
    AFTER fix (new code deployed):   PASSES — wall time ≈ max of both
        individual latencies (concurrent execution).
"""

from __future__ import annotations

import json
import time
import threading

import pytest
import requests


SERVICE_URL = "http://localhost:8642"
# Two OpenClaw agent_backend bots to test concurrently
BOT_A = "vex"
BOT_B = "byte"
USER_ID = "nick"
# Short prompt that still requires a real round-trip through OpenClaw
PROMPT = "Say exactly: ping"


def _is_service_up() -> bool:
    try:
        r = requests.get(f"{SERVICE_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _send_streaming_chat(bot_id: str, results: dict, key: str) -> None:
    """Send a streaming chat request and record timing + first token."""
    url = f"{SERVICE_URL}/v1/botchat/{bot_id}/{USER_ID}/chat/completions"
    payload = {
        "model": "openclaw",
        "stream": True,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    results[key] = {
        "request_sent": time.monotonic(),
        "first_data": None,
        "done": None,
        "error": None,
        "chunks": 0,
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                results[key]["chunks"] += 1
                if results[key]["first_data"] is None:
                    results[key]["first_data"] = time.monotonic()
    except Exception as e:
        results[key]["error"] = str(e)
    results[key]["done"] = time.monotonic()


def _warmup_bot(bot_id: str) -> None:
    """Send a single request to warm up the bot's pipeline."""
    url = f"{SERVICE_URL}/v1/botchat/{bot_id}/{USER_ID}/chat/completions"
    payload = {
        "model": "openclaw",
        "stream": True,
        "messages": [{"role": "user", "content": "Say exactly: warmup"}],
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as resp:
            for _ in resp.iter_lines():
                pass
    except Exception:
        pass


@pytest.mark.skipif(not _is_service_up(), reason="Service not running at localhost:8642")
class TestLiveConcurrentStreaming:
    """Hit the live service with two simultaneous OpenClaw bot requests.

    Measures wall time vs individual latencies.  With serialization,
    wall time ≈ sum of individual times.  With concurrency, wall time
    ≈ max of individual times.
    """

    def test_two_bots_stream_concurrently(self):
        """Two OpenClaw bots must process concurrently, not serially.

        Metric: the ratio of wall time to the sum of individual latencies.
        - Serial:     ratio ≈ 1.0 (wall = a + b)
        - Concurrent: ratio ≈ 0.5 (wall = max(a, b) ≈ (a+b)/2)

        We also send warmup requests first to avoid cold-start effects
        contaminating the timing.
        """
        # Warmup: flush any stale pipeline state
        print("\n  Warming up both bots...")
        warmup_threads = [
            threading.Thread(target=_warmup_bot, args=(BOT_A,)),
            threading.Thread(target=_warmup_bot, args=(BOT_B,)),
        ]
        for t in warmup_threads:
            t.start()
        for t in warmup_threads:
            t.join(timeout=60)
        print("  Warmup complete. Sending concurrent requests...")

        # Actual concurrent test
        results: dict = {}
        t_a = threading.Thread(target=_send_streaming_chat, args=(BOT_A, results, "a"))
        t_b = threading.Thread(target=_send_streaming_chat, args=(BOT_B, results, "b"))

        t0 = time.monotonic()
        t_a.start()
        t_b.start()
        t_a.join(timeout=120)
        t_b.join(timeout=120)
        total_wall = time.monotonic() - t0

        # Both must have completed
        assert results.get("a") and results.get("b"), f"Missing results: {results}"
        assert results["a"]["error"] is None, f"Bot A error: {results['a']['error']}"
        assert results["b"]["error"] is None, f"Bot B error: {results['b']['error']}"
        assert results["a"]["first_data"] is not None, "Bot A never received data"
        assert results["b"]["first_data"] is not None, "Bot B never received data"

        # Timing analysis
        a_latency = results["a"]["first_data"] - t0
        b_latency = results["b"]["first_data"] - t0
        a_done = results["a"]["done"] - t0
        b_done = results["b"]["done"] - t0
        start_gap = abs(a_latency - b_latency)
        sum_latencies = a_latency + b_latency

        # Serialization ratio: wall_time / sum_of_individual_first_data_times
        # Serial:     ratio ≈ 1.0 (second bot waits, so wall ≈ a + b)
        # Concurrent: ratio < 0.75 (both process in parallel, wall ≈ max(a, b))
        ratio = total_wall / sum_latencies if sum_latencies > 0 else 1.0

        print(f"  {'='*60}")
        print(f"  Bot A ({BOT_A}): first data @ {a_latency:.1f}s, done @ {a_done:.1f}s  ({results['a']['chunks']} chunks)")
        print(f"  Bot B ({BOT_B}): first data @ {b_latency:.1f}s, done @ {b_done:.1f}s  ({results['b']['chunks']} chunks)")
        print(f"  Start gap: {start_gap:.1f}s")
        print(f"  Total wall time: {total_wall:.1f}s")
        print(f"  Sum of latencies: {sum_latencies:.1f}s")
        print(f"  Serialization ratio: {ratio:.2f}  (1.0=serial, 0.5=concurrent)")
        print(f"  {'='*60}")

        assert ratio < 0.80, (
            f"Serialization ratio is {ratio:.2f} — bots are running serially! "
            f"Wall time ({total_wall:.1f}s) ≈ sum of latencies ({sum_latencies:.1f}s). "
            f"With concurrent execution, wall time should be ≈ max(individual), "
            f"giving ratio ≈ 0.5."
        )
