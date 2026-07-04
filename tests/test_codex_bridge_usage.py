from codex_bridge.bridge import CodexBridge


def test_sanitize_context_usage_suppresses_impossible_totals():
    usage = {
        "input_tokens": 2_252_466,
        "cache_read_tokens": 2_107_520,
        "cache_creation_tokens": 0,
        "output_tokens": 13_631,
        "context_window": 272_000,
        "max_output_tokens": 128_000,
        "total_cost_usd": None,
    }

    sanitized = CodexBridge._sanitize_context_usage(usage)

    assert sanitized is not None
    assert sanitized["input_tokens"] == 0
    assert sanitized["cache_read_tokens"] == 0
    assert sanitized["cache_creation_tokens"] == 0
    assert sanitized["output_tokens"] == 13_631
    assert sanitized["context_window"] == 272_000


def test_sanitize_context_usage_keeps_plausible_totals():
    usage = {
        "input_tokens": 18_000,
        "cache_read_tokens": 120_000,
        "cache_creation_tokens": 0,
        "output_tokens": 2_400,
        "context_window": 272_000,
        "max_output_tokens": 128_000,
        "total_cost_usd": None,
    }

    sanitized = CodexBridge._sanitize_context_usage(usage)

    assert sanitized == usage
