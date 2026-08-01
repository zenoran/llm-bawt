from llm_bawt.service.pending_tool_calls import PendingToolCallCorrelator


def test_tool_use_id_correlates_parallel_same_name_calls_out_of_order():
    calls = PendingToolCallCorrelator()
    calls.start(call_id="call-a", tool_name="Bash", tool_use_id="tool-a")
    calls.start(call_id="call-b", tool_name="Bash", tool_use_id="tool-b")

    assert calls.finish(tool_name="Bash", tool_use_id="tool-a") == "call-a"
    assert calls.finish(tool_name="Bash", tool_use_id="tool-b") == "call-b"


def test_missing_tool_use_id_preserves_latest_same_name_fallback():
    calls = PendingToolCallCorrelator()
    calls.start(call_id="call-a", tool_name="Read", tool_use_id=None)
    calls.start(call_id="call-b", tool_name="Bash", tool_use_id=None)
    calls.start(call_id="call-c", tool_name="Read", tool_use_id=None)

    assert calls.finish(tool_name="Read", tool_use_id=None) == "call-c"
    assert calls.finish(tool_name="Bash", tool_use_id=None) == "call-b"
    assert calls.finish(tool_name="Unknown", tool_use_id=None) == "call-a"
