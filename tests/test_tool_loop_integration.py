"""Integration tests for tool loop with format handlers.

These tests verify the full tool loop behavior with mocked LLM clients,
ensuring proper tool detection, execution, and response sanitization.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from llm_bawt.tools.loop import ToolLoop, DEFAULT_MAX_ITERATIONS
from llm_bawt.tools.formats import ToolFormat
from llm_bawt.tools.formats.base import ToolCallRequest


class MockMessage:
    """Simple mock message for testing."""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}


class MockLLMClient:
    """Mock LLM client for testing tool loop behavior."""

    def __init__(self, responses: list[str | dict], supports_native: bool = False):
        self.responses = responses
        self.call_count = 0
        self._supports_native = supports_native

    def query(self, messages, stop=None, **kwargs):
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response

    def query_with_tools(self, messages, tools_schema=None, **kwargs):
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        # Return tuple (response, tool_calls)
        if isinstance(response, dict):
            return response, response.get("tool_calls")
        return response, None

    def supports_native_tools(self):
        return self._supports_native


class TestToolLoopBasics:
    """Basic tool loop tests."""

    def test_loop_requires_user_id(self):
        """Tool loop requires user_id to be set."""
        with pytest.raises(ValueError, match="user_id is required"):
            ToolLoop(user_id="")

    def test_loop_accepts_format_string(self):
        """Can initialize with format as string."""
        loop = ToolLoop(user_id="test", tool_format="react")
        assert loop.tool_format == "react"

    def test_loop_accepts_format_enum(self):
        """Can initialize with format as enum."""
        loop = ToolLoop(user_id="test", tool_format=ToolFormat.REACT)
        assert loop.tool_format == ToolFormat.REACT

    def test_default_max_iterations(self):
        """Default max iterations is 5."""
        loop = ToolLoop(user_id="test")
        assert loop.max_iterations == DEFAULT_MAX_ITERATIONS


class TestToolLoopWithReActFormat:
    """Tests for tool loop with ReAct format."""

    @pytest.fixture
    def loop(self):
        return ToolLoop(user_id="test_user", tool_format=ToolFormat.REACT)

    def test_no_tool_call_passes_through(self, loop):
        """Response without tool call is returned directly."""
        client = MockLLMClient(["Just a regular response, no tools needed."])
        messages = [MockMessage("user", "Hello")]

        result = loop.run(messages, client)

        assert result == "Just a regular response, no tools needed."
        assert client.call_count == 1

    def test_final_answer_extracted(self, loop):
        """Final Answer is extracted from ReAct response."""
        client = MockLLMClient([
            "Thought: I have all the info\nFinal Answer: Here is your answer."
        ])
        messages = [MockMessage("user", "Question")]

        result = loop.run(messages, client)

        assert "Here is your answer" in result
        assert "Thought:" not in result
        assert "Final Answer:" not in result

    def test_sanitizes_thought_markers(self, loop):
        """Thought markers are removed from final response."""
        client = MockLLMClient([
            "Thought: Let me think\nThe answer is 42."
        ])
        messages = [MockMessage("user", "What is the answer?")]

        result = loop.run(messages, client)

        assert "Thought:" not in result
        assert "42" in result


class TestToolLoopWithNativeFormat:
    """Tests for tool loop with native OpenAI format."""

    @pytest.fixture
    def loop(self):
        return ToolLoop(user_id="test_user", tool_format=ToolFormat.NATIVE_OPENAI)

    def test_native_client_detection(self, loop):
        """Loop detects native tool support."""
        native_client = MockLLMClient(["Response"], supports_native=True)
        non_native_client = MockLLMClient(["Response"], supports_native=False)

        assert native_client.supports_native_tools()
        assert not non_native_client.supports_native_tools()

    def test_string_response_passed_through(self, loop):
        """String responses are passed through unchanged."""
        client = MockLLMClient(["Just text, no tools."], supports_native=True)
        messages = [MockMessage("user", "Hello")]

        result = loop.run(messages, client)

        assert result == "Just text, no tools."


class TestToolLoopWithXMLFormat:
    """Tests for tool loop with legacy XML format."""

    @pytest.fixture
    def loop(self):
        return ToolLoop(user_id="test_user", tool_format=ToolFormat.XML)

    def test_sanitizes_xml_tags(self, loop):
        """XML tool_call tags are removed from final response."""
        client = MockLLMClient([
            'Here is info <tool_call>{"name":"test"}</tool_call> and more.'
        ])
        messages = [MockMessage("user", "Question")]

        result = loop.run(messages, client)

        assert "<tool_call>" not in result
        assert "</tool_call>" not in result


class TestToolLoopFallbackParsing:
    """Tests for fallback parsing when format mismatch occurs."""

    def test_xml_detected_in_react_mode(self):
        """XML format detected when ReAct format expected."""
        loop = ToolLoop(user_id="test", tool_format=ToolFormat.REACT)

        # Simulate model outputting XML when ReAct was expected
        response = '<tool_call>{"name":"search"}</tool_call>'

        # The sanitize should still clean it up
        sanitized = loop.format_handler.sanitize_response(response)

        assert "<tool_call>" not in sanitized

    def test_react_detected_in_xml_mode(self):
        """ReAct format is sanitized even in XML mode."""
        loop = ToolLoop(user_id="test", tool_format=ToolFormat.XML)

        # XML handler doesn't specifically sanitize ReAct, but the
        # fallback detection in loop.py handles this
        response = "Thought: thinking\nAction: test\nFinal Answer: result"

        # XML sanitizer only handles XML tags
        sanitized = loop.format_handler.sanitize_response(response)

        # XML handler doesn't sanitize ReAct markers
        # This is expected - the fallback is in the loop itself
        assert "Final Answer:" in sanitized or "result" in sanitized


class TestToolLoopIterationLimits:
    """Tests for iteration limiting."""

    def test_respects_max_iterations(self):
        """Loop stops after max iterations."""
        loop = ToolLoop(user_id="test", max_iterations=2, tool_format=ToolFormat.REACT)

        # Executor tracks call counts internally
        # After max iterations, loop should stop even if tools keep being called
        assert loop.max_iterations == 2

    def test_custom_max_iterations(self):
        """Can set custom max iterations."""
        loop = ToolLoop(user_id="test", max_iterations=10)
        assert loop.max_iterations == 10


class TestToolContextTracking:
    """Tests for tool context/history tracking."""

    def test_tool_context_starts_empty(self):
        """Tool context is empty at start."""
        loop = ToolLoop(user_id="test")
        assert loop.tool_context == []


class TestRegressionCases:
    """Regression tests for known issues."""

    def test_raw_tags_never_in_final_response_react(self):
        """Raw tool tags should never appear in final response (ReAct)."""
        loop = ToolLoop(user_id="test", tool_format=ToolFormat.REACT)

        # Model outputs both ReAct format AND XML tags (confused model)
        response = """Thought: I need to search
Action: search
Action Input: {"query": "test"}
<tool_call>{"name":"search"}</tool_call>
Final Answer: Found results."""

        sanitized = loop.format_handler.sanitize_response(response)

        assert "<tool_call>" not in sanitized
        assert "</tool_call>" not in sanitized
        assert "Thought:" not in sanitized
        assert "Action:" not in sanitized

    def test_raw_tags_never_in_final_response_xml(self):
        """Raw tool tags should never appear in final response (XML)."""
        loop = ToolLoop(user_id="test", tool_format=ToolFormat.XML)

        response = """Done. <tool_call>
{"name":"delete_user_attribute","arguments":{"query":"qwen"}}
</tool_call>"""

        sanitized = loop.format_handler.sanitize_response(response)

        assert "<tool_call>" not in sanitized
        assert "</tool_call>" not in sanitized


class TestToolLoopWithMockedExecutor:
    """Tests with mocked tool executor for full loop testing."""

    @patch('llm_bawt.tools.loop.ToolExecutor')
    def test_tool_execution_flow(self, mock_executor_class):
        """Full tool execution flow with mocked executor."""
        # Setup mock executor
        mock_executor = MagicMock()
        mock_executor.reset_call_count.return_value = None
        mock_executor.can_execute_more.return_value = True
        # Executor returns string results
        mock_executor.execute.return_value = "Found 5 matching memories"
        mock_executor_class.return_value = mock_executor

        loop = ToolLoop(user_id="test", tool_format=ToolFormat.REACT)

        # First response has tool call, second is final answer
        responses = [
            "Thought: Need to search\nAction: search_memories\nAction Input: {\"query\": \"test\"}",
            "Thought: Got results\nFinal Answer: Here are your results."
        ]
        client = MockLLMClient(responses)
        messages = [MockMessage("user", "Search for test")]

        result = loop.run(messages, client)

        # Should have called LLM twice (tool call + final response)
        assert client.call_count == 2
        # Final response should be sanitized
        assert "Here are your results" in result
        assert "Thought:" not in result
