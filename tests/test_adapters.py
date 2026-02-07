"""Tests for Model Adapter pattern.

This module tests:
- Adapter registration and retrieval
- PygmalionAdapter output cleaning
- DefaultAdapter passthrough behavior
- Stop sequence combination in streaming
"""

import pytest
from llmbothub.adapters import (
    ModelAdapter,
    DefaultAdapter,
    PygmalionAdapter,
    get_adapter,
    register_adapter,
)


class TestModelAdapter:
    """Test base ModelAdapter interface."""

    def test_default_adapter_name(self):
        """DefaultAdapter has correct name."""
        adapter = DefaultAdapter()
        assert adapter.name == "default"

    def test_default_adapter_passthrough(self):
        """DefaultAdapter returns input unchanged."""
        adapter = DefaultAdapter()
        test_input = "Hello [HUMAN] world"
        assert adapter.clean_output(test_input) == test_input
        assert adapter.get_stop_sequences() == []
        assert adapter.supports_system_role() is True

    def test_pygmalion_adapter_name(self):
        """PygmalionAdapter has correct name."""
        adapter = PygmalionAdapter()
        assert adapter.name == "pygmalion"

    def test_pygmalion_stop_sequences(self):
        """PygmalionAdapter returns expected stop sequences."""
        adapter = PygmalionAdapter()
        stops = adapter.get_stop_sequences()
        
        # Should include role markers
        assert "[HUMAN]" in stops
        assert "[/HUMAN]" in stops
        assert "[INST]" in stops
        assert "[/INST]" in stops
        
        # Should include instruction markers
        assert "### Instruction:" in stops
        assert "### Human:" in stops
        assert "<|im_start|>user" in stops


class TestPygmalionCleaning:
    """Test PygmalionAdapter output cleaning."""

    def test_removes_human_tags_with_content(self):
        """Removes [HUMAN]...[/HUMAN] blocks entirely (hallucinated turns)."""
        adapter = PygmalionAdapter()
        
        # Content inside HUMAN blocks is removed entirely (hallucinated user turn)
        assert adapter.clean_output("Hello [HUMAN]fake turn[/HUMAN] world") == "Hello  world"
        assert adapter.clean_output("[HUMAN]hallucinated[/HUMAN]") == ""

    def test_removes_standalone_human_tags(self):
        """Removes standalone [HUMAN] and [/HUMAN] tags (without matching close/open)."""
        adapter = PygmalionAdapter()
        
        # Unmatched opening tag - content after is preserved
        assert adapter.clean_output("[HUMAN]content") == "content"
        # Unmatched closing tag - content before is preserved
        assert adapter.clean_output("content[/HUMAN]") == "content"

    def test_removes_inst_tags_with_content(self):
        """Removes [INST]...[/INST] blocks entirely (hallucinated instruction blocks)."""
        adapter = PygmalionAdapter()
        
        # Content inside INST blocks is removed entirely
        assert adapter.clean_output("[INST]instruction[/INST]") == ""

    def test_removes_standalone_inst_tags(self):
        """Removes standalone [INST] and [/INST] tags (without matching close/open)."""
        adapter = PygmalionAdapter()
        
        # Unmatched opening tag - content after is preserved
        assert adapter.clean_output("[INST]prompt") == "prompt"

    def test_case_insensitive(self):
        """Tag removal is case insensitive."""
        adapter = PygmalionAdapter()
        
        # Complete blocks are removed entirely
        assert adapter.clean_output("[human]test[/human]") == ""
        assert adapter.clean_output("[Human]test[/Human]") == ""
        assert adapter.clean_output("[inst]test[/inst]") == ""

    def test_removes_bbcode_font(self):
        """Removes BBCode FONT tags."""
        adapter = PygmalionAdapter()
        
        assert adapter.clean_output("[FONT=Arial]text[/FONT]") == "text"
        assert adapter.clean_output("[font=12px]text[/font]") == "text"

    def test_removes_bbcode_bold(self):
        """Removes BBCode bold tags."""
        adapter = PygmalionAdapter()
        
        assert adapter.clean_output("[B]bold[/B]") == "bold"
        assert adapter.clean_output("[b]bold[/b]") == "bold"

    def test_removes_bbcode_italic(self):
        """Removes BBCode italic tags."""
        adapter = PygmalionAdapter()
        
        assert adapter.clean_output("[I]italic[/I]") == "italic"
        assert adapter.clean_output("[i]italic[/i]") == "italic"

    def test_removes_bbcode_color(self):
        """Removes BBCode color tags."""
        adapter = PygmalionAdapter()
        
        assert adapter.clean_output("[COLOR=red]text[/COLOR]") == "text"
        assert adapter.clean_output("[color=#ff0000]text[/color]") == "text"

    def test_cleans_excessive_whitespace(self):
        """Collapses 3+ newlines to 2."""
        adapter = PygmalionAdapter()
        
        assert adapter.clean_output("line1\n\n\nline2") == "line1\n\nline2"
        assert adapter.clean_output("line1\n\n\n\nline2") == "line1\n\nline2"

    def test_strips_whitespace(self):
        """Strips leading/trailing whitespace."""
        adapter = PygmalionAdapter()
        
        assert adapter.clean_output("  text  ") == "text"
        assert adapter.clean_output("\n\ntext\n\n") == "text"

    def test_removes_nested_role_content(self):
        """Removes content inside role blocks entirely."""
        adapter = PygmalionAdapter()
        
        result = adapter.clean_output("Before[HUMAN]fake turn[/HUMAN]After")
        assert "fake turn" not in result
        assert result == "BeforeAfter"

    def test_complex_response(self):
        """Handles complex real-world response."""
        adapter = PygmalionAdapter()
        
        response = """Hello! I'd be happy to help.

[B]Here's what I found:[/B]

[HUMAN]Can you tell me more?[/HUMAN]

[FONT=Arial]Some formatted text[/FONT]

That's all!"""
        
        cleaned = adapter.clean_output(response)
        
        # Should not contain BBCode or role markers
        assert "[B]" not in cleaned
        assert "[/B]" not in cleaned
        assert "[HUMAN]" not in cleaned
        assert "[/HUMAN]" not in cleaned
        assert "[FONT=Arial]" not in cleaned
        assert "[/FONT]" not in cleaned
        
        # Should preserve main content
        assert "Hello!" in cleaned
        assert "I'd be happy to help" in cleaned
        assert "Here's what I found:" in cleaned
        assert "Some formatted text" in cleaned
        assert "That's all!" in cleaned
        
        # Should remove content inside role blocks
        assert "Can you tell me more?" not in cleaned


class TestAdapterRegistry:
    """Test adapter registry functionality."""

    def test_get_adapter_default_no_model_def(self):
        """Returns DefaultAdapter when no model definition."""
        adapter = get_adapter("some_model", None)
        assert isinstance(adapter, DefaultAdapter)

    def test_get_adapter_default_no_adapter_field(self):
        """Returns DefaultAdapter when model def has no adapter field."""
        adapter = get_adapter("some_model", {"type": "gguf"})
        assert isinstance(adapter, DefaultAdapter)

    def test_get_adapter_pygmalion(self):
        """Returns PygmalionAdapter for pygmalion adapter name."""
        adapter = get_adapter("mythomax", {"adapter": "pygmalion"})
        assert isinstance(adapter, PygmalionAdapter)

    def test_get_adapter_mythomax_alias(self):
        """Returns PygmalionAdapter for mythomax adapter name (alias)."""
        adapter = get_adapter("mythomax", {"adapter": "mythomax"})
        assert isinstance(adapter, PygmalionAdapter)

    def test_get_adapter_unknown_uses_default(self):
        """Returns DefaultAdapter for unknown adapter name."""
        adapter = get_adapter("some_model", {"adapter": "unknown_adapter"})
        assert isinstance(adapter, DefaultAdapter)

    def test_register_custom_adapter(self):
        """Can register and retrieve custom adapter."""
        
        class CustomAdapter(ModelAdapter):
            name = "custom"
            
            def get_stop_sequences(self):
                return ["<STOP>"]
        
        register_adapter("custom", CustomAdapter)
        
        adapter = get_adapter("test", {"adapter": "custom"})
        assert isinstance(adapter, CustomAdapter)
        assert adapter.get_stop_sequences() == ["<STOP>"]


class TestStopSequenceCombination:
    """Test that stop sequences are properly combined in streaming."""

    def test_stop_sequences_unique_combination(self):
        """Handler and adapter stops are combined without duplicates."""
        from llmbothub.tools.formats.react import ReActFormatHandler
        
        handler = ReActFormatHandler()
        adapter = PygmalionAdapter()
        
        handler_stops = handler.get_stop_sequences()
        adapter_stops = adapter.get_stop_sequences()
        
        # Combine as done in streaming.py
        combined = list(set(handler_stops + adapter_stops))
        
        # Should include handler stops
        assert "\nObservation:" in combined
        
        # Should include adapter stops
        assert "[HUMAN]" in combined
        assert "[/HUMAN]" in combined


class TestModelAdapterBaseInterface:
    """Test base ModelAdapter ABC interface."""

    def test_abstract_methods_not_required(self):
        """Base class has default implementations."""
        
        # Can instantiate with just name
        class MinimalAdapter(ModelAdapter):
            name = "minimal"
        
        adapter = MinimalAdapter()
        assert adapter.get_stop_sequences() == []
        assert adapter.clean_output("test") == "test"
        assert adapter.supports_system_role() is True
        assert adapter.transform_messages(["msg"]) == ["msg"]

    def test_transform_messages_passthrough(self):
        """Default transform_messages returns messages unchanged."""
        adapter = DefaultAdapter()
        messages = ["msg1", "msg2"]
        assert adapter.transform_messages(messages) == messages
