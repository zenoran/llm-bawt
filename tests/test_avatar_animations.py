"""Tests for avatar animation tool injection."""

import pytest
from unittest.mock import MagicMock

from llm_bawt.service.animation_tool import (
    AvatarAnimation,
    AvatarAnimationStore,
    build_trigger_animation_tool,
    _DEFAULT_ANIMATIONS,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_anim(name, description="Use when testing", sort_order=0, enabled=True):
    return AvatarAnimation(
        name=name,
        description=description,
        sort_order=sort_order,
        enabled=enabled,
    )


# ---------------------------------------------------------------------------
# build_trigger_animation_tool
# ---------------------------------------------------------------------------

class TestBuildTriggerAnimationTool:
    def _build(self, *names):
        anims = [_make_anim(n) for n in names]
        return build_trigger_animation_tool(anims)

    def test_function_name(self):
        tool = self._build("Head Nod Yes", "Weight Shift")
        assert tool["function"]["name"] == "trigger_animation"

    def test_type_is_function(self):
        tool = self._build("Head Nod Yes")
        assert tool["type"] == "function"

    def test_enum_contains_all_names(self):
        tool = self._build("Head Nod Yes", "Shaking Head No", "Weight Shift")
        enum = tool["function"]["parameters"]["properties"]["name"]["enum"]
        assert "Head Nod Yes" in enum
        assert "Shaking Head No" in enum
        assert "Weight Shift" in enum

    def test_enum_count_matches_input(self):
        names = ["Head Nod Yes", "Shaking Head No", "Weight Shift"]
        tool = self._build(*names)
        enum = tool["function"]["parameters"]["properties"]["name"]["enum"]
        assert len(enum) == len(names)

    def test_description_includes_animation_names(self):
        tool = self._build("Happy Hand Gesture", "Angry Gesture")
        desc = tool["function"]["description"]
        assert "Happy Hand Gesture" in desc
        assert "Angry Gesture" in desc

    def test_description_says_must_call(self):
        tool = self._build("Head Nod Yes")
        assert "MUST call this exactly once" in tool["function"]["description"]

    def test_description_mentions_weight_shift_fallback(self):
        tool = self._build("Head Nod Yes", "Weight Shift")
        assert "Weight Shift" in tool["function"]["description"]

    def test_name_field_required(self):
        tool = self._build("Head Nod Yes")
        required = tool["function"]["parameters"]["required"]
        assert "name" in required

    def test_enabled_only_in_enum(self):
        """Disabled animations should never appear — build_trigger_animation_tool
        receives only the list passed to it; callers filter by enabled."""
        # Pass only enabled ones (store.list_enabled() does the filtering)
        tool = build_trigger_animation_tool([
            _make_anim("Head Nod Yes", enabled=True),
        ])
        enum = tool["function"]["parameters"]["properties"]["name"]["enum"]
        assert enum == ["Head Nod Yes"]

    def test_animation_descriptions_in_tool_description(self):
        anims = [_make_anim("Head Nod Yes", description="Use when agreeing")]
        tool = build_trigger_animation_tool(anims)
        assert "Use when agreeing" in tool["function"]["description"]

    def test_empty_list_produces_empty_enum(self):
        tool = build_trigger_animation_tool([])
        enum = tool["function"]["parameters"]["properties"]["name"]["enum"]
        assert enum == []


# ---------------------------------------------------------------------------
# AvatarAnimation model defaults
# ---------------------------------------------------------------------------

class TestAvatarAnimationModel:
    def test_defaults(self):
        anim = AvatarAnimation(name="Test")
        assert anim.enabled is True
        assert anim.sort_order == 0
        assert anim.description is None

    def test_custom_sort_order(self):
        anim = AvatarAnimation(name="Test", sort_order=5)
        assert anim.sort_order == 5

    def test_no_keywords_field(self):
        anim = AvatarAnimation(name="Test")
        assert not hasattr(anim, "keywords") or anim.keywords is None


# ---------------------------------------------------------------------------
# Seed data sanity
# ---------------------------------------------------------------------------

class TestDefaultAnimations:
    def test_seed_count(self):
        assert len(_DEFAULT_ANIMATIONS) == 15

    def test_all_have_names_and_descriptions(self):
        for entry in _DEFAULT_ANIMATIONS:
            assert entry["name"], "Missing name"
            assert entry["description"], f"Missing description for {entry['name']}"

    def test_names_unique(self):
        names = [e["name"] for e in _DEFAULT_ANIMATIONS]
        assert len(names) == len(set(names))

    def test_weight_shift_present_as_fallback(self):
        names = [e["name"] for e in _DEFAULT_ANIMATIONS]
        assert "Weight Shift" in names

    def test_sort_orders_are_sequential(self):
        orders = sorted(e["sort_order"] for e in _DEFAULT_ANIMATIONS)
        assert orders == list(range(1, len(_DEFAULT_ANIMATIONS) + 1))
