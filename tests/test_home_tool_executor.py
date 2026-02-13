"""Tests for Home Assistant tool execution behavior."""

from llm_bawt.tools.executor import ToolExecutor
from llm_bawt.tools.parser import ToolCall


class _FakeHomeClient:
    def __init__(self):
        self.set_calls: list[tuple[str, str, int | None]] = []
        self.query_calls: list[tuple[str | None, str | None]] = []

    def status(self) -> str:
        return "ok"

    def query(self, pattern: str | None = None, domain: str | None = None) -> str:
        self.query_calls.append((pattern, domain))
        if pattern and "sun" in pattern.lower():
            return "Found entities:\n- light.all_sunroom_lights"
        return "No entities found"

    def get(self, entity: str) -> str:
        return f"State for {entity}: on"

    def set(self, entity: str, state: str, brightness: int | None = None) -> str:
        self.set_calls.append((entity, state, brightness))
        if entity == "light.all_sunroom_lights":
            return "Set light.all_sunroom_lights to on"
        return f"Entity '{entity}' not found"

    def scene(self, name: str) -> str:
        return f"Activated scene {name}"


def test_home_set_auto_resolves_guessed_entity() -> None:
    home = _FakeHomeClient()
    executor = ToolExecutor(user_id="test-user", home_client=home)

    result = executor.execute(
        ToolCall(
            name="home",
            arguments={"action": "set", "entity": "light.sun_room_lamp_1", "state": "on"},
            raw_text="",
        )
    )

    assert 'status="success"' in result
    assert "Set light.all_sunroom_lights to on" in result
    # Initial attempt + retry with resolved entity
    assert len(home.set_calls) == 2
    assert home.set_calls[1][0] == "light.all_sunroom_lights"


def test_home_set_not_found_returns_error_with_guidance() -> None:
    home = _FakeHomeClient()
    executor = ToolExecutor(user_id="test-user", home_client=home)

    result = executor.execute(
        ToolCall(
            name="home",
            arguments={"action": "set", "entity": "light.unknown_device", "state": "on"},
            raw_text="",
        )
    )

    assert 'status="error"' in result
    assert "Call home with action='query' first" in result
