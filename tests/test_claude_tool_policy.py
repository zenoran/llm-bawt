"""Regression tests for the data-driven Claude SDK disallowed-tools policy."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from claude_code_bridge.tool_policy import (
    CLAUDE_CODE_DISALLOWED_TOOLS_KEY,
    DEFAULT_DISALLOWED_TOOLS,
    effective_disallowed_tools,
)


class TestToolPolicy:
    def test_default_disables_plan_and_worktree_workflows(self):
        assert effective_disallowed_tools(None, use_proxy=False) == list(
            DEFAULT_DISALLOWED_TOOLS
        )

    def test_configured_policy_replaces_default_and_deduplicates(self):
        raw = json.dumps(["Bash", "Read", "Bash"])
        assert effective_disallowed_tools(raw, use_proxy=False) == ["Bash", "Read"]

    def test_explicit_empty_list_is_respected(self):
        assert effective_disallowed_tools("[]", use_proxy=False) == []

    def test_proxy_additions_are_mandatory_and_deduplicated(self):
        raw = json.dumps(["EnterPlanMode", "WebSearch"])
        assert effective_disallowed_tools(raw, use_proxy=True) == [
            "EnterPlanMode",
            "WebSearch",
            "WebFetch",
        ]

    @pytest.mark.parametrize("raw", ["not-json", json.dumps("Bash"), json.dumps([1])])
    def test_malformed_transport_falls_back_to_default(self, raw):
        assert effective_disallowed_tools(raw, use_proxy=False) == list(
            DEFAULT_DISALLOWED_TOOLS
        )


class TestSettingDefinition:
    def test_setting_is_registered_as_string_list(self):
        from llm_bawt.setting_definitions import (
            SETTING_DEFINITIONS,
            STORAGE_RUNTIME_SETTING,
        )

        definition = SETTING_DEFINITIONS[CLAUDE_CODE_DISALLOWED_TOOLS_KEY]
        assert definition.type == "string_list"
        assert definition.storage == STORAGE_RUNTIME_SETTING
        assert definition.default == list(DEFAULT_DISALLOWED_TOOLS)


class TestGlobalResolution:
    def test_resolver_ignores_bot_scope(self, monkeypatch):
        from llm_bawt import runtime_settings
        from llm_bawt.runtime_setting_resolution import resolve_global_runtime_setting
        from llm_bawt.utils.config import Config

        captured: dict = {}

        class FakeResolver:
            def __init__(self, config, bot=None):
                captured["bot"] = bot

            def resolve(self, key, fallback=None):
                captured["key"] = key
                captured["fallback"] = fallback
                return fallback

        monkeypatch.setattr(runtime_settings, "RuntimeSettingsResolver", FakeResolver)
        result = resolve_global_runtime_setting(
            Config(), CLAUDE_CODE_DISALLOWED_TOOLS_KEY
        )

        assert captured["bot"] is None
        assert result == list(DEFAULT_DISALLOWED_TOOLS)


class TestTransport:
    @pytest.mark.asyncio
    async def test_subscriber_json_encodes_explicit_empty_list(self):
        from agent_bridge.subscriber import RedisSubscriber

        subscriber = RedisSubscriber("redis://localhost:6379/0")
        captured: dict = {}

        async def fake_xadd(_stream, fields, **_kwargs):
            captured.update(fields)
            return b"1"

        subscriber._pub_redis = SimpleNamespace(xadd=fake_xadd)
        await subscriber.send_command(
            session_key="test-session",
            message="hello",
            request_id="req-test",
            disallowed_tools=[],
        )

        assert captured["disallowed_tools"] == "[]"

    def test_agent_bridge_forwards_configured_list(self, monkeypatch):
        from llm_bawt.agent_backends import agent_bridge as module

        captured: dict = {}

        class FakeRedisSubscriber:
            def __init__(self, _url):
                self._redis = SimpleNamespace(
                    connection_pool=SimpleNamespace(
                        connection_kwargs={"url": "redis://localhost:6379/0"}
                    )
                )

            async def connect(self):
                return None

            async def send_command(self, **kwargs):
                captured.update(kwargs)

            async def subscribe_run(self, *_args, **_kwargs):
                if False:
                    yield None

            async def close(self):
                return None

        root_subscriber = SimpleNamespace(
            _redis=SimpleNamespace(
                connection_pool=SimpleNamespace(
                    connection_kwargs={"url": "redis://localhost:6379/0"}
                )
            )
        )
        monkeypatch.setattr(module, "get_agent_subscriber", lambda: root_subscriber)
        monkeypatch.setattr(
            "agent_bridge.subscriber.RedisSubscriber", FakeRedisSubscriber
        )

        backend = module.AgentBridgeBackend()
        list(
            backend.stream_raw(
                "hello",
                {
                    "bot_id": "test",
                    "disallowed_tools": ["EnterPlanMode"],
                    "timeout_seconds": 1,
                },
            )
        )
        assert captured["disallowed_tools"] == ["EnterPlanMode"]


class TestClaudeBackendResolution:
    def test_claude_backend_adds_resolved_policy_before_dispatch(self):
        from llm_bawt.agent_backends.claude_code import ClaudeCodeBackend

        backend = ClaudeCodeBackend()
        configured = ["EnterPlanMode", "ExitWorktree"]

        with (
            patch(
                "llm_bawt.runtime_setting_resolution.resolve_global_runtime_setting",
                return_value=configured,
            ),
            patch(
                "llm_bawt.agent_backends.agent_bridge.AgentBridgeBackend.stream_raw",
                return_value=iter(()),
            ) as parent_stream,
        ):
            list(
                backend.stream_raw(
                    "hello",
                    {"model": "claude-opus-4-7", "bot_id": "loopy"},
                )
            )

        dispatched = parent_stream.call_args.args[1]
        assert dispatched["disallowed_tools"] == configured
