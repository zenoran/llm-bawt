"""Unit tests for OpenClaw bridge-based backend."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_bawt.agent_backends.openclaw import (
    OpenClawBackend,
    OpenClawToolCall,
    _friendly_tool_name,
    set_openclaw_subscriber,
    get_openclaw_subscriber,
)


def test_friendly_tool_name_exec():
    assert _friendly_tool_name("exec", {"command": "find /tmp -name foo"}) == "find"
    assert _friendly_tool_name("exec", {"command": "/usr/bin/grep pattern"}) == "grep"
    assert _friendly_tool_name("exec", {}) == "exec"


def test_friendly_tool_name_passthrough():
    assert _friendly_tool_name("search_web", {"query": "ai"}) == "search_web"


def test_tool_call_display_name():
    tc = OpenClawToolCall(name="exec", arguments={"command": "curl http://example.com"})
    assert tc.display_name == "curl"


def test_subscriber_module_singleton():
    """Test that set/get_openclaw_subscriber works."""
    original = get_openclaw_subscriber()
    try:
        mock_sub = MagicMock()
        set_openclaw_subscriber(mock_sub)
        assert get_openclaw_subscriber() is mock_sub
    finally:
        set_openclaw_subscriber(original)


def test_stream_raw_no_subscriber():
    """stream_raw raises when no subscriber is set."""
    original = get_openclaw_subscriber()
    try:
        set_openclaw_subscriber(None)
        backend = OpenClawBackend()
        with pytest.raises(RuntimeError, match="subscriber not initialized"):
            list(backend.stream_raw("hello", {}))
    finally:
        set_openclaw_subscriber(original)


def test_health_check_no_subscriber():
    """health_check returns False when no subscriber."""
    original = get_openclaw_subscriber()
    try:
        set_openclaw_subscriber(None)
        backend = OpenClawBackend()
        result = asyncio.run(backend.health_check({}))
        assert result is False
    finally:
        set_openclaw_subscriber(original)


def test_health_check_with_subscriber():
    """health_check returns True when subscriber is connected."""
    original = get_openclaw_subscriber()
    try:
        mock_sub = MagicMock()
        mock_sub.connected = True
        set_openclaw_subscriber(mock_sub)
        backend = OpenClawBackend()
        result = asyncio.run(backend.health_check({}))
        assert result is True
    finally:
        set_openclaw_subscriber(original)


def test_resolve_session_key():
    backend = OpenClawBackend()
    assert backend._resolve_session_key({}) == "main"
    assert backend._resolve_session_key({"session_key": "agent"}) == "agent"
