"""Bridge-neutral changed-file lifecycle coverage (TASK-661)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from agent_bridge.changed_file_lifecycle import ChangedFileLifecycleMixin
from agent_bridge.events import AgentEventKind
from codex_bridge.bridge import CodexBridge
from openclaw_bridge.bridge import SessionBridge


class _Tracker:
    async def finalize(self):
        return {
            "files": [{"repo_key": "r", "path": "a.py"}],
            "overlapping_repos": ["r"],
            "truncated": True,
        }


class _Bridge(ChangedFileLifecycleMixin):
    def __init__(self):
        self._publisher = MagicMock()
        self._backend_name = "test-bridge"
        self._trigger_message_ids = {"req": "msg"}


def test_shared_lifecycle_publishes_manifest_event():
    bridge = _Bridge()
    seq = asyncio.run(bridge._publish_changed_files(
        _Tracker(), request_id="req", session_key="s", seq=7,
    ))
    assert seq == 8
    event = bridge._publisher.publish_run_event.call_args.args[1]
    assert event.kind is AgentEventKind.CHANGED_FILES
    assert event.raw["overlapping_repos"] == ["r"]
    assert event.raw["truncated"] is True
    assert event.provider == "test-bridge"
    assert event.trigger_message_id == "msg"


def test_active_bridge_classes_share_the_lifecycle_contract():
    assert issubclass(CodexBridge, ChangedFileLifecycleMixin)
    assert issubclass(SessionBridge, ChangedFileLifecycleMixin)
