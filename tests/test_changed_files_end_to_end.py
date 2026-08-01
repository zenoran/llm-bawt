from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import SQLModel, create_engine

from agent_bridge.changed_files import ToolChangedFileCapture
from llm_bawt.service.changed_files_store import (
    TurnChangedFile,
    TurnDiffPayloadRecord,
    reset_diff_blob_backend,
)
from llm_bawt.service.tool_changed_files import ToolChangedFilesCoordinator


@pytest.fixture()
def coordinator(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_BAWT_DIFFBLOBS_FS_ROOT", str(tmp_path / "diffs"))
    monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "fs")
    reset_diff_blob_backend()
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(
        engine, tables=[TurnChangedFile.__table__, TurnDiffPayloadRecord.__table__]
    )
    yield ToolChangedFilesCoordinator(engine)
    reset_diff_blob_backend()


def test_tool_capture_to_durable_summary_and_diff(tmp_path: Path, coordinator):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    path = repo / "app.py"
    path.write_text("print('one')\n")

    capture = ToolChangedFileCapture(cwd=repo)
    capture.start(request_id="turn-1", tool_use_id="tool-1", tool_name="Edit", arguments={"file_path": str(path)})
    path.write_text("print('one')\nprint('two')\n")
    finished = capture.finish(request_id="turn-1", tool_use_id="tool-1", tool_name="Edit")
    assert finished is not None

    persisted = coordinator.persist({
        "turn_id": "turn-1",
        "bot_id": "snark",
        "user_id": "nick",
        "trigger_message_id": "msg-1",
        "tool_use_id": "tool-1",
        "tool_name": finished[0],
        "arguments": finished[1],
        "file": finished[2],
    })
    assert persisted is not None
    assert persisted["summary"]["total_files"] == 1
    assert persisted["summary"]["total_additions"] == 1
    assert persisted["file"]["has_content"] is True

    repo_key = persisted["file"]["repo_key"]
    before, after = coordinator.store.get_file_content(
        turn_id="turn-1", repo_key=repo_key, path=str(path)
    )
    assert before.text == "print('one')\n"
    assert after.text == "print('one')\nprint('two')\n"


def test_overlapping_turns_keep_only_their_declared_files(tmp_path: Path, coordinator):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    a = repo / "a.py"
    b = repo / "b.py"
    a.write_text("a0\n")
    b.write_text("b0\n")

    first = ToolChangedFileCapture(cwd=repo)
    second = ToolChangedFileCapture(cwd=repo)
    first.start(request_id="turn-a", tool_use_id="tool-a", tool_name="Edit", arguments={"file_path": str(a)})
    second.start(request_id="turn-b", tool_use_id="tool-b", tool_name="Edit", arguments={"file_path": str(b)})
    a.write_text("a1\n")
    b.write_text("b1\n")

    for turn_id, tool_id, capture, tool_name in (
        ("turn-a", "tool-a", first, "Edit"),
        ("turn-b", "tool-b", second, "Edit"),
    ):
        finished = capture.finish(request_id=turn_id, tool_use_id=tool_id, tool_name=tool_name)
        assert finished is not None
        assert coordinator.persist({
            "turn_id": turn_id,
            "bot_id": "snark",
            "user_id": "nick",
            "trigger_message_id": f"msg-{turn_id}",
            "tool_use_id": tool_id,
            "tool_name": finished[0],
            "arguments": finished[1],
            "file": finished[2],
        }) is not None

    assert [f["path"] for f in coordinator.store.summary_for_turn("turn-a")["files"]] == [str(a)]
    assert [f["path"] for f in coordinator.store.summary_for_turn("turn-b")["files"]] == [str(b)]


def test_metadata_fallback_uses_completed_tool_arguments(coordinator):
    persisted = coordinator.persist({
        "turn_id": "turn-meta",
        "bot_id": "snark",
        "user_id": "nick",
        "trigger_message_id": "msg-meta",
        "tool_use_id": None,
        "tool_name": "Write",
        "arguments": {"file_path": "/workspace/generated.py"},
    })
    assert persisted is not None
    assert persisted["file"]["path"] == "/workspace/generated.py"
    assert persisted["file"]["has_content"] is False


def test_bridge_content_enriches_metadata_first_row(tmp_path: Path, coordinator):
    path = tmp_path / "enriched.py"
    path.write_text("before\n")
    capture = ToolChangedFileCapture(cwd=tmp_path)
    capture.start(
        request_id="turn-enrich",
        tool_use_id="tool-enrich",
        tool_name="Edit",
        arguments={"file_path": str(path)},
    )
    path.write_text("after\n")

    first = coordinator.persist({
        "turn_id": "turn-enrich",
        "bot_id": "snark",
        "user_id": "nick",
        "trigger_message_id": "msg-enrich",
        "tool_use_id": "tool-enrich",
        "tool_name": "Edit",
        "arguments": {"file_path": str(path)},
    })
    assert first is not None
    assert first["file"]["has_content"] is False

    finished = capture.finish(
        request_id="turn-enrich", tool_use_id="tool-enrich", tool_name="Edit"
    )
    assert finished is not None
    second = coordinator.persist({
        "turn_id": "turn-enrich",
        "bot_id": "snark",
        "user_id": "nick",
        "trigger_message_id": "msg-enrich",
        "tool_use_id": "tool-enrich",
        "tool_name": finished[0],
        "arguments": finished[1],
        "file": finished[2],
    })
    assert second is not None
    assert second["summary"]["total_files"] == 1
    assert second["file"]["has_content"] is True
    before, after = coordinator.store.get_file_content(
        turn_id="turn-enrich", repo_key="workspace", path=str(path)
    )
    assert before.text == "before\n"
    assert after.text == "after\n"
