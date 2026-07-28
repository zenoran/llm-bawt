"""TASK-661: the bridge→app seam, wired for real.

The tracker (bridge side) and the store (app side) are developed in separate
processes that never import each other — the only thing holding them together
is the JSON manifest shape. That contract is exactly the sort of thing that
drifts silently, so this exercises the whole chain with no mocks in the middle:

    real git repo -> ChangedFileTracker -> manifest dict
                  -> AgentEvent round-trip (Redis wire format)
                  -> decode_manifest_files -> ChangedFilesStore
                  -> build_turn_summary (what the chat UI renders)
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest
from sqlmodel import SQLModel, create_engine

from agent_bridge.changed_files import ChangedFileTracker
from agent_bridge.events import AgentEvent, AgentEventKind
from llm_bawt.service.changed_files_store import (
    ChangedFilesStore,
    TurnChangedFile,
    TurnDiffPayloadRecord,
    decode_manifest_files,
    reset_diff_blob_backend,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "ws" / "myrepo"
    r.mkdir(parents=True)
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "T")
    (r / "app.py").write_text("print('one')\n")
    (r / "doomed.py").write_text("delete me\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "init")
    return r


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_BAWT_DIFFBLOBS_FS_ROOT", str(tmp_path / "diffs"))
    monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "fs")
    reset_diff_blob_backend()
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(
        engine, tables=[TurnChangedFile.__table__, TurnDiffPayloadRecord.__table__]
    )
    yield ChangedFilesStore(engine)
    reset_diff_blob_backend()


def _over_the_wire(manifest: dict) -> dict:
    """Push the manifest through the exact AgentEvent serialization the Redis
    transport uses, so an unencodable field would fail here rather than in prod."""
    event = AgentEvent(
        event_id="e1",
        session_key="claude-code:snark",
        run_id="run-1",
        kind=AgentEventKind.CHANGED_FILES,
        origin="system",
        trigger_message_id="msg-abc",
        raw=manifest,
    )
    import json

    revived = AgentEvent.from_dict(json.loads(json.dumps(event.to_dict())))
    assert revived.kind is AgentEventKind.CHANGED_FILES
    return revived.raw


def test_full_chain_tracker_to_rendered_summary(repo: Path, store: ChangedFilesStore):
    async def go():
        tracker = ChangedFileTracker(roots=[repo.parent])
        await tracker.start()
        (repo / "app.py").write_text("print('one')\nprint('two')\n")
        (repo / "new_module.py").write_text("fresh\n")
        (repo / "doomed.py").unlink()
        return await tracker.finalize()

    manifest = _over_the_wire(asyncio.run(go()))

    saved = store.save_turn_files(
        turn_id="turn-1",
        bot_id="snark",
        user_id="nick",
        trigger_message_id="msg-abc",
        files=decode_manifest_files(manifest["files"]),
    )
    assert saved == 3

    summary = store.summary_for_turn("turn-1")
    assert summary["total_files"] == 3
    assert summary["total_additions"] == 2  # +1 app.py, +1 new_module.py
    assert summary["total_deletions"] == 1  # -1 doomed.py (app.py only appended)

    by_path = {f["path"]: f for f in summary["files"]}
    assert by_path["app.py"]["change_kind"] == "modified"
    assert by_path["new_module.py"]["change_kind"] == "added"
    assert by_path["doomed.py"]["change_kind"] == "deleted"
    # Every text file must be openable in the diff modal.
    assert all(f["has_content"] for f in summary["files"])

    # The bytes that come back are the bytes that were on disk for this turn.
    before, after = store.get_file_content(
        turn_id="turn-1", repo_key="myrepo", path="app.py"
    )
    assert before.text == "print('one')\n"
    assert after.text == "print('one')\nprint('two')\n"

    # A deleted file has a before and an empty after (not a missing row).
    before, after = store.get_file_content(
        turn_id="turn-1", repo_key="myrepo", path="doomed.py"
    )
    assert before.text == "delete me\n"
    assert after.text == ""


def test_trigger_keyed_lookup_matches_live_event_shape(
    repo: Path, store: ChangedFilesStore
):
    """History hydration keys by trigger_message_id; it must produce byte-identical
    output to the turn-keyed summary the live turn_complete event carries."""

    async def go():
        tracker = ChangedFileTracker(roots=[repo.parent])
        await tracker.start()
        (repo / "app.py").write_text("changed\n")
        return await tracker.finalize()

    manifest = _over_the_wire(asyncio.run(go()))
    store.save_turn_files(
        turn_id="turn-9",
        bot_id="snark",
        user_id="nick",
        trigger_message_id="msg-xyz",
        files=decode_manifest_files(manifest["files"]),
    )

    live = store.summary_for_turn("turn-9")
    hydrated = store.summaries_for_triggers(["msg-xyz"])["msg-xyz"]
    assert live == hydrated


def test_rename_survives_the_whole_chain(repo: Path, store: ChangedFilesStore):
    async def go():
        tracker = ChangedFileTracker(roots=[repo.parent])
        await tracker.start()
        (repo / "app.py").rename(repo / "renamed_app.py")
        return await tracker.finalize()

    manifest = _over_the_wire(asyncio.run(go()))
    store.save_turn_files(
        turn_id="turn-r",
        bot_id="snark",
        user_id="nick",
        trigger_message_id="msg-r",
        files=decode_manifest_files(manifest["files"]),
    )
    entry = next(
        f for f in store.summary_for_turn("turn-r")["files"]
        if f["path"] == "renamed_app.py"
    )
    assert entry["change_kind"] == "renamed"
    assert entry["old_path"] == "app.py"


def test_binary_file_is_stored_without_content(repo: Path, store: ChangedFilesStore):
    async def go():
        tracker = ChangedFileTracker(roots=[repo.parent])
        await tracker.start()
        (repo / "icon.bin").write_bytes(bytes(range(256)) * 4)
        return await tracker.finalize()

    manifest = _over_the_wire(asyncio.run(go()))
    store.save_turn_files(
        turn_id="turn-b",
        bot_id="snark",
        user_id="nick",
        trigger_message_id="msg-b",
        files=decode_manifest_files(manifest["files"]),
    )
    entry = next(
        f for f in store.summary_for_turn("turn-b")["files"] if f["path"] == "icon.bin"
    )
    assert entry["binary"] is True
    # Binary rows render as non-clickable — no diff to open.
    assert entry["has_content"] is False


def test_resaving_a_turn_is_idempotent(repo: Path, store: ChangedFilesStore):
    """A retried turn must overwrite its rows, not duplicate them."""

    async def go():
        tracker = ChangedFileTracker(roots=[repo.parent])
        await tracker.start()
        (repo / "app.py").write_text("v2\n")
        return await tracker.finalize()

    manifest = _over_the_wire(asyncio.run(go()))
    files = decode_manifest_files(manifest["files"])
    for _ in range(3):
        store.save_turn_files(
            turn_id="turn-dup",
            bot_id="snark",
            user_id="nick",
            trigger_message_id="msg-dup",
            files=files,
        )
    assert store.summary_for_turn("turn-dup")["total_files"] == 1


def test_empty_turn_produces_no_manifest_and_no_rows(
    repo: Path, store: ChangedFilesStore
):
    """A turn that only talks must not leave an empty changed-files row behind —
    the UI gates on total_files > 0."""

    async def go():
        tracker = ChangedFileTracker(roots=[repo.parent])
        await tracker.start()
        return await tracker.finalize()

    assert asyncio.run(go()) is None
    assert store.summary_for_turn("turn-empty")["total_files"] == 0
