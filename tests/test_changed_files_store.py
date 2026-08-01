"""Tests for per-turn changed-file persistence (TASK-661).

Exercises the store's save/read roundtrip against both the object-store blob
path (a temp FS backend) and the bounded Postgres fallback, plus the canonical
serializer shape used by both the live event and the read API.
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text
from sqlmodel import SQLModel, create_engine

from llm_bawt.service import changed_files_store as cfs
from llm_bawt.service.changed_files_store import (
    MAX_BYTES_PER_SIDE,
    MAX_FILES_PER_TURN,
    ChangedFileInput,
    ChangedFilesStore,
    TurnChangedFile,
    TurnDiffPayloadRecord,
    build_turn_summary,
    reset_diff_blob_backend,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    # Point the diff blob backend at a temp FS dir and clear the cache so the
    # store writes/reads content-addressed blobs on disk (the prod path).
    monkeypatch.setenv("LLM_BAWT_DIFFBLOBS_FS_ROOT", str(tmp_path / "diffs"))
    monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "fs")
    reset_diff_blob_backend()
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(
        engine,
        tables=[
            TurnChangedFile.__table__,
            TurnDiffPayloadRecord.__table__,
        ],
    )
    s = ChangedFilesStore(engine)
    yield s
    reset_diff_blob_backend()


def _mod(path, before, after, **kw):
    return ChangedFileInput(
        repo_key="repoA", repo_label="llm-bawt", path=path,
        change_kind="modified",
        before=before.encode() if before is not None else None,
        after=after.encode() if after is not None else None,
        **kw,
    )


def test_ensure_schema_removes_legacy_overlap_columns():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE turn_changed_files (
                id INTEGER PRIMARY KEY,
                created_at DATETIME NOT NULL,
                turn_id VARCHAR(128) NOT NULL,
                bot_id VARCHAR,
                user_id VARCHAR,
                trigger_message_id VARCHAR,
                repo_key VARCHAR(128) NOT NULL,
                repo_label VARCHAR(255),
                path TEXT NOT NULL,
                old_path TEXT,
                change_kind VARCHAR(16) NOT NULL,
                additions INTEGER,
                deletions INTEGER,
                binary BOOLEAN NOT NULL,
                truncated BOOLEAN NOT NULL,
                overlapping BOOLEAN NOT NULL,
                manifest_truncated BOOLEAN NOT NULL,
                before_sha256 VARCHAR(64),
                before_bytes BIGINT,
                before_blob_key VARCHAR(128),
                after_sha256 VARCHAR(64),
                after_bytes BIGINT,
                after_blob_key VARCHAR(128),
                content_type VARCHAR(128),
                source_tool_call_ids TEXT
            )
        """))

    legacy = ChangedFilesStore(engine)
    legacy.ensure_schema()

    columns = {column["name"] for column in inspect(engine).get_columns("turn_changed_files")}
    assert "overlapping" not in columns
    assert "manifest_truncated" not in columns
    assert legacy.save_turn_files(
        turn_id="turn-legacy", bot_id="b", user_id="u", trigger_message_id="m",
        files=[_mod("a.py", "old", "new")],
    ) == 1


def test_save_and_read_roundtrip(store):
    n = store.save_turn_files(
        turn_id="turn-1", bot_id="snark", user_id="nick",
        trigger_message_id="msg-1",
        files=[_mod("a.py", "old\n", "new\n", additions=1, deletions=1)],
    )
    assert n == 1
    resolved = store.get_file_content(turn_id="turn-1", repo_key="repoA", path="a.py")
    assert resolved is not None
    before, after = resolved
    assert before.text == "old\n"
    assert after.text == "new\n"
    assert not before.binary and not after.binary


def test_summary_shape_and_totals(store):
    store.save_turn_files(
        turn_id="turn-2", bot_id="snark", user_id="nick", trigger_message_id="msg-2",
        files=[
            _mod("a.py", "x", "y", additions=3, deletions=2),
            ChangedFileInput(repo_key="repoA", repo_label="llm-bawt", path="new.py",
                             change_kind="added", before=None, after=b"hi", additions=1, deletions=0),
        ],
    )
    summary = store.summary_for_turn("turn-2")
    assert summary["total_files"] == 2
    assert summary["total_additions"] == 4
    assert summary["total_deletions"] == 2
    files = {f["path"]: f for f in summary["files"]}
    assert files["a.py"]["turn_id"] == "turn-2"
    assert files["a.py"]["has_content"] is True
    assert files["new.py"]["change_kind"] == "added"
    # Added file: before side absent, after present -> still has content.
    assert files["new.py"]["has_content"] is True


def test_added_and_deleted_have_empty_absent_side(store):
    store.save_turn_files(
        turn_id="turn-3", bot_id="b", user_id="u", trigger_message_id="m",
        files=[
            ChangedFileInput(repo_key="r", path="added.py", change_kind="added",
                             before=None, after=b"content"),
            ChangedFileInput(repo_key="r", path="gone.py", change_kind="deleted",
                             before=b"was here", after=None),
        ],
    )
    add = store.get_file_content(turn_id="turn-3", repo_key="r", path="added.py")
    assert add[0].text == "" and add[1].text == "content"
    delete = store.get_file_content(turn_id="turn-3", repo_key="r", path="gone.py")
    assert delete[0].text == "was here" and delete[1].text == ""


def test_binary_stores_no_content(store):
    store.save_turn_files(
        turn_id="turn-4", bot_id="b", user_id="u", trigger_message_id="m",
        files=[ChangedFileInput(repo_key="r", path="img.png", change_kind="modified",
                                binary=True, before=b"\x00\x01", after=b"\x00\x02")],
    )
    summary = store.summary_for_turn("turn-4")
    f = summary["files"][0]
    assert f["binary"] is True
    assert f["has_content"] is False
    before, after = store.get_file_content(turn_id="turn-4", repo_key="r", path="img.png")
    assert before.binary and after.binary
    assert before.text == "" and after.text == ""


def test_upsert_is_idempotent(store):
    for after in ("v1", "v2"):
        store.save_turn_files(
            turn_id="turn-5", bot_id="b", user_id="u", trigger_message_id="m",
            files=[_mod("a.py", "base", after)],
        )
    summary = store.summary_for_turn("turn-5")
    assert summary["total_files"] == 1  # not duplicated
    _, after = store.get_file_content(turn_id="turn-5", repo_key="repoA", path="a.py")
    assert after.text == "v2"  # last write wins


def test_truncation_flag_and_cap(store):
    big = "a" * (MAX_BYTES_PER_SIDE + 100)
    store.save_turn_files(
        turn_id="turn-6", bot_id="b", user_id="u", trigger_message_id="m",
        files=[_mod("big.txt", "", big)],
    )
    f = store.summary_for_turn("turn-6")["files"][0]
    assert f["truncated"] is True
    _, after = store.get_file_content(turn_id="turn-6", repo_key="repoA", path="big.txt")
    assert len(after.text) == MAX_BYTES_PER_SIDE


def test_file_count_cap(store):
    files = [_mod(f"f{i}.py", "a", "b") for i in range(MAX_FILES_PER_TURN + 20)]
    n = store.save_turn_files(
        turn_id="turn-7", bot_id="b", user_id="u", trigger_message_id="m", files=files,
    )
    assert n == MAX_FILES_PER_TURN


def test_repeated_file_upsert_keeps_first_before_latest_after(store):
    store.save_turn_files(
        turn_id="turn-repeat", bot_id="b", user_id="u", trigger_message_id="m",
        files=[_mod("a.py", "v1", "v2", source_tool_call_ids=["tool-1"])],
    )
    store.save_turn_files(
        turn_id="turn-repeat", bot_id="b", user_id="u", trigger_message_id="m",
        files=[_mod("a.py", "v1", "v3", source_tool_call_ids=["tool-2"])],
    )
    before, after = store.get_file_content(
        turn_id="turn-repeat", repo_key="repoA", path="a.py"
    )
    assert before.text == "v1"
    assert after.text == "v3"


def test_summaries_for_triggers_keys_by_message(store):
    store.save_turn_files(
        turn_id="turn-8", bot_id="b", user_id="u", trigger_message_id="msg-8",
        files=[_mod("a.py", "x", "y")],
    )
    out = store.summaries_for_triggers(["msg-8", "nope"])
    assert set(out.keys()) == {"msg-8"}
    assert out["msg-8"]["turn_id"] == "turn-8"


def test_postgres_fallback_when_no_object_store(store, monkeypatch):
    # Force "no object store": content must round-trip via turn_diff_payloads.
    monkeypatch.setattr(cfs, "_get_diff_blob_backend", lambda: None)
    store.save_turn_files(
        turn_id="turn-9", bot_id="b", user_id="u", trigger_message_id="m",
        files=[_mod("a.py", "before-text", "after-text")],
    )
    before, after = store.get_file_content(turn_id="turn-9", repo_key="repoA", path="a.py")
    assert before.text == "before-text"
    assert after.text == "after-text"


def test_missing_file_returns_none(store):
    assert store.get_file_content(turn_id="nope", repo_key="r", path="x") is None


def test_empty_summary_shape():
    summary = build_turn_summary("turn-x", [])
    assert summary == {
        "turn_id": "turn-x", "files": [],
        "total_files": 0, "total_additions": 0, "total_deletions": 0,
    }
