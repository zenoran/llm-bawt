"""TASK-661 slice 3: bridge-side changed-file capture.

These drive real git repos in tmpdirs — the parsing is the part most likely to
rot silently, and mocking `git` would only test my idea of its output format.
"""

from __future__ import annotations

import asyncio
import base64
import subprocess
from pathlib import Path

import pytest

from agent_bridge.changed_files import (
    ChangedFileTracker,
    _parse_numstat_z,
    _parse_raw_z,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True,
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "workspaces"
    root.mkdir()
    r = root / "demo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "T")
    (r / "keep.txt").write_text("one\ntwo\nthree\n")
    (r / ".gitignore").write_text("ignored/\n")
    (r / "ignored").mkdir()
    (r / "ignored" / "junk.txt").write_text("noise\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "init")
    return r


def _tracker(repo: Path) -> ChangedFileTracker:
    return ChangedFileTracker(roots=[repo.parent])


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def test_parse_raw_z_handles_modify_and_rename():
    payload = (
        b":100644 100644 aaa111 bbb222 M\x00src/app.py\x00"
        b":100644 100644 ccc333 ddd444 R100\x00old/name.py\x00new/name.py\x00"
        b":000000 100644 " + b"0" * 40 + b" eee555 A\x00added.py\x00"
    )
    entries = _parse_raw_z(payload)
    assert [e.path for e in entries] == ["src/app.py", "new/name.py", "added.py"]
    assert entries[1].status == "R"
    assert entries[1].old_path == "old/name.py"
    assert entries[2].before_sha == "0" * 40


def test_parse_numstat_z_handles_rename_and_binary():
    payload = (
        b"3\t1\tsrc/app.py\x00"
        b"5\t0\t\x00old/name.py\x00new/name.py\x00"
        b"-\t-\tlogo.png\x00"
    )
    stats = _parse_numstat_z(payload)
    assert stats["src/app.py"] == (3, 1, False)
    assert stats["new/name.py"] == (5, 0, False)
    assert stats["logo.png"] == (None, None, True)


# ---------------------------------------------------------------------------
# End-to-end capture against a real repo
# ---------------------------------------------------------------------------


def test_no_changes_yields_no_manifest(repo: Path):
    async def go():
        t = _tracker(repo)
        await t.start()
        return await t.finalize()

    assert _run(go()) is None


def test_captures_modified_added_and_deleted(repo: Path):
    async def go():
        t = _tracker(repo)
        await t.start()
        (repo / "keep.txt").write_text("one\ntwo\nCHANGED\n")
        (repo / "brand_new.txt").write_text("fresh\n")
        (repo / ".gitignore").unlink()
        return await t.finalize()

    manifest = _run(go())
    assert manifest is not None
    by_path = {f["path"]: f for f in manifest["files"]}

    assert by_path["keep.txt"]["change_kind"] == "modified"
    assert by_path["keep.txt"]["additions"] == 1
    assert by_path["keep.txt"]["deletions"] == 1
    before = base64.b64decode(by_path["keep.txt"]["before_b64"]).decode()
    after = base64.b64decode(by_path["keep.txt"]["after_b64"]).decode()
    assert before == "one\ntwo\nthree\n"
    assert after == "one\ntwo\nCHANGED\n"

    assert by_path["brand_new.txt"]["change_kind"] == "added"
    assert by_path["brand_new.txt"]["before_b64"] is None
    assert base64.b64decode(by_path["brand_new.txt"]["after_b64"]).decode() == "fresh\n"

    assert by_path[".gitignore"]["change_kind"] == "deleted"
    assert by_path[".gitignore"]["after_b64"] is None


def test_gitignored_files_are_never_reported(repo: Path):
    async def go():
        t = _tracker(repo)
        await t.start()
        (repo / "ignored" / "junk.txt").write_text("still noise, now different\n")
        (repo / "ignored" / "more.txt").write_text("also ignored\n")
        return await t.finalize()

    assert _run(go()) is None


def test_real_index_is_not_mutated(repo: Path):
    """The whole point of GIT_INDEX_FILE — a snapshot must not stage anything."""

    async def go():
        t = _tracker(repo)
        await t.start()
        (repo / "untracked.txt").write_text("do not stage me\n")
        await t.finalize()

    _run(go())
    staged = subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--name-only"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert staged == ""
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "?? untracked.txt" in status


def test_staged_work_survives_a_turn(repo: Path):
    """A sibling bot's staged-but-uncommitted work must still be staged after."""

    async def go():
        (repo / "keep.txt").write_text("staged edit\n")
        _git(repo, "add", "keep.txt")
        t = _tracker(repo)
        await t.start()
        (repo / "other.txt").write_text("turn work\n")
        return await t.finalize()

    manifest = _run(go())
    assert manifest is not None
    assert {f["path"] for f in manifest["files"]} == {"other.txt"}
    staged = subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--name-only"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    assert staged == ["keep.txt"]


def test_rename_is_detected(repo: Path):
    async def go():
        t = _tracker(repo)
        await t.start()
        (repo / "keep.txt").rename(repo / "renamed.txt")
        return await t.finalize()

    manifest = _run(go())
    assert manifest is not None
    entry = next(f for f in manifest["files"] if f["path"] == "renamed.txt")
    assert entry["change_kind"] == "renamed"
    assert entry["old_path"] == "keep.txt"


def test_binary_file_carries_no_content(repo: Path):
    async def go():
        t = _tracker(repo)
        await t.start()
        (repo / "blob.bin").write_bytes(bytes(range(256)) * 8)
        return await t.finalize()

    manifest = _run(go())
    assert manifest is not None
    entry = next(f for f in manifest["files"] if f["path"] == "blob.bin")
    assert entry["binary"] is True
    assert entry["before_b64"] is None
    assert entry["after_b64"] is None


def test_oversized_file_is_truncated_and_flagged(repo: Path):
    async def go():
        t = ChangedFileTracker(roots=[repo.parent], max_bytes_per_side=64)
        await t.start()
        (repo / "big.txt").write_text("x" * 4096 + "\n")
        return await t.finalize()

    manifest = _run(go())
    assert manifest is not None
    entry = next(f for f in manifest["files"] if f["path"] == "big.txt")
    assert entry["truncated"] is True
    assert len(base64.b64decode(entry["after_b64"])) == 64


def test_total_content_budget_drops_bytes_but_keeps_metadata(repo: Path):
    async def go():
        t = ChangedFileTracker(roots=[repo.parent], max_total_content=10)
        await t.start()
        (repo / "a.txt").write_text("a" * 200 + "\n")
        (repo / "b.txt").write_text("b" * 200 + "\n")
        return await t.finalize()

    manifest = _run(go())
    assert manifest is not None
    assert manifest["truncated"] is True
    assert len(manifest["files"]) == 2
    # Metadata survives even when the bytes were dropped.
    assert all(f["additions"] == 1 for f in manifest["files"])
    assert all(f["after_b64"] is None for f in manifest["files"])


def test_file_cap_truncates_manifest(repo: Path):
    async def go():
        t = ChangedFileTracker(roots=[repo.parent], max_files=3)
        await t.start()
        for i in range(10):
            (repo / f"f{i}.txt").write_text(f"{i}\n")
        return await t.finalize()

    manifest = _run(go())
    assert manifest is not None
    assert len(manifest["files"]) == 3
    assert manifest["truncated"] is True


def test_concurrent_turns_are_flagged_overlapping(repo: Path):
    """Union + flag: both turns see the change, both admit they overlapped."""

    async def go():
        first = _tracker(repo)
        await first.start()
        (repo / "from_first.txt").write_text("first\n")

        second = _tracker(repo)
        await second.start()
        (repo / "from_second.txt").write_text("second\n")

        return await first.finalize(), await second.finalize()

    m1, m2 = _run(go())
    assert m1 is not None and m2 is not None
    # First turn started before the overlap and is retro-flagged.
    assert m1["overlapping_repos"] == ["demo"]
    assert m2["overlapping_repos"] == ["demo"]
    # Second only ever saw its own file; first saw both (the union caveat).
    assert {f["path"] for f in m2["files"]} == {"from_second.txt"}
    assert {f["path"] for f in m1["files"]} == {"from_first.txt", "from_second.txt"}


def test_sequential_turns_are_not_flagged(repo: Path):
    async def go():
        first = _tracker(repo)
        await first.start()
        (repo / "one.txt").write_text("1\n")
        m1 = await first.finalize()

        second = _tracker(repo)
        await second.start()
        (repo / "two.txt").write_text("2\n")
        m2 = await second.finalize()
        return m1, m2

    m1, m2 = _run(go())
    assert m1["overlapping_repos"] == []
    assert m2["overlapping_repos"] == []
    assert {f["path"] for f in m2["files"]} == {"two.txt"}


def test_finalize_without_start_is_a_noop(repo: Path):
    assert _run(_tracker(repo).finalize()) is None


def test_disabled_tracker_reports_nothing(repo: Path, monkeypatch):
    monkeypatch.setenv("AGENT_BRIDGE_CHANGED_FILES_DISABLED", "1")

    async def go():
        t = _tracker(repo)
        await t.start()
        (repo / "keep.txt").write_text("changed\n")
        return await t.finalize()

    assert _run(go()) is None


def test_stale_worktree_is_skipped_quietly(repo: Path, caplog):
    """`~/dev` collects linked worktrees whose admin dir is gone. Those must not
    emit a WARNING on every turn (they surfaced 3/turn in the live smoke)."""
    stale = repo.parent / "stale-worktree"
    stale.mkdir()
    # A .git *file* pointing at an admin dir that doesn't exist — exactly what a
    # worktree left behind by a deleted checkout looks like.
    (stale / ".git").write_text("gitdir: /nonexistent/.git/worktrees/stale\n")
    (stale / "file.txt").write_text("hi\n")

    async def go():
        t = ChangedFileTracker(roots=[repo.parent])
        await t.start()
        (repo / "keep.txt").write_text("real change\n")
        return await t.finalize()

    with caplog.at_level("WARNING"):
        manifest = _run(go())

    # The healthy repo is still captured...
    assert manifest is not None
    assert {f["path"] for f in manifest["files"]} == {"keep.txt"}
    # ...and the stale worktree stayed silent.
    assert caplog.records == []


def test_same_named_repositories_get_distinct_keys(tmp_path: Path):
    roots = [tmp_path / "one" / "same", tmp_path / "two" / "same"]
    for idx, r in enumerate(roots):
        r.mkdir(parents=True)
        _git(r, "init", "-q", "-b", "main")
        _git(r, "config", "user.email", "t@example.com")
        _git(r, "config", "user.name", "T")
        (r / "file.txt").write_text(f"base-{idx}\n")
        _git(r, "add", "-A")
        _git(r, "commit", "-qm", "init")

    async def go():
        t = ChangedFileTracker(roots=roots)
        await t.start()
        for idx, r in enumerate(roots):
            (r / "file.txt").write_text(f"changed-{idx}\n")
        return await t.finalize()

    manifest = _run(go())
    assert manifest is not None
    assert len(manifest["files"]) == 2
    assert len({f["repo_key"] for f in manifest["files"]}) == 2
    assert {f["repo_label"] for f in manifest["files"]} == {"same"}


def test_non_git_directories_are_skipped(tmp_path: Path):
    root = tmp_path / "roots"
    (root / "not_a_repo").mkdir(parents=True)
    (root / "not_a_repo" / "file.txt").write_text("hi\n")

    async def go():
        t = ChangedFileTracker(roots=[root])
        await t.start()
        (root / "not_a_repo" / "file.txt").write_text("changed\n")
        return await t.finalize()

    assert _run(go()) is None
