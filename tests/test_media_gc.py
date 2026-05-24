"""Unit tests for :mod:`llm_bawt.service.jobs.media_gc` (TASK-231).

These tests exercise the GC orchestration and SQL builder in isolation
— without a live Postgres. The orphan-finding query itself uses
Postgres-only constructs (``jsonb_array_elements``, ``make_interval``)
so its on-DB behavior is covered by the integration smoke test
documented in TASK-231; here we verify:

* the orphan-eviction loop calls ``MediaStore.delete`` once per row
* counts reported in the result dict (``deleted_count`` /
  ``freed_bytes``) match what we deleted
* dry-run mode never touches the store
* the SQL builder degrades safely when there are zero messages tables

The 7-day grace + ``expires_at`` semantics live entirely in the SQL
query, so we verify those via the contract test: when the find-orphans
hook returns row X, the GC loop deletes row X regardless of *why* the
query picked it. Both the "8-days-old & unreferenced" and "expired"
branches map to the same downstream behavior.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pytest

from llm_bawt.media.store import MediaStore
from llm_bawt.service.jobs import media_gc
from llm_bawt.service.jobs.media_gc import (
    _build_referenced_query,
    run_media_gc,
)


# ---------------------------------------------------------------------------
# In-memory fakes — mirror tests/test_media_store.py
# ---------------------------------------------------------------------------


class FakeMediaAssetStore:
    """Stand-in for :class:`MediaAssetStore` used by :class:`MediaStore`.

    Tracks rows in a dict so ``MediaStore.delete`` can resolve the
    sha256 and call back into ``delete``.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.by_sha: dict[str, str] = {}

    def add(self, *, asset_id: str, sha256: str, size_bytes: int) -> None:
        self.rows[asset_id] = {
            "id": asset_id,
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
        self.by_sha[sha256] = asset_id

    def get_by_id(self, asset_id: str) -> Optional[dict[str, Any]]:
        row = self.rows.get(asset_id)
        return dict(row) if row else None

    def get_by_sha256(self, sha256: str) -> Optional[dict[str, Any]]:
        rid = self.by_sha.get(sha256)
        return dict(self.rows[rid]) if rid else None

    def delete(self, asset_id: str) -> bool:
        row = self.rows.pop(asset_id, None)
        if row is None:
            return False
        self.by_sha.pop(row["sha256"], None)
        return True


def _write_blob_set(root: Path, sha: str) -> list[Path]:
    """Write the three blob variants the deletion path expects.

    Mirrors :data:`llm_bawt.media.store.VARIANT_DIRS` so we can assert
    they're all gone after a GC run.
    """
    from llm_bawt.media.store import VARIANT_DIRS

    paths: list[Path] = []
    for subdir in VARIANT_DIRS.values():
        p = root / subdir / sha[:2] / sha[2:4] / f"{sha}.webp"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake-webp-bytes")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_store(tmp_path: Path) -> tuple[MediaStore, FakeMediaAssetStore]:
    """Build a real MediaStore over a tmp-path tree + in-memory DB fake.

    Returns ``(media_store, asset_store)`` so tests can both manipulate
    rows directly and assert what the GC removed.
    """
    asset_store = FakeMediaAssetStore()
    store = MediaStore(root=tmp_path, db=asset_store)
    return store, asset_store


@pytest.fixture
def patched_orphan_finder(monkeypatch: pytest.MonkeyPatch):
    """Stub out the Postgres-only SQL hook.

    Tests configure ``state["orphans"]`` to whatever orphan rows the SQL
    "would have" returned, and ``state["tables"]`` to the discovered
    messages-table count.
    """
    state: dict[str, Any] = {"orphans": [], "tables": ["bot1_messages"]}

    def fake_find_orphans(conn, tables, grace_days):
        return list(state["orphans"])

    def fake_discover(conn):
        return list(state["tables"])

    def fake_engine(config, **kwargs):
        # Any sentinel that supports ``connect()`` as a context manager —
        # the real query callbacks above ignore the conn entirely.
        class _DummyConn:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class _DummyEngine:
            def connect(self):
                return _DummyConn()

        return _DummyEngine()

    monkeypatch.setattr(media_gc, "_find_orphan_assets", fake_find_orphans)
    monkeypatch.setattr(media_gc, "_discover_message_tables", fake_discover)
    monkeypatch.setattr(media_gc, "get_shared_engine", fake_engine, raising=False)
    # ``run_media_gc`` does ``from llm_bawt.utils.db import get_shared_engine``
    # late, so we have to patch the source module too.
    import llm_bawt.utils.db as utils_db

    monkeypatch.setattr(utils_db, "get_shared_engine", fake_engine)
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunMediaGc:
    """End-to-end orchestration tests with a faked SQL layer."""

    def test_no_orphans_returns_zero_counts(
        self,
        fake_store: tuple[MediaStore, FakeMediaAssetStore],
        patched_orphan_finder: dict,
    ):
        """No orphan rows → deleted_count=0, freed_bytes=0, errors=[].

        Validates: "running it twice in a row should be a no-op the
        second time" — same shape as zero orphans on the second pass.
        """
        store, _ = fake_store
        patched_orphan_finder["orphans"] = []

        result = run_media_gc(
            config=object(),
            media_store=store,
        )

        assert result["orphan_count"] == 0
        assert result["deleted_count"] == 0
        assert result["freed_bytes"] == 0
        assert result["dry_run"] is False
        assert result["errors"] == []
        assert result["scanned_tables"] == 1

    def test_orphan_evicted_deletes_row_and_blobs(
        self,
        fake_store: tuple[MediaStore, FakeMediaAssetStore],
        tmp_path: Path,
        patched_orphan_finder: dict,
    ):
        """Orphan from 8 days ago with no references → row + 3 blobs gone.

        Covers TASK-231 acceptance test "orphan from 8 days ago gets
        GC'd" plus the blob-cleanup requirement.
        """
        store, asset_store = fake_store

        sha = "a" * 64
        asset_id = "ma_OLD_ORPHAN"
        asset_store.add(asset_id=asset_id, sha256=sha, size_bytes=12345)
        blob_paths = _write_blob_set(tmp_path, sha)
        for p in blob_paths:
            assert p.exists()

        patched_orphan_finder["orphans"] = [
            {"id": asset_id, "sha256": sha, "size_bytes": 12345}
        ]

        result = run_media_gc(
            config=object(),
            media_store=store,
        )

        assert result["orphan_count"] == 1
        assert result["deleted_count"] == 1
        assert result["freed_bytes"] == 12345
        assert result["errors"] == []
        # Row gone
        assert asset_store.get_by_id(asset_id) is None
        # All three blob variants gone
        for p in blob_paths:
            assert not p.exists(), f"variant blob lingered: {p}"

    def test_referenced_asset_not_passed_to_loop(
        self,
        fake_store: tuple[MediaStore, FakeMediaAssetStore],
        tmp_path: Path,
        patched_orphan_finder: dict,
    ):
        """Asset NOT in the orphan list survives — referenced/young.

        Mirrors "orphan from yesterday survives" + "referenced asset
        survives even after 8 days": both are characterized by the SQL
        not returning the row, which is the only signal the loop sees.
        """
        store, asset_store = fake_store

        sha_keep = "b" * 64
        keep_id = "ma_KEEP_ME"
        asset_store.add(asset_id=keep_id, sha256=sha_keep, size_bytes=999)
        keep_paths = _write_blob_set(tmp_path, sha_keep)

        # SQL hook returns nothing — the GC loop never sees this row.
        patched_orphan_finder["orphans"] = []

        result = run_media_gc(
            config=object(),
            media_store=store,
        )

        assert result["deleted_count"] == 0
        # Row + blobs all still there
        assert asset_store.get_by_id(keep_id) is not None
        for p in keep_paths:
            assert p.exists()

    def test_expires_at_path_uses_same_deletion(
        self,
        fake_store: tuple[MediaStore, FakeMediaAssetStore],
        tmp_path: Path,
        patched_orphan_finder: dict,
    ):
        """An ``expires_at < now()`` row is GC'd the same way as an aged orphan.

        The two SQL branches funnel into the same downstream loop, so
        the test stages the row as if the SQL had picked it for the
        ``expires_at`` reason: shows the deletion path is identical.
        """
        store, asset_store = fake_store

        sha = "c" * 64
        asset_id = "ma_EXPIRED"
        asset_store.add(asset_id=asset_id, sha256=sha, size_bytes=2048)
        expired_paths = _write_blob_set(tmp_path, sha)

        # The SQL would return this asset because expires_at < now() —
        # we simulate that by having the find-orphans hook return it
        # even though it's "fresh" age-wise.
        patched_orphan_finder["orphans"] = [
            {"id": asset_id, "sha256": sha, "size_bytes": 2048}
        ]

        result = run_media_gc(
            config=object(),
            media_store=store,
        )

        assert result["deleted_count"] == 1
        assert result["freed_bytes"] == 2048
        assert asset_store.get_by_id(asset_id) is None
        for p in expired_paths:
            assert not p.exists()

    def test_dry_run_does_not_delete(
        self,
        fake_store: tuple[MediaStore, FakeMediaAssetStore],
        tmp_path: Path,
        patched_orphan_finder: dict,
    ):
        """Dry-run reports counts without touching DB or disk."""
        store, asset_store = fake_store

        sha = "d" * 64
        asset_id = "ma_DRYRUN"
        asset_store.add(asset_id=asset_id, sha256=sha, size_bytes=4096)
        dry_paths = _write_blob_set(tmp_path, sha)

        patched_orphan_finder["orphans"] = [
            {"id": asset_id, "sha256": sha, "size_bytes": 4096}
        ]

        result = run_media_gc(
            config=object(),
            media_store=store,
            dry_run=True,
        )

        assert result["orphan_count"] == 1
        assert result["deleted_count"] == 0
        assert result["dry_run"] is True
        # Dry-run reports what would have been freed (so ops can size the run).
        assert result["freed_bytes"] == 4096
        # Nothing actually removed.
        assert asset_store.get_by_id(asset_id) is not None
        for p in dry_paths:
            assert p.exists()

    def test_counts_aggregate_across_multiple_orphans(
        self,
        fake_store: tuple[MediaStore, FakeMediaAssetStore],
        tmp_path: Path,
        patched_orphan_finder: dict,
    ):
        """``deleted_count`` and ``freed_bytes`` sum correctly across rows."""
        store, asset_store = fake_store

        sizes = [100, 500, 4000]
        for i, size in enumerate(sizes):
            sha = (f"{i:x}" * 64)[:64]
            asset_store.add(asset_id=f"ma_M_{i}", sha256=sha, size_bytes=size)
            _write_blob_set(tmp_path, sha)

        patched_orphan_finder["orphans"] = [
            {"id": f"ma_M_{i}", "sha256": (f"{i:x}" * 64)[:64], "size_bytes": s}
            for i, s in enumerate(sizes)
        ]

        result = run_media_gc(
            config=object(),
            media_store=store,
        )

        assert result["orphan_count"] == 3
        assert result["deleted_count"] == 3
        assert result["freed_bytes"] == sum(sizes)
        for i in range(3):
            assert asset_store.get_by_id(f"ma_M_{i}") is None

    def test_delete_error_continues_sweep(
        self,
        fake_store: tuple[MediaStore, FakeMediaAssetStore],
        tmp_path: Path,
        patched_orphan_finder: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A single failed delete is recorded but doesn't abort the sweep.

        Re-entrancy guarantee: a flaky delete leaves the remaining
        orphans for the next nightly run, but doesn't stop us from
        cleaning up the rest of *this* run.
        """
        store, asset_store = fake_store

        sha_bad = "e" * 64
        sha_ok = "f" * 64
        asset_store.add(asset_id="ma_BAD", sha256=sha_bad, size_bytes=1)
        asset_store.add(asset_id="ma_OK", sha256=sha_ok, size_bytes=200)
        _write_blob_set(tmp_path, sha_bad)
        _write_blob_set(tmp_path, sha_ok)

        original_delete = store.delete

        def maybe_explode(asset_id: str) -> None:
            if asset_id == "ma_BAD":
                raise RuntimeError("simulated blob unlink failure")
            return original_delete(asset_id)

        monkeypatch.setattr(store, "delete", maybe_explode)

        patched_orphan_finder["orphans"] = [
            {"id": "ma_BAD", "sha256": sha_bad, "size_bytes": 1},
            {"id": "ma_OK", "sha256": sha_ok, "size_bytes": 200},
        ]

        result = run_media_gc(
            config=object(),
            media_store=store,
        )

        assert result["orphan_count"] == 2
        assert result["deleted_count"] == 1
        assert result["freed_bytes"] == 200
        assert len(result["errors"]) == 1
        assert result["errors"][0]["asset_id"] == "ma_BAD"
        assert "simulated blob unlink failure" in result["errors"][0]["error"]
        # Successful asset is gone; failed one stays for the next run.
        assert asset_store.get_by_id("ma_BAD") is not None
        assert asset_store.get_by_id("ma_OK") is None


class TestSqlBuilder:
    """SQL-construction tests that don't need a live Postgres."""

    def test_empty_tables_returns_safe_fallback(self):
        """Zero ``*_messages`` tables → the WITH clause still parses.

        A brand-new install before any bot has booted has no per-bot
        tables yet. The GC should still run and find any orphans by
        ``expires_at`` alone.
        """
        sql = _build_referenced_query([])
        assert "FALSE" in sql.upper() or "WHERE FALSE" in sql.upper()
        assert "UNION ALL" not in sql

    def test_single_table_emits_one_fragment(self):
        sql = _build_referenced_query(["nova_messages"])
        assert "nova_messages" in sql
        assert "jsonb_array_elements" in sql
        assert "UNION ALL" not in sql

    def test_multiple_tables_unioned(self):
        sql = _build_referenced_query(
            ["nova_messages", "echo_messages", "pip_messages"]
        )
        assert "nova_messages" in sql
        assert "echo_messages" in sql
        assert "pip_messages" in sql
        # N tables → N-1 UNION ALL connectors.
        assert sql.count("UNION ALL") == 2
        # asset_id selector wired correctly on every fragment.
        assert sql.count("asset_id") >= 3

    def test_query_filters_null_asset_ids(self):
        """Defensive: malformed attachments shouldn't surface as NULL ids
        that then fail the orphan join (``r.asset_id = a.id`` against
        NULL is always NULL/false but we still want them stripped at
        the source)."""
        sql = _build_referenced_query(["nova_messages"])
        assert "IS NOT NULL" in sql
