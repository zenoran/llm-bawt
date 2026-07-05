"""TASK-391 — agent-visible 'Attached Images' manifest builder.

Covers :func:`build_agent_image_manifest`: the plain-text block appended
to an agent backend's prompt so its shell/tools can curl the same image
assets the model sees inline.
"""

from __future__ import annotations

from typing import Any

from llm_bawt.media.serializers import build_agent_image_manifest


def _row(asset_id: str, **over: Any) -> dict[str, Any]:
    """media_assets row mirroring what MediaAssetStore.get_by_id returns."""
    row = {
        "id": asset_id,
        "sha256": f"sha-{asset_id}",
        "mime_type": "image/webp",
        "original_mime_type": "image/png",
        "size_bytes": 128_500,
        "width": 1568,
        "height": 882,
        "source": "chat_upload",
        "owner_user_id": "u1",
    }
    row.update(over)
    return row


class FakeStore:
    """Minimal asset store exposing get_many + get_by_id over a dict."""

    def __init__(self, rows: dict[str, dict[str, Any]], *, batch: bool = True):
        self._rows = rows
        self.batch = batch
        self.many_calls = 0
        self.by_id_calls = 0

    def get_many(self, ids: list[str]) -> list[dict[str, Any]]:
        if not self.batch:
            raise AttributeError("no batch")
        self.many_calls += 1
        return [self._rows[i] for i in ids if i in self._rows]

    def get_by_id(self, asset_id: str) -> dict[str, Any] | None:
        self.by_id_calls += 1
        return self._rows.get(asset_id)


def test_empty_refs_returns_empty_string():
    store = FakeStore({})
    assert build_agent_image_manifest([], store, origin="http://app:8642") == ""


def test_manifest_with_absolute_urls_and_metadata():
    store = FakeStore({"ma_AAA": _row("ma_AAA")})
    refs = [{"asset_id": "ma_AAA", "kind": "image"}]

    out = build_agent_image_manifest(refs, store, origin="http://app:8642")

    assert "[Attached Images]" in out
    assert "attached 1 image(s)" in out
    assert "asset_id=ma_AAA" in out
    assert "type=image/webp" in out
    assert "1568x882" in out
    assert "125.5 KB" in out  # 128500 / 1024
    assert "original: http://app:8642/v1/uploads/ma_AAA" in out
    assert "preview:  http://app:8642/v1/uploads/ma_AAA/preview" in out
    assert "thumb:    http://app:8642/v1/uploads/ma_AAA/thumb" in out


def test_trailing_slash_on_origin_is_normalized():
    store = FakeStore({"ma_AAA": _row("ma_AAA")})
    out = build_agent_image_manifest(
        [{"asset_id": "ma_AAA", "kind": "image"}], store, origin="http://app:8642/"
    )
    assert "http://app:8642/v1/uploads/ma_AAA" in out
    assert "http://app:8642//v1/uploads" not in out


def test_empty_origin_emits_relative_paths():
    store = FakeStore({"ma_AAA": _row("ma_AAA")})
    out = build_agent_image_manifest(
        [{"asset_id": "ma_AAA", "kind": "image"}], store, origin=""
    )
    assert "original: /v1/uploads/ma_AAA" in out
    assert "http://" not in out


def test_multiple_refs_numbered_and_deduped():
    store = FakeStore({"ma_AAA": _row("ma_AAA"), "ma_BBB": _row("ma_BBB")})
    refs = [
        {"asset_id": "ma_AAA", "kind": "image"},
        {"asset_id": "ma_BBB", "kind": "image"},
        {"asset_id": "ma_AAA", "kind": "image"},  # duplicate collapses
    ]
    out = build_agent_image_manifest(refs, store, origin="http://app:8642")

    assert "attached 2 image(s)" in out
    assert "1. asset_id=ma_AAA" in out
    assert "2. asset_id=ma_BBB" in out
    assert "3." not in out  # dedup means no third entry


def test_unresolvable_ref_is_dropped():
    store = FakeStore({"ma_AAA": _row("ma_AAA")})
    refs = [
        {"asset_id": "ma_AAA", "kind": "image"},
        {"asset_id": "ma_GONE", "kind": "image"},  # not in store
    ]
    out = build_agent_image_manifest(refs, store, origin="http://app:8642")

    assert "attached 1 image(s)" in out
    assert "ma_AAA" in out
    assert "ma_GONE" not in out


def test_all_unresolvable_returns_empty():
    store = FakeStore({})
    out = build_agent_image_manifest(
        [{"asset_id": "ma_GONE", "kind": "image"}], store, origin="http://app:8642"
    )
    assert out == ""


def test_missing_dimensions_omitted():
    store = FakeStore({"ma_AAA": _row("ma_AAA", width=None, height=None)})
    out = build_agent_image_manifest(
        [{"asset_id": "ma_AAA", "kind": "image"}], store, origin=""
    )
    assert "asset_id=ma_AAA" in out
    assert "x" not in out.split("asset_id=ma_AAA")[1].split("\n")[0]


def test_falls_back_to_get_by_id_without_batch():
    store = FakeStore({"ma_AAA": _row("ma_AAA")}, batch=False)
    out = build_agent_image_manifest(
        [{"asset_id": "ma_AAA", "kind": "image"}], store, origin="http://app:8642"
    )
    assert "asset_id=ma_AAA" in out
    assert store.by_id_calls == 1
