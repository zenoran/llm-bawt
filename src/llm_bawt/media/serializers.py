"""Shared dict-shape helpers for media-attachment JSON payloads.

Used by:

- ``POST /v1/uploads`` response (TASK-224, this module's primary caller).
- ``/v1/chat/completions`` persistence layer (TASK-225) — when an LLM call
  includes an image attachment, the row written to ``{bot}_messages.attachments``
  uses the same shape.
- ``/v1/history`` responses (TASK-226) — history JSON returns the attachment
  dict embedded in each message.
- bawthub frontend renderers via API docs and TypeScript types.

Keep this stable. **Other code reads it**, so do not rename keys or change
URL shapes without updating every downstream consumer first.

Two functions, two shapes:

- :func:`asset_to_attachment_dict` — the *minimum* envelope every API
  surface ships. Goes into the ``attachments`` array on chat/history rows.
- :func:`asset_to_upload_response_dict` — superset returned by
  ``POST /v1/uploads`` only; adds ``sha256``, ``size_bytes``, and
  ``original_mime_type`` so the uploader can dedup client-side, show file
  size, and render a "you uploaded a JPEG; we store WebP" hint.

Both forms share the same ``urls`` block so any consumer can use a single
helper to pick a variant URL.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from .assets import MediaAsset

logger = logging.getLogger(__name__)


def asset_to_attachment_dict(asset: MediaAsset) -> dict:
    """Return the canonical 'attachment' dict used on every API surface.

    This is the shape persisted in ``{bot}_messages.attachments`` and the
    shape returned inside ``/v1/history`` messages. Keep field names and
    URL structure stable — bawthub renderers read this directly.
    """
    return {
        "asset_id": asset.id,
        "kind": "image",
        "mime_type": asset.mime_type,
        "width": asset.width,
        "height": asset.height,
        "urls": {
            "thumb": f"/v1/uploads/{asset.id}/thumb",
            "preview": f"/v1/uploads/{asset.id}/preview",
            "original": f"/v1/uploads/{asset.id}",
        },
    }


def asset_to_upload_response_dict(asset: MediaAsset) -> dict:
    """Shape returned by ``POST /v1/uploads``. Superset of attachment dict.

    Adds ``sha256`` (for client-side dedup checks), ``size_bytes`` (for
    file-size display), and ``original_mime_type`` (the pre-normalization
    MIME the client sent — useful for "you uploaded a JPEG; we store WebP"
    UX) on top of the canonical attachment envelope.
    """
    base = asset_to_attachment_dict(asset)
    base["sha256"] = asset.sha256
    base["size_bytes"] = asset.size_bytes
    base["original_mime_type"] = asset.original_mime_type
    return base


def asset_row_to_attachment_dict(row: dict[str, Any]) -> dict:
    """Same canonical attachment dict, but from a DB-row mapping.

    ``MediaAssetStore`` returns plain dicts (not ``MediaAsset`` SQLModel
    instances), so the history-enrichment path on TASK-226 needs an
    overload that takes a mapping. Keep this in sync with
    :func:`asset_to_attachment_dict` — both shapes are wire-identical.
    """
    asset_id = row.get("id")
    return {
        "asset_id": asset_id,
        "kind": "image",
        "mime_type": row.get("mime_type"),
        "width": row.get("width"),
        "height": row.get("height"),
        "urls": {
            "thumb": f"/v1/uploads/{asset_id}/thumb",
            "preview": f"/v1/uploads/{asset_id}/preview",
            "original": f"/v1/uploads/{asset_id}",
        },
    }


def enrich_attachments_for_messages(
    messages: Iterable[dict[str, Any]],
    asset_store,
) -> None:
    """Resolve tiny ``attachments`` refs into full URL-block dicts in place.

    TASK-226 — history endpoints persist a tiny shape per row:
    ``[{"asset_id": "ma_xxx", "kind": "image"}, ...]``. Outbound HTTP
    consumers want the canonical attachment envelope including
    ``mime_type``/``width``/``height``/``urls`` so the frontend can
    render thumbnails without a second round-trip per message.

    This helper:

    1. Collects every distinct ``asset_id`` referenced by ``messages``.
    2. Runs a single ``SELECT * FROM media_assets WHERE id = ANY(:ids)``
       through ``asset_store`` — O(1) regardless of page size.
    3. Rewrites each message's ``attachments`` list to the full shape,
       preserving order.

    Missing asset IDs (e.g. deleted blobs) drop out of the rewritten
    list and trigger a single warning log per call. The message keeps
    ``attachments=[]`` rather than failing so partial deletes can never
    take down the whole history response.

    The ``attachments`` key is always set on every message after this
    call (even on messages that had no refs to begin with), so frontend
    code can iterate without defensive checks.

    Args:
        messages: Iterable of message dicts. Mutated in place.
        asset_store: Anything exposing ``get_by_id(asset_id) -> dict | None``
            **and** (optionally) ``get_many(ids) -> list[dict]``. The
            former is required; the batched variant is preferred when
            available.
    """
    # Materialize to a list so we can iterate twice (collect ids, rewrite).
    msg_list = list(messages)

    seen_ids: list[str] = []
    seen_set: set[str] = set()
    for msg in msg_list:
        for ref in msg.get("attachments") or []:
            if not isinstance(ref, dict):
                continue
            aid = ref.get("asset_id")
            if aid and aid not in seen_set:
                seen_set.add(aid)
                seen_ids.append(aid)

    # Batch-load. Fall back to per-id lookups if the store doesn't expose
    # a batch helper (the production MediaAssetStore does not at TASK-226
    # time, so we go per-id but in a single connection cycle below).
    asset_map: dict[str, dict[str, Any]] = {}
    if seen_ids:
        getter_many = getattr(asset_store, "get_many", None)
        if callable(getter_many):
            try:
                rows = getter_many(seen_ids) or []
                for r in rows:
                    if r and r.get("id"):
                        asset_map[r["id"]] = r
            except Exception as e:
                logger.warning(
                    "asset_store.get_many failed (%s); falling back to per-id lookup",
                    e,
                )

        if not asset_map:
            for aid in seen_ids:
                try:
                    row = asset_store.get_by_id(aid)
                except Exception as e:
                    logger.warning("asset_store.get_by_id(%s) failed: %s", aid, e)
                    row = None
                if row:
                    asset_map[aid] = row

    missing: list[str] = [aid for aid in seen_ids if aid not in asset_map]
    if missing:
        logger.warning(
            "history attachments: %s asset id(s) not in media_assets (deleted?): %s",
            len(missing),
            ",".join(missing[:5]),
        )

    for msg in msg_list:
        raw_refs = msg.get("attachments") or []
        resolved: list[dict[str, Any]] = []
        for ref in raw_refs:
            if not isinstance(ref, dict):
                continue
            aid = ref.get("asset_id")
            row = asset_map.get(aid) if aid else None
            if row:
                resolved.append(asset_row_to_attachment_dict(row))
            # else: silently drop unresolved refs (already logged above)
        msg["attachments"] = resolved
