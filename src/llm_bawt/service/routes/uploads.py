"""HTTP routes for the content-addressed media store (TASK-224).

Endpoints
---------

::

    POST   /v1/uploads
        body: multipart `file=...`  OR
              application/json {"data_url": "...", "filename": "..."}
        query: ?source=chat_upload | tool_generated | agent_attachment
        auth:  X-Entity-Id header (owner scoping)
        returns: asset metadata + variant URLs (see
                 ``asset_to_upload_response_dict``).

    GET    /v1/uploads/{asset_id}            -> original WebP bytes
    GET    /v1/uploads/{asset_id}/thumb      -> 256px variant
    GET    /v1/uploads/{asset_id}/preview    -> 1024px variant
    DELETE /v1/uploads/{asset_id}            -> owner-only delete

These routes wrap :class:`MediaStore` (TASK-223). The store handles
normalization, dedup, variant generation, and disk layout; this module
adds the HTTP surface: auth, content negotiation, caching headers, and
error mapping.

Auth model
----------

- ``POST`` requires the ``X-Entity-Id`` header — the value is recorded as
  ``owner_user_id`` and used by ``DELETE`` to gate access.
- ``DELETE`` requires ``X-Entity-Id`` to match the asset's owner; mismatch
  returns ``403 Forbidden``.
- ``GET`` is *unauthenticated* by design. Asset IDs are ``ma_<ulid>`` and
  the underlying paths are sha256-derived (unguessable), which mirrors the
  bawthub ``Upload`` model's behaviour. If you want stronger ACLs, do not
  put the asset URL in a place where unauthenticated callers can see it.

Limits
------

- 15 MB max raw upload — the store normalises down to a 1568px-cap WebP,
  but we reject obvious abuse before reading the full body.
- Accept ``image/jpeg``, ``image/png``, ``image/gif``, ``image/webp``;
  anything else returns ``415 Unsupported Media Type``.
"""

from __future__ import annotations

import base64
import logging
import re
from typing import Optional

from fastapi import APIRouter, File, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from ...media import (
    MediaAssetNotFound,
    MediaStore,
    asset_to_upload_response_dict,
    get_media_store,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Uploads"])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Hard cap on raw upload size. We normalise down to ~200-400 KB WebP, so
#: anything above this is either a misuse of the API or an attack.
MAX_RAW_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB

#: MIME types we'll accept on POST. Everything else returns 415.
ACCEPTED_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
    }
)

#: Stored variants always WebP; we serve them out unchanged.
RESPONSE_MIME = "image/webp"

#: One year, immutable — the URL is content-addressed so the bytes can
#: never change. Browsers can cache aggressively.
CACHE_CONTROL = "public, max-age=31536000, immutable"

#: Data URL regex: ``data:<mime>;base64,<payload>``. We require base64
#: encoding (the common case) and surface anything else as 400.
_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[\w.+/-]+);base64,(?P<payload>[A-Za-z0-9+/=\s]+)$"
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _store() -> MediaStore:
    """Resolve the process-wide :class:`MediaStore`.

    Pulled out as a tiny wrapper so tests can monkeypatch a different
    accessor if they want isolation from the singleton.
    """
    return get_media_store()


def _require_entity_id(x_entity_id: Optional[str]) -> str:
    """Return the entity ID or raise 401 if missing.

    The existing service has no centralised auth middleware; we enforce
    presence here so producers know to wire the header. Treat empty or
    whitespace-only values the same as missing.
    """
    if not x_entity_id or not x_entity_id.strip():
        raise HTTPException(
            status_code=401,
            detail="X-Entity-Id header is required",
        )
    return x_entity_id.strip()


# ---------------------------------------------------------------------------
# Body parsing
# ---------------------------------------------------------------------------


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    """Return ``(raw_bytes, declared_mime)`` from a ``data:<mime>;base64,<...>`` URL.

    Raises 400 on malformed input. We trust the declared MIME only enough
    to gate it through the 415 check — the store re-derives the real
    format from the bytes via Pillow.
    """
    match = _DATA_URL_RE.match(data_url.strip())
    if not match:
        raise HTTPException(
            status_code=400,
            detail="data_url must be of form 'data:<mime>;base64,<payload>'",
        )
    declared_mime = match.group("mime").lower()
    payload = re.sub(r"\s+", "", match.group("payload"))
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as e:  # ValueError or binascii.Error
        raise HTTPException(status_code=400, detail=f"data_url base64 decode failed: {e}")
    return raw, declared_mime


def _check_mime(declared_mime: str) -> str:
    """Normalise and validate the declared MIME, or raise 415."""
    mime = declared_mime.split(";")[0].strip().lower()
    if mime not in ACCEPTED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type {mime!r}. "
                f"Accepted: {sorted(ACCEPTED_MIME_TYPES)}"
            ),
        )
    return mime


def _check_size(raw: bytes) -> None:
    """Reject raw uploads over :data:`MAX_RAW_UPLOAD_BYTES`."""
    if len(raw) > MAX_RAW_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Upload too large: {len(raw)} bytes > "
                f"{MAX_RAW_UPLOAD_BYTES} byte limit"
            ),
        )


# ---------------------------------------------------------------------------
# POST /v1/uploads
# ---------------------------------------------------------------------------


@router.post("/v1/uploads")
async def upload_asset(
    request: Request,
    source: str = Query(
        default="chat_upload",
        description="Provenance tag; one of chat_upload | tool_generated | agent_attachment",
    ),
    x_entity_id: Optional[str] = Header(default=None, alias="X-Entity-Id"),
    file: Optional[UploadFile] = File(default=None),
):
    """Accept an image upload via multipart or JSON-with-data-URL.

    Two body shapes are supported (we sniff the content-type):

    - ``multipart/form-data`` with a ``file`` part. Standard browser form
      upload path.
    - ``application/json`` with ``{"data_url": "data:image/png;base64,...",
      "filename": "<optional>"}``. Used by paste/clipboard flows that
      already have a data URL in hand and don't want to round-trip
      through multipart.

    Returns the canonical upload-response dict — see
    :func:`asset_to_upload_response_dict` for the exact shape.
    """
    entity_id = _require_entity_id(x_entity_id)

    content_type = (request.headers.get("content-type") or "").lower()

    raw: bytes
    declared_mime: str

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        data_url = body.get("data_url")
        if not isinstance(data_url, str) or not data_url:
            raise HTTPException(
                status_code=400,
                detail="JSON body must include 'data_url' (string)",
            )
        raw, declared_mime = _decode_data_url(data_url)
    elif file is not None:
        # FastAPI parses multipart for us; ``file.content_type`` may be
        # missing on hand-crafted requests, in which case fall back to a
        # bytes sniff so we still 415 cleanly instead of crashing in
        # Pillow.
        raw = await file.read()
        declared_mime = (file.content_type or "application/octet-stream").lower()
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Provide either a multipart 'file' field or a JSON body with 'data_url'"
            ),
        )

    _check_mime(declared_mime)
    _check_size(raw)

    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload body")

    try:
        asset = _store().upload(
            raw_bytes=raw,
            original_mime=declared_mime,
            source=source,
            owner_user_id=entity_id,
        )
    except ValueError as e:
        # ``source`` not in ALLOWED_SOURCES surfaces here.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("MediaStore.upload failed")
        raise HTTPException(status_code=500, detail=f"upload failed: {e}")

    return JSONResponse(asset_to_upload_response_dict(asset))


# ---------------------------------------------------------------------------
# GET /v1/uploads/{asset_id}[/variant]
# ---------------------------------------------------------------------------


def _etag_matches(if_none_match: Optional[str], etag: str) -> bool:
    """Return True if any tag in an ``If-None-Match`` header matches ``etag``.

    Per RFC 7232 the header is a comma-separated list of entity-tags or the
    literal ``*``. We compare strong-only (we never emit a weak ETag) and
    tolerate optional surrounding whitespace.
    """
    if not if_none_match:
        return False
    candidates = [c.strip() for c in if_none_match.split(",")]
    if "*" in candidates:
        return True
    return etag in candidates


def _serve_variant(
    asset_id: str,
    variant: str,
    if_none_match: Optional[str] = None,
) -> Response:
    """Read a variant from the store and wrap it in a cacheable response.

    Common helper for the three GET endpoints; consolidates the headers so
    the original / thumb / preview routes can't drift on caching policy.
    Honours ``If-None-Match`` by returning 304 *before* reading the blob —
    the DB-only ``stat`` call is enough to compute the ETag.
    """
    store = _store()

    # Compute ETag from the asset row first so a conditional request never
    # has to read disk. ``stat`` is a single SELECT; if the asset is gone
    # we fall through to ``read_variant`` for a consistent 404 path.
    asset = store.stat(asset_id)
    etag = f'"{asset.sha256}"' if asset is not None else None

    if etag and _etag_matches(if_none_match, etag):
        return Response(
            status_code=304,
            headers={"ETag": etag, "Cache-Control": CACHE_CONTROL},
        )

    try:
        data, mime = store.read_variant(asset_id, variant)  # type: ignore[arg-type]
    except MediaAssetNotFound:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id!r} not found")
    except FileNotFoundError:
        # Row exists but blob is gone — treat as 404 from the client's
        # perspective; the actionable detail is logged for ops.
        logger.error("Asset %s has DB row but missing blob (variant=%s)", asset_id, variant)
        raise HTTPException(status_code=404, detail=f"Asset {asset_id!r} blob missing")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    headers = {
        "Cache-Control": CACHE_CONTROL,
    }
    if etag:
        headers["ETag"] = etag

    return Response(content=data, media_type=mime or RESPONSE_MIME, headers=headers)


@router.get("/v1/uploads/{asset_id}")
async def get_original(
    asset_id: str,
    if_none_match: Optional[str] = Header(default=None, alias="If-None-Match"),
) -> Response:
    """Return the original WebP variant (1568px-cap, Q85)."""
    return _serve_variant(asset_id, "original", if_none_match)


@router.get("/v1/uploads/{asset_id}/thumb")
async def get_thumb(
    asset_id: str,
    if_none_match: Optional[str] = Header(default=None, alias="If-None-Match"),
) -> Response:
    """Return the 256px thumb variant (Q80)."""
    return _serve_variant(asset_id, "thumb", if_none_match)


@router.get("/v1/uploads/{asset_id}/preview")
async def get_preview(
    asset_id: str,
    if_none_match: Optional[str] = Header(default=None, alias="If-None-Match"),
) -> Response:
    """Return the 1024px preview variant (Q82) — vision-model feed size."""
    return _serve_variant(asset_id, "preview", if_none_match)


# ---------------------------------------------------------------------------
# DELETE /v1/uploads/{asset_id}
# ---------------------------------------------------------------------------


@router.delete("/v1/uploads/{asset_id}", status_code=204)
async def delete_asset(
    asset_id: str,
    x_entity_id: Optional[str] = Header(default=None, alias="X-Entity-Id"),
) -> Response:
    """Owner-only delete. Removes all three blob variants + the DB row.

    Returns ``204 No Content`` on success. Mismatched owner is ``403``;
    unknown asset is ``404``.
    """
    entity_id = _require_entity_id(x_entity_id)
    store = _store()

    asset = store.stat(asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id!r} not found")

    # ``owner_user_id`` is nullable for tool-generated assets that have no
    # human owner. Those can't be deleted via this owner-scoped endpoint —
    # the GC sweep handles them. 403 (not 404) so the caller knows the
    # asset exists but they can't touch it.
    if asset.owner_user_id is None or asset.owner_user_id != entity_id:
        raise HTTPException(
            status_code=403,
            detail="X-Entity-Id does not match asset owner",
        )

    try:
        store.delete(asset_id)
    except Exception as e:
        logger.exception("MediaStore.delete failed for %s", asset_id)
        raise HTTPException(status_code=500, detail=f"delete failed: {e}")

    return Response(status_code=204)
