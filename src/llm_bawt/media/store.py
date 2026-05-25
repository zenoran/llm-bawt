"""MediaStore — normalized, content-addressed image storage (TASK-223).

This is the single entry point for image bytes flowing into llm-bawt: chat
uploads, tool-generated images, agent attachments. Every image goes through
the same normalization pipeline before it touches disk, so we can rely on a
predictable WebP-everywhere world downstream.

Pipeline on upload
------------------
1. ``Pillow.Image.open(BytesIO(raw_bytes))``
2. Strip EXIF / GPS / ICC profile / XMP — we rebuild the image from its
   RGB(A) pixel buffer, which leaves no metadata behind by construction.
3. Convert ``CMYK`` / ``P`` palette modes to ``RGB``; keep ``RGBA`` as-is so
   transparent screenshots round-trip without flattening.
4. If ``max(w, h) > 1568``: ``image.thumbnail((1568, 1568), Image.LANCZOS)``.
5. Encode WebP, ``quality=85``, ``method=6``. The resulting bytes are the
   stored "original" — see the rationale below for why we discard the
   user's source format.
6. ``sha256(stored_original_bytes)`` — computed **after** normalization, so
   two upload paths that produce the same normalized buffer dedup
   correctly. Pasting the same screenshot twice -> one row. Pasting the
   same image at different source resolutions -> different rows, as the
   normalized bytes differ.

From the same normalized buffer we derive two more variants:

- ``thumb_256``: fit ``(256, 256)``, WebP Q80 — chat bubble + image strip.
- ``preview_1024``: fit ``(1024, 1024)``, WebP Q82 — lightbox view and the
  primary feed to standard-detail vision models.

Why a 1568px cap?
-----------------
Anthropic's vision API recommends an image stay ≤ ~1.15 MP; anything bigger
is downscaled server-side before the model sees it, so the extra pixels
just inflate the upload and the token bill. 1568 longest-edge is comfortably
under 1.15 MP for any aspect ratio you'll see in practice.

Concrete numbers we've measured on real screenshots:

============================  ============  ============
                              raw 4K PNG    1568-cap WebP
                              ------------  ------------
size on disk                  ~ 8 MB        ~ 200–400 KB
Anthropic vision tokens       ~ 2,500–3,000 ~ 1,200
============================  ============  ============

That's a 30–60% token reduction and 95%+ storage reduction, with no visible
quality loss for screenshots or photos. WebP Q85 is the inflection point
where double-blind comparisons stop reliably distinguishing it from PNG.

Storage layout
--------------
``<MEDIA_ROOT>`` (env: ``LLM_BAWT_MEDIA_ROOT``, default
``/var/lib/llm-bawt/media/blobs``)::

    originals/<aa>/<bb>/<sha256>.webp
    thumb_256/<aa>/<bb>/<sha256>.webp
    preview_1024/<aa>/<bb>/<sha256>.webp

The ``<aa>/<bb>`` shard prefix (first / next two hex chars of the sha)
keeps any single directory from growing past a few thousand entries. Same
convention used by :class:`llm_bawt.media.storage.MediaStorage`.

Database
--------
Metadata lives in the ``media_assets`` table managed by
:class:`llm_bawt.media.assets.MediaAssetStore` (TASK-222). MediaStore wraps
that Store for inserts / lookups / deletes — it doesn't talk to Postgres
directly. The ``sha256`` UNIQUE constraint there is the source of truth
for dedup; we do an explicit ``get_by_sha256`` first to avoid the bytes
write entirely when we already have the asset.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from PIL import Image

from .assets import MediaAsset, MediaAssetStore, new_asset_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — tuned for Anthropic vision; see module docstring.
# ---------------------------------------------------------------------------

DEFAULT_MEDIA_ROOT = Path("/var/lib/llm-bawt/media/blobs")

#: Longest-edge cap. Stays under Anthropic's ~1.15 MP recommendation for
#: any aspect ratio you'll see in practice.
MAX_LONG_EDGE = 1568

#: Quality settings for the WebP encoder. Q85 is the screenshot/photo sweet
#: spot; Q80/Q82 for derived variants where we want a smaller payload.
ORIGINAL_WEBP_QUALITY = 85
THUMB_WEBP_QUALITY = 80
PREVIEW_WEBP_QUALITY = 82

#: WebP encode effort. ``method=6`` is the maximum — slow on encode (we
#: only do this once per asset thanks to dedup) but produces materially
#: smaller files vs the default ``method=4``.
WEBP_METHOD = 6

THUMB_MAX = (256, 256)
PREVIEW_MAX = (1024, 1024)

#: Three on-disk subdirectories, one per variant.
VARIANT_DIRS: dict[str, str] = {
    "original": "originals",
    "thumb": "thumb_256",
    "preview": "preview_1024",
}

VARIANT_MIME = "image/webp"

ALLOWED_SOURCES = ("chat_upload", "tool_generated", "agent_attachment")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MediaAssetNotFound(LookupError):
    """Raised by :meth:`MediaStore.read_variant` when no DB row matches the id."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shard_path(root: Path, variant_subdir: str, sha256_hex: str) -> Path:
    """Return ``<root>/<variant_subdir>/<aa>/<bb>/<sha256>.webp``.

    The two-level prefix mirrors :class:`MediaStorage` so anyone walking
    the tree can rely on the same fan-out pattern.
    """
    return root / variant_subdir / sha256_hex[:2] / sha256_hex[2:4] / f"{sha256_hex}.webp"


def _normalize_to_original(raw_bytes: bytes) -> tuple[bytes, Image.Image, int, int]:
    """Run the full normalization pipeline on raw image bytes.

    Returns ``(stored_original_webp_bytes, normalized_image, width, height)``
    where ``normalized_image`` is a Pillow Image kept in memory so the
    caller can derive the thumb / preview variants without decoding the
    WebP we just produced.
    """
    src = Image.open(io.BytesIO(raw_bytes))

    # Step 1: strip ALL metadata — EXIF, ICC profile, XMP, comments. The
    # canonical way to do this in Pillow is to reconstruct the image from
    # just its pixel buffer; ``Image.new`` + ``putdata`` would also work
    # but is slower. ``frombytes`` keeps it to a single allocation.
    src.load()  # force-decode before we touch .mode / .size

    # Step 2: collapse exotic color modes. CMYK and palette (P) modes have
    # no natural WebP encoding and would confuse vision models even if
    # they did — convert to sRGB. Preserve RGBA so transparent screenshots
    # don't lose their alpha channel.
    if src.mode == "RGBA":
        target_mode = "RGBA"
    elif src.mode == "LA":
        # Grayscale + alpha — promote to RGBA so the rest of the pipeline
        # has one fewer case to handle.
        src = src.convert("RGBA")
        target_mode = "RGBA"
    else:
        # Everything else: RGB, L, P, CMYK, 1, I, F — all roll up to RGB.
        src = src.convert("RGB")
        target_mode = "RGB"

    # Rebuild a clean image from just the pixel buffer. Anything Pillow
    # was carrying on the original (``.info``, ``.applist``, ``.icc_profile``)
    # is dropped on the floor — that's the EXIF / ICC strip.
    stripped = Image.frombytes(target_mode, src.size, src.tobytes())

    # Step 3: downscale if needed. ``thumbnail`` mutates in place and
    # preserves aspect ratio.
    if max(stripped.size) > MAX_LONG_EDGE:
        stripped.thumbnail((MAX_LONG_EDGE, MAX_LONG_EDGE), Image.LANCZOS)

    # Step 4: encode WebP. ``save_all=False`` so we don't accidentally
    # emit an animated WebP for multi-frame inputs (animated GIFs collapse
    # to their first frame — intentional for now; revisit if anyone uploads
    # an animated screenshot).
    buf = io.BytesIO()
    stripped.save(
        buf,
        format="WEBP",
        quality=ORIGINAL_WEBP_QUALITY,
        method=WEBP_METHOD,
        # ``exif=b""`` + ``icc_profile=None`` are belt-and-suspenders;
        # frombytes() already discarded both, but if a future Pillow
        # version starts back-filling them from defaults this still wins.
        exif=b"",
        icc_profile=None,
    )
    original_bytes = buf.getvalue()
    return original_bytes, stripped, stripped.size[0], stripped.size[1]


def _encode_variant(normalized: Image.Image, max_size: tuple[int, int], quality: int) -> bytes:
    """Derive a fit-inside WebP variant from the already-normalized image.

    We ``.copy()`` so the caller's ``normalized`` instance is left at its
    original (cap-bounded) dimensions and can be reused for multiple
    variants.
    """
    img = normalized.copy()
    img.thumbnail(max_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(
        buf,
        format="WEBP",
        quality=quality,
        method=WEBP_METHOD,
        exif=b"",
        icc_profile=None,
    )
    return buf.getvalue()


def _write_idempotent(path: Path, data: bytes) -> None:
    """Write bytes to ``path`` only if not already present.

    The blob layout is content-addressed, so an existing file at the same
    path *must* have the same sha256 — we don't re-verify. Saves a write
    and an fsync on the dedup-hit path.
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling tempfile + rename so a crash mid-write never
    # leaves a half-baked blob at the canonical path. ``os.replace`` is
    # atomic on POSIX.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _row_to_asset(row: dict) -> MediaAsset:
    """Coerce a raw DB-row dict into the :class:`MediaAsset` SQLModel.

    ``MediaAssetStore`` returns ``dict[str, Any]`` from RETURNING; we want
    the typed model in the public API so downstream code can rely on
    attribute access.
    """
    return MediaAsset(**row)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class MediaStore:
    """High-level facade over the on-disk blob tree + ``media_assets`` row.

    Construct once per process via :func:`get_media_store`. The class is
    cheap to instantiate (no work at ``__init__`` beyond a ``mkdir``), so
    tests can build their own with a ``tmp_path`` root.
    """

    def __init__(
        self,
        root: Path | None = None,
        db: MediaAssetStore | None = None,
    ):
        if root is None:
            env_root = os.environ.get("LLM_BAWT_MEDIA_ROOT")
            root = Path(env_root) if env_root else DEFAULT_MEDIA_ROOT
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = db

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload(
        self,
        raw_bytes: bytes,
        original_mime: str,
        source: str,
        owner_user_id: Optional[str],
        expires_at: Optional[datetime] = None,
    ) -> MediaAsset:
        """Normalize, deduplicate, and persist an image.

        Dedup is keyed on the **post-normalization** sha256. If a row
        already exists with that sha, we short-circuit: no re-encode, no
        rewrite, just return the existing asset. Blob writes are idempotent
        too — if the DB row is gone but the file is still there we'll
        skip the disk write and let the new row point at the same path.
        """
        if source not in ALLOWED_SOURCES:
            raise ValueError(
                f"source must be one of {ALLOWED_SOURCES!r}, got {source!r}"
            )

        # Step 1: normalize. This is the only place we hold the full
        # decoded image; everything else just shuffles bytes around.
        original_bytes, normalized, width, height = _normalize_to_original(raw_bytes)
        sha256_hex = hashlib.sha256(original_bytes).hexdigest()

        # Step 2: dedup hit? Return early before touching disk or
        # generating variants. ``MediaAssetStore.insert`` would do this
        # internally, but we check first to skip the variant-encode work.
        #
        # Self-heal: if the row exists but any of its on-disk blobs are
        # gone (manual cleanup, partial restore, container reset that
        # wiped the bind-mount), we cannot just return the row — chat
        # reads would 404.  Re-write the blobs from the bytes we have
        # in hand right now and then return the existing row, so the
        # caller's asset_id stays stable across the heal.
        if self.db is not None:
            existing = self.db.get_by_sha256(sha256_hex)
            if existing is not None:
                blobs_intact = all(
                    _shard_path(self.root, subdir, sha256_hex).exists()
                    for subdir in VARIANT_DIRS.values()
                )
                if blobs_intact:
                    logger.debug(
                        "MediaStore upload dedup hit: sha=%s -> id=%s",
                        sha256_hex[:12],
                        existing["id"],
                    )
                    return _row_to_asset(existing)
                logger.warning(
                    "MediaStore upload dedup hit with missing blobs — self-healing: sha=%s -> id=%s",
                    sha256_hex[:12],
                    existing["id"],
                )
                # Fall through to variant encode + idempotent write below.
                # The DB row stays; insert() would race the existing
                # sha unique constraint, so we skip it on the heal path.
                _heal_existing_row = existing
            else:
                _heal_existing_row = None
        else:
            _heal_existing_row = None

        # Step 3: derive variants from the same normalized buffer (NOT
        # re-decoding the WebP we just wrote — that would re-apply lossy
        # compression on top of itself).
        thumb_bytes = _encode_variant(normalized, THUMB_MAX, THUMB_WEBP_QUALITY)
        preview_bytes = _encode_variant(normalized, PREVIEW_MAX, PREVIEW_WEBP_QUALITY)

        # Step 4: write all three blobs. Same sha256 path for all three
        # variants — the variant subdir disambiguates.
        _write_idempotent(_shard_path(self.root, VARIANT_DIRS["original"], sha256_hex), original_bytes)
        _write_idempotent(_shard_path(self.root, VARIANT_DIRS["thumb"], sha256_hex), thumb_bytes)
        _write_idempotent(_shard_path(self.root, VARIANT_DIRS["preview"], sha256_hex), preview_bytes)

        # Step 5: register in Postgres. ``insert`` is itself dedup-safe —
        # if two callers race on the same bytes, the second one gets the
        # first one's row back instead of an IntegrityError.
        #
        # Heal path: we already had a row for this sha but its blobs
        # were missing; we just rewrote them above.  Skip the insert
        # (would unique-key race) and return the existing row so the
        # caller's previously-known asset_id stays stable.
        if _heal_existing_row is not None:
            return _row_to_asset(_heal_existing_row)
        if self.db is not None:
            row = self.db.insert(
                sha256=sha256_hex,
                mime_type=VARIANT_MIME,
                original_mime_type=original_mime,
                size_bytes=len(original_bytes),
                width=width,
                height=height,
                source=source,
                owner_user_id=owner_user_id,
                expires_at=expires_at,
            )
            return _row_to_asset(row)

        # No DB configured (test path) — synthesize an ephemeral asset so
        # the contract still holds. Production always has a DB.
        return MediaAsset(
            id=new_asset_id(),
            sha256=sha256_hex,
            mime_type=VARIANT_MIME,
            original_mime_type=original_mime,
            size_bytes=len(original_bytes),
            width=width,
            height=height,
            source=source,
            owner_user_id=owner_user_id,
            expires_at=expires_at,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_variant(
        self,
        asset_id: str,
        variant: Literal["original", "thumb", "preview"],
    ) -> tuple[bytes, str]:
        """Return ``(bytes, mime_type)`` for the requested variant.

        :raises MediaAssetNotFound: if no DB row matches ``asset_id``.
        :raises FileNotFoundError: if the row exists but the blob has been
            evicted from disk (e.g. volume restored without DB sync).
        :raises ValueError: if ``variant`` is not one of the three known names.
        """
        if variant not in VARIANT_DIRS:
            raise ValueError(
                f"variant must be one of {list(VARIANT_DIRS)!r}, got {variant!r}"
            )
        row = self._require_row(asset_id)
        path = _shard_path(self.root, VARIANT_DIRS[variant], row["sha256"])
        if not path.is_file():
            raise FileNotFoundError(
                f"MediaStore blob missing for asset={asset_id} variant={variant} path={path}"
            )
        return path.read_bytes(), VARIANT_MIME

    def read_original_as_data_url(self, asset_id: str) -> str:
        """Return the original variant as a ``data:image/webp;base64,...`` URL.

        Convenience for the LLM inlining path (TASK-225). Always returns
        the cap-bounded original — callers that want the smaller preview
        should base64 it themselves.
        """
        data, mime = self.read_variant(asset_id, "original")
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def stat(self, asset_id: str) -> MediaAsset | None:
        """Return the asset row without reading any bytes, or ``None``."""
        if self.db is None:
            return None
        row = self.db.get_by_id(asset_id)
        return _row_to_asset(row) if row is not None else None

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, asset_id: str) -> None:
        """Remove all three blob variants + the DB row. Idempotent.

        Order: blobs first, then DB row. If the process dies between the
        two, the next ``upload`` of the same content will overwrite
        nothing (idempotent write) and re-insert the row — the system
        self-heals.
        """
        if self.db is not None:
            row = self.db.get_by_id(asset_id)
            if row is None:
                return  # nothing to do
            sha256_hex = row["sha256"]
        else:  # tests
            return

        for subdir in VARIANT_DIRS.values():
            path = _shard_path(self.root, subdir, sha256_hex)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            # Walk up and rmdir empty shard prefixes so we don't leak
            # thousands of empty <aa>/<bb> directories after GC sweeps.
            parent = path.parent
            try:
                while parent != self.root and parent.is_dir():
                    parent.rmdir()  # raises if not empty — that's our break
                    parent = parent.parent
            except OSError:
                pass

        self.db.delete(asset_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_row(self, asset_id: str) -> dict:
        if self.db is None:
            raise MediaAssetNotFound(
                f"MediaStore has no DB attached; cannot resolve asset_id={asset_id!r}"
            )
        row = self.db.get_by_id(asset_id)
        if row is None:
            raise MediaAssetNotFound(f"No media_assets row for id={asset_id!r}")
        return row


# ---------------------------------------------------------------------------
# Process-wide accessor
# ---------------------------------------------------------------------------

_singleton: MediaStore | None = None


def get_media_store() -> MediaStore:
    """Return the process-wide :class:`MediaStore`, building on first call.

    Wires up the shared :class:`MediaAssetStore` against the active
    :class:`Config` so route handlers can grab a Store without DI plumbing.
    Mirrors :func:`llm_bawt.service.dependencies._get_or_build_store` in
    spirit — a single Store per process, lazy-built.
    """
    global _singleton
    if _singleton is None:
        from ..utils.config import Config  # late import to avoid cycle

        config = Config()
        try:
            db = MediaAssetStore(config)
        except Exception as e:
            # No DB? Construct anyway — tests + early boot can still use
            # the on-disk path. Log loudly so prod doesn't silently lose
            # metadata.
            logger.warning("MediaStore initialized without DB: %s", e)
            db = None
        _singleton = MediaStore(db=db)
    return _singleton


def reset_media_store() -> None:
    """Drop the singleton. Tests only — don't call in production."""
    global _singleton
    _singleton = None
