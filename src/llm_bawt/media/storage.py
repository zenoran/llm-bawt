"""Filesystem blob manager for media generation outputs.

Content-addressed storage using SHA256 hashes. No binary data in the database;
files are organized on disk with a two-level directory prefix for shard distribution.

Layout:
  media/videos/{sha256[:2]}/{sha256[2:4]}/{sha256}.mp4
  media/thumbnails/{sha256[:2]}/{sha256[2:4]}/{sha256}.jpg
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_PATH = "/app/storage/media"


class MediaStorage:
    """Content-addressed filesystem storage for media blobs."""

    def __init__(self, base_path: str | None = None):
        self.base_path = Path(
            base_path or os.environ.get("MEDIA_STORAGE_PATH", DEFAULT_STORAGE_PATH)
        )
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info("MediaStorage initialized at %s", self.base_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, data: bytes, subdir: str, extension: str) -> tuple[str, str]:
        """Write data to content-addressed storage.

        Args:
            data: Raw bytes to store.
            subdir: Subdirectory name (e.g. 'videos', 'thumbnails').
            extension: File extension including dot (e.g. '.mp4', '.jpg').

        Returns:
            (relative_path, sha256_hex) where relative_path is relative to base_path.
        """
        sha256_hex = hashlib.sha256(data).hexdigest()
        rel_path = self._build_relative_path(sha256_hex, subdir, extension)
        abs_path = self.base_path / rel_path

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(data)

        logger.debug("Wrote %d bytes to %s (sha256=%s)", len(data), rel_path, sha256_hex[:12])
        return str(rel_path), sha256_hex

    async def write_async(self, data: bytes, subdir: str, extension: str) -> tuple[str, str]:
        """Async wrapper around write (runs in thread pool)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.write, data, subdir, extension)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative path to an absolute filesystem path.

        Args:
            relative_path: Path relative to the storage base.

        Returns:
            Absolute Path object.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        abs_path = self.base_path / relative_path
        if not abs_path.is_file():
            raise FileNotFoundError(f"Media file not found: {abs_path}")
        return abs_path

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, relative_path: str) -> bool:
        """Delete a file and remove empty parent directories.

        Args:
            relative_path: Path relative to the storage base.

        Returns:
            True if file was deleted, False if it didn't exist.
        """
        abs_path = self.base_path / relative_path
        if not abs_path.is_file():
            return False

        abs_path.unlink()
        logger.debug("Deleted %s", relative_path)

        # Clean up empty parent directories up to base_path
        parent = abs_path.parent
        while parent != self.base_path:
            try:
                parent.rmdir()  # Only removes if empty
                parent = parent.parent
            except OSError:
                break

        return True

    async def delete_async(self, relative_path: str) -> bool:
        """Async wrapper around delete."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.delete, relative_path)

    # ------------------------------------------------------------------
    # Poster frame extraction
    # ------------------------------------------------------------------

    async def extract_poster_frame(self, video_path: str) -> str | None:
        """Extract the first frame of a video as a JPEG thumbnail.

        Uses ffmpeg subprocess to extract a single frame. The thumbnail is
        stored in the thumbnails subdirectory with the same SHA256 naming.

        Args:
            video_path: Relative path to the video file in storage.

        Returns:
            Relative path to the thumbnail, or None on failure.
        """
        try:
            abs_video = self.resolve(video_path)
        except FileNotFoundError:
            logger.error("Cannot extract poster frame: video not found at %s", video_path)
            return None

        # Read video to compute its SHA for thumbnail naming
        video_data = abs_video.read_bytes()
        sha256_hex = hashlib.sha256(video_data).hexdigest()

        thumb_rel = self._build_relative_path(sha256_hex, "thumbnails", ".jpg")
        thumb_abs = self.base_path / thumb_rel
        thumb_abs.parent.mkdir(parents=True, exist_ok=True)

        # If thumbnail already exists (same video content), skip extraction
        if thumb_abs.is_file():
            logger.debug("Thumbnail already exists: %s", thumb_rel)
            return str(thumb_rel)

        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-i", str(abs_video),
                "-vframes", "1", "-f", "image2",
                "-y",  # Overwrite
                str(thumb_abs),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.error("ffmpeg poster extraction failed (rc=%d): %s", proc.returncode, stderr.decode()[:500])
                return None

            logger.debug("Extracted poster frame to %s", thumb_rel)
            return str(thumb_rel)

        except FileNotFoundError:
            logger.warning("ffmpeg not found; cannot extract poster frame")
            return None
        except Exception as e:
            logger.error("Failed to extract poster frame: %s", e)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_relative_path(sha256_hex: str, subdir: str, extension: str) -> Path:
        """Build the content-addressed relative path.

        Pattern: {subdir}/{sha256[:2]}/{sha256[2:4]}/{sha256}{extension}
        """
        return Path(subdir) / sha256_hex[:2] / sha256_hex[2:4] / f"{sha256_hex}{extension}"
