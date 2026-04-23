"""FastAPI routes for the unified media generation API.

Endpoints:
  POST   /v1/media/generations          — Submit a generation job
  GET    /v1/media/generations           — List generations (paginated)
  GET    /v1/media/generations/{id}      — Get status/result
  DELETE /v1/media/generations/{id}      — Delete generation + files
  GET    /v1/media/generations/{id}/content   — Stream raw media file
  GET    /v1/media/generations/{id}/thumbnail — Serve poster frame
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ...media.clients.grok_media import GrokMediaClient
from ...media.db import MediaGenerationStore
from ...media.schemas import (
    MediaGenerationListResponse,
    MediaGenerationRequest,
    MediaGenerationResponse,
    MediaOutput,
)
from ...media.storage import MediaStorage
from ...utils.config import Config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Media"])

# Module-level singletons initialised lazily
_store: MediaGenerationStore | None = None
_storage: MediaStorage | None = None
_grok_client: GrokMediaClient | None = None
# Track active polling tasks so they can be cancelled on shutdown
_poll_tasks: dict[str, asyncio.Task] = {}


def _get_store() -> MediaGenerationStore:
    global _store
    if _store is None:
        _store = MediaGenerationStore(Config())
    return _store


def _get_storage() -> MediaStorage:
    global _storage
    if _storage is None:
        _storage = MediaStorage()
    return _storage


def _get_grok_client() -> GrokMediaClient:
    global _grok_client
    if _grok_client is None:
        _grok_client = GrokMediaClient()
    return _grok_client


def _gen_id() -> str:
    """Generate a prefixed unique ID for a media generation."""
    return f"gen_{uuid.uuid4().hex[:20]}"


# ------------------------------------------------------------------
# Helpers: row -> response
# ------------------------------------------------------------------

def _row_to_response(row: dict) -> MediaGenerationResponse:
    """Convert a database row dict to a MediaGenerationResponse."""
    outputs: list[MediaOutput] = []
    if row.get("status") == "completed" and row.get("file_path"):
        gen_id = row["id"]
        output = MediaOutput(
            file_path=row.get("file_path"),
            thumbnail_path=row.get("thumbnail_path"),
            file_size_bytes=row.get("file_size_bytes"),
            mime_type=row.get("mime_type"),
            content_url=f"/v1/media/generations/{gen_id}/content",
            thumbnail_url=f"/v1/media/generations/{gen_id}/thumbnail" if row.get("thumbnail_path") else None,
            width=row.get("width"),
            height=row.get("height"),
            actual_duration=row.get("actual_duration"),
        )
        outputs.append(output)

    return MediaGenerationResponse(
        id=row["id"],
        status=row["status"],
        media_type=row["media_type"],
        prompt=row["prompt"],
        revised_prompt=row.get("revised_prompt"),
        progress=row.get("progress", 0),
        outputs=outputs,
        provider=row.get("provider"),
        model=row.get("model"),
        aspect_ratio=row.get("aspect_ratio"),
        duration=row.get("duration"),
        resolution=row.get("resolution"),
        error=row.get("error"),
        created_at=row["created_at"].isoformat() if isinstance(row.get("created_at"), datetime) else str(row.get("created_at", "")),
        completed_at=row["completed_at"].isoformat() if isinstance(row.get("completed_at"), datetime) else (str(row["completed_at"]) if row.get("completed_at") else None),
    )


# ------------------------------------------------------------------
# Background polling
# ------------------------------------------------------------------

async def _poll_video_job(gen_id: str) -> None:
    """Background task that polls xAI for video completion.

    Runs every 5 seconds until the job completes or fails, then
    downloads the video, stores it, and extracts a poster frame.
    """
    store = _get_store()
    storage = _get_storage()
    client = _get_grok_client()

    row = store.get_by_id(gen_id)
    if not row:
        logger.error("Poll task: generation %s not found", gen_id)
        return

    provider_job_id = row["provider_job_id"]
    if not provider_job_id:
        logger.error("Poll task: no provider_job_id for %s", gen_id)
        store.update(gen_id, {"status": "failed", "error": "Missing provider job ID"})
        return

    logger.info("Starting poll loop for generation %s (job %s)", gen_id, provider_job_id)

    max_polls = 120  # 120 * 5s = 10 minutes max
    transient_errors = 0
    max_transient_errors = 5

    try:
        for poll_num in range(max_polls):
            await asyncio.sleep(5)

            try:
                result = await client.poll_status(provider_job_id)
                transient_errors = 0  # Reset on success
            except Exception as e:
                transient_errors += 1
                logger.warning("Poll error for %s (%d/%d): %s", gen_id, transient_errors, max_transient_errors, e)
                if transient_errors >= max_transient_errors:
                    store.update(gen_id, {"status": "failed", "error": f"Too many poll errors: {e}"})
                    return
                continue

            # Update progress
            store.update(gen_id, {
                "status": result.status,
                "progress": result.progress,
                "revised_prompt": result.revised_prompt,
            })

            if result.status == "failed":
                store.update(gen_id, {"error": result.error or "Generation failed"})
                logger.warning("Generation %s failed: %s", gen_id, result.error)
                return

            if result.status == "completed":
                break
        else:
            # Exhausted all polls
            store.update(gen_id, {"status": "failed", "error": "Generation timed out after 10 minutes"})
            logger.error("Generation %s timed out", gen_id)
            return

        # Download the completed video
        if not result.media_url:
            store.update(gen_id, {"status": "failed", "error": "No media URL in completed response"})
            logger.error("Generation %s completed but no media URL", gen_id)
            return

        logger.info("Downloading video for %s from %s", gen_id, result.media_url[:80])
        video_data = await client.download(result.media_url)

        # Determine extension from media type
        media_type = row.get("media_type", "video")
        ext = ".mp4" if media_type == "video" else ".png"
        subdir = "videos" if media_type == "video" else "images"
        mime = "video/mp4" if media_type == "video" else "image/png"

        # Store to filesystem
        rel_path, sha256 = await storage.write_async(video_data, subdir, ext)

        updates: dict = {
            "status": "completed",
            "progress": 100,
            "file_path": rel_path,
            "file_size_bytes": len(video_data),
            "mime_type": mime,
            "completed_at": datetime.now(timezone.utc),
            "width": result.width,
            "height": result.height,
            "actual_duration": result.actual_duration,
            "revised_prompt": result.revised_prompt,
        }

        # Extract poster frame for videos
        if media_type == "video":
            thumb_path = await storage.extract_poster_frame(rel_path)
            if thumb_path:
                updates["thumbnail_path"] = thumb_path

        store.update(gen_id, updates)
        logger.info("Generation %s completed: %s (%d bytes)", gen_id, rel_path, len(video_data))

    except asyncio.CancelledError:
        logger.info("Poll task cancelled for %s", gen_id)
        raise
    except Exception as e:
        logger.exception("Unexpected error polling generation %s", gen_id)
        store.update(gen_id, {"status": "failed", "error": str(e)})
    finally:
        _poll_tasks.pop(gen_id, None)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.post("/v1/media/generations", response_model=MediaGenerationResponse)
async def create_generation(request: MediaGenerationRequest):
    """Submit a new media generation job."""
    store = _get_store()
    client = _get_grok_client()

    gen_id = _gen_id()

    # Auto-select model based on media type
    model = request.model
    if not model:
        if request.media_type == "video":
            model = "grok-imagine-video"
        else:
            model = "grok-imagine"

    # Determine provider from model name
    provider = "xai"

    # Insert the initial DB record
    row = {
        "id": gen_id,
        "status": "pending",
        "media_type": request.media_type,
        "prompt": request.prompt,
        "provider": provider,
        "model": model,
        "aspect_ratio": request.aspect_ratio,
        "duration": request.duration if request.media_type == "video" else None,
        "resolution": request.resolution,
        "progress": 0,
        "created_at": datetime.now(timezone.utc),
    }
    store.insert(row)

    # Submit to the provider
    try:
        result = await client.generate(
            prompt=request.prompt,
            media_type=request.media_type,
            model=model,
            source_image=request.source_image,
            aspect_ratio=request.aspect_ratio,
            duration=request.duration or 5,
            resolution=request.resolution,
            num_outputs=request.num_outputs,
        )

        store.update(gen_id, {
            "status": result.status,
            "provider_job_id": result.provider_job_id,
            "progress": result.progress,
            "revised_prompt": result.revised_prompt,
        })

        # Start background polling for async jobs (video)
        if result.status in ("pending", "processing"):
            task = asyncio.create_task(_poll_video_job(gen_id))
            _poll_tasks[gen_id] = task

    except Exception as e:
        logger.exception("Failed to submit generation %s", gen_id)
        store.update(gen_id, {"status": "failed", "error": str(e)})

    # Return current state
    db_row = store.get_by_id(gen_id)
    if not db_row:
        raise HTTPException(status_code=500, detail="Failed to retrieve generation after creation")
    return _row_to_response(db_row)


@router.get("/v1/media/generations/{gen_id}", response_model=MediaGenerationResponse)
async def get_generation(gen_id: str):
    """Get the status and result of a media generation."""
    store = _get_store()
    row = store.get_by_id(gen_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Generation {gen_id} not found")
    return _row_to_response(row)


@router.get("/v1/media/generations", response_model=MediaGenerationListResponse)
async def list_generations(
    media_type: str | None = Query(default=None, description="Filter by media type"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List media generations with optional filtering and pagination."""
    store = _get_store()
    rows, total = store.list_generations(media_type=media_type, limit=limit, offset=offset)
    items = [_row_to_response(r) for r in rows]
    return MediaGenerationListResponse(items=items, total=total)


@router.delete("/v1/media/generations/{gen_id}")
async def delete_generation(gen_id: str):
    """Delete a media generation and its associated files."""
    store = _get_store()
    storage = _get_storage()

    row = store.get_by_id(gen_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Generation {gen_id} not found")

    # Cancel any active polling task
    poll_task = _poll_tasks.pop(gen_id, None)
    if poll_task and not poll_task.done():
        poll_task.cancel()

    # Delete files from disk
    if row.get("file_path"):
        await storage.delete_async(row["file_path"])
    if row.get("thumbnail_path"):
        await storage.delete_async(row["thumbnail_path"])

    # Delete DB record
    store.delete(gen_id)

    return {"deleted": True, "id": gen_id}


@router.get("/v1/media/generations/{gen_id}/content")
async def stream_content(gen_id: str):
    """Stream the raw media file (video/mp4, image/png, etc.)."""
    store = _get_store()
    storage = _get_storage()

    row = store.get_by_id(gen_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Generation {gen_id} not found")

    file_path = row.get("file_path")
    if not file_path:
        raise HTTPException(status_code=404, detail="Media file not yet available (generation may still be in progress)")

    try:
        abs_path = storage.resolve(file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Media file not found on disk")

    mime_type = row.get("mime_type", "application/octet-stream")
    return FileResponse(
        path=str(abs_path),
        media_type=mime_type,
        filename=abs_path.name,
    )


@router.get("/v1/media/generations/{gen_id}/thumbnail")
async def serve_thumbnail(gen_id: str):
    """Serve the poster frame / thumbnail image."""
    store = _get_store()
    storage = _get_storage()

    row = store.get_by_id(gen_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Generation {gen_id} not found")

    thumb_path = row.get("thumbnail_path")
    if not thumb_path:
        raise HTTPException(status_code=404, detail="Thumbnail not available")

    try:
        abs_path = storage.resolve(thumb_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Thumbnail file not found on disk")

    return FileResponse(
        path=str(abs_path),
        media_type="image/jpeg",
        filename=abs_path.name,
    )
