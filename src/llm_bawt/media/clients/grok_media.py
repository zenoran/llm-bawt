"""xAI (Grok) media generation client.

Uses httpx to call the xAI video/image generation API directly,
since these endpoints are not part of the OpenAI SDK.

API flow for video generation:
1. POST https://api.x.ai/v1/videos/generations  -> { request_id, status }
2. GET  https://api.x.ai/v1/videos/{request_id} -> poll until status == "done"
3. Download video from the returned URL
"""

from __future__ import annotations

import logging
import os

import httpx

from .base import GenerationResult, MediaClient

logger = logging.getLogger(__name__)

XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_TIMEOUT = 60.0


class GrokMediaClient(MediaClient):
    """xAI media generation client using httpx."""

    def __init__(self, api_key: str | None = None, base_url: str = XAI_BASE_URL):
        self.api_key = api_key or os.environ.get("XAI_API_KEY") or os.environ.get("LLM_BAWT_XAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")

        if not self.api_key:
            logger.warning(
                "GrokMediaClient: No xAI API key found. "
                "Set XAI_API_KEY or LLM_BAWT_XAI_API_KEY environment variable."
            )

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(DEFAULT_TIMEOUT, connect=10.0),
        )

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # MediaClient interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        media_type: str,
        model: str,
        *,
        source_image: str | None = None,
        aspect_ratio: str = "16:9",
        duration: float = 5,
        resolution: str = "720p",
        num_outputs: int = 1,
    ) -> GenerationResult:
        """Submit a video or image generation request to xAI."""
        if media_type == "video":
            return await self._generate_video(
                prompt=prompt,
                model=model,
                source_image=source_image,
                aspect_ratio=aspect_ratio,
                duration=duration,
                resolution=resolution,
            )
        else:
            # Image generation placeholder (xAI may add this later)
            raise NotImplementedError(
                f"xAI image generation is not yet supported. media_type={media_type}"
            )

    async def poll_status(self, provider_job_id: str) -> GenerationResult:
        """Poll the status of a video generation job."""
        resp = await self._client.get(f"/videos/{provider_job_id}")

        # 4xx errors are permanent — return as failed instead of raising
        if 400 <= resp.status_code < 500:
            error_text = resp.text[:300]
            logger.error("xAI poll returned %s: %s", resp.status_code, error_text)
            return GenerationResult(
                provider_job_id=provider_job_id,
                status="failed",
                progress=0,
                error=f"Provider error ({resp.status_code}): {error_text}",
            )

        resp.raise_for_status()
        data = resp.json()

        status = self._map_status(data.get("status", ""))
        progress = self._estimate_progress(status, data)

        result = GenerationResult(
            provider_job_id=provider_job_id,
            status=status,
            progress=progress,
            revised_prompt=data.get("revised_prompt"),
        )

        if status == "completed":
            # xAI nests video info under data["video"]
            video_obj = data.get("video") or {}
            result.media_url = (
                video_obj.get("url")
                or data.get("url")
                or data.get("video_url")
            )
            result.progress = 100
            result.width = data.get("width") or video_obj.get("width")
            result.height = data.get("height") or video_obj.get("height")
            result.actual_duration = video_obj.get("duration") or data.get("duration")
        elif status == "failed":
            result.error = data.get("error", {}).get("message", "Unknown error") if isinstance(data.get("error"), dict) else str(data.get("error", "Unknown error"))

        return result

    async def download(self, media_url: str) -> bytes:
        """Download media from xAI's temporary URL."""
        # Use a separate client without base_url for absolute URLs
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            resp = await client.get(media_url)
            resp.raise_for_status()
            return resp.content

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate_video(
        self,
        prompt: str,
        model: str,
        source_image: str | None,
        aspect_ratio: str,
        duration: float,
        resolution: str,
    ) -> GenerationResult:
        """POST to /videos/generations."""
        payload: dict = {
            "prompt": prompt,
            "model": model,
            "duration": int(duration),
            "resolution": resolution,
        }

        # Always send aspect_ratio — xAI doesn't auto-detect from image
        payload["aspect_ratio"] = aspect_ratio

        if source_image:
            # xAI expects image as an object: {"url": "data:image/...;base64,..."}
            payload["image"] = {"url": source_image}

        logger.info(
            "xAI video generation request: model=%s, duration=%s, aspect_ratio=%s, has_image=%s",
            model, duration, aspect_ratio, bool(source_image),
        )

        resp = await self._client.post("/videos/generations", json=payload)
        if resp.status_code >= 400:
            logger.error(
                "xAI video generation failed: status=%s body=%s",
                resp.status_code, resp.text[:500],
            )
        resp.raise_for_status()
        data = resp.json()

        provider_job_id = data.get("request_id") or data.get("id", "")
        status = self._map_status(data.get("status", "pending"))

        logger.info(
            "xAI video generation submitted: job_id=%s, status=%s",
            provider_job_id, status,
        )

        return GenerationResult(
            provider_job_id=provider_job_id,
            status=status,
            progress=0 if status == "pending" else 10,
            revised_prompt=data.get("revised_prompt"),
        )

    @staticmethod
    def _map_status(xai_status: str) -> str:
        """Map xAI-specific status strings to our canonical statuses."""
        mapping = {
            "pending": "pending",
            "queued": "pending",
            "in_progress": "processing",
            "processing": "processing",
            "running": "processing",
            "done": "completed",
            "completed": "completed",
            "complete": "completed",
            "failed": "failed",
            "error": "failed",
            "cancelled": "failed",
        }
        return mapping.get(xai_status.lower(), "processing")

    @staticmethod
    def _estimate_progress(status: str, data: dict) -> int:
        """Estimate progress percentage from status and response data."""
        if status == "completed":
            return 100
        if status == "failed":
            return 0
        # If the API provides a progress field, use it
        if "progress" in data:
            return min(100, max(0, int(data["progress"])))
        # Otherwise estimate based on status
        if status == "pending":
            return 5
        if status == "processing":
            return 50
        return 0
