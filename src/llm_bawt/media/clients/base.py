"""Abstract base class for media generation clients."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationResult:
    """Result returned by a media generation client."""

    provider_job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: int = 0  # 0-100
    media_url: Optional[str] = None  # Temporary URL to download completed media
    revised_prompt: Optional[str] = None
    error: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    actual_duration: Optional[float] = None
    extra: dict = field(default_factory=dict)  # Provider-specific metadata


class MediaClient(abc.ABC):
    """Abstract base class for provider-specific media generation clients.

    Each provider (xAI, OpenAI, Stability, etc.) implements this interface
    to provide a unified media generation API.
    """

    @abc.abstractmethod
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
        """Submit a media generation request.

        Args:
            prompt: Text prompt for generation.
            media_type: 'video' or 'image'.
            model: Model identifier.
            source_image: Optional base64 data URI or URL for image-to-video.
            aspect_ratio: Desired aspect ratio.
            duration: Duration in seconds (video only).
            resolution: Desired resolution.
            num_outputs: Number of outputs (image only).

        Returns:
            GenerationResult with at least provider_job_id and status.
        """
        ...

    @abc.abstractmethod
    async def poll_status(self, provider_job_id: str) -> GenerationResult:
        """Check the status of a previously submitted generation job.

        Args:
            provider_job_id: The job ID returned by generate().

        Returns:
            Updated GenerationResult.
        """
        ...

    @abc.abstractmethod
    async def download(self, media_url: str) -> bytes:
        """Download completed media from the provider's temporary URL.

        Args:
            media_url: URL to download the media from.

        Returns:
            Raw bytes of the media file.
        """
        ...
