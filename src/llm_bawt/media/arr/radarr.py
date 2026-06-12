"""Async client for Radarr v3 API.

Mirrors the Sonarr client shape — same lookup/add/queue/history surface,
adapted for movies.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import _BaseArrClient, _read_env_pair

logger = logging.getLogger(__name__)


class RadarrClient(_BaseArrClient):
    """Radarr v3 async client."""

    service = "radarr"
    api_prefix = "/api/v3"

    # ---- discovery -------------------------------------------------------

    async def system_status(self) -> dict:
        return await self._get(f"{self.api_prefix}/system/status")

    async def lookup(self, term: str) -> list[dict]:
        """Search for a movie by title/imdb id (``imdb:tt...``)/tmdb id."""
        return await self._get(f"{self.api_prefix}/movie/lookup", term=term) or []

    async def list_quality_profiles(self) -> list[dict]:
        return await self._get(f"{self.api_prefix}/qualityprofile") or []

    async def list_root_folders(self) -> list[dict]:
        return await self._get(f"{self.api_prefix}/rootfolder") or []

    # ---- library ---------------------------------------------------------

    async def list_movies(self) -> list[dict]:
        return await self._get(f"{self.api_prefix}/movie") or []

    async def get_movie(self, movie_id: int) -> dict | None:
        return await self._get(f"{self.api_prefix}/movie/{movie_id}")

    async def add_movie(
        self,
        lookup_result: dict,
        *,
        quality_profile_id: int,
        root_folder_path: str,
        monitored: bool = True,
        minimum_availability: str = "released",
        search_now: bool = True,
    ) -> dict:
        """Add a movie to Radarr.

        Like Sonarr's ``add_series``, this overlays user fields on top of
        a lookup result so the schema matches whatever the running Radarr
        version expects. ``minimum_availability`` is one of
        ``announced``, ``inCinemas``, ``released``, ``preDB``.
        """
        payload = dict(lookup_result)
        payload.update(
            {
                "qualityProfileId": quality_profile_id,
                "rootFolderPath": root_folder_path,
                "monitored": monitored,
                "minimumAvailability": minimum_availability,
                "addOptions": {"searchForMovie": search_now},
            }
        )
        return await self._post(f"{self.api_prefix}/movie", json=payload)

    async def trigger_movie_search(self, movie_id: int) -> dict:
        return await self._post(
            f"{self.api_prefix}/command",
            json={"name": "MoviesSearch", "movieIds": [movie_id]},
        )

    # ---- queue / history -------------------------------------------------

    async def queue(self, *, page_size: int = 50) -> dict:
        return await self._get(
            f"{self.api_prefix}/queue",
            pageSize=page_size,
            includeUnknownMovieItems=str(True).lower(),
            includeMovie=str(True).lower(),
        ) or {"records": [], "totalRecords": 0}

    async def history(
        self,
        *,
        page_size: int = 20,
        event_type: int | None = None,
    ) -> dict:
        """Radarr history (event types differ slightly from Sonarr).

        1 = grabbed, 2 = downloadFolderImported, 3 = downloadFailed,
        4 = movieFileDeleted, 5 = movieFolderImported, 6 = movieFileRenamed,
        7 = downloadIgnored.
        """
        params: dict[str, Any] = {
            "page": 1,
            "pageSize": page_size,
            "sortKey": "date",
            "sortDirection": "descending",
        }
        if event_type is not None:
            params["eventType"] = event_type
        return await self._get(f"{self.api_prefix}/history", **params) or {
            "records": [],
            "totalRecords": 0,
        }

    async def missing(self, *, page_size: int = 20, monitored: bool = True) -> dict:
        return await self._get(
            f"{self.api_prefix}/wanted/missing",
            page=1,
            pageSize=page_size,
            sortKey="releaseDate",
            sortDirection="descending",
            monitored=str(monitored).lower(),
        ) or {"records": [], "totalRecords": 0}


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_singleton: RadarrClient | None = None


def get_radarr_client() -> RadarrClient:
    global _singleton
    if _singleton is None:
        url, key = _read_env_pair(
            "radarr",
            "LLM_BAWT_RADARR_URL",
            "LLM_BAWT_RADARR_API_KEY",
        )
        _singleton = RadarrClient(
            url, key, timeout=float(os.getenv("LLM_BAWT_ARR_TIMEOUT", "20"))
        )
    return _singleton
