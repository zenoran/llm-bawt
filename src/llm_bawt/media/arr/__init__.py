"""Async clients for the *arr media stack.

Wraps Sonarr (TV), Radarr (movies), and SABnzbd (download client) HTTP
APIs behind small, typed async clients. The MCP server's ``media_*``
tools layer on top of these to give bots a unified media library
interface.

Configuration is read from env vars:

    LLM_BAWT_SONARR_URL       e.g. http://sonarr-host:8989
    LLM_BAWT_SONARR_API_KEY
    LLM_BAWT_RADARR_URL       e.g. http://radarr-host:7878
    LLM_BAWT_RADARR_API_KEY
    LLM_BAWT_SABNZBD_URL      e.g. http://sabnzbd-host:8080
    LLM_BAWT_SABNZBD_API_KEY

Each client is a lazy singleton; the first ``get_*_client()`` call
creates the underlying ``httpx.AsyncClient`` and reuses it for the
lifetime of the process.
"""

from .base import ArrAPIError, ArrNotConfigured
from .radarr import RadarrClient, get_radarr_client
from .sabnzbd import SabnzbdClient, get_sabnzbd_client
from .sonarr import SonarrClient, get_sonarr_client

__all__ = [
    "ArrAPIError",
    "ArrNotConfigured",
    "RadarrClient",
    "SabnzbdClient",
    "SonarrClient",
    "get_radarr_client",
    "get_sabnzbd_client",
    "get_sonarr_client",
]
