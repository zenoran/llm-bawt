"""Shared base for Sonarr/Radarr/SABnzbd async clients.

Pulls common concerns into one place:

- env-var-driven configuration with a clear error if missing
- one ``httpx.AsyncClient`` per service (lazy)
- consistent error wrapping (``ArrAPIError``) so MCP tools can return
  a structured failure instead of leaking httpx tracebacks
- a small ``api_key`` header injection helper for Sonarr/Radarr
  (SABnzbd takes its key as a query param, so it overrides)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 20.0


class ArrAPIError(RuntimeError):
    """Raised when an *arr or SAB API returns a non-2xx or invalid payload."""

    def __init__(self, service: str, status: int | None, message: str):
        super().__init__(f"[{service}] HTTP {status}: {message}")
        self.service = service
        self.status = status
        self.message = message


class ArrNotConfigured(RuntimeError):
    """Raised when the env vars for a given service are missing."""

    def __init__(self, service: str, missing: list[str]):
        super().__init__(
            f"[{service}] not configured — missing env vars: {', '.join(missing)}"
        )
        self.service = service
        self.missing = missing


class _BaseArrClient:
    """Common HTTP plumbing for Sonarr/Radarr v3-style APIs.

    Subclasses set ``service`` (used in errors/logs), ``base_url``, and
    ``api_key``, then call ``_get``/``_post``/``_put``/``_delete``.
    """

    service: str = "arr"

    def __init__(self, base_url: str, api_key: str, timeout: float = _DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={"X-Api-Key": api_key, "Accept": "application/json"},
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: Any = None,
    ) -> Any:
        try:
            resp = await self._client.request(method, path, params=params, json=json)
        except httpx.HTTPError as e:
            raise ArrAPIError(self.service, None, f"transport: {e!r}") from e
        if resp.status_code >= 400:
            # Try to surface the API's own error body — *arr usually returns
            # a list of {errorMessage, propertyName} on 400.
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:200]
            raise ArrAPIError(self.service, resp.status_code, str(detail))
        if not resp.content:
            return None
        try:
            return resp.json()
        except Exception as e:
            raise ArrAPIError(self.service, resp.status_code, f"non-JSON body: {e!r}") from e

    async def _get(self, path: str, **params: Any) -> Any:
        return await self._request("GET", path, params=params or None)

    async def _post(self, path: str, json: Any = None) -> Any:
        return await self._request("POST", path, json=json)

    async def _put(self, path: str, json: Any = None) -> Any:
        return await self._request("PUT", path, json=json)

    async def _delete(self, path: str, **params: Any) -> Any:
        return await self._request("DELETE", path, params=params or None)

    async def aclose(self) -> None:
        await self._client.aclose()


def _read_env_pair(service: str, url_var: str, key_var: str) -> tuple[str, str]:
    """Read a (url, api_key) pair from env, raising ArrNotConfigured if missing."""
    url = os.getenv(url_var)
    key = os.getenv(key_var)
    missing = [n for n, v in [(url_var, url), (key_var, key)] if not v]
    if missing:
        raise ArrNotConfigured(service, missing)
    return url, key  # type: ignore[return-value]
