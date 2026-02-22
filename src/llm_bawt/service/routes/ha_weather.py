"""Home Assistant weather proxy route."""

import time
from typing import Any

import httpx
from fastapi import APIRouter, Query

from ..dependencies import get_service

router = APIRouter()

_cache: dict[str, Any] = {}
_cache_ts: float = 0.0
_CACHE_TTL = 900  # 15 minutes


def _ha_base_url(config: Any) -> str | None:
    """Derive HA base URL from native MCP URL."""
    url = config.HA_NATIVE_MCP_URL
    if url and "/api/mcp" in url:
        return url.rsplit("/api/mcp", 1)[0]
    return None


@router.get("/v1/ha/weather", tags=["Home Assistant"])
async def get_ha_weather(entity_id: str = Query(default="weather.home")):
    """Fetch current weather state from Home Assistant."""
    global _cache, _cache_ts

    now = time.monotonic()
    cache_key = entity_id
    if cache_key in _cache and (now - _cache_ts) < _CACHE_TTL:
        return _cache[cache_key]

    service = get_service()
    base_url = _ha_base_url(service.config)
    token = service.config.HA_NATIVE_MCP_TOKEN

    if not base_url or not token:
        return {"error": "Home Assistant not configured"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{base_url}/api/states/{entity_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code != 200:
                return {"error": f"HA returned {resp.status_code} for {entity_id}"}
            data = resp.json()
    except Exception as exc:
        return {"error": str(exc)}

    attrs = data.get("attributes", {})
    result = {
        "state": data.get("state", "unknown"),
        "temperature": attrs.get("temperature"),
        "temperature_unit": attrs.get("temperature_unit", "°F"),
        "apparent_temperature": attrs.get("apparent_temperature"),
        "humidity": attrs.get("humidity"),
        "wind_speed": attrs.get("wind_speed"),
        "wind_speed_unit": attrs.get("wind_speed_unit"),
        "friendly_name": attrs.get("friendly_name", entity_id),
    }

    _cache[cache_key] = result
    _cache_ts = now
    return result
