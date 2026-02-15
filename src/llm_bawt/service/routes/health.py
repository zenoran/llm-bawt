"""Health and service status routes."""

import dataclasses

from fastapi import APIRouter

from ..dependencies import get_service
from ..schemas import HealthResponse, ServiceStatusResponse, SystemStatusResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@router.get("/status", response_model=ServiceStatusResponse, tags=["System"])
async def get_status():
    """Get detailed service status."""
    return get_service().get_status()


@router.get("/v1/status", response_model=SystemStatusResponse, tags=["System"])
async def get_system_status():
    """Get full system status (config, service, memory, dependencies)."""
    from llm_bawt.core.status import collect_system_status

    service = get_service()
    status = collect_system_status(service.config)
    return SystemStatusResponse(**dataclasses.asdict(status))


# -------------------------------------------------------------------------
# Nextcloud Talk Webhook
# -------------------------------------------------------------------------
