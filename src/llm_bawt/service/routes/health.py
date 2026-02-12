"""Health and service status routes."""

from fastapi import APIRouter

from ..dependencies import get_service
from ..schemas import HealthResponse, ServiceStatusResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse()

@router.get("/status", response_model=ServiceStatusResponse, tags=["System"])
async def get_status():
    """Get detailed service status."""
    return get_service().get_status()

# -------------------------------------------------------------------------
# Nextcloud Talk Webhook
# -------------------------------------------------------------------------
