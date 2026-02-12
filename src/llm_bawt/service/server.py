"""Background service compatibility re-exports."""

from .api import DEFAULT_HTTP_PORT, SERVICE_VERSION, app, main
from .background_service import BackgroundService

__all__ = [
    "BackgroundService",
    "app",
    "main",
    "DEFAULT_HTTP_PORT",
    "SERVICE_VERSION",
]
