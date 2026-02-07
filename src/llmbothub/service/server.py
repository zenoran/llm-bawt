"""
Background service for llmbothub.

This module re-exports from api.py for backward compatibility.
The actual implementation is in api.py.

Run with: llm-service (requires: pip install llmbothub[service])
"""

# Re-export everything from api.py
from .api import (
    BackgroundService,
    app,
    main,
    DEFAULT_HTTP_PORT,
    SERVICE_VERSION,
)

__all__ = [
    "BackgroundService",
    "app",
    "main",
    "DEFAULT_HTTP_PORT",
    "SERVICE_VERSION",
]
