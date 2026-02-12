"""Shared FastAPI dependencies and service helpers."""

from typing import TYPE_CHECKING, Any

try:
    from fastapi import Depends, HTTPException
except ImportError:  # pragma: no cover - FastAPI is optional at import time
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    def Depends(dep):  # type: ignore[misc]
        return dep

from .logging import generate_request_id as _generate_request_id
from .schemas import UserProfileAttribute

if TYPE_CHECKING:
    from .background_service import BackgroundService

_service: "BackgroundService | None" = None


def set_service(service: "BackgroundService | None") -> None:
    """Set the global background service instance."""
    global _service
    _service = service


def get_service() -> "BackgroundService":
    """Get the initialized background service instance."""
    if _service is None:
        raise RuntimeError("Service not initialized")
    return _service


def get_effective_bot_id(bot_id: str | None = None) -> str:
    """Resolve optional bot_id to the configured service default."""
    return bot_id or get_service()._default_bot


def require_memory_client(bot_id: str = Depends(get_effective_bot_id)):
    """Dependency that requires a live memory client for the effective bot."""
    client = get_service().get_memory_client(bot_id)
    if not client:
        raise HTTPException(status_code=503, detail="Memory service unavailable")
    return client


def attribute_to_response(attr: Any) -> UserProfileAttribute:
    """Convert a profile attribute ORM/domain object to API schema."""
    return UserProfileAttribute(
        id=attr.id,
        category=attr.category.value if hasattr(attr.category, "value") else str(attr.category),
        key=attr.key,
        value=attr.value,
        confidence=attr.confidence,
        source=attr.source,
        created_at=attr.created_at.isoformat() if attr.created_at else None,
        updated_at=attr.updated_at.isoformat() if attr.updated_at else None,
    )


def generate_request_id() -> str:
    """Expose request ID generation as a shared dependency helper."""
    return _generate_request_id()
