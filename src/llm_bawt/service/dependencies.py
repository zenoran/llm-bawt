"""Shared FastAPI dependencies and service helpers.

Process-wide singleton stores
-----------------------------
The ``get_*_store`` / ``get_profile_manager`` helpers below return cached
instances of Stores so that route handlers don't pay the construction
cost (table-existence checks, schema migrations) on every request.

After the TASK-202 refactor, **all Stores share a single SQLAlchemy
engine** (see :func:`llm_bawt.utils.db.get_shared_engine`), so caching
the Store is no longer necessary to avoid connection-pool leaks — the
underlying pool is the same regardless of how many Store instances
exist. The cache survives mainly to avoid repeating
``CREATE TABLE IF NOT EXISTS`` on every call.

The cache is keyed on ``id(config)`` so a fresh config (e.g. after
``/v1/admin/reload``) yields a fresh Store. **Do NOT** call
``engine.dispose()`` on evicted Stores — the engine is shared with every
other Store and disposing it would break the whole process.
"""

import threading
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

# --------------------------------------------------------------------------
# Process-wide store singletons (TASK-202: prevent connection-pool leaks).
# --------------------------------------------------------------------------
_store_lock = threading.Lock()
_store_cache: dict[tuple[str, int], Any] = {}


def _get_or_build_store(name: str, config: Any, factory):
    """Return a cached store instance, building it on first request.

    Keyed on ``(name, id(config))`` so the cache invalidates when the
    config object is replaced (e.g. settings reload).

    NOTE (TASK-202): every Store now shares one process-wide SQLAlchemy
    engine. We do **not** call ``engine.dispose()`` on evicted Stores
    because that would tear down the shared pool used by every other
    Store. Eviction simply drops the Python reference; the shared engine
    keeps running.
    """
    key = (name, id(config))
    cached = _store_cache.get(key)
    if cached is not None:
        return cached
    with _store_lock:
        cached = _store_cache.get(key)
        if cached is not None:
            return cached
        # Drop stale instances under the same name with a different config.
        # Do NOT dispose their engine — it's the shared pool.
        for existing_key in [k for k in _store_cache if k[0] == name]:
            _store_cache.pop(existing_key, None)
        instance = factory()
        _store_cache[key] = instance
        return instance


def reset_store_cache() -> None:
    """Drop all cached stores. Used at shutdown / reload.

    Does NOT dispose the underlying shared engine — that's owned by
    :mod:`llm_bawt.utils.db` and disposing it would break every other
    Store. Call ``llm_bawt.utils.db.reset_shared_engines`` separately if
    you really need to recycle the pool (tests only).
    """
    with _store_lock:
        _store_cache.clear()


def get_profile_manager(config: Any):
    """Process-wide ``ProfileManager`` singleton. See module docstring."""
    from ..profiles import ProfileManager

    return _get_or_build_store("profile_manager", config, lambda: ProfileManager(config))


def get_bot_profile_store(config: Any):
    """Process-wide ``BotProfileStore`` singleton."""
    from ..runtime_settings import BotProfileStore

    return _get_or_build_store("bot_profile_store", config, lambda: BotProfileStore(config))


def get_runtime_settings_store(config: Any):
    """Process-wide ``RuntimeSettingsStore`` singleton."""
    from ..runtime_settings import RuntimeSettingsStore

    return _get_or_build_store(
        "runtime_settings_store", config, lambda: RuntimeSettingsStore(config)
    )


def get_model_definition_store(config: Any):
    """Process-wide ``ModelDefinitionStore`` singleton."""
    from ..runtime_settings import ModelDefinitionStore

    return _get_or_build_store(
        "model_definition_store", config, lambda: ModelDefinitionStore(config)
    )


# TASK-214: get_avatar_animation_store() was removed. The avatar animation
# catalog now lives in bawthub Prisma and travels on each chat request as
# `animations: list[ChatRequestAnimation]`. llm-bawt is stateless w.r.t.
# the catalog.


def get_prompt_template_store(config: Any):
    """Process-wide ``PromptTemplateStore`` singleton."""
    from ..prompt_registry import PromptTemplateStore

    return _get_or_build_store(
        "prompt_template_store", config, lambda: PromptTemplateStore(config)
    )


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
