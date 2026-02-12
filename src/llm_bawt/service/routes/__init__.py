"""Route router collection for app registration."""

from .chat import router as chat_router
from .health import router as health_router
from .history import router as history_router
from .llm import router as llm_router
from .memory import router as memory_router
from .models import router as models_router
from .nextcloud import router as nextcloud_router
from .profiles import router as profiles_router
from .tasks import router as tasks_router

all_routers = [
    health_router,
    nextcloud_router,
    models_router,
    chat_router,
    tasks_router,
    history_router,
    memory_router,
    profiles_router,
    llm_router,
]

__all__ = ["all_routers"]
