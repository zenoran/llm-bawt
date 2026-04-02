"""Route router collection for app registration."""

from .botchat import router as botchat_router
from .chat import router as chat_router
from .health import router as health_router
from .history import router as history_router
from .jobs import router as jobs_router
from .llm import router as llm_router
from .memory import router as memory_router
from .models import router as models_router
from .openclaw_ws import router as openclaw_ws_router
from .nextcloud import router as nextcloud_router
from .prompts import router as prompts_router
from .profiles import router as profiles_router
from .settings import router as settings_router
from .tasks import router as tasks_router
from .ha_weather import router as ha_weather_router
from .turn_logs import router as turn_logs_router
from .avatar_animations import router as avatar_animations_router
from .ws_gateway import router as ws_gateway_router

all_routers = [
    health_router,
    ha_weather_router,
    nextcloud_router,
    models_router,
    openclaw_ws_router,
    ws_gateway_router,
    botchat_router,
    chat_router,
    tasks_router,
    turn_logs_router,
    jobs_router,
    history_router,
    memory_router,
    prompts_router,
    settings_router,
    profiles_router,
    llm_router,
    avatar_animations_router,
]

__all__ = ["all_routers"]
