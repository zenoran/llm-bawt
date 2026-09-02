"""Virgin-tenant seeding (TASK-350 / TASK-355).

Public surface:

- :func:`llm_bawt.seeding.engine.seed_tenant` — idempotent infrastructure
  seed (catalog schema, standard access paths, prompt defaults). Called by
  the installer AND the setup wizard.
- :meth:`llm_bawt.seeding.engine.TenantSeeder.register_model` — registers the
  operator's chosen model + endpoint for the first bot.
- ``python -m llm_bawt.seeding`` — thin CLI wrapper around ``seed_tenant``.

No bots or model rows are seed data: the wizard creates the first bot only
after a provider credential exists (see :mod:`llm_bawt.seeding.data`).
"""

from .data import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_FIRST_BOT_PROMPT,
    FIRST_BOT_PROVIDERS,
    FirstBotProvider,
    first_bot_provider,
)
from .engine import TenantSeeder, seed_tenant

__all__ = [
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_FIRST_BOT_PROMPT",
    "FIRST_BOT_PROVIDERS",
    "FirstBotProvider",
    "TenantSeeder",
    "first_bot_provider",
    "seed_tenant",
]
