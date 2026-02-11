import logging
from typing import Type
from .base import ModelAdapter
from .default import DefaultAdapter

logger = logging.getLogger(__name__)

_ADAPTERS: dict[str, Type[ModelAdapter]] = {}


def register_adapter(name: str, adapter_cls: Type[ModelAdapter]):
    """Register an adapter class."""
    _ADAPTERS[name] = adapter_cls


def get_adapter(model_alias: str, model_def: dict | None = None) -> ModelAdapter:
    """Get adapter instance for a model.

    Priority:
    1. Explicit 'adapter' field in model definition
    2. Auto-detection based on model alias or repo_id
    3. Default adapter
    """
    adapter_name = None
    if model_def:
        adapter_name = model_def.get("adapter")

    if adapter_name and adapter_name in _ADAPTERS:
        logger.debug(f"Using adapter '{adapter_name}' for model '{model_alias}'")
        return _ADAPTERS[adapter_name]()

    # Auto-detect adapter from model alias or repo_id
    detected = _auto_detect_adapter(model_alias, model_def)
    if detected:
        logger.debug(f"Auto-detected adapter '{detected}' for model '{model_alias}'")
        return _ADAPTERS[detected]()

    return DefaultAdapter()


def _auto_detect_adapter(model_alias: str, model_def: dict | None) -> str | None:
    """Auto-detect adapter based on model name patterns."""
    # Normalize for matching
    alias_lower = model_alias.lower()
    repo_id = (model_def.get("repo_id", "") if model_def else "").lower()

    # Pygmalion/MythoMax/Lewd models - these have specific markers like [HUMAN], [INST]
    if any(x in alias_lower for x in ("pygmalion", "mytho", "mythomax", "lewd")):
        return "pygmalion"
    if any(x in repo_id for x in ("pygmalion", "mytho", "mythomax", "lewd")):
        return "pygmalion"

    return None


# Auto-register adapters when module loads
def _register_builtins():
    from .default import DefaultAdapter
    from .pygmalion import PygmalionAdapter
    from .dolphin import DolphinAdapter

    register_adapter("default", DefaultAdapter)
    register_adapter("pygmalion", PygmalionAdapter)
    register_adapter("mythomax", PygmalionAdapter)  # Alias
    register_adapter("dolphin", DolphinAdapter)  # Available if explicitly configured


_register_builtins()
