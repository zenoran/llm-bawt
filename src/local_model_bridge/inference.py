"""Local model loading + lifecycle on the bridge side (TASK-276).

Owns the single-local-model-at-a-time loader that the bridge uses to turn a
model *alias* into a live ``LLMClient`` (``LlamaCppClient`` for gguf,
``VLLMClient`` for vllm or gguf-with-backend=vllm).

Model definitions are resolved by querying the main app:
    GET {app_api_url}/v1/models/definitions/{alias}
which returns the catalog row (type, repo_id, filename, description, extra).
The ``extra`` blob carries the runtime fields (backend, chat_format,
context_window, n_gpu_layers, ...) the clients expect to find nested in the
model definition.  Results are cached; a miss triggers a single refetch.

Only one local model is held in memory at a time.  Switching aliases unloads
the previous client first (freeing VRAM) before loading the new one.  Callers
must serialize model loads themselves — the bridge does this by running all
inference on a single-worker ThreadPoolExecutor.

Heavy deps (llama_cpp / vllm / torch) are imported lazily inside the client
constructors, so importing this module never requires them.
"""

from __future__ import annotations

import gc
import logging
import threading
from typing import TYPE_CHECKING, Any

import httpx

from llm_bawt.utils.config import Config

if TYPE_CHECKING:
    from llm_bawt.clients.base import LLMClient

logger = logging.getLogger(__name__)


class _ModelDefinitionCache:
    """Resolve a model alias -> full model definition via the main app.

    Mirrors codex_bridge's ``_ModelInfoCache`` shape but fetches the full
    catalog row (we need repo_id/filename/backend/chat_format, not just the
    context window).
    """

    def __init__(self, app_api_url: str) -> None:
        self._app_api_url = (app_api_url or "").rstrip("/")
        self._defs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _fetch(self, alias: str) -> dict[str, Any] | None:
        if not self._app_api_url:
            return None
        url = f"{self._app_api_url}/v1/models/definitions/{alias}"
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(url)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                payload = resp.json() or {}
        except Exception as e:
            logger.warning("Model definition fetch failed for %s: %s", alias, e)
            return None

        # Flatten the catalog row into the runtime model_definition shape the
        # local clients expect: top-level type/model_id/repo_id/filename plus
        # the ``extra`` blob (backend, chat_format, context_window, ...).
        definition: dict[str, Any] = {
            "type": payload.get("type"),
            "model_id": payload.get("model_id"),
            "repo_id": payload.get("repo_id"),
            "filename": payload.get("filename"),
            "description": payload.get("description"),
        }
        extra = payload.get("extra")
        if isinstance(extra, dict):
            for k, v in extra.items():
                definition.setdefault(k, v)
        # Drop None top-level keys so client `.get(..., default)` fallbacks work.
        return {k: v for k, v in definition.items() if v is not None}

    def get(self, alias: str, *, fallback: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Return the resolved definition for *alias*, or *fallback*.

        Caches successful lookups.  On a cache miss, fetches once; if the
        fetch fails and a *fallback* (e.g. the local_model_definition carried
        in the chat.send command) is supplied, that is used instead.
        """
        with self._lock:
            cached = self._defs.get(alias)
            if cached is not None:
                return cached
            resolved = self._fetch(alias)
            if resolved:
                self._defs[alias] = resolved
                return resolved
        # Fetch miss — fall back to the caller-supplied definition if any.
        if fallback:
            return fallback
        return None


class LocalModelLoader:
    """Single-local-model-at-a-time loader.

    Holds at most one loaded ``LLMClient``.  Loading a different alias unloads
    the current one first (freeing VRAM).  Re-requesting the currently loaded
    alias returns the cached client without reloading.

    Not internally thread-safe for concurrent loads — the bridge serializes
    all calls onto a single-worker executor.  A lock still guards against the
    health-server thread racing ``current_alias`` reads.
    """

    def __init__(self, config: Config, app_api_url: str = "") -> None:
        self._config = config
        self._defs = _ModelDefinitionCache(app_api_url)
        self._current_alias: str | None = None
        self._current_client: "LLMClient | None" = None
        self._lock = threading.RLock()

    @property
    def current_alias(self) -> str | None:
        return self._current_alias

    def get_client(
        self,
        alias: str,
        *,
        fallback_definition: dict[str, Any] | None = None,
    ) -> "LLMClient":
        """Return a loaded client for *alias*, loading/switching as needed.

        ``fallback_definition`` is the local model definition the app nested in
        the chat.send command (under ``local_model_definition``); used only if
        the /v1/models lookup fails.
        """
        with self._lock:
            if self._current_alias == alias and self._current_client is not None:
                return self._current_client

            definition = self._defs.get(alias, fallback=fallback_definition)
            if not definition:
                raise ValueError(
                    f"Local model alias '{alias}' could not be resolved against "
                    f"the app catalog and no fallback definition was provided."
                )

            # Unload the previous model before loading a new one so we never
            # hold two local models in VRAM simultaneously.
            if self._current_client is not None:
                self._unload_locked()

            client = self._build_client(alias, definition)
            self._current_alias = alias
            self._current_client = client
            logger.info("Local model loaded: %s (type=%s)", alias, definition.get("type"))
            return client

    def _build_client(self, alias: str, definition: dict[str, Any]) -> "LLMClient":
        """Construct the right local client for *definition*.

        Mirrors the gguf/vllm branch logic from the app's
        ``service/core.py:_initialize_client`` (pre-TASK-276), which now lives
        here in the bridge.
        """
        model_type = str(definition.get("type") or "").strip().lower()

        if model_type == "gguf":
            from llm_bawt.gguf_handler import get_or_download_gguf_model

            repo_id = definition.get("repo_id")
            filename = definition.get("filename")
            if not repo_id or not filename:
                raise ValueError(
                    f"Missing 'repo_id' or 'filename' in GGUF definition for '{alias}'"
                )

            model_path = get_or_download_gguf_model(repo_id, filename, self._config)
            if not model_path:
                raise FileNotFoundError(
                    f"Could not download GGUF model: {repo_id}/{filename}"
                )

            backend = definition.get("backend", "llama-cpp")
            if backend == "vllm":
                from .vllm_client import VLLMClient

                return VLLMClient(
                    str(model_path),  # local GGUF path for vLLM
                    config=self._config,
                    model_definition=definition,
                )

            from .llama_cpp_client import LlamaCppClient

            chat_format = definition.get("chat_format")
            return LlamaCppClient(
                str(model_path),
                config=self._config,
                chat_format=chat_format,
                model_definition=definition,
            )

        if model_type == "vllm":
            from .vllm_client import VLLMClient

            # model_id may be a HuggingFace model id or a GGUF path.
            model_id = definition.get("model_id") or alias
            return VLLMClient(
                model_id,
                config=self._config,
                model_definition=definition,
            )

        raise ValueError(
            f"Local model '{alias}' has unsupported type '{model_type}'. "
            f"Supported: gguf, vllm."
        )

    def unload(self) -> bool:
        """Unload the current model (if any). Returns True if one was unloaded."""
        with self._lock:
            return self._unload_locked()

    def _unload_locked(self) -> bool:
        if self._current_client is None:
            return False
        previous = self._current_alias
        logger.info("Unloading local model: %s", previous)
        client = self._current_client
        self._current_client = None
        self._current_alias = None
        try:
            unload = getattr(client, "unload", None)
            if callable(unload):
                unload()
        except Exception as e:
            logger.warning("Error during client unload: %s", e)
        del client
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        except Exception as e:
            logger.debug("Could not clear CUDA cache: %s", e)
        logger.info("Local model unloaded: %s", previous)
        return True
