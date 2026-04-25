"""LLM client and bot instance management mixin.

Extracted from background_service.py — handles model resolution,
LLMBawt caching, memory clients, and background client lifecycle.
"""

from __future__ import annotations

from typing import Any

from ..bots import BotManager
from .logging import get_service_logger

log = get_service_logger(__name__)


class InstanceManagerMixin:
    """Mixin providing model/bot instance management for BackgroundService."""

    def _on_model_unloaded(self, model_alias: str):
        """Called when a model is unloaded - clears related caches."""
        log.debug(f"Model '{model_alias}' unloaded - clearing caches")

        # Clear client cache for this model
        if model_alias in self._client_cache:
            del self._client_cache[model_alias]

        # Clear all LLMBawt instances that use this model
        keys_to_remove = [
            key for key in self._llm_bawt_cache
            if key[0] == model_alias
        ]
        for key in keys_to_remove:
            del self._llm_bawt_cache[key]

        log.debug(f"Cleared {len(keys_to_remove)} cached instances for model '{model_alias}'")

    def invalidate_bot_instances(self, bot_id: str) -> int:
        """Invalidate cached ServiceLLMBawt instances for a bot across models/users."""
        normalized_bot_id = (bot_id or "").strip().lower()
        if not normalized_bot_id:
            return 0
        keys_to_remove = [
            key for key in self._llm_bawt_cache
            if key[1] == normalized_bot_id
        ]
        for key in keys_to_remove:
            del self._llm_bawt_cache[key]
        if keys_to_remove:
            log.debug(
                "Cleared %s cached instances for bot '%s'",
                len(keys_to_remove),
                normalized_bot_id,
            )
        return len(keys_to_remove)

    def invalidate_all_instances(self) -> int:
        """Invalidate all cached ServiceLLMBawt instances."""
        cleared = len(self._llm_bawt_cache)
        self._llm_bawt_cache.clear()
        if cleared:
            log.debug("Cleared all cached instances (%s)", cleared)
        return cleared

    def clear_session_model_overrides(self, bot_id: str | None = None, user_id: str | None = None) -> int:
        """Clear session model overrides, optionally scoped by bot and/or user."""
        if not self._session_model_overrides:
            return 0

        normalized_bot = (bot_id or "").strip().lower() if bot_id is not None else None
        normalized_user = (user_id or "").strip() if user_id is not None else None

        keys_to_remove: list[tuple[str, str]] = []
        for key in self._session_model_overrides:
            key_bot, key_user = key
            if normalized_bot is not None and key_bot != normalized_bot:
                continue
            if normalized_user is not None and key_user != normalized_user:
                continue
            keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._session_model_overrides[key]

        if keys_to_remove:
            scope = []
            if normalized_bot is not None:
                scope.append(f"bot={normalized_bot}")
            if normalized_user is not None:
                scope.append(f"user={normalized_user}")
            detail = " ".join(scope) if scope else "all sessions"
            log.info("Cleared %s session model override(s) for %s", len(keys_to_remove), detail)

        return len(keys_to_remove)

    def _load_available_models(self):
        """Load list of available models from config.

        Builds the canonical model catalog for the service:
        1. All models from models.yaml / DB
        2. Virtual model aliases for agent-backend bots
        3. A mapping from bot slug → backend model for agent-backend bots
           (stored separately — does NOT mutate bot objects)
        """
        models = self.config.defined_models.get("models", {})
        self._available_models = list(models.keys())

        # Agent-backend bot → virtual model mapping (no bot mutation)
        self._agent_backend_models: dict[str, str] = {}

        bot_manager = BotManager(self.config)
        for bot in bot_manager.list_bots():
            backend_name = getattr(bot, "agent_backend", None)
            if not backend_name:
                continue
            backend_config = getattr(bot, "agent_backend_config", {}) or {}
            # Register the virtual model definition if not already a real model
            if backend_name not in models:
                models[backend_name] = {
                    "type": "agent_backend",
                    "backend": backend_name,
                    "bot_config": backend_config,
                    "tool_support": "none",
                }
                if backend_name not in self._available_models:
                    self._available_models.append(backend_name)
                log.info(
                    "Registered virtual model '%s' for agent backend (bot=%s)",
                    backend_name,
                    bot.slug,
                )
            # Track the mapping without mutating the bot object
            self._agent_backend_models[bot.slug] = backend_name

        # Resolve default model: bot default → config default → first available
        default_bot = bot_manager.get_bot(self._default_bot) or bot_manager.get_default_bot()
        if default_bot:
            if default_bot.slug in self._agent_backend_models:
                self._default_model = self._agent_backend_models[default_bot.slug]
            elif default_bot.default_model and default_bot.default_model in self._available_models:
                self._default_model = default_bot.default_model
        if not self._default_model:
            config_default = getattr(self.config, "DEFAULT_MODEL_ALIAS", None)
            if config_default and config_default in self._available_models:
                self._default_model = config_default
        if not self._default_model and self._available_models:
            self._default_model = self._available_models[0]

        log.debug(
            "Loaded %d models (default=%s, agent_backends=%s)",
            len(self._available_models), self._default_model,
            list(self._agent_backend_models.values()),
        )

    def _resolve_request_model(
        self,
        requested_model: str | None,
        bot_id: str,
        local_mode: bool,
    ) -> tuple[str, list[str]]:
        """Resolve the model alias for a request.

        This is the single authoritative resolution point. The CLI in service
        mode sends only the user's explicit --model (or None).

        Priority:
        1. Agent-backend bots are locked to their backend model
        2. Explicit requested model (if available on service)
        3. Bot's default_model (from bots.yaml / DB)
        4. Service default model
        5. First available model (with warning)

        Returns:
            Tuple of (resolved_model_alias, list_of_warnings)
        """
        warnings: list[str] = []
        bot_manager = BotManager(self.config)
        bot = bot_manager.get_bot(bot_id) or bot_manager.get_default_bot()

        # 1. Agent-backend bots are locked to their backend model
        backend_model = self._agent_backend_models.get(bot.slug if bot else bot_id)
        if backend_model and backend_model in self._available_models:
            if requested_model and requested_model != backend_model:
                warnings.append(
                    f"Bot '{bot_id}' uses agent backend; ignoring requested model '{requested_model}'"
                )
            return backend_model, warnings

        # 2. Explicit requested model
        model = requested_model.strip() if requested_model else None
        if model:
            if model in self._available_models:
                return model, warnings
            # Requested model not available — fall through with warning
            warnings.append(f"Model '{model}' not available on service")

        # 3. Bot's default_model
        if bot and bot.default_model and bot.default_model in self._available_models:
            if warnings:
                warnings[-1] += f", using bot default '{bot.default_model}'"
            return bot.default_model, warnings

        # 4. Service default model
        if self._default_model and self._default_model in self._available_models:
            if warnings:
                warnings[-1] += f", using service default '{self._default_model}'"
            return self._default_model, warnings

        # 5. First available
        if self._available_models:
            fallback = self._available_models[0]
            msg = f"No preferred model available, using '{fallback}'"
            log.warning(msg)
            warnings.append(msg)
            return fallback, warnings

        raise ValueError("No models available on service.")

    def get_memory_client(self, bot_id: str, user_id: str | None = None):
        """Get or create memory client for a bot/user pair."""
        cache_key = (bot_id, user_id)

        if cache_key not in self._memory_clients:
            try:
                from ..mcp_server.client import get_memory_client
                self._memory_clients[cache_key] = get_memory_client(
                    config=self.config,
                    bot_id=bot_id,
                    user_id=user_id,
                    server_url=self.config.MCP_SERVER_URL,
                )
                log.memory_operation("client_init", bot_id, details="MemoryClient created")
            except Exception as e:
                log.warning(f"Memory client unavailable for {bot_id}: {e}")
                self._memory_clients[cache_key] = None
        return self._memory_clients.get(cache_key)

    def _get_llm_bawt(self, model_alias: str, bot_id: str, user_id: str, local_mode: bool = False):
        """Get or create an LLMBawt instance with caching.

        This method enforces single-model loading through the ModelLifecycleManager.
        If a different model is requested, the current model will be unloaded first.

        Model selection priority:
        1. Pending model switch (from switch_model tool) - becomes the new session model
        2. Current session model override (from previous switch)
        3. Model from API request
        """
        from .core import ServiceLLMBawt

        # Session key for model overrides (per bot+user)
        session_key = (bot_id, user_id)

        # Check for pending model switch (from switch_model tool)
        pending = self._model_lifecycle.clear_pending_switch()
        if pending:
            log.info(f"🔄 Switching to model: {pending}")
            # Store as session override so subsequent requests use this model
            self._session_model_overrides[session_key] = pending
            model_alias = pending
        elif session_key in self._session_model_overrides:
            # Use the session model override from a previous switch
            model_alias = self._session_model_overrides[session_key]
            log.debug(f"Using session model override: {model_alias}")

        cache_key = (model_alias, bot_id, user_id)

        # Check if we need to switch models (different model requested)
        current_model = self._model_lifecycle.current_model
        if current_model and current_model != model_alias:
            log.info(f"🔄 Model: {current_model} → {model_alias}")
            # Unloading will trigger _on_model_unloaded callback which clears caches
            self._model_lifecycle.unload_current_model()

        if cache_key in self._llm_bawt_cache:
            cached = self._llm_bawt_cache[cache_key]
            # If cached instance has no memory (local_mode=True) but this request
            # needs memory (local_mode=False), discard the stale instance so we
            # create a fresh memory-enabled one.
            #
            # The reverse (memory-enabled instance reused for a local_mode=True
            # request) is harmless — extra context won't break anything, and
            # memory extraction is gated by request.extract_memory separately.
            cached_local = getattr(cached, "local_mode", False)
            discard = False
            if cached_local and not local_mode:
                log.info(
                    "Upgrading cached local-mode instance → memory-enabled for %s/%s/%s",
                    model_alias, bot_id, user_id,
                )
                discard = True
            elif not local_mode and not cached_local:
                # Non-local instance whose memory init failed (e.g. transient DB
                # outage).  _db_available=False + memory=None means the init
                # exception path was hit — discard so we retry the connection.
                # (requires_memory=False sets _db_available=True, so won't match.)
                from ..utils.config import has_database_credentials
                has_memory = getattr(cached, "memory", None) is not None
                db_available = getattr(cached, "_db_available", False)
                if not has_memory and not db_available and has_database_credentials(self.config):
                    log.info(
                        "Discarding cached instance with failed memory init for %s/%s/%s — retrying",
                        model_alias, bot_id, user_id,
                    )
                    discard = True

            if discard:
                del self._llm_bawt_cache[cache_key]
            else:
                log.cache_hit("llm_bawt", f"{model_alias}/{bot_id}/{user_id}")
                log.debug(f"Reusing cached ServiceLLMBawt instance for {cache_key}")
                return cached

        log.cache_miss("llm_bawt", f"{model_alias}/{bot_id}/{user_id}")

        # Get model type for logging and cache decisions
        model_def = self.config.defined_models.get("models", {}).get(model_alias, {})
        model_type = model_def.get("type", "unknown")

        # Check if we already have a client for this model in the client cache
        # If so, reuse it to avoid reloading GGUF models into VRAM.
        # Skip for agent_backend models — they're lightweight and hold per-bot
        # state (_bot_config with session_key) that must not be shared.
        existing_client = None
        if model_type not in ("agent_backend", "claude-code"):
            existing_client = self._client_cache.get(model_alias)

        # Only log model loading if we don't have the client cached
        if not existing_client:
            log.model_loading(model_alias, model_type, cached=False)
        else:
            log.model_loading(model_alias, model_type, cached=True)
        try:
            # Create a copy of config for each ServiceLLMBawt instance
            # This is necessary because it modifies config.SYSTEM_MESSAGE
            # based on the bot's system prompt
            instance_config = self.config.model_copy(deep=True)
            llm_bawt = ServiceLLMBawt(
                resolved_model_alias=model_alias,
                config=instance_config,
                local_mode=local_mode,
                bot_id=bot_id,
                user_id=user_id,
                existing_client=existing_client,  # Reuse cached client if available
            )
            self._llm_bawt_cache[cache_key] = llm_bawt

            # Also cache the client for future reuse by extraction tasks.
            # Skip agent_backend clients — they hold per-bot state.
            if model_alias not in self._client_cache and model_type != "agent_backend":
                self._client_cache[model_alias] = llm_bawt.client
            # Note: BaseLLMBawt.__init__ already registers with lifecycle manager

        except Exception as e:
            log.model_error(model_alias, str(e))
            raise

        return self._llm_bawt_cache[cache_key]

    def get_client(self, model_alias: str):
        """Get LLM client for a given model (for extraction tasks).

        Uses a dedicated client cache to avoid reloading models.
        GGUF models especially are expensive to load into VRAM,
        so we cache the client independently from LLMBawt instances.
        """
        if model_alias in self._client_cache:
            log.cache_hit("llm_client", model_alias)
            return self._client_cache[model_alias]

        log.cache_miss("llm_client", model_alias)

        # Check if we already have an LLMBawt instance with this model
        # and can reuse its client
        for (cached_model, _, _), llm_bawt in self._llm_bawt_cache.items():
            if cached_model == model_alias:
                log.debug(f"Reusing client from existing LLMBawt instance for '{model_alias}'")
                self._client_cache[model_alias] = llm_bawt.client
                return llm_bawt.client

        # Need to create a new client - use spark bot (no memory overhead)
        log.debug(f"Creating new client for model '{model_alias}' (extraction context)")
        llm_bawt = self._get_llm_bawt(model_alias, "spark", "system", local_mode=True)
        self._client_cache[model_alias] = llm_bawt.client
        return llm_bawt.client

    def _get_background_client(self, model_override: str | None = None) -> tuple[Any | None, str | None]:
        """Get or create an API client for background tasks (summarization, extraction).

        This client is completely isolated from the main chat model lifecycle —
        it never triggers model switches or unloads the active chat model.

        Only API-based models (openai/grok) are supported.

        Returns:
            ``(client, model_alias)`` or ``(None, None)`` if unavailable.
        """
        preferred = (
            model_override
            or getattr(self.config, "EXTRACTION_MODEL", None)
            or getattr(self.config, "MAINTENANCE_MODEL", None)
            or getattr(self.config, "SUMMARIZATION_MODEL", None)
        )

        # Reuse cached client if same model
        if self._bg_client is not None and (not preferred or preferred == self._bg_client_model):
            return self._bg_client, self._bg_client_model

        # Resolve alias
        try:
            model_alias, _ = self._resolve_request_model(
                preferred, bot_id="nova", local_mode=False,
            )
        except Exception as e:
            log.error(f"Failed to resolve background model '{preferred}': {e}")
            return None, None

        models = self.config.defined_models.get("models", {})
        model_def = models.get(model_alias, {})
        model_type = model_def.get("type")

        if model_type not in ("openai", "grok"):
            if preferred:
                log.warning(
                    "Background model '%s' is type '%s' (only openai/grok supported). "
                    "Falling back to heuristics.", preferred, model_type,
                )
            return None, None

        # Create a standalone client — no model lifecycle involvement
        try:
            from ..clients.openai_client import OpenAIClient
            from ..clients.grok_client import GrokClient

            model_id = model_def.get("model_id", model_alias)
            if model_type == "grok":
                client = GrokClient(
                    model=model_id, config=self.config, model_definition=model_def,
                )
            else:
                client = OpenAIClient(
                    model=model_id, config=self.config, model_definition=model_def,
                )

            self._bg_client = client
            self._bg_client_model = model_alias
            log.info("Background client ready: %s (%s)", model_alias, model_type)
            return client, model_alias

        except Exception as e:
            log.error("Failed to create background client for '%s': %s", model_alias, e)
            return None, None
