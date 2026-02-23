"""Runtime settings storage and resolution.

Supports layered runtime tuning with precedence:
request overrides > bot DB settings > global DB settings > bot.yaml settings > Config fallback.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import Column, DateTime, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Session, SQLModel, create_engine, select

from .utils.config import Config, has_database_credentials

if TYPE_CHECKING:
    from .bots import Bot

logger = logging.getLogger(__name__)


class RuntimeSetting(SQLModel, table=True):
    """DB-backed runtime setting."""

    __tablename__ = "runtime_settings"

    id: int | None = Field(default=None, primary_key=True)
    scope_type: str = Field(
        sa_column=Column(String(16), nullable=False, index=True),
        description="global or bot",
    )
    scope_id: str = Field(
        sa_column=Column(String(128), nullable=False, index=True),
        description="* for global, bot slug for bot scope",
    )
    key: str = Field(sa_column=Column(String(128), nullable=False, index=True))
    value_json: str = Field(sa_column=Column(Text, nullable=False))
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class BotProfile(SQLModel, table=True):
    """DB-backed bot personality profile."""

    __tablename__ = "bot_profiles"

    id: int | None = Field(default=None, primary_key=True)
    slug: str = Field(sa_column=Column(String(128), nullable=False, index=True, unique=True))
    name: str = Field(sa_column=Column(String(255), nullable=False))
    description: str = Field(sa_column=Column(Text, nullable=False, default=""))
    system_prompt: str = Field(sa_column=Column(Text, nullable=False))
    requires_memory: bool = Field(default=True)
    voice_optimized: bool = Field(default=False)
    uses_tools: bool = Field(default=False)
    uses_search: bool = Field(default=False)
    uses_home_assistant: bool = Field(default=False)
    default_model: str | None = Field(default=None, sa_column=Column(String(255), nullable=True))
    nextcloud_config: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB, nullable=True))
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class BotProfileStore:
    """DB access for bot personality profiles."""

    def __init__(self, config: Config):
        self.config = config
        self.engine = None
        if not has_database_credentials(config):
            return

        try:
            host = getattr(config, "POSTGRES_HOST", "localhost")
            port = int(getattr(config, "POSTGRES_PORT", 5432))
            user = getattr(config, "POSTGRES_USER", "llm_bawt")
            password = getattr(config, "POSTGRES_PASSWORD", "")
            database = getattr(config, "POSTGRES_DATABASE", "llm_bawt")
            encoded_password = quote_plus(password)
            connection_url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{database}"
            self.engine = create_engine(connection_url, echo=False)
            self._ensure_tables_exist()
        except Exception as e:
            self.engine = None
            logger.warning("Bot profiles DB unavailable: %s", e)

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return
        SQLModel.metadata.create_all(self.engine, tables=[BotProfile.__table__])

    def get(self, slug: str) -> BotProfile | None:
        if self.engine is None:
            return None
        normalized_slug = (slug or "").strip().lower()
        if not normalized_slug:
            return None
        with Session(self.engine) as session:
            return session.exec(
                select(BotProfile).where(BotProfile.slug == normalized_slug)
            ).first()

    def list_all(self) -> list[BotProfile]:
        if self.engine is None:
            return []
        with Session(self.engine) as session:
            return list(session.exec(select(BotProfile).order_by(BotProfile.slug)).all())

    def upsert(self, profile: BotProfile | dict[str, Any]) -> BotProfile:
        if self.engine is None:
            raise RuntimeError("Bot profiles DB unavailable")

        if isinstance(profile, BotProfile):
            payload = profile.model_dump()
        else:
            payload = dict(profile)

        slug = str(payload.get("slug", "")).strip().lower()
        if not slug:
            raise ValueError("Bot profile slug is required")

        now = datetime.utcnow()
        with Session(self.engine) as session:
            row = session.exec(select(BotProfile).where(BotProfile.slug == slug)).first()
            if row is None:
                row = BotProfile(
                    slug=slug,
                    name=str(payload.get("name") or slug.title()),
                    description=str(payload.get("description") or ""),
                    system_prompt=str(payload.get("system_prompt") or "You are a helpful assistant."),
                    requires_memory=bool(payload.get("requires_memory", True)),
                    voice_optimized=bool(payload.get("voice_optimized", False)),
                    uses_tools=bool(payload.get("uses_tools", False)),
                    uses_search=bool(payload.get("uses_search", False)),
                    uses_home_assistant=bool(payload.get("uses_home_assistant", False)),
                    default_model=payload.get("default_model"),
                    nextcloud_config=payload.get("nextcloud_config"),
                    created_at=now,
                    updated_at=now,
                )
            else:
                row.name = str(payload.get("name", row.name) or row.name)
                row.description = str(payload.get("description", row.description) or "")
                row.system_prompt = str(payload.get("system_prompt", row.system_prompt) or row.system_prompt)
                row.requires_memory = bool(payload.get("requires_memory", row.requires_memory))
                row.voice_optimized = bool(payload.get("voice_optimized", row.voice_optimized))
                row.uses_tools = bool(payload.get("uses_tools", row.uses_tools))
                row.uses_search = bool(payload.get("uses_search", row.uses_search))
                row.uses_home_assistant = bool(payload.get("uses_home_assistant", row.uses_home_assistant))
                row.default_model = payload.get("default_model", row.default_model)
                row.nextcloud_config = payload.get("nextcloud_config", row.nextcloud_config)
                row.updated_at = now

            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def delete(self, slug: str) -> bool:
        if self.engine is None:
            raise RuntimeError("Bot profiles DB unavailable")
        normalized_slug = (slug or "").strip().lower()
        if not normalized_slug:
            return False
        with Session(self.engine) as session:
            row = session.exec(
                select(BotProfile).where(BotProfile.slug == normalized_slug)
            ).first()
            if not row:
                return False
            session.delete(row)
            session.commit()
            return True

    def seed_from_yaml(self, bots_dict: dict[str, Any]) -> int:
        if self.engine is None:
            raise RuntimeError("Bot profiles DB unavailable")
        if not isinstance(bots_dict, dict):
            return 0

        seeded = 0
        for slug, bot_data in bots_dict.items():
            if not isinstance(bot_data, dict):
                continue
            payload = {
                "slug": slug,
                "name": bot_data.get("name", str(slug).title()),
                "description": bot_data.get("description", ""),
                "system_prompt": bot_data.get("system_prompt", "You are a helpful assistant."),
                "requires_memory": bot_data.get("requires_memory", True),
                "voice_optimized": bot_data.get("voice_optimized", False),
                "uses_tools": bot_data.get("uses_tools", False),
                "uses_search": bot_data.get("uses_search", False),
                "uses_home_assistant": bot_data.get("uses_home_assistant", False),
                "default_model": bot_data.get("default_model"),
                "nextcloud_config": bot_data.get("nextcloud"),
            }
            self.upsert(payload)
            seeded += 1
        return seeded


class RuntimeSettingsStore:
    """Low-level DB access for runtime settings."""

    def __init__(self, config: Config):
        self.config = config
        self.engine = None
        if not has_database_credentials(config):
            return

        try:
            host = getattr(config, "POSTGRES_HOST", "localhost")
            port = int(getattr(config, "POSTGRES_PORT", 5432))
            user = getattr(config, "POSTGRES_USER", "llm_bawt")
            password = getattr(config, "POSTGRES_PASSWORD", "")
            database = getattr(config, "POSTGRES_DATABASE", "llm_bawt")
            encoded_password = quote_plus(password)
            connection_url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{database}"
            self.engine = create_engine(connection_url, echo=False)
            self._ensure_tables_exist()
        except Exception as e:
            self.engine = None
            logger.warning("Runtime settings DB unavailable: %s", e)

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return
        SQLModel.metadata.create_all(
            self.engine,
            tables=[RuntimeSetting.__table__, BotProfile.__table__],
        )
        with self.engine.connect() as conn:
            conn.execute(
                text(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_runtime_settings_scope_key
                    ON runtime_settings (scope_type, scope_id, key)
                    """
                )
            )
            conn.commit()

    def get_scope_settings(self, scope_type: str, scope_id: str) -> dict[str, Any]:
        """Return all settings for a scope."""
        if self.engine is None:
            return {}
        with Session(self.engine) as session:
            rows = session.exec(
                select(RuntimeSetting).where(
                    RuntimeSetting.scope_type == scope_type,
                    RuntimeSetting.scope_id == scope_id,
                )
            ).all()
        out: dict[str, Any] = {}
        for row in rows:
            try:
                out[row.key] = json.loads(row.value_json)
            except Exception:
                logger.warning(
                    "Invalid runtime setting JSON for %s/%s key=%s",
                    scope_type,
                    scope_id,
                    row.key,
                )
        return out

    def set_value(self, scope_type: str, scope_id: str, key: str, value: Any) -> None:
        """Upsert setting value."""
        if self.engine is None:
            raise RuntimeError("Runtime settings DB unavailable")
        value_json = json.dumps(value, ensure_ascii=False)
        with Session(self.engine) as session:
            row = session.exec(
                select(RuntimeSetting).where(
                    RuntimeSetting.scope_type == scope_type,
                    RuntimeSetting.scope_id == scope_id,
                    RuntimeSetting.key == key,
                )
            ).first()
            if row:
                row.value_json = value_json
                row.updated_at = datetime.utcnow()
            else:
                row = RuntimeSetting(
                    scope_type=scope_type,
                    scope_id=scope_id,
                    key=key,
                    value_json=value_json,
                )
            session.add(row)
            session.commit()

    def delete_value(self, scope_type: str, scope_id: str, key: str) -> bool:
        """Delete one setting value. Returns True if deleted."""
        if self.engine is None:
            raise RuntimeError("Runtime settings DB unavailable")
        with Session(self.engine) as session:
            row = session.exec(
                select(RuntimeSetting).where(
                    RuntimeSetting.scope_type == scope_type,
                    RuntimeSetting.scope_id == scope_id,
                    RuntimeSetting.key == key,
                )
            ).first()
            if not row:
                return False
            session.delete(row)
            session.commit()
            return True


class ModelDefinition(SQLModel, table=True):
    """DB-backed model definition — mirrors models.yaml but DB takes priority."""

    __tablename__ = "model_definitions"

    id: int | None = Field(default=None, primary_key=True)
    alias: str = Field(sa_column=Column(String(128), nullable=False, index=True, unique=True))
    type: str = Field(sa_column=Column(String(64), nullable=False))
    model_id: str | None = Field(default=None, sa_column=Column(String(512), nullable=True))
    repo_id: str | None = Field(default=None, sa_column=Column(String(512), nullable=True))
    filename: str | None = Field(default=None, sa_column=Column(String(512), nullable=True))
    description: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    extra: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
        description="Optional fields: chat_format, context_window, max_tokens, n_gpu_layers, tool_support, tool_format",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    def to_model_dict(self) -> dict[str, Any]:
        """Convert to the dict format used by config.defined_models['models'][alias]."""
        d: dict[str, Any] = {"type": self.type}
        if self.model_id is not None:
            d["model_id"] = self.model_id
        if self.repo_id is not None:
            d["repo_id"] = self.repo_id
        if self.filename is not None:
            d["filename"] = self.filename
        if self.description is not None:
            d["description"] = self.description
        if self.extra:
            d.update(self.extra)
        return d


class ModelDefinitionStore:
    """DB access for model definitions. DB always takes priority over models.yaml."""

    def __init__(self, config: Config):
        self.config = config
        self.engine = None
        if not has_database_credentials(config):
            return

        try:
            host = getattr(config, "POSTGRES_HOST", "localhost")
            port = int(getattr(config, "POSTGRES_PORT", 5432))
            user = getattr(config, "POSTGRES_USER", "llm_bawt")
            password = getattr(config, "POSTGRES_PASSWORD", "")
            database = getattr(config, "POSTGRES_DATABASE", "llm_bawt")
            encoded_password = quote_plus(password)
            connection_url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{database}"
            self.engine = create_engine(connection_url, echo=False)
            self._ensure_tables_exist()
        except Exception as e:
            self.engine = None
            logger.warning("Model definitions DB unavailable: %s", e)

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return
        SQLModel.metadata.create_all(self.engine, tables=[ModelDefinition.__table__])

    def count(self) -> int:
        if self.engine is None:
            return 0
        with Session(self.engine) as session:
            return session.exec(select(ModelDefinition)).all().__len__()

    def list_all(self) -> list[ModelDefinition]:
        if self.engine is None:
            return []
        with Session(self.engine) as session:
            return list(session.exec(select(ModelDefinition).order_by(ModelDefinition.alias)).all())

    def to_config_dict(self) -> dict[str, dict[str, Any]]:
        """Return all DB models as {alias: definition_dict} for merging into config."""
        return {row.alias: row.to_model_dict() for row in self.list_all()}

    def get(self, alias: str) -> ModelDefinition | None:
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            return session.exec(
                select(ModelDefinition).where(ModelDefinition.alias == alias)
            ).first()

    def upsert(self, alias: str, model_data: dict[str, Any]) -> ModelDefinition:
        if self.engine is None:
            raise RuntimeError("Model definitions DB unavailable")

        known_fields = {"type", "model_id", "repo_id", "filename", "description"}
        extra = {k: v for k, v in model_data.items() if k not in known_fields}
        now = datetime.utcnow()

        with Session(self.engine) as session:
            row = session.exec(
                select(ModelDefinition).where(ModelDefinition.alias == alias)
            ).first()
            if row is None:
                row = ModelDefinition(
                    alias=alias,
                    type=str(model_data.get("type", "openai")),
                    model_id=model_data.get("model_id"),
                    repo_id=model_data.get("repo_id"),
                    filename=model_data.get("filename"),
                    description=model_data.get("description"),
                    extra=extra or None,
                    created_at=now,
                    updated_at=now,
                )
            else:
                row.type = str(model_data.get("type", row.type))
                row.model_id = model_data.get("model_id", row.model_id)
                row.repo_id = model_data.get("repo_id", row.repo_id)
                row.filename = model_data.get("filename", row.filename)
                row.description = model_data.get("description", row.description)
                row.extra = extra or None
                row.updated_at = now

            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def delete(self, alias: str) -> bool:
        if self.engine is None:
            raise RuntimeError("Model definitions DB unavailable")
        with Session(self.engine) as session:
            row = session.exec(
                select(ModelDefinition).where(ModelDefinition.alias == alias)
            ).first()
            if not row:
                return False
            session.delete(row)
            session.commit()
            return True

    def seed_from_yaml(self, models_dict: dict[str, Any]) -> int:
        """Seed DB from models.yaml format {alias: definition}. Skips existing aliases."""
        if self.engine is None:
            raise RuntimeError("Model definitions DB unavailable")
        seeded = 0
        for alias, model_data in models_dict.items():
            if not isinstance(model_data, dict):
                continue
            if self.get(alias) is not None:
                continue
            self.upsert(alias, model_data)
            seeded += 1
        return seeded


class RuntimeSettingsResolver:
    """Resolve runtime settings with layered precedence and short-lived cache."""

    def __init__(
        self,
        config: Config,
        bot: Bot | None = None,
        bot_id: str | None = None,
        cache_ttl_seconds: float = 5.0,
    ):
        self.config = config
        self.bot = bot
        self.bot_id = (bot_id or getattr(bot, "slug", "") or "").strip().lower() or None
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self.store = RuntimeSettingsStore(config)
        self._bot_cache: dict[str, Any] = {}
        self._global_cache: dict[str, Any] = {}
        self._cache_loaded_at: float = 0.0

    def _refresh_cache_if_needed(self) -> None:
        if self.store.engine is None:
            return
        now = time.time()
        if self._cache_loaded_at and (now - self._cache_loaded_at) < self.cache_ttl_seconds:
            return
        self._global_cache = self.store.get_scope_settings("global", "*")
        self._bot_cache = (
            self.store.get_scope_settings("bot", self.bot_id)
            if self.bot_id
            else {}
        )
        self._cache_loaded_at = now

    def resolve(
        self,
        key: str,
        fallback: Any = None,
        request_overrides: dict[str, Any] | None = None,
    ) -> Any:
        """Resolve a setting value using precedence chain."""
        if request_overrides and key in request_overrides:
            return request_overrides[key]

        self._refresh_cache_if_needed()
        if key in self._bot_cache:
            return self._bot_cache[key]
        if key in self._global_cache:
            return self._global_cache[key]

        bot_settings = getattr(self.bot, "settings", {}) if self.bot else {}
        if isinstance(bot_settings, dict) and key in bot_settings:
            return bot_settings[key]

        if fallback is not None:
            return fallback
        return None

    def resolve_from_config_attr(
        self,
        key: str,
        config_attr: str,
        request_overrides: dict[str, Any] | None = None,
    ) -> Any:
        """Resolve setting using config attribute as final fallback."""
        fallback = getattr(self.config, config_attr, None)
        return self.resolve(key=key, fallback=fallback, request_overrides=request_overrides)
