"""Bot system for llm-bawt.

Bots are AI personalities with their own system prompts and isolated memory.
Bot definitions are loaded from bots.yaml in this package directory,
with user overrides from ~/.config/llm-bawt/bots.yaml taking priority.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_bawt.utils.config import Config, has_database_credentials

logger = logging.getLogger(__name__)

# Path to the repo bots.yaml file (in the same directory as this module)
REPO_BOTS_YAML_PATH = Path(__file__).parent / "bots.yaml"


def get_repo_bots_yaml_path() -> Path:
    """Get the path to repo bots.yaml."""
    return REPO_BOTS_YAML_PATH


def get_user_bots_yaml_path() -> Path:
    """Get the path to user bots.yaml (~/.config/llm-bawt/bots.yaml)."""
    from llm_bawt.utils.config import get_default_config_dir
    return get_default_config_dir() / "bots.yaml"


# Backwards compatibility alias
def get_bots_yaml_path() -> Path:
    """Get the path to bots.yaml (repo path for backwards compat)."""
    return REPO_BOTS_YAML_PATH


@dataclass
class Bot:
    """A bot personality with its own system prompt and capabilities."""
    
    slug: str  # Unique identifier (e.g., "nova", "spark", "mira")
    name: str  # Display name (e.g., "Nova", "Spark", "Mira")
    description: str  # Short description for --list-bots
    system_prompt: str  # The system message sent to the LLM
    requires_memory: bool = True  # Whether this bot needs database/memory persistence
    voice_optimized: bool = False  # Whether output is optimized for TTS
    default_model: str | None = None  # Default model alias for this bot
    uses_tools: bool = False  # Whether this bot can use tools (memory search, etc.)
    uses_search: bool = False  # Whether this bot can search the web
    uses_home_assistant: bool = False  # Whether this bot can control Home Assistant via MCP
    nextcloud: dict | None = None  # Nextcloud integration config (bot_id, secret, etc.)
    settings: dict[str, Any] = field(default_factory=dict)  # Effective bot settings (template + overrides)
    
    def __post_init__(self):
        # Ensure slug is lowercase and valid
        self.slug = self.slug.lower().strip()


@dataclass
class ModelSelection:
    """Resolved model selection with source metadata."""

    alias: str | None
    source: str  # explicit | bot_default | config_default | none


# Global bot registry - populated from YAML on module load
BUILTIN_BOTS: dict[str, Bot] = {}
_DEFAULTS: dict[str, str] = {"standard": "nova", "local": "spark"}
_SYSTEM_PROMPTS: dict[str, str] = {}
_RAW_BOT_DATA: dict[str, dict] = {}  # Raw effective bot data for integrations
_BOT_SETTINGS_TEMPLATE: dict[str, Any] = {}
_LAST_REPO_MTIME: float = 0
_LAST_USER_MTIME: float = 0
_FORCE_RELOAD: bool = False


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, with override taking priority.
    
    For nested dicts, merges recursively. For other types, override wins.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_file(yaml_path: Path) -> dict | None:
    """Load a YAML file and return its contents, or None if not found."""
    try:
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing {yaml_path}: {e}")
    except Exception as e:
        logger.error(f"Error reading {yaml_path}: {e}")
    return None


def invalidate_bots_cache() -> None:
    """Force the next _check_reload() to reload bots from DB + YAML."""
    global _FORCE_RELOAD
    _FORCE_RELOAD = True


def _check_reload() -> bool:
    """Check if config files have changed and reload if needed.

    Returns True if reload was performed.
    """
    global _LAST_REPO_MTIME, _LAST_USER_MTIME, _FORCE_RELOAD

    if _FORCE_RELOAD:
        _FORCE_RELOAD = False
        _load_bots_config()
        return True

    repo_path = get_repo_bots_yaml_path()
    user_path = get_user_bots_yaml_path()

    repo_mtime = repo_path.stat().st_mtime if repo_path.exists() else 0
    user_mtime = user_path.stat().st_mtime if user_path.exists() else 0

    if repo_mtime > _LAST_REPO_MTIME or user_mtime > _LAST_USER_MTIME:
        _load_bots_config()
        return True
    return False


def _extract_bot_settings_template(merged_data: dict[str, Any]) -> dict[str, Any]:
    """Extract bot settings template from merged YAML.

    Preferred key: `bot_settings_template`.
    Compatibility aliases: `bot_settings_defaults`, `bot_defaults`.
    """
    for key in ("bot_settings_template", "bot_settings_defaults", "bot_defaults"):
        value = merged_data.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _load_bots_config() -> None:
    """Load bot definitions DB-first with YAML fallback for missing slugs."""
    global BUILTIN_BOTS, _DEFAULTS, _SYSTEM_PROMPTS, _RAW_BOT_DATA, _BOT_SETTINGS_TEMPLATE
    global _LAST_REPO_MTIME, _LAST_USER_MTIME
    
    repo_path = get_repo_bots_yaml_path()
    user_path = get_user_bots_yaml_path()
    
    # Track mtimes for hot reload
    _LAST_REPO_MTIME = repo_path.stat().st_mtime if repo_path.exists() else 0
    _LAST_USER_MTIME = user_path.stat().st_mtime if user_path.exists() else 0
    
    # Load repo config (defaults)
    repo_data = _load_yaml_file(repo_path) or {}
    
    # Load user config (overrides)
    user_data = _load_yaml_file(user_path) or {}
    
    # Merge: user overrides repo
    merged_data = _deep_merge(repo_data, user_data)
    
    if not merged_data:
        logger.warning("No bot configuration found in repo or user config")
        return
    
    # Clear existing data
    BUILTIN_BOTS.clear()
    _RAW_BOT_DATA.clear()
    _BOT_SETTINGS_TEMPLATE = _extract_bot_settings_template(merged_data)

    runtime_store = None
    db_profiles_by_slug: dict[str, Any] = {}
    db_bot_settings: dict[str, dict[str, Any]] = {}

    # Load profiles from DB first (if available)
    try:
        config = Config()
        if has_database_credentials(config):
            from llm_bawt.runtime_settings import BotProfileStore, RuntimeSettingsStore

            profile_store = BotProfileStore(config)
            runtime_store = RuntimeSettingsStore(config)
            if profile_store.engine is not None:
                db_profiles = profile_store.list_all()
                db_profiles_by_slug = {p.slug: p for p in db_profiles}

            if runtime_store.engine is not None:
                for slug in db_profiles_by_slug:
                    db_bot_settings[slug] = runtime_store.get_scope_settings("bot", slug)
    except Exception as e:
        logger.warning("Failed loading bot profiles from DB: %s", e)

    def _build_settings(slug: str, yaml_bot_data: dict[str, Any] | None = None) -> dict[str, Any]:
        yaml_settings = {}
        if isinstance(yaml_bot_data, dict):
            yaml_settings = yaml_bot_data.get("settings", {}) or {}
        return _deep_merge(
            _deep_merge(_BOT_SETTINGS_TEMPLATE, yaml_settings),
            db_bot_settings.get(slug, {}),
        )

    def _register_bot(slug: str, payload: dict[str, Any], yaml_source: dict[str, Any] | None = None) -> None:
        normalized_slug = (slug or "").strip().lower()
        if not normalized_slug:
            return
        effective_settings = _build_settings(normalized_slug, yaml_source)
        merged_bot_data = dict(payload)
        merged_bot_data["settings"] = effective_settings
        _RAW_BOT_DATA[normalized_slug] = merged_bot_data
        BUILTIN_BOTS[normalized_slug] = Bot(
            slug=normalized_slug,
            name=payload.get("name", normalized_slug.title()),
            description=payload.get("description", ""),
            system_prompt=payload.get("system_prompt", "You are a helpful assistant."),
            requires_memory=payload.get("requires_memory", True),
            voice_optimized=payload.get("voice_optimized", False),
            default_model=payload.get("default_model"),
            uses_tools=payload.get("uses_tools", False),
            uses_search=payload.get("uses_search", False),
            uses_home_assistant=payload.get("uses_home_assistant", False),
            nextcloud=payload.get("nextcloud"),
            settings=effective_settings,
        )

    # 1) DB bot profiles are authoritative for personality fields
    for slug, profile in db_profiles_by_slug.items():
        _register_bot(
            slug,
            {
                "name": profile.name,
                "description": profile.description,
                "system_prompt": profile.system_prompt,
                "requires_memory": profile.requires_memory,
                "voice_optimized": profile.voice_optimized,
                "default_model": profile.default_model,
                "uses_tools": profile.uses_tools,
                "uses_search": profile.uses_search,
                "uses_home_assistant": profile.uses_home_assistant,
                "nextcloud": profile.nextcloud_config,
            },
        )
    
    # 2) YAML fallback only for slugs missing in DB
    for slug, bot_data in merged_data.get("bots", {}).items():
        normalized_slug = (slug or "").strip().lower()
        if not normalized_slug or normalized_slug in BUILTIN_BOTS:
            continue
        _register_bot(normalized_slug, bot_data, yaml_source=bot_data)
    
    # Load defaults (merged)
    if "defaults" in merged_data:
        _DEFAULTS.update(merged_data["defaults"])
    
    # Load system prompts (merged)
    if "system_prompts" in merged_data:
        _SYSTEM_PROMPTS.update(merged_data["system_prompts"])
    
    logger.debug(
        f"Loaded {len(BUILTIN_BOTS)} bots from config "
        f"(repo: {repo_path.exists()}, user: {user_path.exists()}, db_profiles: {len(db_profiles_by_slug)})"
    )


def get_raw_bot_data(slug: str) -> dict | None:
    """Get raw bot config data by slug (for integrations).
    
    Returns the merged raw YAML data for a bot, including integration-specific
    sections like 'nextcloud'.
    """
    _check_reload()
    return _RAW_BOT_DATA.get(slug.lower().strip())


def get_all_raw_bot_data() -> dict[str, dict]:
    """Get all raw bot config data (for integrations)."""
    _check_reload()
    return _RAW_BOT_DATA.copy()


def get_bot_settings_template() -> dict[str, Any]:
    """Get the merged bot settings template/defaults."""
    _check_reload()
    return _BOT_SETTINGS_TEMPLATE.copy()


def save_user_bot_config(slug: str, section: str, data: dict) -> None:
    """Save bot-specific integration config to DB-backed bot profile."""
    normalized_slug = (slug or "").strip().lower()
    if not normalized_slug:
        raise ValueError("Bot slug is required")

    if section != "nextcloud":
        raise ValueError(f"Unsupported bot config section: {section}")

    config = Config()
    if not has_database_credentials(config):
        raise RuntimeError("Database credentials required to persist bot profile config")

    from llm_bawt.runtime_settings import BotProfileStore

    profile_store = BotProfileStore(config)
    if profile_store.engine is None:
        raise RuntimeError("Bot profiles DB unavailable")

    profile = profile_store.get(normalized_slug)
    if profile is None:
        source = get_raw_bot_data(normalized_slug) or {}
        profile_payload = {
            "slug": normalized_slug,
            "name": source.get("name", normalized_slug.title()),
            "description": source.get("description", ""),
            "system_prompt": source.get("system_prompt", "You are a helpful assistant."),
            "requires_memory": source.get("requires_memory", True),
            "voice_optimized": source.get("voice_optimized", False),
            "default_model": source.get("default_model"),
            "uses_tools": source.get("uses_tools", False),
            "uses_search": source.get("uses_search", False),
            "uses_home_assistant": source.get("uses_home_assistant", False),
            "nextcloud_config": data,
        }
    else:
        profile_payload = {
            "slug": profile.slug,
            "name": profile.name,
            "description": profile.description,
            "system_prompt": profile.system_prompt,
            "requires_memory": profile.requires_memory,
            "voice_optimized": profile.voice_optimized,
            "default_model": profile.default_model,
            "uses_tools": profile.uses_tools,
            "uses_search": profile.uses_search,
            "uses_home_assistant": profile.uses_home_assistant,
            "nextcloud_config": data,
        }

    profile_store.upsert(profile_payload)
    _load_bots_config()


def remove_user_bot_section(slug: str, section: str) -> bool:
    """Remove a section from DB-backed bot profile."""
    normalized_slug = (slug or "").strip().lower()
    if not normalized_slug:
        return False

    if section != "nextcloud":
        return False

    config = Config()
    if not has_database_credentials(config):
        return False

    from llm_bawt.runtime_settings import BotProfileStore

    profile_store = BotProfileStore(config)
    if profile_store.engine is None:
        return False

    profile = profile_store.get(normalized_slug)
    if profile is None or profile.nextcloud_config is None:
        return False

    profile_store.upsert(
        {
            "slug": profile.slug,
            "name": profile.name,
            "description": profile.description,
            "system_prompt": profile.system_prompt,
            "requires_memory": profile.requires_memory,
            "voice_optimized": profile.voice_optimized,
            "default_model": profile.default_model,
            "uses_tools": profile.uses_tools,
            "uses_search": profile.uses_search,
            "uses_home_assistant": profile.uses_home_assistant,
            "nextcloud_config": None,
        }
    )

    _load_bots_config()
    return True


def get_system_prompt(name: str) -> str | None:
    """Get a system prompt by name (e.g., 'refine')."""
    _check_reload()
    return _SYSTEM_PROMPTS.get(name)


# Initialize bots on module load
_load_bots_config()


class BotManager:
    """Manages bot loading and retrieval."""
    
    def __init__(self, config: Any = None):
        self.config = config
        logger.debug(f"BotManager initialized with {len(BUILTIN_BOTS)} bots")
    
    @property
    def _bots(self) -> dict[str, Bot]:
        """Get bots from global registry, checking for reload."""
        _check_reload()
        return BUILTIN_BOTS
    
    def get_bot(self, slug: str) -> Bot | None:
        """Get a bot by slug.
        
        Args:
            slug: The bot identifier (case-insensitive)
            
        Returns:
            The Bot instance, or None if not found
        """
        return self._bots.get(slug.lower().strip())
    
    def get_default_bot(self, local_mode: bool = False) -> Bot:
        """Get the default bot based on mode.
        
        Args:
            local_mode: If True, return local default; otherwise return standard default
            
        Returns:
            The default Bot instance
        """
        if local_mode:
            default_slug = _DEFAULTS.get("local", "spark")
        else:
            # Check config for default bot override
            if self.config and hasattr(self.config, 'DEFAULT_BOT'):
                config_default = getattr(self.config, 'DEFAULT_BOT', None)
                if config_default and config_default in self._bots:
                    return self._bots[config_default]
            default_slug = _DEFAULTS.get("standard", "nova")
        
        bot = self._bots.get(default_slug)
        if not bot:
            # Fallback to first available bot
            if self._bots:
                return next(iter(self._bots.values()))
            # Ultimate fallback - create a minimal bot
            return Bot(
                slug="assistant",
                name="Assistant",
                description="Default assistant",
                system_prompt="You are a helpful assistant.",
                requires_memory=False,
            )
        return bot

    def _is_model_alias(self, alias: str) -> bool:
        """Check if an alias exists in models.yaml."""
        if not self.config or not hasattr(self.config, "defined_models"):
            return False
        models = getattr(self.config, "defined_models", {}).get("models", {})
        return alias in models

    def select_model(
        self,
        requested_model: str | None,
        bot_slug: str | None = None,
        local_mode: bool = False,
    ) -> ModelSelection:
        """Resolve the effective model alias for a request.

        Priority:
        1) Explicit request model (unless it matches a bot slug and isn't a model alias)
        2) Bot default_model
        3) Config DEFAULT_MODEL_ALIAS
        """
        model_alias = requested_model.strip() if requested_model else None
        if model_alias == "":
            model_alias = None

        # If the provided model matches a bot slug (and isn't a model alias),
        # treat it as no model specified.
        if model_alias:
            bot_match = self.get_bot(model_alias)
            if bot_match and not self._is_model_alias(model_alias):
                model_alias = None

        if model_alias:
            return ModelSelection(alias=model_alias, source="explicit")

        bot = self.get_bot(bot_slug) if bot_slug else None
        if not bot:
            bot = self.get_default_bot(local_mode=local_mode)

        if bot and bot.default_model:
            return ModelSelection(alias=bot.default_model, source="bot_default")

        config_default = getattr(self.config, "DEFAULT_MODEL_ALIAS", None) if self.config else None
        if config_default:
            return ModelSelection(alias=config_default, source="config_default")

        return ModelSelection(alias=None, source="none")
    
    def list_bots(self) -> list[Bot]:
        """List all available bots.
        
        Returns:
            List of all Bot instances, sorted by slug
        """
        return sorted(self._bots.values(), key=lambda b: b.slug)

    def get_effective_settings(self, slug: str) -> dict[str, Any]:
        """Get effective bot settings (template hydrated + bot overrides)."""
        bot = self.get_bot(slug)
        return bot.settings.copy() if bot else {}
    
    def bot_exists(self, slug: str) -> bool:
        """Check if a bot exists.
        
        Args:
            slug: The bot identifier (case-insensitive)
            
        Returns:
            True if the bot exists
        """
        return slug.lower().strip() in self._bots
    
    def get_default_slug(self, local_mode: bool = False) -> str:
        """Get the default bot slug for the current mode."""
        if local_mode:
            return _DEFAULTS.get("local", "spark")
        return _DEFAULTS.get("standard", "nova")


def get_bot(slug: str) -> Bot | None:
    """Convenience function to get a bot by slug.
    
    Args:
        slug: The bot identifier (case-insensitive)
        
    Returns:
        The Bot instance, or None if not found
    """
    return BUILTIN_BOTS.get(slug.lower().strip())


def get_bot_system_prompt(slug: str) -> str | None:
    """Get the system prompt for a bot.
    
    Args:
        slug: The bot identifier (case-insensitive)
        
    Returns:
        The system prompt string, or None if bot not found
    """
    bot = get_bot(slug)
    return bot.system_prompt if bot else None


def strip_emotes(text: str) -> str:
    """Strip roleplay emotes/actions from text for TTS output.
    
    RP-tuned models often include *action* text despite prompt instructions.
    This function removes them for clean TTS output.
    
    Patterns removed:
    - *action text* (asterisk-wrapped actions)
    - ::action:: (colon-wrapped actions)
    - (action) when on its own line or at sentence boundaries
    - Multiple consecutive whitespace normalized
    
    Args:
        text: The raw LLM response text
        
    Returns:
        Clean text suitable for TTS
    """
    if not text:
        return text
    
    # Remove *action* patterns (asterisk-wrapped)
    # Matches: *smiles warmly*, *pauses*, etc.
    text = re.sub(r'\*[^*]+\*', '', text)
    
    # Remove ::action:: patterns (colon-wrapped, less common)
    text = re.sub(r'::[^:]+::', '', text)
    
    # Remove standalone (action) patterns (parentheses on their own)
    # Only remove if it's the whole line or at sentence boundaries
    # Be careful not to remove legitimate parenthetical content
    text = re.sub(r'^\s*\([^)]+\)\s*$', '', text, flags=re.MULTILINE)
    
    # Normalize multiple spaces/newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


class StreamingEmoteFilter:
    """Buffer-based filter for stripping *emotes* from streaming text.
    
    Handles cases where emote markers span chunk boundaries by buffering
    text between asterisks until we know if it's an emote or not.
    
    Usage:
        filter = StreamingEmoteFilter()
        for chunk in stream:
            filtered = filter.process(chunk)
            if filtered:
                yield filtered
        # Flush any remaining buffered content
        final = filter.flush()
        if final:
            yield final
    """
    
    def __init__(self):
        self.buffer = ""
        self.in_emote = False
    
    def process(self, chunk: str) -> str:
        """Process a chunk and return filtered text.
        
        Returns text that is safe to emit. May buffer text that could
        be part of an emote until we know for sure.
        """
        result = []
        
        for char in chunk:
            if char == '*':
                if self.in_emote:
                    # End of emote - discard buffered content
                    self.buffer = ""
                    self.in_emote = False
                else:
                    # Start of potential emote
                    # First, emit any buffered content
                    if self.buffer:
                        result.append(self.buffer)
                        self.buffer = ""
                    self.in_emote = True
            elif self.in_emote:
                # Inside potential emote - buffer it
                self.buffer += char
                # If the emote gets too long (>100 chars) it's probably not an emote
                if len(self.buffer) > 100:
                    # Not an emote - emit the asterisk and buffer
                    result.append('*')
                    result.append(self.buffer)
                    self.buffer = ""
                    self.in_emote = False
            else:
                # Normal character outside emote
                result.append(char)
        
        return ''.join(result)
    
    def flush(self) -> str:
        """Flush any remaining buffered content.
        
        Call this when the stream ends to get any remaining text.
        """
        if self.in_emote and self.buffer:
            # Stream ended mid-emote - emit what we have
            result = '*' + self.buffer
            self.buffer = ""
            self.in_emote = False
            return result
        return ""
