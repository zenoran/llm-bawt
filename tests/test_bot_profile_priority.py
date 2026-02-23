"""Regression tests for bot profile precedence and prompt cache invalidation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_bawt import bots as bots_module
from llm_bawt.service.background_service import BackgroundService


def test_db_bot_profile_overrides_yaml(monkeypatch, tmp_path: Path) -> None:
    """DB bot_profiles should override YAML bot fields and support DB-only bots."""
    repo_yaml = tmp_path / "repo-bots.yaml"
    user_yaml = tmp_path / "user-bots.yaml"
    repo_yaml.write_text("{}", encoding="utf-8")
    user_yaml.write_text("{}", encoding="utf-8")

    repo_data = {
        "bot_settings_template": {"temperature": 0.55, "ui_color": "blue"},
        "bots": {
            "nova": {
                "name": "Nova YAML",
                "description": "from repo",
                "system_prompt": "repo prompt",
                "requires_memory": True,
                "voice_optimized": False,
                "uses_tools": False,
                "uses_search": False,
                "uses_home_assistant": False,
                "default_model": "repo-model",
                "color": "red",
                "settings": {"temperature": 0.8},
            }
        },
    }
    user_data = {"bots": {"nova": {"description": "from user"}}}
    db_overrides = {
        "nova": {
            "name": "Nova DB",
            "description": "from db",
            "system_prompt": "db prompt",
            "requires_memory": False,
            "voice_optimized": True,
            "uses_tools": True,
            "uses_search": True,
            "uses_home_assistant": True,
            "default_model": "db-model",
            "nextcloud": {"bot_id": "nc-nova"},
        },
        "ember": {
            "name": "Ember",
            "description": "db only bot",
            "system_prompt": "ember prompt",
            "requires_memory": True,
            "voice_optimized": False,
            "uses_tools": False,
            "uses_search": False,
            "uses_home_assistant": False,
            "default_model": None,
            "nextcloud": None,
        },
    }

    def fake_load_yaml_file(path: Path) -> dict[str, Any]:
        if path == repo_yaml:
            return repo_data
        if path == user_yaml:
            return user_data
        return {}

    monkeypatch.setattr(bots_module, "get_repo_bots_yaml_path", lambda: repo_yaml)
    monkeypatch.setattr(bots_module, "get_user_bots_yaml_path", lambda: user_yaml)
    monkeypatch.setattr(bots_module, "_load_yaml_file", fake_load_yaml_file)
    monkeypatch.setattr(bots_module, "_load_db_bot_overrides", lambda: db_overrides)

    bots_module._load_bots_config()

    nova = bots_module.get_bot("nova")
    assert nova is not None
    assert nova.system_prompt == "db prompt"
    assert nova.description == "from db"
    assert nova.requires_memory is False
    assert nova.voice_optimized is True
    assert nova.default_model == "db-model"
    assert nova.color == "red"  # Keep YAML-presentational metadata.
    assert nova.settings["temperature"] == 0.8  # Keep YAML settings.
    assert bots_module.get_raw_bot_data("nova")["nextcloud"]["bot_id"] == "nc-nova"

    ember = bots_module.get_bot("ember")
    assert ember is not None
    assert ember.system_prompt == "ember prompt"
    assert ember.settings["temperature"] == 0.55
    assert ember.color == "blue"


def test_background_service_invalidates_bot_instances() -> None:
    """Bot-level cache invalidation should only evict matching bot keys."""
    service = BackgroundService.__new__(BackgroundService)
    service._llm_bawt_cache = {
        ("model-a", "nova", "u1"): object(),
        ("model-b", "nova", "u2"): object(),
        ("model-a", "spark", "u1"): object(),
    }

    removed = service.invalidate_bot_instances("nova")

    assert removed == 2
    assert ("model-a", "spark", "u1") in service._llm_bawt_cache
    assert ("model-a", "nova", "u1") not in service._llm_bawt_cache
    assert ("model-b", "nova", "u2") not in service._llm_bawt_cache


def test_background_service_invalidates_all_instances() -> None:
    """Global cache invalidation should clear all cached instances."""
    service = BackgroundService.__new__(BackgroundService)
    service._llm_bawt_cache = {
        ("model-a", "nova", "u1"): object(),
        ("model-b", "spark", "u2"): object(),
    }

    removed = service.invalidate_all_instances()

    assert removed == 2
    assert service._llm_bawt_cache == {}
