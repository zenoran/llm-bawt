from __future__ import annotations

import logging
from pathlib import Path

from codex_bridge.local_plugins import install_repo_local_plugins


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Build a minimal repo-managed plugin tree under ``tmp_path``.

    Returns ``(home, dev_root, codex_home, source_root, repo_skill_md,
    system_skill_md)`` so callers can assert against any of them.
    """
    home = tmp_path / "home"
    dev_root = tmp_path / "dev"
    codex_home = tmp_path / ".codex"
    source_root = dev_root / "agent-skills" / "codex"

    repo_skill = dev_root / "agent-skills" / "repo-skill" / "SKILL.md"
    system_skill = codex_home / "skills" / ".system" / "system-skill" / "SKILL.md"

    _write(repo_skill, "# repo skill\n")
    _write(system_skill, "# system skill\n")
    _write(
        source_root / ".codex-plugin" / "marketplace.json",
        '{"name":"local-agent-skills","plugins":[]}\n',
    )
    _write(
        source_root / "plugins" / "agent-skills" / ".codex-plugin" / "plugin.json",
        '{"name":"agent-skills"}\n',
    )
    (source_root / "plugins" / "agent-skills" / "skills" / "repo-skill").mkdir(parents=True)
    (source_root / "plugins" / "agent-skills" / "skills" / "system-skill").mkdir(parents=True)

    return home, dev_root, codex_home, source_root, repo_skill, system_skill


def test_install_repo_local_plugins_stages_marketplace_and_skill_links(
    monkeypatch,
    tmp_path: Path,
):
    home, dev_root, codex_home, source_root, repo_skill, system_skill = _build_fixture(
        tmp_path
    )

    monkeypatch.setenv("CODEX_DEV_ROOT", str(dev_root))
    monkeypatch.setenv("CODEX_LOCAL_PLUGINS_SRC", str(source_root))

    install_repo_local_plugins(
        logger=logging.getLogger("test.codex_local_plugins"),
        codex_home=codex_home,
        home=home,
    )

    staged_marketplace = home / ".agents" / "plugins" / "marketplace.json"
    staged_plugin_meta_dir = home / "plugins" / "agent-skills" / ".codex-plugin"
    staged_repo_skill = home / "plugins" / "agent-skills" / "skills" / "repo-skill"
    staged_system_skill = home / "plugins" / "agent-skills" / "skills" / "system-skill"

    assert staged_marketplace.is_symlink()
    assert staged_marketplace.resolve() == source_root / ".codex-plugin" / "marketplace.json"

    assert staged_plugin_meta_dir.is_symlink()
    assert staged_plugin_meta_dir.resolve() == (
        source_root / "plugins" / "agent-skills" / ".codex-plugin"
    )

    assert staged_repo_skill.is_symlink()
    assert staged_repo_skill.resolve() == repo_skill.parent

    assert staged_system_skill.is_symlink()
    assert staged_system_skill.resolve() == system_skill.parent


def test_install_repo_local_plugins_populates_codex_cache(monkeypatch, tmp_path: Path):
    """Codex 0.128 only loads plugins from
    ``<codex_home>/plugins/cache/<marketplace>/<plugin>/<version>/``.
    The home-local layout is registration scaffolding; the cache is
    where the runtime loader actually reads from.
    """
    home, dev_root, codex_home, source_root, repo_skill, _ = _build_fixture(tmp_path)

    monkeypatch.setenv("CODEX_DEV_ROOT", str(dev_root))
    monkeypatch.setenv("CODEX_LOCAL_PLUGINS_SRC", str(source_root))

    install_repo_local_plugins(
        logger=logging.getLogger("test.codex_local_plugins"),
        codex_home=codex_home,
        home=home,
    )

    plugin_cache_root = (
        codex_home / "plugins" / "cache" / "local-agent-skills" / "agent-skills" / "v1"
    )

    cached_manifest = plugin_cache_root / ".codex-plugin" / "plugin.json"
    assert cached_manifest.exists()
    # Must be a real file, not a symlink — Codex's loader is strict.
    assert not cached_manifest.is_symlink()
    assert cached_manifest.read_text().startswith("{")

    cached_repo_skill = plugin_cache_root / "skills" / "repo-skill"
    assert cached_repo_skill.is_symlink()
    assert cached_repo_skill.resolve() == repo_skill.parent


def test_install_repo_local_plugins_writes_config_toml_entries(
    monkeypatch, tmp_path: Path
):
    home, dev_root, codex_home, source_root, _, _ = _build_fixture(tmp_path)

    monkeypatch.setenv("CODEX_DEV_ROOT", str(dev_root))
    monkeypatch.setenv("CODEX_LOCAL_PLUGINS_SRC", str(source_root))

    # Pre-existing config that the staging code must not clobber.
    config_toml = codex_home / "config.toml"
    _write(config_toml, 'model = "gpt-5.4"\n\n[plugins."github@openai-curated"]\nenabled = true\n')

    install_repo_local_plugins(
        logger=logging.getLogger("test.codex_local_plugins"),
        codex_home=codex_home,
        home=home,
    )

    contents = config_toml.read_text()
    # Existing user config preserved verbatim
    assert 'model = "gpt-5.4"' in contents
    assert '[plugins."github@openai-curated"]' in contents
    # New entries appended
    assert "[marketplaces.local-agent-skills]" in contents
    assert 'source_type = "local"' in contents
    assert f'source = "{home}"' in contents
    assert '[plugins."agent-skills@local-agent-skills"]' in contents


def test_install_repo_local_plugins_is_idempotent(monkeypatch, tmp_path: Path):
    home, dev_root, codex_home, source_root, _, _ = _build_fixture(tmp_path)

    monkeypatch.setenv("CODEX_DEV_ROOT", str(dev_root))
    monkeypatch.setenv("CODEX_LOCAL_PLUGINS_SRC", str(source_root))

    config_toml = codex_home / "config.toml"

    install_repo_local_plugins(
        logger=logging.getLogger("test.codex_local_plugins"),
        codex_home=codex_home,
        home=home,
    )
    first = config_toml.read_text()

    # Running again must not duplicate the marketplace/plugin sections —
    # otherwise every container restart would balloon config.toml.
    install_repo_local_plugins(
        logger=logging.getLogger("test.codex_local_plugins"),
        codex_home=codex_home,
        home=home,
    )
    second = config_toml.read_text()

    assert first == second
    assert second.count("[marketplaces.local-agent-skills]") == 1
    assert second.count('[plugins."agent-skills@local-agent-skills"]') == 1
