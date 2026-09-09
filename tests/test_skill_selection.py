from pathlib import Path
from unittest.mock import patch

import pytest

from agent_bridge.skill_codex import codex_skill_env, prepare_codex
from agent_bridge.skill_packages import SkillPackageError
from agent_bridge.skill_registry import SkillRegistry
from agent_bridge.skill_selection import claude_skill_options
from tests.test_skill_packages import package


@pytest.fixture
def registry(tmp_path, monkeypatch):
    registry = SkillRegistry(tmp_path / 'registry')
    registry.install(package(tmp_path / 'source'), visibility='public', provenance='fixture@1')
    registry.select('tenant', ['example'], audience='public')
    monkeypatch.setenv('AGENT_SKILL_REGISTRY', str(registry.root))
    return registry


def test_bundle_change_requires_fresh_conversation(registry):
    from agent_bridge.skill_selection import guard_resumed_bundle
    guard_resumed_bundle('tenant', thread='thread-1', resume=None)
    guard_resumed_bundle('tenant', thread='thread-1', resume='sdk-session')
    registry.select('tenant', [], audience='public')
    with pytest.raises(SkillPackageError, match='changed'):
        guard_resumed_bundle('tenant', thread='thread-1', resume='sdk-session')
    guard_resumed_bundle('tenant', thread='thread-2', resume=None)


def test_selecting_bundle_on_legacy_resume_fails(registry):
    from agent_bridge.skill_selection import guard_resumed_bundle
    with pytest.raises(SkillPackageError, match='new conversation'):
        guard_resumed_bundle('tenant', thread='legacy', resume='sdk-session')


def test_legacy_settings_unchanged():
    assert claude_skill_options(None) == {}
    assert codex_skill_env(None) is None


def test_legacy_session_never_touches_configured_registry(tmp_path, monkeypatch):
    from agent_bridge.skill_selection import guard_resumed_bundle

    missing = tmp_path / 'must-not-be-created'
    monkeypatch.setenv('AGENT_SKILL_REGISTRY', str(missing))
    guard_resumed_bundle(None, thread='legacy-thread', resume='legacy-session')
    assert not missing.exists()


def test_claude_explicit_bundle_filters_discovery(registry):
    options = claude_skill_options('tenant')
    assert options['skills'] == ['example:example']
    assert options['setting_sources'] == []
    assert options['plugins'] == [{'type': 'local', 'path': str(registry.bundle_paths('tenant')[0])}]


def test_missing_codex_preparation_fails_closed(registry):
    with pytest.raises(SkillPackageError, match='not prepared'):
        codex_skill_env('tenant')


def test_codex_preparation_preserves_base_config_and_auth(registry, tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    config = '# Operator comment\nmodel = "test"\n[plugins."private@old"]\nenabled = true\n'
    (base / 'config.toml').write_text(config)
    (base / 'auth.json').write_text('{}')
    with patch('agent_bridge.skill_codex.subprocess.run') as run:
        run.return_value.stdout = '{}'
        receipt = prepare_codex(registry, 'tenant', binary='/bin/codex', base_home=base)
    assert (base / 'config.toml').read_text() == config
    generated = (Path(receipt['codex_home']) / 'config.toml').read_text()
    assert 'Operator comment' in generated and 'private@old' not in generated
    assert (Path(receipt['codex_home']) / 'auth.json').resolve() == base / 'auth.json'
    assert run.call_count == 2
    assert codex_skill_env('tenant')['CODEX_HOME'] == receipt['codex_home']


def test_config_update_creates_generation_and_guards_resume(registry, tmp_path):
    from agent_bridge.skill_selection import guard_resumed_bundle
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'config.toml').write_text('model = "old"\n')
    with patch('agent_bridge.skill_codex.subprocess.run') as run:
        run.return_value.stdout = '{}'
        first = prepare_codex(registry, 'tenant', binary='/bin/codex', base_home=base)
        guard_resumed_bundle('tenant', thread='codex-thread', resume=None, harness='codex')
        (base / 'config.toml').write_text('model = "new"\n')
        second = prepare_codex(registry, 'tenant', binary='/bin/codex', base_home=base)
    assert first['codex_home'] != second['codex_home']
    assert Path(first['codex_home']).is_dir()
    assert codex_skill_env('tenant')['CODEX_HOME'] == second['codex_home']
    with pytest.raises(SkillPackageError, match='changed'):
        guard_resumed_bundle('tenant', thread='codex-thread', resume='old-session', harness='codex')


def test_failed_codex_preparation_not_selectable(registry, tmp_path):
    with patch('agent_bridge.skill_codex.subprocess.run', side_effect=RuntimeError('install failed')):
        with pytest.raises(RuntimeError):
            prepare_codex(registry, 'tenant', binary='/bin/codex', base_home=tmp_path)
    with pytest.raises(SkillPackageError, match='not prepared'):
        codex_skill_env('tenant')
