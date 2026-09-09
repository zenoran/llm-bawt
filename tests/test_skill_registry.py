
import pytest

from agent_bridge.skill_packages import SkillPackageError
from agent_bridge.skill_registry import SkillRegistry
from tests.test_skill_packages import package


def install(registry, source, visibility='public'):
    return registry.install(source, visibility=visibility, provenance='fixture@abc123')


def test_install_snapshot_select_and_offline_read(tmp_path):
    source = package(tmp_path / 'source')
    registry = SkillRegistry(tmp_path / 'registry')
    receipt = install(registry, source)
    registry.select('tenant', ['example'], audience='public')
    path = registry.bundle_paths('tenant')[0]
    assert path != source
    assert (path / 'skills/example/SKILL.md').is_file()
    assert registry.bundle('tenant')['packages'] == [receipt]
    (source / 'skills/example/SKILL.md').write_text('source changed')
    assert 'Do the task' in (path / 'skills/example/SKILL.md').read_text()


def test_update_does_not_mutate_existing_bundle_and_rollback(tmp_path):
    source = package(tmp_path / 'source')
    registry = SkillRegistry(tmp_path / 'registry')
    first = install(registry, source)
    registry.select('tenant', ['example'], audience='public')
    (source / 'skills/example/SKILL.md').write_text(
        '---\nname: example\ndescription: New content\n---\nUpdated\n')
    second = install(registry, source)
    assert second['sha256'] != first['sha256']
    assert registry.bundle('tenant')['packages'] == [first]
    assert registry.rollback('example') == first
    registry.select('next', ['example'], audience='public')
    assert registry.bundle('next')['packages'] == [first]


def test_failed_install_preserves_last_good(tmp_path):
    source = package(tmp_path / 'source')
    registry = SkillRegistry(tmp_path / 'registry')
    first = install(registry, source)
    (source / '.codex-plugin/plugin.json').write_text('{}')
    with pytest.raises(SkillPackageError):
        install(registry, source)
    assert registry.package('example') == first


def test_public_bundle_rejects_private(tmp_path):
    registry = SkillRegistry(tmp_path / 'registry')
    install(registry, package(tmp_path / 'source'), 'private')
    with pytest.raises(SkillPackageError, match='private'):
        registry.select('tenant', ['example'], audience='public')
    assert not (registry.root / 'bundles/tenant.json').exists()


def test_remove_preserves_active_bundle(tmp_path):
    registry = SkillRegistry(tmp_path / 'registry')
    install(registry, package(tmp_path / 'source'))
    registry.select('tenant', ['example'], audience='public')
    registry.remove('example')
    assert registry.bundle_paths('tenant')[0].exists()
    with pytest.raises(FileNotFoundError):
        registry.select('next', ['example'], audience='public')


@pytest.mark.parametrize('name', ['../escape', '/absolute', '.', 'name/other', 'UPPER'])
def test_reject_invalid_names(tmp_path, name):
    with pytest.raises(SkillPackageError):
        SkillRegistry(tmp_path).select(name, [], audience='public')


def test_empty_selection_disables_packages(tmp_path):
    registry = SkillRegistry(tmp_path)
    assert registry.select('none', [], audience='public')['packages'] == []
    assert registry.bundle_paths('none') == []


def test_reinstall_rejects_modified_release(tmp_path):
    registry = SkillRegistry(tmp_path / 'registry')
    source = package(tmp_path / 'source')
    receipt = install(registry, source)
    (registry.root / 'releases' / receipt['release'] / 'package/extra').write_text('tamper')
    with pytest.raises(SkillPackageError, match='modified'):
        install(registry, source)


@pytest.mark.parametrize('mutation', ['content', 'mode'])
def test_selected_bundle_rejects_modified_release(tmp_path, mutation):
    registry = SkillRegistry(tmp_path / 'registry')
    receipt = install(registry, package(tmp_path / 'source'))
    registry.select('tenant', ['example'], audience='public')
    skill = registry.root / 'releases' / receipt['release'] / 'package/skills/example/SKILL.md'
    if mutation == 'content':
        skill.write_text(skill.read_text() + '\ntampered\n')
    else:
        skill.chmod(skill.stat().st_mode ^ 0o100)
    with pytest.raises(SkillPackageError, match='modified'):
        registry.bundle_paths('tenant')
