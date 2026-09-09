import json

import pytest

from agent_bridge.skill_packages import SkillPackageError, validate_package


def package(root):
    for vendor in ('claude', 'codex'):
        path = root / f'.{vendor}-plugin/plugin.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(name='example', version='1.0.0',
                                       description='Example skills', skills='./skills/')))
    skill = root / 'skills/example/SKILL.md'
    skill.parent.mkdir(parents=True)
    skill.write_text('---\nname: example\ndescription: Example task\n---\nDo the task.\n')
    return root


def test_dual_manifest_package(tmp_path):
    result = validate_package(package(tmp_path))
    assert result.name == 'example'
    assert result.skills == ('example',)
    assert len(result.sha256) == 64
    assert result == validate_package(tmp_path)


def test_digest_changes_with_resource_and_mode(tmp_path):
    package(tmp_path)
    helper = tmp_path / 'skills/example/helper.py'
    helper.write_text('print(1)')
    first = validate_package(tmp_path).sha256
    helper.write_text('print(2)')
    second = validate_package(tmp_path).sha256
    assert first != second
    helper.chmod(helper.stat().st_mode ^ 0o100)
    assert second != validate_package(tmp_path).sha256


@pytest.mark.parametrize('component', ['hooks', 'agents', 'commands', '.mcp.json', '.lsp.json'])
def test_reject_autoload_components(tmp_path, component):
    package(tmp_path)
    (tmp_path / component).write_text('{}')
    with pytest.raises(SkillPackageError, match='skills-only'):
        validate_package(tmp_path)


def test_reject_private_symlink(tmp_path):
    root = package(tmp_path / 'package')
    private = tmp_path / 'private.txt'
    private.write_text('private content')
    (root / 'skills/example/private').symlink_to(private)
    with pytest.raises(SkillPackageError, match='Symlink'):
        validate_package(root)


@pytest.mark.parametrize('field,value', [('version', '2.0.0'), ('name', 'other'),
                                       ('skills', '../private'), ('hooks', {})])
def test_reject_manifest_drift_and_components(tmp_path, field, value):
    package(tmp_path)
    path = tmp_path / '.codex-plugin/plugin.json'
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(SkillPackageError):
        validate_package(tmp_path)


@pytest.mark.parametrize('text', ['# no metadata', '---\n[]\n---\n',
                                 '---\nname: other\ndescription: valid\n---\n',
                                 '---\nname: example\ndescription: 42\n---\n'])
def test_reject_invalid_skill(tmp_path, text):
    package(tmp_path)
    (tmp_path / 'skills/example/SKILL.md').write_text(text)
    with pytest.raises(SkillPackageError):
        validate_package(tmp_path)
