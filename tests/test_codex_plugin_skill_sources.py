"""Package sources must not be shadowed by legacy private-repo mappings."""
from pathlib import Path

from codex_bridge.local_plugins import _resolve_skill_target


def _skill(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text("---\nname: shared\ndescription: Test skill\n---\n")
    return path


def _resolve(root: Path, entry: Path) -> Path | None:
    return _resolve_skill_target(
        skill_name="shared",
        dev_root=root / "dev",
        codex_home=root / "codex",
        source_skill_entry=entry,
    )


def test_bundled_skill_wins_over_private_and_system_names(tmp_path: Path):
    _skill(tmp_path / "dev/agent-skills/shared")
    _skill(tmp_path / "codex/skills/.system/shared")
    bundled = _skill(tmp_path / "public-plugin/skills/shared")
    assert _resolve(tmp_path, bundled) == bundled


def test_explicit_other_repository_link_wins_over_private_name(tmp_path: Path):
    _skill(tmp_path / "dev/agent-skills/shared")
    target = _skill(tmp_path / "other-repo/shared")
    entry = tmp_path / "linked-skill"
    entry.symlink_to(target, target_is_directory=True)
    assert _resolve(tmp_path, entry) == target


def test_broken_explicit_link_does_not_fall_back_to_private_skill(tmp_path: Path):
    _skill(tmp_path / "dev/agent-skills/shared")
    entry = tmp_path / "broken-skill"
    entry.symlink_to(tmp_path / "missing", target_is_directory=True)
    assert _resolve(tmp_path, entry) is None


def test_explicit_link_without_skill_does_not_fall_back(tmp_path: Path):
    _skill(tmp_path / "dev/agent-skills/shared")
    target = tmp_path / "empty-target"
    target.mkdir()
    entry = tmp_path / "linked-skill"
    entry.symlink_to(target, target_is_directory=True)
    assert _resolve(tmp_path, entry) is None


def test_legacy_empty_mapping_still_resolves_repo_skill(tmp_path: Path):
    target = _skill(tmp_path / "dev/agent-skills/shared")
    entry = tmp_path / "legacy-entry"
    entry.mkdir()
    assert _resolve(tmp_path, entry) == target


def test_legacy_empty_mapping_still_resolves_system_skill(tmp_path: Path):
    target = _skill(tmp_path / "codex/skills/.system/shared")
    entry = tmp_path / "legacy-entry"
    entry.mkdir()
    assert _resolve(tmp_path, entry) == target
