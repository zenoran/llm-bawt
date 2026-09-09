"""Validate portable, skills-only packages before native harness installation.

Validation is deliberately local and read-only. It neither installs packages nor
turns skill metadata into authorization. Native plugin managers own their caches.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
# Reject executable plugin components, including auto-discovered ones. Bundled
# skill scripts remain possible; installation still requires trusted provenance.
_COMPONENTS = {
    "hooks", "mcpServers", "lspServers", "agents", "commands", "settings",
    "userConfig", "outputStyles", "experimental", "dependencies", "apps",
    "channels", "workflows", "monitors",
}
_ROOT_COMPONENTS = {
    "hooks", "agents", "commands", "settings.json", ".mcp.json", ".app.json",
    ".lsp.json", "workflows", "monitors", "output-styles",
}


@dataclass(frozen=True)
class SkillPackage:
    name: str
    version: str
    root: Path
    skills: tuple[str, ...]
    sha256: str


class SkillPackageError(ValueError):
    """Package cannot safely participate in the portable skills-only contract."""


def validate_package(root: Path) -> SkillPackage:
    """Validate dual manifests, contained resources and portable skill metadata.

    Symlinks are rejected rather than followed: native installers differ in
    whether they copy or preserve them, and external links can leak private data.
    Authoring trees may use links, but distributable artifacts must be real files.
    """
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise SkillPackageError("Package root must be a directory")
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SkillPackageError(f"Symlink not allowed in package: {path.relative_to(root)}")
        if not path.is_dir() and not path.is_file():
            raise SkillPackageError(f"Special file not allowed: {path.relative_to(root)}")
        if path.is_file():
            files.append(path)
    for component in sorted(_ROOT_COMPONENTS):
        if (root / component).exists():
            raise SkillPackageError(f"Not a skills-only package: {component}")

    manifests = []
    for vendor in ("claude", "codex"):
        path = root / f".{vendor}-plugin" / "plugin.json"
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise SkillPackageError(f"Invalid {vendor} manifest: {exc}") from exc
        if not isinstance(data, dict):
            raise SkillPackageError(f"{vendor} manifest must be an object")
        forbidden = _COMPONENTS.intersection(data)
        if forbidden:
            raise SkillPackageError(f"Not skills-only: {', '.join(sorted(forbidden))}")
        name = data.get("name")
        if not isinstance(name, str) or len(name) > 64 or not _NAME.fullmatch(name):
            raise SkillPackageError("Package name must be lowercase kebab-case, at most 64 characters")
        if not isinstance(data.get("version"), str) or not data["version"].strip():
            raise SkillPackageError("A stable package version is required")
        if not isinstance(data.get("description"), str) or not data["description"].strip():
            raise SkillPackageError("Package description is required")
        if data.get("skills") != "./skills/":
            raise SkillPackageError("Portable packages must declare skills: './skills/'")
        manifests.append(data)
    if any(manifests[0][k] != manifests[1][k] for k in ("name", "version")):
        raise SkillPackageError("Harness manifests disagree on package identity/version")

    skills_root = root / "skills"
    skills = []
    if not skills_root.is_dir():
        raise SkillPackageError("Missing skills directory")
    for directory in sorted(skills_root.iterdir()):
        if not directory.is_dir():
            raise SkillPackageError(f"Unexpected file in skills directory: {directory.name}")
        path = directory / "SKILL.md"
        try:
            text = path.read_text()
            lines = text.splitlines()
            if not lines or lines[0] != "---":
                raise ValueError("missing YAML frontmatter")
            try:
                end = lines.index("---", 1)
            except ValueError as exc:
                raise ValueError("missing closing frontmatter delimiter") from exc
            metadata = yaml.safe_load("\n".join(lines[1:end]))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise SkillPackageError(f"Invalid skill {directory.name}: {exc}") from exc
        if not isinstance(metadata, dict):
            raise SkillPackageError(f"Skill metadata must be an object: {directory.name}")
        name = metadata.get("name")
        if (not isinstance(name, str) or name != directory.name
                or len(name) > 64 or not _NAME.fullmatch(name)):
            raise SkillPackageError(f"Invalid skill name: {directory.name}")
        description = metadata.get("description")
        if not isinstance(description, str) or not description.strip() or len(description) > 1024:
            raise SkillPackageError(f"Invalid description: {directory.name}")
        skills.append(name)
    if not skills:
        raise SkillPackageError("Package must contain at least one skill")

    digest = hashlib.sha256()
    for path in files:
        content = path.read_bytes()
        relative = path.relative_to(root).as_posix().encode()
        for value in (relative, str(path.stat().st_mode & 0o111).encode(), content):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return SkillPackage(manifests[0]["name"], manifests[0]["version"], root,
                        tuple(skills), digest.hexdigest())
