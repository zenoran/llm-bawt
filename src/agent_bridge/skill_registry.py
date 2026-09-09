"""Immutable local package registry and atomic named bundle selection.

The operator supplies already-fetched, trusted, skills-only package directories.
No network operations occur on chat turns. Registry writes are explicit admin
operations; selections are plain JSON suitable for backup/container mounts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from .skill_packages import SkillPackageError, validate_package

_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


def _name(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value) or len(value) > 64:
        raise SkillPackageError("Invalid registry name")
    return value


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class SkillRegistry:
    def __init__(self, root: Path):
        self.root = Path(root).absolute()

    @contextmanager
    def _lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def install(self, source: Path, *, visibility: str, provenance: str) -> dict:
        """Copy, validate and activate one immutable package release.

        Updating the named package never changes previously constructed bundles.
        Public/private classification is an operator decision, not a grep result.
        """
        if visibility not in {"public", "private"}:
            raise SkillPackageError("Visibility must be public or private")
        if not provenance.strip():
            raise SkillPackageError("Source provenance is required (repo and resolved commit)")
        package = validate_package(source)
        with self._lock():
            releases = self.root / "releases"
            releases.mkdir(exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=releases))
            try:
                shutil.copytree(package.root, stage / "package")
                copied = validate_package(stage / "package")
                if copied.sha256 != package.sha256:
                    raise SkillPackageError("Source changed during package snapshot")
                release_id = f"{package.name}-{package.sha256}"
                target = releases / release_id
                receipt = dict(name=package.name, version=package.version,
                               sha256=package.sha256, skills=list(package.skills),
                               visibility=visibility, provenance=provenance,
                               release=release_id)
                _atomic_json(stage / "receipt.json", receipt)
                if target.exists():
                    prior = json.loads((target / "receipt.json").read_text())
                    if prior != receipt:
                        raise SkillPackageError("Existing release has different provenance/classification")
                    if validate_package(target / "package").sha256 != package.sha256:
                        raise SkillPackageError("Existing immutable release was modified")
                else:
                    os.rename(stage, target)
                index = self.root / "packages" / f"{package.name}.json"
                old = json.loads(index.read_text()) if index.exists() else None
                if not old or old.get("current") != receipt:
                    _atomic_json(index, {"current": receipt, "previous": old.get("current") if old else None})
                return receipt
            finally:
                if stage.exists():
                    shutil.rmtree(stage)

    def package(self, name: str) -> dict:
        return json.loads((self.root / "packages" / f"{_name(name)}.json").read_text())["current"]

    def rollback(self, name: str) -> dict:
        with self._lock():
            index = self.root / "packages" / f"{_name(name)}.json"
            state = json.loads(index.read_text())
            if not state.get("previous"):
                raise SkillPackageError("No previous package release")
            _atomic_json(index, {"current": state["previous"], "previous": state["current"]})
            return state["previous"]

    def select(self, name: str, packages: list[str], *, audience: str) -> dict:
        """Atomically publish a bundle of exact releases; never mutate old ones."""
        _name(name)
        if audience not in {"public", "private"}:
            raise SkillPackageError("Audience must be public or private")
        if len(packages) != len(set(packages)):
            raise SkillPackageError("Duplicate package selection")
        with self._lock():
            receipts = [self.package(item) for item in sorted(packages)]
            if audience == "public" and any(r["visibility"] != "public" for r in receipts):
                raise SkillPackageError("Public bundle cannot include private packages")
            state = {"name": name, "audience": audience, "packages": receipts}
            index = self.root / "bundles" / f"{name}.json"
            old = json.loads(index.read_text()) if index.exists() else None
            if not old or old.get("current") != state:
                _atomic_json(index, {"current": state, "previous": old.get("current") if old else None})
            return state

    def bundle(self, name: str) -> dict:
        state = json.loads((self.root / "bundles" / f"{_name(name)}.json").read_text())["current"]
        for receipt in state["packages"]:
            release = receipt["release"]
            if Path(release).name != release or release in {".", ".."}:
                raise SkillPackageError("Invalid release path")
            root = self.root / "releases" / release / "package"
            if root.is_symlink() or not root.is_dir():
                raise SkillPackageError(f"Missing package release: {release}")
            verified = validate_package(root)
            if (
                verified.name != receipt.get("name")
                or verified.version != receipt.get("version")
                or list(verified.skills) != receipt.get("skills")
                or verified.sha256 != receipt.get("sha256")
            ):
                raise SkillPackageError(f"Installed package release was modified: {release}")
        return state

    def rollback_bundle(self, name: str) -> dict:
        with self._lock():
            index = self.root / "bundles" / f"{_name(name)}.json"
            state = json.loads(index.read_text())
            if not state.get("previous"):
                raise SkillPackageError("No previous bundle selection")
            _atomic_json(index, {"current": state["previous"], "previous": state["current"]})
            return state["previous"]

    def bundle_paths(self, name: str) -> list[Path]:
        return [self.root / "releases" / r["release"] / "package"
                for r in self.bundle(name)["packages"]]

    def remove(self, name: str) -> None:
        """Remove package from future selection; retain files used by sessions."""
        with self._lock():
            (self.root / "packages" / f"{_name(name)}.json").unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("install")
    install.add_argument("source", type=Path)
    install.add_argument("--visibility", choices=["public", "private"], required=True)
    install.add_argument("--provenance", required=True)
    select = commands.add_parser("select")
    select.add_argument("name")
    select.add_argument("packages", nargs="*")
    select.add_argument("--audience", choices=["public", "private"], required=True)
    prepare = commands.add_parser("prepare-codex")
    prepare.add_argument("name")
    prepare.add_argument("--binary", required=True)
    prepare.add_argument("--base-home", type=Path, required=True)
    for command in ("inspect", "rollback", "rollback-bundle", "remove"):
        commands.add_parser(command).add_argument("name")
    args = parser.parse_args()
    registry = SkillRegistry(args.root)
    if args.command == "install":
        result = registry.install(args.source, visibility=args.visibility, provenance=args.provenance)
    elif args.command == "select":
        result = registry.select(args.name, args.packages, audience=args.audience)
    elif args.command == "prepare-codex":
        from .skill_codex import prepare_codex
        result = prepare_codex(registry, args.name, binary=args.binary, base_home=args.base_home)
    elif args.command == "inspect":
        result = registry.bundle(args.name)
    elif args.command == "rollback-bundle":
        result = registry.rollback_bundle(args.name)
    elif args.command == "rollback":
        result = registry.rollback(args.name)
    else:
        registry.remove(args.name)
        result = {"removed": args.name, "release_files_retained": True}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
