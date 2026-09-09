"""Resolve an explicitly selected bundle; legacy bots remain unchanged."""
from __future__ import annotations

import os
from pathlib import Path

from .skill_registry import SkillRegistry, _name


def guard_resumed_bundle(bundle: str | None, *, thread: str | None, resume: str | None,
                         harness: str = 'claude') -> None:
    """Require a fresh conversation when its configured skill bundle changes."""
    import hashlib
    import json
    from .skill_registry import _atomic_json
    from .skill_packages import SkillPackageError

    # Opt-in means legacy turns have no registry dependency at all: no reads,
    # locks, directories or session-binding files when no bundle is selected.
    if bundle is None:
        return
    root = os.environ.get("AGENT_SKILL_REGISTRY")
    if not root or not thread:
        if resume:
            raise SkillPackageError("Start a new conversation before selecting a skill bundle")
        return
    registry = SkillRegistry(Path(root))
    state = registry.bundle(bundle)
    generation = None
    if harness == 'codex':
        from .skill_codex import codex_skill_env
        generation = codex_skill_env(bundle)['CODEX_HOME']
    digest = hashlib.sha256(json.dumps([state, generation], sort_keys=True).encode()).hexdigest()
    key = hashlib.sha256((harness + ':' + thread).encode()).hexdigest()
    path = registry.root / "session-bindings" / f"{key}.json"
    with registry._lock():
        if resume:
            if path.exists():
                old = json.loads(path.read_text())
                if old["bundle_digest"] != digest:
                    raise SkillPackageError("Skill bundle changed; start a new conversation to activate it")
            else:
                raise SkillPackageError("Start a new conversation before selecting a skill bundle")
        _atomic_json(path, {"bundle_digest": digest, "bundle": bundle})


def claude_skill_options(bundle: str | None) -> dict:
    if bundle is None:
        return {}
    _name(bundle)
    root = os.environ.get("AGENT_SKILL_REGISTRY")
    if not root:
        raise ValueError("AGENT_SKILL_REGISTRY is required for skill_bundle selection")
    registry = SkillRegistry(Path(root))
    state = registry.bundle(bundle)
    paths = registry.bundle_paths(bundle)
    return {
        "plugins": [{"type": "local", "path": str(path)} for path in paths],
        "skills": [f"{receipt['name']}:{skill}"
                   for receipt in state['packages'] for skill in receipt['skills']],
        # Explicitly supplied skills must not opt into ambient user/project
        # discovery. The bridge still provides its MCP and hook settings.
        "setting_sources": [],
    }
