"""Native Codex preparation in a dedicated home, never on the chat hot path.

Auth/session files remain owned by the existing bridge. This isolates plugin
configuration, not filesystem access or repository-scoped skills.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .skill_registry import SkillRegistry, _atomic_json
from .skill_packages import SkillPackageError


def bundle_key(state: dict) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def prepare_codex(registry: SkillRegistry, bundle: str, *, binary: str,
                  base_home: Path) -> dict:
    """Explicit admin action: native installation into a fresh isolated home.

    Does not mutate base_home, its config, native plugin caches or credentials.
    Generated homes retain the base configuration text and override only plugin
    and marketplace tables using a TOML-aware round trip.
    """
    import tomlkit

    state = registry.bundle(bundle)
    selection_key = bundle_key(state)
    config_path = base_home / 'config.toml'
    base_config = config_path.read_text() if config_path.exists() else ''
    config_digest = hashlib.sha256(base_config.encode()).hexdigest()
    key = hashlib.sha256((selection_key + str(base_home.resolve()) + config_digest).encode()).hexdigest()
    pointer = registry.root / 'codex-prepared' / f'{selection_key}.json'
    with registry._lock():
        homes = registry.root / 'codex'
        homes.mkdir(exist_ok=True)
        target = homes / key
        receipt_path = target / 'prepared.json'
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            _atomic_json(pointer, receipt)
            return receipt
        if target.exists():
            raise SkillPackageError('Incomplete Codex home; inspect before retrying')
        # Native manager stores absolute paths: build at the FINAL path, but
        # make it selectable only after every command succeeds.
        target.mkdir()
        try:
            codex_home = target / '.codex'
            codex_home.mkdir()
            config = tomlkit.parse(base_config)
            for section in ('plugins', 'marketplaces'):
                config.pop(section, None)
            (codex_home / 'config.toml').write_text(tomlkit.dumps(config))
            for name in ('auth.json', 'sessions'):
                if (base_home / name).exists():
                    (codex_home / name).symlink_to((base_home / name).resolve())
            marketplace = target / 'marketplace'
            entries = []
            for path, receipt in zip(registry.bundle_paths(bundle), state['packages']):
                destination = marketplace / 'plugins' / receipt['name']
                shutil.copytree(path, destination)
                entries.append({'name': receipt['name'],
                                'source': {'source': 'local', 'path': './plugins/' + receipt['name']},
                                'policy': {'installation': 'AVAILABLE', 'authentication': 'ON_INSTALL'},
                                'category': 'Productivity'})
            _atomic_json(marketplace / '.agents/plugins/marketplace.json',
                         {'name': 'bawt-managed', 'plugins': entries})
            env = dict(os.environ, HOME=str(target), CODEX_HOME=str(codex_home))
            def run(*args):
                result = subprocess.run([binary, 'plugin', *args, '--json'],
                                        env=env, cwd=target, text=True,
                                        capture_output=True, timeout=120, check=True)
                return json.loads(result.stdout)
            run('marketplace', 'add', str(marketplace))
            installed = [run('add', r['name'] + '@bawt-managed') for r in state['packages']]
            receipt = {'bundle_key': selection_key, 'generation': key,
                       'config_sha256': config_digest, 'bundle': bundle, 'home': str(target),
                       'codex_home': str(codex_home), 'installed': installed,
                       'base_home': str(base_home.resolve())}
            _atomic_json(receipt_path, receipt)
            _atomic_json(pointer, receipt)
            return receipt
        except BaseException:
            shutil.rmtree(target)
            raise


def codex_skill_env(bundle: str | None) -> dict | None:
    if bundle is None:
        return None
    root = os.environ.get('AGENT_SKILL_REGISTRY')
    if not root:
        raise SkillPackageError('AGENT_SKILL_REGISTRY is required for skill_bundle selection')
    registry = SkillRegistry(Path(root))
    key = bundle_key(registry.bundle(bundle))
    receipt_path = registry.root / 'codex-prepared' / f'{key}.json'
    if not receipt_path.is_file():
        raise SkillPackageError(f'Codex bundle {bundle!r} is not prepared; run prepare-codex first')
    receipt = json.loads(receipt_path.read_text())
    expected = registry.root / 'codex' / receipt['generation']
    if (Path(receipt['generation']).name != receipt['generation']
            or Path(receipt['home']) != expected
            or Path(receipt['codex_home']) != expected / '.codex'
            or not (expected / 'prepared.json').is_file()):
        raise SkillPackageError('Invalid prepared Codex home receipt')
    return dict(os.environ, HOME=receipt['home'], CODEX_HOME=receipt['codex_home'])
