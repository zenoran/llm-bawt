"""Read-only preflight for probes claiming to exercise the deployed bridge.

A host Codex login is not evidence of the bridge's app-owned credential state.
Never fetch, copy, refresh, or print credential values here.
"""
from __future__ import annotations

import os
from pathlib import Path

from .transport import auth_path


def validate_probe_context(*, base_home: Path, binary: str) -> dict[str, str]:
    if not os.environ.get('LLM_BAWT_API_URL'):
        raise ValueError(
            'Run this probe inside the configured Codex bridge: LLM_BAWT_API_URL '
            'is missing. A host ~/.codex/auth.json is not the bridge credential source.'
        )
    configured_home = os.environ.get('CODEX_HOME')
    configured_binary = os.environ.get('CODEX_BIN')
    if not configured_home or not configured_binary:
        raise ValueError('Probe requires the live bridge CODEX_HOME and CODEX_BIN configuration')
    base_home = base_home.expanduser().resolve()
    if base_home != Path(configured_home).expanduser().resolve():
        raise ValueError('Probe base home differs from configured bridge CODEX_HOME')
    selected_auth = base_home / 'auth.json'
    if selected_auth.resolve() != auth_path().resolve():
        raise ValueError('Probe auth path differs from configured bridge CODEX_AUTH_PATH')
    if not selected_auth.is_file():
        raise ValueError('Bridge auth file is missing; inspect app broker materialization')
    if Path(binary).resolve() != Path(configured_binary).resolve():
        raise ValueError('Probe binary differs from configured bridge CODEX_BIN')
    if not Path(binary).is_file():
        raise ValueError('Configured bridge executable is missing')
    return {'auth_source': 'configured-bridge-file', 'base_home': str(base_home),
            'auth_path': str(selected_auth), 'binary': str(Path(binary).resolve())}
