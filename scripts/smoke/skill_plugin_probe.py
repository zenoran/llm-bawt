"""Opt-in native Codex SDK skill invocation probe, with an isolated config home.

Requires the actual bridge's configured executable and container-local auth
path; refuses unrelated host credentials before creating files or invoking the
SDK. Uses bridge authentication through a symlink. Does not alter live plugin
config, bot profiles, or existing conversations. Run only when
an actual inference probe is authorized. Receipts contain no credential values.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import tempfile

from agent_bridge.skill_registry import SkillRegistry
from agent_bridge.skill_codex import prepare_codex


async def run(binary: str, base_home: Path, work_root: Path):
    from openai_codex_sdk import Codex
    from codex_bridge.probe_context import validate_probe_context
    context = validate_probe_context(base_home=base_home, binary=binary)
    print(json.dumps({'probe_context': context}), flush=True)
    root = Path(tempfile.mkdtemp(prefix='skill-probe-', dir=work_root))
    source = root / 'source'
    for vendor in ('claude', 'codex'):
        path = source / f'.{vendor}-plugin/plugin.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(name='compat-probe', version='1.0.0',
                                       description='Isolated skill probe', skills='./skills/')))
    skill = source / 'skills/probe'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text(
        '---\nname: probe\ndescription: Run the isolated TASK-844 compatibility probe when explicitly requested.\n---\n'
        'Read references/receipt.txt and execute scripts/receipt.py with Python. '
        'Resolve both relative to this SKILL.md. Report both outputs. Do not edit files.\n')
    (skill / 'references').mkdir()
    (skill / 'references/receipt.txt').write_text('TASK844_REFERENCE_OK\n')
    (skill / 'scripts').mkdir()
    (skill / 'scripts/receipt.py').write_text('print("TASK844_HELPER_OK")\n')
    registry = SkillRegistry(root / 'registry')
    registry.install(source, visibility='public', provenance='isolated-fixture@1')
    registry.select('probe', ['compat-probe'], audience='public')
    receipt = prepare_codex(registry, 'probe', binary=binary, base_home=base_home)
    env = dict(os.environ, HOME=receipt['home'], CODEX_HOME=receipt['codex_home'])
    # Existing MCPs are irrelevant to this fixture: strip from the isolated copy
    # only, so no bot memory or application services are contacted by the probe.
    import tomlkit
    config_path = Path(receipt['codex_home']) / 'config.toml'
    config = tomlkit.parse(config_path.read_text())
    for field in ('mcp_servers', 'hooks'):
        config.pop(field, None)
    config_path.write_text(tomlkit.dumps(config))
    codex = Codex({'codex_path_override': binary, 'env': env})
    thread = codex.start_thread({'working_directory': str(root),
                                'skip_git_repo_check': True,
                                'sandbox_mode': 'read-only', 'approval_policy': 'never'})
    streamed = await thread.run_streamed(
        'Use $compat-probe:probe to read its bundled receipt and execute its harmless '
        'Python helper. Return the two receipt strings. Do not modify any files or use MCP tools.')
    output = []
    async for event in streamed.events:
        if hasattr(event, 'model_dump'):
            data = event.model_dump()
            if data.get('type') in {'item.completed', 'turn.failed', 'error'}:
                output.append(data)
                print(json.dumps(data, default=str), flush=True)
    (root / 'probe-results.json').write_text(json.dumps(output, indent=2, default=str))
    successful_commands = [
        event['item'].get('aggregated_output', '')
        for event in output
        if event.get('item', {}).get('type') == 'command_execution'
        and event['item'].get('exit_code') == 0
    ]
    verified = all(marker in '\n'.join(successful_commands)
                   for marker in ('TASK844_REFERENCE_OK', 'TASK844_HELPER_OK'))
    print(json.dumps({'probe_root': str(root), 'installed': receipt['installed'],
                      'receipts_verified': verified}))
    if not verified:
        raise RuntimeError('Plugin probe did not verify both receipts through successful command execution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--base-home', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(run(args.binary, args.base_home, args.work_root), 150))


if __name__ == '__main__':
    main()
