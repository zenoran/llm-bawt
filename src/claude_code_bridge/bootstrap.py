"""Bootstrap persistent Claude bridge config inside the container.

This normalizes the on-disk ~/.claude layout every time the bridge starts so
skills and MCP config survive image rebuilds and host config drift.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_SKILLS_PATH = "/home/bridge/dev/agent-skills"
_DEFAULT_BAWTHUB_MCP_URL = "http://app:8001/mcp"
_DEFAULT_PLAYWRIGHT_MCP_URL = "http://playwright-mcp:8931/mcp"


def _write_json_if_changed(path: Path, payload: dict) -> bool:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        try:
            if path.read_text() == rendered:
                return False
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered)
    return True


def _ensure_skills_link(claude_dir: Path, target: Path) -> None:
    skills_link = claude_dir / "skills"
    if target.exists():
        if skills_link.is_symlink():
            try:
                if skills_link.resolve() == target.resolve():
                    return
            except OSError:
                pass
            skills_link.unlink()
        elif skills_link.exists():
            raise RuntimeError(
                f"Refusing to replace non-symlink Claude skills path: {skills_link}"
            )
        skills_link.symlink_to(target)
        logger.info("Claude skills linked: %s -> %s", skills_link, target)
        return
    logger.warning("Claude skills source missing; leaving %s untouched", target)


def _normalize_mcp_settings(settings: dict) -> dict:
    mcp_servers = settings.get("mcpServers")
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
    else:
        mcp_servers = dict(mcp_servers)

    legacy = mcp_servers.pop("llm-bawt-memory", None)
    legacy_url = legacy.get("url") if isinstance(legacy, dict) else None

    bawthub_url = (
        os.getenv("CLAUDE_CODE_BAWTHUB_MCP_URL")
        or legacy_url
        or _DEFAULT_BAWTHUB_MCP_URL
    )
    playwright_url = (
        os.getenv("CLAUDE_CODE_PLAYWRIGHT_MCP_URL") or _DEFAULT_PLAYWRIGHT_MCP_URL
    )

    # "http" is the only URL-based type the Agent SDK/CLI accepts here —
    # "url" is silently dropped (server never registers).
    mcp_servers["bawthub"] = {
        "type": "http",
        "url": bawthub_url,
    }
    mcp_servers.setdefault(
        "playwright",
        {
            "type": "http",
            "url": playwright_url,
        },
    )

    normalized = dict(settings)
    normalized["mcpServers"] = mcp_servers
    return normalized


def bootstrap_claude_home() -> None:
    claude_dir = Path.home() / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)

    skills_target = Path(
        os.getenv("CLAUDE_CODE_SKILLS_PATH") or _DEFAULT_SKILLS_PATH
    )
    _ensure_skills_link(claude_dir, skills_target)

    settings_path = claude_dir / "settings.json"
    settings: dict = {}
    if settings_path.exists():
        try:
            raw = json.loads(settings_path.read_text())
            if isinstance(raw, dict):
                settings = raw
        except Exception as exc:
            logger.warning("Failed to parse %s, rebuilding it: %s", settings_path, exc)

    normalized = _normalize_mcp_settings(settings)
    if _write_json_if_changed(settings_path, normalized):
        logger.info("Claude settings bootstrapped at %s", settings_path)
