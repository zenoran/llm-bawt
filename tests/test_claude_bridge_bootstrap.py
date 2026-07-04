from __future__ import annotations

import json

from claude_code_bridge.bootstrap import bootstrap_claude_home


def test_bootstrap_creates_skills_symlink_and_settings(tmp_path, monkeypatch):
    home = tmp_path / "home"
    skills = tmp_path / "agent-skills"
    skills.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CODE_SKILLS_PATH", str(skills))
    monkeypatch.delenv("CLAUDE_CODE_BAWTHUB_MCP_URL", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_PLAYWRIGHT_MCP_URL", raising=False)

    bootstrap_claude_home()

    claude_dir = home / ".claude"
    skills_link = claude_dir / "skills"
    assert skills_link.is_symlink()
    assert skills_link.resolve() == skills.resolve()

    settings = json.loads((claude_dir / "settings.json").read_text())
    assert settings["mcpServers"]["bawthub"]["url"] == "http://app:8001/mcp"
    assert settings["mcpServers"]["playwright"]["url"] == "http://playwright-mcp:8931/mcp"


def test_bootstrap_migrates_legacy_mcp_name_and_preserves_other_settings(tmp_path, monkeypatch):
    home = tmp_path / "home"
    skills = tmp_path / "agent-skills"
    skills.mkdir(parents=True)
    claude_dir = home / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "settings.json").write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Bash(*)"]},
                "mcpServers": {
                    "llm-bawt-memory": {"type": "url", "url": "http://172.18.0.1:8001/mcp"},
                    "custom": {"type": "url", "url": "http://example.test/mcp"},
                },
            }
        )
    )

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CODE_SKILLS_PATH", str(skills))
    monkeypatch.delenv("CLAUDE_CODE_BAWTHUB_MCP_URL", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_PLAYWRIGHT_MCP_URL", raising=False)

    bootstrap_claude_home()

    settings = json.loads((claude_dir / "settings.json").read_text())
    assert "llm-bawt-memory" not in settings["mcpServers"]
    assert settings["mcpServers"]["bawthub"]["url"] == "http://172.18.0.1:8001/mcp"
    assert settings["mcpServers"]["custom"]["url"] == "http://example.test/mcp"
    assert settings["permissions"]["allow"] == ["Bash(*)"]
