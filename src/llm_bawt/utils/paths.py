import os
from pathlib import Path


def _find_repo_root(start: Path) -> Path | None:
    """Walk up from start to find a repo root (pyproject.toml or .git)."""
    start = start.resolve()
    for current in (start, *start.parents):
        if (current / "pyproject.toml").is_file() or (current / ".git").exists():
            return current
    return None


def resolve_log_dir() -> Path:
    """Resolve log directory for debug files.

    Priority:
    1) LLM_BAWT_LOG_DIR env var (explicit override)
    2) Repo root discovered from CWD
    3) Repo root discovered from this file location
    4) CWD fallback
    """
    env_dir = os.environ.get("LLM_BAWT_LOG_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    cwd_root = _find_repo_root(Path.cwd())
    if cwd_root:
        return cwd_root / ".logs"

    file_root = _find_repo_root(Path(__file__))
    if file_root:
        return file_root / ".logs"

    return Path.cwd() / ".logs"
