"""Thin async wrapper around ``codex_app_server.AsyncCodex``.

The transport encapsulates the lazy construction, restart-on-crash, and
auth.json sanity-check logic so ``bridge.py`` can stay focused on Redis
command dispatch and event translation.

A single ``AsyncCodex`` per bridge container is shared across all sessions
(per architectural decision #2). Concurrency between sessions is bounded by
``[agents] max_threads`` in ``~/.codex/config.toml``; per-session ordering
is enforced by ``SessionQueue.lock`` in ``bridge.py``.

If the codex Rust subprocess dies or the OAuth tokens expire mid-turn, the
supervisor in ``bridge.py`` detects it, calls ``CodexTransport.shutdown()``
to tear down, and the next ``ensure_codex()`` call rebuilds with fresh
auth.json on disk — so ``codex login`` on the host self-heals the bridge
without a docker restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codex_app_server import AsyncCodex

logger = logging.getLogger(__name__)


CODEX_AUTH_PATH_ENV = "CODEX_AUTH_PATH"
DEFAULT_AUTH_PATH = Path("/home/bridge/.codex/auth.json")
SCRUB_ENV_VARS = ("CODEX_API_KEY", "OPENAI_API_KEY")


def auth_path() -> Path:
    raw = os.getenv(CODEX_AUTH_PATH_ENV)
    return Path(raw).expanduser() if raw else DEFAULT_AUTH_PATH


def scrub_api_key_env() -> list[str]:
    """Remove API key env vars so AsyncCodex falls back to OAuth.

    Returns the list of env var names that were actually set (so callers can
    log a warning — these should never be set when running OAuth-only).
    """
    scrubbed: list[str] = []
    for name in SCRUB_ENV_VARS:
        if os.environ.pop(name, None) is not None:
            scrubbed.append(name)
    return scrubbed


def validate_auth_json(path: Path | None = None) -> dict:
    """Read + sanity-check the OAuth auth.json bundle.

    Raises RuntimeError with a clear message on any structural problem so
    the entrypoint can hard-fail at startup. Returns the parsed dict on
    success.
    """
    p = path or auth_path()
    if not p.exists():
        raise RuntimeError(
            f"Codex auth.json missing at {p}. Run 'codex login' on the host "
            f"to provision OAuth credentials, then restart the bridge."
        )
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Codex auth.json at {p} is not valid JSON: {e}") from e

    tokens = data.get("tokens") or {}
    auth_mode = (tokens.get("auth_mode") or data.get("auth_mode") or "").lower()
    if auth_mode and auth_mode != "chatgpt":
        raise RuntimeError(
            f"Codex auth.json auth_mode is {auth_mode!r}; only 'chatgpt' "
            f"OAuth is supported. Re-run 'codex login' with ChatGPT mode."
        )

    refresh_token = tokens.get("refresh_token") or data.get("refresh_token")
    if not refresh_token:
        raise RuntimeError(
            f"Codex auth.json at {p} has no refresh_token. Re-run 'codex login' "
            f"to refresh the OAuth bundle."
        )
    return data


class CodexTransport:
    """Lazy + restartable wrapper around AsyncCodex.

    Use ``ensure_codex()`` to obtain a live instance; ``shutdown()`` to tear
    it down (e.g. after a process-death detection). The next ensure call
    rebuilds and re-reads auth.json off disk.
    """

    def __init__(self, *, codex_bin: str | None = None) -> None:
        self._codex_bin = codex_bin
        self._codex: AsyncCodex | None = None
        self._lock = asyncio.Lock()

    @property
    def codex(self) -> AsyncCodex | None:
        return self._codex

    async def ensure_codex(self) -> AsyncCodex:
        """Return the live AsyncCodex, lazy-constructing on first use."""
        if self._codex is not None:
            return self._codex
        async with self._lock:
            if self._codex is not None:
                return self._codex
            # Validate auth.json on every (re)build so the failure surfaces
            # before we hand a doomed AsyncCodex back to the caller.
            validate_auth_json()
            from codex_app_server import AppServerConfig, AsyncCodex

            cfg_kwargs: dict = {}
            if self._codex_bin:
                cfg_kwargs["codex_bin"] = self._codex_bin
            cfg = AppServerConfig(**cfg_kwargs) if cfg_kwargs else AppServerConfig()

            codex = AsyncCodex(cfg)
            await codex.__aenter__()
            self._codex = codex
            logger.info("AsyncCodex started (codex_bin=%s)", self._codex_bin or "<bundled>")
            return codex

    async def shutdown(self) -> None:
        """Tear down the AsyncCodex; next ensure rebuilds."""
        codex = self._codex
        self._codex = None
        if codex is None:
            return
        try:
            await codex.__aexit__(None, None, None)
        except Exception:
            logger.warning("AsyncCodex shutdown raised", exc_info=True)
        else:
            logger.info("AsyncCodex shut down")
