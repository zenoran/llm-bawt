"""Codex SDK lifecycle: auth validation + lazy ``Codex`` construction.

The real ``openai_codex_sdk.Codex`` is a thin sync handle around the bundled
``codex`` Rust binary; each turn spawns a fresh subprocess, so there is no
long-lived app-server to supervise. We hold a single ``Codex`` per bridge
container (concurrency between sessions is bounded by ``[agents]
max_threads`` in ``~/.codex/config.toml``; per-session ordering by
``SessionQueue.lock`` in ``bridge.py``).

If the OAuth bundle is invalid at startup we hard-fail. Mid-run auth
failures surface as ``CodexAuthError`` from ``thread.run_streamed`` and the
bridge translates them to an ERROR event for the user.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai_codex_sdk import Codex

logger = logging.getLogger(__name__)


CODEX_AUTH_PATH_ENV = "CODEX_AUTH_PATH"
DEFAULT_AUTH_PATH = Path("/home/bridge/.codex/auth.json")
SCRUB_ENV_VARS = ("CODEX_API_KEY", "OPENAI_API_KEY")


def auth_path() -> Path:
    raw = os.getenv(CODEX_AUTH_PATH_ENV)
    return Path(raw).expanduser() if raw else DEFAULT_AUTH_PATH


def scrub_api_key_env() -> list[str]:
    """Remove API key env vars so the codex binary falls back to OAuth.

    Returns the list of env var names that were actually set.
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
    """Lazy holder for a shared ``Codex`` instance.

    The SDK's ``Codex`` is sync-constructed and spawns a fresh ``codex exec``
    subprocess per turn, so there's no long-lived server to recycle. The
    only state we hold is the SDK handle itself, lazily built on first use
    (and rebuilt only if explicitly torn down — useful when a fresh
    ``codex login`` rewrites ``auth.json``).
    """

    def __init__(self, *, codex_bin: str | None = None) -> None:
        self._codex_bin = codex_bin
        self._codex: "Codex | None" = None

    @property
    def codex(self) -> "Codex | None":
        return self._codex

    def ensure_codex(self) -> "Codex":
        """Return the live ``Codex``, lazy-constructing on first use."""
        if self._codex is not None:
            return self._codex
        # Validate auth.json on every (re)build so the failure surfaces
        # before we hand a doomed Codex back to the caller.
        validate_auth_json()
        from openai_codex_sdk import Codex

        opts: dict | None = (
            {"codex_path_override": self._codex_bin} if self._codex_bin else None
        )
        self._codex = Codex(opts)
        logger.info("Codex SDK ready (codex_bin=%s)", self._codex_bin or "<bundled>")
        return self._codex

    def reset(self) -> None:
        """Drop the cached ``Codex`` so the next ensure rebuilds.

        Used after auth-recovery so a fresh ``codex login`` is picked up
        without restarting the container. There is no subprocess to tear
        down — the next ``run_streamed`` call will spawn a fresh codex
        binary that re-reads ``auth.json``.
        """
        self._codex = None
        logger.info("Codex SDK handle reset; next request will rebuild")
