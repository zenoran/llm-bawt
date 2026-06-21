"""Claude OAuth credential for the usage endpoint.

The usage endpoint needs a ``user:profile``-scoped credential, which only a
full subscription ``claude login`` produces (a ``claudeAiOauth`` bundle). The
bridge's *inference* token (env ``CLAUDE_CODE_OAUTH_TOKEN``, a ``setup-token``)
is ``user:inference`` only and 403s on ``/api/oauth/usage`` — see
``docs/usage-endpoint.md``.

TWO MODES (``CLAUDE_USAGE_CREDENTIALS_MODE``):

* ``shared`` (default) — **read-only reuse of an existing login bundle**, e.g.
  the one your interactive Claude Code (TUI) maintains at
  ``~/.claude/.credentials.json``. We NEVER refresh or write it: bundle refresh
  rotates the refresh token and rewrites the file, which would invalidate the
  TUI's copy. Instead we ride whatever access token the TUI last wrote; your
  organic TUI usage keeps it fresh. If it lapses (no TUI use for ~8h) the
  adapter reports ``stale`` until the next TUI use refreshes it. No second
  credential to manage.

* ``owned`` — a DEDICATED bundle this app exclusively refreshes + rewrites.
  Use only if the credential is NOT shared with anything else.

PATH (``CLAUDE_USAGE_CREDENTIALS_PATH``): the bundle file. For ``shared`` reuse
of the host TUI login, mount the host ``~/.claude`` DIRECTORY (read-only) into
the app and point this at ``<mount>/.credentials.json`` — mount the directory,
NOT the single file, or the TUI's atomic rename-on-refresh pins the old inode
and the container reads stale forever (same class of bug as the codex
``auth.json`` EBUSY note). Default path:
``~/.config/llm-bawt/claude-usage-credentials.json``.

The file may be the standard wrapper ``{"claudeAiOauth": {...}}`` or a bare
bundle ``{"accessToken": ..., "refreshToken": ...}`` — both are accepted.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Same OAuth client the Claude CLI / bridge use for subscription refresh.
_OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_REFRESH_BUFFER_MS = 5 * 60 * 1000


def usage_credentials_path() -> Path:
    override = os.getenv("CLAUDE_USAGE_CREDENTIALS_PATH")
    if override:
        return Path(override)
    return Path.home() / ".config" / "llm-bawt" / "claude-usage-credentials.json"


def _mode() -> str:
    return (os.getenv("CLAUDE_USAGE_CREDENTIALS_MODE") or "shared").strip().lower()


@dataclass
class UsageToken:
    """Result of resolving the usage credential.

    ``state`` is one of:
      * ``ok``      — usable access token (``token`` set)
      * ``missing`` — no credential file / bundle configured
      * ``stale``   — credential present but its access token has expired and
                      we won't refresh it (shared mode); the holder (your TUI)
                      will refresh it on next use
    """

    token: str | None
    state: str


def _load() -> tuple[dict, dict | None]:
    """Return (raw_file_json, oauth_bundle). bundle is None if absent."""
    path = usage_credentials_path()
    if not path.exists():
        return {}, None
    try:
        data = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to read usage credential %s: %s", path, e)
        return {}, None
    bundle = data.get("claudeAiOauth") if isinstance(data, dict) else None
    if bundle is None and isinstance(data, dict) and data.get("accessToken"):
        bundle = data
    return (data if isinstance(data, dict) else {}), bundle


def _save(raw: dict, bundle: dict) -> None:
    path = usage_credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if "claudeAiOauth" in raw or not raw:
        out = dict(raw)
        out["claudeAiOauth"] = bundle
    else:
        out = bundle
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(path)


def _expired(expires_at: int | None, *, buffer_ms: int = 0) -> bool:
    if not expires_at:
        return False
    return (int(time.time() * 1000) + buffer_ms) >= int(expires_at)


def _refresh(bundle: dict, raw: dict) -> dict:
    """Refresh + persist the bundle. ONLY called in 'owned' mode."""
    refresh_token = bundle.get("refreshToken")
    if not refresh_token:
        raise RuntimeError("usage credential has no refreshToken")
    scopes = bundle.get("scopes") or []
    resp = httpx.post(
        _OAUTH_TOKEN_URL,
        json={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": _OAUTH_CLIENT_ID,
            "scope": " ".join(scopes),
        },
        headers={"Content-Type": "application/json"},
        timeout=15.0,
    )
    if resp.is_error:
        detail = (resp.text or "").strip().replace("\n", " ")[:300]
        raise RuntimeError(f"usage OAuth refresh failed ({resp.status_code}): {detail}")
    payload = resp.json()
    refreshed = {
        **bundle,
        "accessToken": payload["access_token"],
        "refreshToken": payload.get("refresh_token", refresh_token),
        "expiresAt": int(time.time() * 1000) + int(payload["expires_in"]) * 1000,
        "scopes": payload.get("scope", "").split() if payload.get("scope") else scopes,
    }
    try:
        _save(raw, refreshed)
    except Exception as e:  # noqa: BLE001
        logger.warning("Refreshed usage token but could not persist file: %s", e)
    return refreshed


def load_usage_token() -> UsageToken:
    """Resolve the usage access token without ever breaking a shared login.

    In ``shared`` mode (default) we never refresh/write — we return whatever
    access token the credential's owner (your TUI) last wrote, marking it
    ``stale`` once expired. In ``owned`` mode we refresh + rewrite the file.
    """
    raw, bundle = _load()
    if not bundle:
        return UsageToken(None, "missing")

    expired = _expired(bundle.get("expiresAt"))
    if _mode() == "owned" and _expired(bundle.get("expiresAt"), buffer_ms=_REFRESH_BUFFER_MS):
        try:
            bundle = _refresh(bundle, raw)
            expired = False
            logger.info("Refreshed Claude usage OAuth token (owned mode)")
        except Exception as e:  # noqa: BLE001 — fall back to existing token
            logger.warning("Failed to refresh owned usage token: %s", e)
            expired = _expired(bundle.get("expiresAt"))

    token = bundle.get("accessToken")
    if not token:
        return UsageToken(None, "missing")
    return UsageToken(token, "stale" if expired else "ok")
