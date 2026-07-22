"""The app-owned Claude OAuth credential — single source for inference + usage.

TASK-635: there is ONE Claude login for the whole deployment. The app owns the
full-scope ``claudeAiOauth`` bundle (``user:inference`` + ``user:profile`` + …,
minted by the ``claude`` provider adapter's wizard login) and is the SOLE
refresher of its rotate-on-use refresh-token chain. Everything else is a
read-only consumer:

* the ``/v1/usage`` Claude adapter (same process — calls :func:`load_usage_token`),
* the claude-code bridge (reads the bundle file via a read-only mount, falling
  back to ``GET /v1/providers/claude/token`` when the file looks stale — see
  ``claude_code_bridge/_bridge_helpers.py``; it NEVER refreshes).

Refresh is serialized behind a process-wide lock and re-checks freshness after
acquiring it, so concurrent callers (usage fetch, broker endpoint, proactive
loop) can never race the single-use refresh token. A proactive background loop
(:func:`proactive_refresh_loop`, started from the app lifespan) refreshes at
``expiresAt - buffer`` so the access token never lapses even when idle.

TWO MODES (``CLAUDE_USAGE_CREDENTIALS_MODE``):

* ``owned`` — the deployment default: the dedicated bundle this app exclusively
  refreshes + rewrites.
* ``shared`` — legacy read-only reuse of a bundle some OTHER owner (e.g. an
  interactive TUI) maintains. We never write it; it goes stale if the owner is
  idle. Kept for setups without a dedicated login.

PATH: ``CLAUDE_CREDENTIALS_PATH`` (preferred) or the legacy
``CLAUDE_USAGE_CREDENTIALS_PATH`` env; default
``~/.config/llm-bawt/claude-usage-credentials.json``. When sharing across
containers, mount the DIRECTORY (not the single file) — refresh persists via
tmp+rename, and a single-file bind mount pins the old inode.

The file may be the standard wrapper ``{"claudeAiOauth": {...}}`` or a bare
bundle ``{"accessToken": ..., "refreshToken": ...}`` — both are accepted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Same OAuth client the Claude CLI / bridge use for subscription refresh.
_OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_REFRESH_BUFFER_MS = 5 * 60 * 1000
# Proactive loop: check cadence + how early before expiry we refresh. The
# refresh window is deliberately wider than the on-demand buffer so the loop
# (not a request) is what normally performs the refresh.
_PROACTIVE_CHECK_S = 300.0
_PROACTIVE_BUFFER_MS = 20 * 60 * 1000

# Serializes refresh across the usage fetch, the broker endpoint, and the
# proactive loop — the refresh token is single-use, so two concurrent
# refreshes would invalidate each other (the exact race TASK-635 exists to
# prevent).
_REFRESH_LOCK = threading.Lock()

# Refresh-outcome tracking (TASK-637) — the invisible-failure fix. The app is
# the SOLE refresher (TASK-635), so module-level state is authoritative for
# this process. When refresh starts failing while the access token is still
# valid, there is up to ~expiresAt of runway — these fields let the health
# layer surface that warning window instead of burying it in logs.
_last_refresh_ok_at: int | None = None  # epoch ms
_last_refresh_error: str | None = None
_last_refresh_error_at: int | None = None


def _record_refresh_outcome(error: str | None) -> None:
    global _last_refresh_ok_at, _last_refresh_error, _last_refresh_error_at
    now = int(time.time() * 1000)
    if error is None:
        _last_refresh_ok_at = now
        _last_refresh_error = None
        _last_refresh_error_at = None
    else:
        _last_refresh_error = error[:300]
        _last_refresh_error_at = now


def refresh_health() -> dict:
    """Cleartext refresh-chain health for the provider health layer (no tokens)."""
    return {
        "last_refresh_at": _last_refresh_ok_at,
        "last_refresh_error": _last_refresh_error,
        "last_refresh_error_at": _last_refresh_error_at,
    }


def claude_credentials_path() -> Path:
    """The app-owned Claude credential bundle file."""
    override = os.getenv("CLAUDE_CREDENTIALS_PATH") or os.getenv(
        "CLAUDE_USAGE_CREDENTIALS_PATH"
    )
    if override:
        return Path(override)
    return Path.home() / ".config" / "llm-bawt" / "claude-usage-credentials.json"


# Backwards-compatible alias — existing callers/docs use the usage-era name.
usage_credentials_path = claude_credentials_path


def credentials_mode() -> str:
    """Active credential mode: ``shared`` (default) or ``owned``."""
    return (os.getenv("CLAUDE_USAGE_CREDENTIALS_MODE") or "shared").strip().lower()


# Backwards-compatible private alias (internal callers use _mode()).
_mode = credentials_mode


@dataclass
class UsageToken:
    """Result of resolving the app-owned credential.

    ``state`` is one of:
      * ``ok``      — usable access token (``token`` set)
      * ``missing`` — no credential file / bundle configured
      * ``stale``   — credential present but its access token has expired and
                      we couldn't (owned) or won't (shared) refresh it
    """

    token: str | None
    state: str
    expires_at: int | None = None


def _load() -> tuple[dict, dict | None]:
    """Return (raw_file_json, oauth_bundle). bundle is None if absent."""
    path = claude_credentials_path()
    if not path.exists():
        return {}, None
    try:
        data = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to read Claude credential %s: %s", path, e)
        return {}, None
    bundle = data.get("claudeAiOauth") if isinstance(data, dict) else None
    if bundle is None and isinstance(data, dict) and data.get("accessToken"):
        bundle = data
    return (data if isinstance(data, dict) else {}), bundle


def _save(raw: dict, bundle: dict) -> None:
    path = claude_credentials_path()
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


def _refresh_upstream(bundle: dict) -> dict:
    """Exchange the refresh token upstream. Caller must hold _REFRESH_LOCK."""
    refresh_token = bundle.get("refreshToken")
    if not refresh_token:
        raise RuntimeError("Claude credential has no refreshToken")
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
        raise RuntimeError(f"Claude OAuth refresh failed ({resp.status_code}): {detail}")
    payload = resp.json()
    return {
        **bundle,
        "accessToken": payload["access_token"],
        "refreshToken": payload.get("refresh_token", refresh_token),
        "expiresAt": int(time.time() * 1000) + int(payload["expires_in"]) * 1000,
        "scopes": payload.get("scope", "").split() if payload.get("scope") else scopes,
    }


def _refresh_serialized(*, buffer_ms: int, force: bool = False) -> dict | None:
    """Refresh + persist the bundle under the lock (owned mode only).

    Re-loads inside the lock and skips the upstream call if another caller
    already refreshed while we waited. Returns the current bundle (refreshed
    or not), or None if no bundle exists. Raises on upstream refresh failure.
    """
    with _REFRESH_LOCK:
        raw, bundle = _load()
        if not bundle:
            return None
        if not force and not _expired(bundle.get("expiresAt"), buffer_ms=buffer_ms):
            return bundle  # someone else refreshed while we waited
        try:
            refreshed = _refresh_upstream(bundle)
        except Exception as e:  # noqa: BLE001 — record then re-raise for callers
            _record_refresh_outcome(str(e))
            raise
        _record_refresh_outcome(None)
        try:
            _save(raw, refreshed)
        except Exception as e:  # noqa: BLE001
            logger.warning("Refreshed Claude token but could not persist file: %s", e)
        logger.info("Refreshed Claude OAuth token (owned mode)")
        return refreshed


def get_access_token(*, force_refresh: bool = False) -> UsageToken:
    """Resolve the app-owned Claude access token.

    In ``owned`` mode this refreshes (serialized) when expired-or-near-expiry,
    or unconditionally with ``force_refresh`` (e.g. a reader got a 401). In
    ``shared`` mode we never refresh/write — the file's owner does.
    """
    raw, bundle = _load()
    if not bundle:
        return UsageToken(None, "missing")

    expired = _expired(bundle.get("expiresAt"))
    needs = force_refresh or _expired(bundle.get("expiresAt"), buffer_ms=_REFRESH_BUFFER_MS)
    if _mode() == "owned" and needs:
        try:
            bundle = _refresh_serialized(buffer_ms=_REFRESH_BUFFER_MS, force=force_refresh) or bundle
            expired = _expired(bundle.get("expiresAt"))
        except Exception as e:  # noqa: BLE001 — fall back to existing token
            logger.warning("Failed to refresh owned Claude token: %s", e)
            expired = _expired(bundle.get("expiresAt"))

    token = bundle.get("accessToken")
    if not token:
        return UsageToken(None, "missing")
    return UsageToken(token, "stale" if expired else "ok", bundle.get("expiresAt"))


def load_usage_token() -> UsageToken:
    """Legacy name used by the usage adapter — same resolution."""
    return get_access_token()


def bundle_status() -> dict:
    """Cleartext facts about the stored bundle (no tokens) — for honest UI status."""
    _, bundle = _load()
    if not bundle:
        return {"present": False}
    return {
        "present": True,
        "expired": _expired(bundle.get("expiresAt")),
        "expires_at": bundle.get("expiresAt"),
        "scopes": bundle.get("scopes") or [],
        "subscription": bundle.get("subscriptionType"),
        "mode": _mode(),
        **refresh_health(),
    }


async def proactive_refresh_loop() -> None:
    """Keep the owned bundle fresh forever — refresh at expiresAt - buffer.

    Started from the app lifespan. No-op (sleep only) in shared mode or when
    no bundle exists yet. Failures are logged and retried next tick; the
    on-demand path in :func:`get_access_token` remains the backstop.
    """
    logger.info(
        "Claude credential proactive refresh loop started (mode=%s, path=%s)",
        _mode(),
        claude_credentials_path(),
    )
    while True:
        try:
            if _mode() == "owned":
                _, bundle = _load()
                if bundle and _expired(bundle.get("expiresAt"), buffer_ms=_PROACTIVE_BUFFER_MS):
                    await asyncio.to_thread(
                        _refresh_serialized, buffer_ms=_PROACTIVE_BUFFER_MS
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("Proactive Claude token refresh failed (will retry): %s", e)
        await asyncio.sleep(_PROACTIVE_CHECK_S)
