"""OpenAI Responses API adapter authenticated by the ChatGPT subscription OAuth bundle.

The codex CLI's ``codex login`` does the browser-based PKCE dance and writes
the OAuth bundle to ``~/.codex/auth.json``. We consume that artifact, refresh
the access_token ourselves against ``auth.openai.com/oauth/token`` when it
ages out, and persist the refreshed bundle back so the codex CLI stays in
lockstep.

OAuth constants (CLIENT_ID, refresh URL, env override) come from
``openai/codex@main:codex-rs/login/src/auth/manager.rs`` (Apache-2.0).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

import httpx

from .base import ProviderAdapter

logger = logging.getLogger(__name__)


# ── Constants lifted from codex-rs/login/src/auth/manager.rs ──────────────
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEFAULT_REFRESH_URL = "https://auth.openai.com/oauth/token"
REFRESH_URL_ENV = "CODEX_REFRESH_TOKEN_URL_OVERRIDE"

# Responses API base. ChatGPT subscription OAuth tokens authenticate against
# the standard public API. ``OPENAI_BASE_URL`` matches the codex CLI's env
# var name so an operator that points codex elsewhere also points the proxy.
DEFAULT_API_BASE = "https://api.openai.com/v1"
API_BASE_ENV = "OPENAI_BASE_URL"

# OAuth bundle location. ``CODEX_AUTH_PATH`` matches the existing codex
# bridge's env var so a single override moves both.
DEFAULT_AUTH_PATH = Path(
    os.getenv("CODEX_AUTH_PATH") or str(Path.home() / ".codex" / "auth.json")
)

# Refresh window — codex CLI's tokens have a 1-hour TTL; we refresh 5 min
# early to avoid borderline races on long-running tool turns.
REFRESH_SAFETY_SECONDS = 300
TOKEN_TTL_SECONDS = 3600


class OpenAIChatGPTAdapter(ProviderAdapter):
    """ChatGPT-subscription-authenticated OpenAI Responses API client.

    Tokens are read from ``~/.codex/auth.json`` (the artifact ``codex login``
    produces), refreshed in-place when stale, and persisted back so the
    codex CLI keeps working.
    """

    name: ClassVar[str] = "openai_chatgpt"

    def __init__(self, auth_path: Path | None = None) -> None:
        self.auth_path = auth_path or DEFAULT_AUTH_PATH
        self._cached_bundle: dict | None = None
        # Serializes refresh attempts so a burst of concurrent requests
        # doesn't trigger N parallel refreshes.
        self._refresh_lock = asyncio.Lock()

    # ── bundle I/O ────────────────────────────────────────────────────────
    def _load_bundle(self) -> dict:
        if not self.auth_path.exists():
            raise RuntimeError(
                f"ChatGPT OAuth bundle missing at {self.auth_path}. "
                f"Run `codex login` on the host to provision tokens."
            )
        try:
            return json.loads(self.auth_path.read_text())
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"ChatGPT OAuth bundle at {self.auth_path} is not valid JSON: {e}. "
                f"Re-run `codex login`."
            ) from e

    def _save_bundle(self, bundle: dict) -> None:
        # tmp + atomic rename so a crash mid-write doesn't brick auth.json.
        tmp = self.auth_path.with_suffix(self.auth_path.suffix + ".tmp")
        tmp.write_text(json.dumps(bundle, indent=2))
        tmp.replace(self.auth_path)
        # auth.json owns secrets; keep 600 perms even if the codex CLI does
        # the same.
        try:
            os.chmod(self.auth_path, 0o600)
        except OSError:
            pass

    # ── refresh ───────────────────────────────────────────────────────────
    def _expires_soon(self, bundle: dict) -> bool:
        last_refresh = bundle.get("last_refresh")
        if not last_refresh:
            return True
        try:
            issued = datetime.fromisoformat(
                last_refresh.replace("Z", "+00:00")
            ).timestamp()
        except (TypeError, ValueError):
            return True
        age = time.time() - issued
        return age > (TOKEN_TTL_SECONDS - REFRESH_SAFETY_SECONDS)

    async def _refresh(self, bundle: dict) -> dict:
        refresh_token = (bundle.get("tokens") or {}).get("refresh_token")
        if not refresh_token:
            raise RuntimeError(
                f"ChatGPT OAuth bundle at {self.auth_path} has no refresh_token. "
                f"Re-run `codex login`."
            )
        url = os.getenv(REFRESH_URL_ENV) or DEFAULT_REFRESH_URL
        body = {
            "client_id": CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=body)
        if resp.status_code != 200:
            raise RuntimeError(
                f"ChatGPT OAuth refresh failed ({resp.status_code}): {resp.text[:300]}. "
                f"Re-run `codex login`."
            )
        payload = resp.json()
        tokens = bundle.setdefault("tokens", {})
        for key in ("id_token", "access_token", "refresh_token"):
            val = payload.get(key)
            if val:
                tokens[key] = val
        bundle["last_refresh"] = datetime.now(timezone.utc).isoformat()
        self._save_bundle(bundle)
        logger.info("ChatGPT OAuth bundle refreshed (path=%s)", self.auth_path)
        return bundle

    # ── ProviderAdapter ───────────────────────────────────────────────────
    async def authorize(self) -> tuple[str, str]:
        async with self._refresh_lock:
            bundle = self._cached_bundle or self._load_bundle()
            if self._expires_soon(bundle):
                bundle = await self._refresh(bundle)
            else:
                # Re-read the file periodically so a fresh `codex login`
                # on the host is picked up without a daemon restart.
                self._cached_bundle = bundle
            self._cached_bundle = bundle

        access_token = (bundle.get("tokens") or {}).get("access_token")
        if not access_token:
            raise RuntimeError(
                f"ChatGPT OAuth bundle at {self.auth_path} has no access_token "
                f"after refresh."
            )
        base_url = os.getenv(API_BASE_ENV) or DEFAULT_API_BASE
        return access_token, base_url
