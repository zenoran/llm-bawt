"""GitHub provider adapter — device-flow OAuth + GitHub App installation tokens.

Auth model (locked with Nick): a single ``bawthub`` GitHub App, authorized by
each tenant user via **device flow** (no callback URL needed). The user approves
once at ``github.com/login/device``; we then read their App installations, let
them pick repos, and mint **short-lived installation tokens** on demand for git
over HTTPS. No static secret ever lands on the tenant.

Required env (the App Nick registers once):
  * ``GITHUB_APP_CLIENT_ID``   — OAuth client id (device flow)
  * ``GITHUB_APP_ID``          — numeric App id (JWT ``iss``)
  * ``GITHUB_APP_PRIVATE_KEY`` — PEM contents  (or ``GITHUB_APP_PRIVATE_KEY_PATH``)

Network calls are sync (httpx) so the adapter stays testable without an event
loop; the async router invokes them via ``run_in_threadpool``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from .base import (
    AUTH_DEVICE_OAUTH,
    STATUS_CONNECTED,
    ConnectionRecord,
    DeviceFlowStart,
    DevicePollResult,
    ProviderAdapter,
    ValidateResult,
)

logger = logging.getLogger(__name__)

_DEVICE_CODE_URL = "https://github.com/login/device/code"
_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
_API = "https://api.github.com"
_DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
# Scope the user token needs to enumerate installations / repos.
_SCOPE = "read:user"
_UA = "llm-bawt-provider-connect"
_TIMEOUT = 15.0


class GitHubConfigError(RuntimeError):
    """Raised when the GitHub App env is not configured."""


def _client_id() -> str:
    cid = os.getenv("GITHUB_APP_CLIENT_ID")
    if not cid:
        raise GitHubConfigError(
            "GITHUB_APP_CLIENT_ID not set — register the bawthub GitHub App and "
            "provide its client id."
        )
    return cid


def _app_id() -> str:
    aid = os.getenv("GITHUB_APP_ID")
    if not aid:
        raise GitHubConfigError("GITHUB_APP_ID not set (numeric App id for JWT).")
    return aid


def _private_key() -> str:
    pem = os.getenv("GITHUB_APP_PRIVATE_KEY")
    if pem:
        return pem.replace("\\n", "\n")
    path = os.getenv("GITHUB_APP_PRIVATE_KEY_PATH")
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read()
    raise GitHubConfigError(
        "GITHUB_APP_PRIVATE_KEY (or GITHUB_APP_PRIVATE_KEY_PATH) not set — needed "
        "to mint installation tokens."
    )


def _headers(token: str | None = None, *, bearer: bool = False) -> dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "User-Agent": _UA}
    if token:
        h["Authorization"] = f"{'Bearer' if bearer else 'token'} {token}"
    return h


def _app_jwt() -> str:
    """Signed RS256 JWT for App-level API calls (short-lived, <=10min)."""
    import jwt  # lazy — pyjwt[crypto]

    now = int(time.time())
    payload = {"iat": now - 60, "exp": now + 9 * 60, "iss": _app_id()}
    return jwt.encode(payload, _private_key(), algorithm="RS256")


class GitHubAdapter(ProviderAdapter):
    id = "github"
    label = "GitHub"
    auth_methods = (AUTH_DEVICE_OAUTH,)

    @staticmethod
    def is_configured() -> bool:
        try:
            _client_id()
            return True
        except GitHubConfigError:
            return False

    # --- device flow --------------------------------------------------------
    def start_device_flow(self) -> DeviceFlowStart:
        resp = httpx.post(
            _DEVICE_CODE_URL,
            data={"client_id": _client_id(), "scope": _SCOPE},
            headers={"Accept": "application/json", "User-Agent": _UA},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        d = resp.json()
        return DeviceFlowStart(
            user_code=d["user_code"],
            verification_uri=d.get("verification_uri", "https://github.com/login/device"),
            device_code=d["device_code"],
            interval=int(d.get("interval", 5)),
            expires_in=int(d.get("expires_in", 900)),
        )

    def poll_device_flow(self, device_code: str) -> DevicePollResult:
        resp = httpx.post(
            _ACCESS_TOKEN_URL,
            data={
                "client_id": _client_id(),
                "device_code": device_code,
                "grant_type": _DEVICE_GRANT,
            },
            headers={"Accept": "application/json", "User-Agent": _UA},
            timeout=_TIMEOUT,
        )
        d = resp.json() if resp.content else {}
        err = d.get("error")
        if err == "authorization_pending":
            return DevicePollResult(status="pending")
        if err == "slow_down":
            return DevicePollResult(status="slow_down")
        if err in ("expired_token", "incorrect_device_code"):
            return DevicePollResult(status="expired", detail=err)
        if err == "access_denied":
            return DevicePollResult(status="denied", detail=err)
        if err:
            return DevicePollResult(status="error", detail=str(d.get("error_description") or err))

        token = d.get("access_token")
        if not token:
            return DevicePollResult(status="error", detail="no access_token in response")

        record = self._build_record(token)
        self.store.save(record)
        return DevicePollResult(status="authorized", record=record)

    # --- record assembly ----------------------------------------------------
    def _build_record(self, user_token: str) -> ConnectionRecord:
        account = None
        installations: list[dict[str, Any]] = []
        try:
            u = httpx.get(f"{_API}/user", headers=_headers(user_token), timeout=_TIMEOUT)
            if u.is_success:
                account = u.json().get("login")
            inst = httpx.get(
                f"{_API}/user/installations",
                headers=_headers(user_token),
                timeout=_TIMEOUT,
            )
            if inst.is_success:
                installations = inst.json().get("installations", [])
        except Exception as e:  # noqa: BLE001 — connection still recorded
            logger.warning("GitHub post-auth enrichment failed: %s", e)

        meta: dict[str, Any] = {
            "installations": [
                {
                    "id": i.get("id"),
                    "account": (i.get("account") or {}).get("login"),
                }
                for i in installations
            ],
        }
        # Auto-select when exactly one installation exists (common case).
        if len(installations) == 1:
            meta["installation_id"] = installations[0].get("id")

        return ConnectionRecord(
            provider=self.id,
            status=STATUS_CONNECTED,
            auth_method=AUTH_DEVICE_OAUTH,
            account=account,
            meta=meta,
            secret={"user_token": user_token},
        )

    # --- installation token (for git credential helper) ---------------------
    def mint_installation_token(self, record: ConnectionRecord | None = None) -> str:
        """Mint a fresh, short-lived installation token. Never persisted."""
        rec = record or self.store.load(self.id)
        if rec is None or rec.status != STATUS_CONNECTED:
            raise RuntimeError("GitHub is not connected")
        installation_id = rec.meta.get("installation_id")
        if not installation_id:
            raise RuntimeError(
                "No GitHub App installation selected — pick repos/installation first"
            )
        resp = httpx.post(
            f"{_API}/app/installations/{installation_id}/access_tokens",
            headers=_headers(_app_jwt(), bearer=True),
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["token"]

    def list_installations(self) -> list[dict[str, Any]]:
        rec = self.store.load(self.id)
        if rec is None:
            return []
        return rec.meta.get("installations", [])

    def select_installation(self, installation_id: int, repos: list[str] | None = None) -> ConnectionRecord:
        rec = self.store.load(self.id)
        if rec is None:
            raise RuntimeError("GitHub is not connected")
        rec.meta["installation_id"] = installation_id
        if repos is not None:
            rec.meta["repos"] = repos
        self.store.save(rec)
        return rec

    def validate(self, record: ConnectionRecord) -> ValidateResult:
        token = record.secret.get("user_token")
        if not token:
            return ValidateResult(ok=False, detail="no stored token")
        try:
            u = httpx.get(f"{_API}/user", headers=_headers(token), timeout=_TIMEOUT)
            if u.is_success:
                return ValidateResult(ok=True, account=u.json().get("login"))
            return ValidateResult(ok=False, detail=f"github /user -> {u.status_code}")
        except Exception as e:  # noqa: BLE001
            return ValidateResult(ok=False, detail=str(e))
