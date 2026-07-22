"""Codex (ChatGPT subscription) — STATUS-ONLY provider adapter (TASK-637).

There is no in-app login flow for codex yet (that's TASK-636 phase 2 — today
the credential is minted by ``codex login`` on the host and bind-mounted into
the app + bridges at ``CODEX_AUTH_PATH``). This adapter exists so the health
layer can see the credential *before* codex-routed bots start failing:

* the access token is a JWT with an ``exp`` claim (~days-lived),
* the bridge proxy refreshes it lazily on use — an expired token self-heals on
  the next codex turn *if* the refresh chain is alive, and 401s the turn if not.

So: expired → ``warning`` (may self-heal; tells the operator what to run if it
doesn't), missing/unreadable → ``unconfigured``. No secrets ever leave this
module — only claims-derived facts (email, plan, exp).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .base import (
    HEALTH_OK,
    HEALTH_UNCONFIGURED,
    HEALTH_WARNING,
    STATUS_CONNECTED,
    ProviderAdapter,
    health_block,
)

logger = logging.getLogger(__name__)


def codex_auth_path() -> Path:
    return Path(os.getenv("CODEX_AUTH_PATH") or (Path.home() / ".codex" / "auth.json"))


def _jwt_claims(token: str) -> dict[str, Any]:
    """Best-effort unverified decode of a JWT payload — scheduling facts only,
    never a trust decision (same rationale as the bridge proxy's ``_jwt_exp``)."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return claims if isinstance(claims, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def codex_auth_status() -> dict[str, Any]:
    """Cleartext facts about the codex OAuth bundle (no tokens)."""
    path = codex_auth_path()
    if not path.exists():
        return {"present": False}
    try:
        data = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to read codex auth bundle %s: %s", path, e)
        return {"present": False, "error": str(e)}
    tokens = data.get("tokens") or {}
    access = tokens.get("access_token")
    if not access:
        return {"present": False}
    claims = _jwt_claims(access)
    id_claims = _jwt_claims(tokens.get("id_token") or "")
    auth_claims = id_claims.get("https://api.openai.com/auth") or {}
    exp = claims.get("exp")
    return {
        "present": True,
        # epoch ms, matching the health-block convention.
        "expires_at": int(exp) * 1000 if exp else None,
        "account": id_claims.get("email"),
        "plan": auth_claims.get("chatgpt_plan_type"),
        "last_refresh": data.get("last_refresh"),
    }


class CodexAdapter(ProviderAdapter):
    id = "codex"
    label = "Codex (ChatGPT)"
    auth_methods = ()  # status-only until TASK-636 phase 2 lands a real flow

    def descriptor(self) -> dict[str, Any]:
        desc = super().descriptor()
        status = codex_auth_status()
        if status.get("present"):
            conn = desc["connection"]
            conn["connected"] = True
            conn["status"] = STATUS_CONNECTED
            plan = status.get("plan")
            conn["account"] = status.get("account") or "chatgpt-subscription"
            conn["meta"] = {
                **(conn.get("meta") or {}),
                "credentials_path": str(codex_auth_path()),
                "plan": plan,
                "expires_at": status.get("expires_at"),
                "last_refresh": status.get("last_refresh"),
            }
        return desc

    def health(self) -> dict[str, Any]:
        status = codex_auth_status()
        if not status.get("present"):
            return health_block(
                HEALTH_UNCONFIGURED,
                detail=(
                    "No codex credential — run `codex login` on the host "
                    "(writes ~/.codex/auth.json) to enable codex-routed bots."
                ),
                fix="cli",
            )
        expires_at = status.get("expires_at")
        if expires_at and int(time.time() * 1000) >= expires_at:
            return health_block(
                HEALTH_WARNING,
                detail=(
                    "Access token expired — the bridge refreshes it on the next "
                    "codex turn. If codex bots fail with auth errors, re-run "
                    "`codex login` on the host."
                ),
                expires_at=expires_at,
                fix="cli",
            )
        return health_block(HEALTH_OK, expires_at=expires_at)

    def disconnect(self) -> bool:  # no wizard-owned state to clear
        return self.store.delete(self.id)
