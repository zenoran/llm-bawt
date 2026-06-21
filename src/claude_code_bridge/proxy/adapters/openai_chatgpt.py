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
import base64
import hashlib
import json
import logging
import os
import time
import uuid
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

# Responses API base. ChatGPT *subscription* OAuth tokens do NOT carry the
# platform ``api.responses.write`` scope, so they 401 against
# ``api.openai.com/v1/responses``. They DO authenticate against the ChatGPT
# backend's codex endpoint (the same surface ``codex exec`` uses), which the
# ``openai`` client reaches by appending ``/responses`` to this base.
# Verified empirically against a live ``codex login`` bundle (TASK-270).
# ``OPENAI_BASE_URL`` overrides for operators who proxy elsewhere.
DEFAULT_API_BASE = "https://chatgpt.com/backend-api/codex"
API_BASE_ENV = "OPENAI_BASE_URL"

# Params the ChatGPT codex backend rejects with HTTP 400 "Unsupported
# parameter" even though the standard Responses API accepts them. The Claude
# Agent SDK always sends ``max_tokens`` (→ ``max_output_tokens``) and may send
# ``temperature``; both must be dropped for this upstream. (Verified live.)
_UNSUPPORTED_PARAMS = ("temperature", "top_p", "max_output_tokens")

# The codex backend hard-requires a non-empty ``instructions`` field and
# ``stream: true``. The Claude SDK populates ``instructions`` from its system
# prompt in practice; this fallback covers the rare system-less request.
_FALLBACK_INSTRUCTIONS = "You are a helpful coding assistant."

# Reasoning effort for gpt-5.x. The codex backend defaults to effort "none"
# when ``reasoning`` is omitted, which makes a reasoning model behave badly —
# no chain of thought at all. The codex CLI itself defaults gpt-5.4 to
# "high" (see ~/.codex/config.toml `model_reasoning_effort`); match that.
# Override per-deploy with OPENAI_CHATGPT_REASONING_EFFORT. Valid backend
# values for GPT-5.4: none | low | medium | high | xhigh.
DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORT_ENV = "OPENAI_CHATGPT_REASONING_EFFORT"
_VALID_EFFORT = {"none", "low", "medium", "high", "xhigh"}


def _prompt_cache_key(responses_body: dict) -> str:
    """Derive a stable per-conversation cache key from the opening prefix.

    The Claude SDK doesn't pass a conversation/thread id through
    ``/v1/messages``, so for the ChatGPT codex backend we approximate codex
    CLI's stable thread key by hashing the invariant opening: instructions plus
    the first user content item. Later history turns are intentionally ignored
    so the key remains stable across the conversation.
    """
    instructions = responses_body.get("instructions") or ""
    first_user_json = ""
    for item in responses_body.get("input") or []:
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        first_user_json = json.dumps(
            item.get("content") or [], sort_keys=True, separators=(",", ":")
        )
        break
    seed = f"{instructions}\n\n{first_user_json}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()


def _jwt_exp(token: str) -> float | None:
    """Best-effort decode of a JWT's ``exp`` claim (seconds since epoch).

    Returns None if the token isn't a decodable JWT. We only read the
    unverified payload to schedule refresh — never to make a trust
    decision, so skipping signature verification is fine here.
    """
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = claims.get("exp")
        return float(exp) if exp is not None else None
    except Exception:  # noqa: BLE001 — any malformed token → "unknown expiry"
        return None

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
        self._account_id: str | None = None
        # Stable per-process session id, mirroring codex CLI telemetry. Not
        # security-relevant; the chatgpt-account-id header is what authorizes.
        self._session_id = uuid.uuid4().hex
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
        text = json.dumps(bundle, indent=2)
        # Prefer tmp + atomic rename so a crash mid-write doesn't brick
        # auth.json. BUT auth.json is typically a *bind-mounted single file*
        # in the bridge container, and rename() over a bind-mount target
        # fails with EBUSY ([Errno 16] Device or resource busy). Fall back to
        # an in-place rewrite, which keeps the bind-mount's shared inode so
        # the host's ~/.codex/auth.json (and the codex CLI) see the update.
        tmp = self.auth_path.with_suffix(self.auth_path.suffix + ".tmp")
        try:
            tmp.write_text(text)
            os.replace(tmp, self.auth_path)
        except OSError:
            # Clean up the tmp file, then rewrite the bind-mounted target
            # in place. Not torn-write-atomic, but the only option for a
            # bind-mounted file and the write is a single small payload.
            try:
                tmp.unlink()
            except OSError:
                pass
            self.auth_path.write_text(text)
        # auth.json owns secrets; keep 600 perms even if the codex CLI does
        # the same.
        try:
            os.chmod(self.auth_path, 0o600)
        except OSError:
            pass

    # ── refresh ───────────────────────────────────────────────────────────
    def _expires_soon(self, bundle: dict) -> bool:
        """True when the access_token needs refreshing.

        Source of truth is the access_token JWT's own ``exp`` claim —
        ChatGPT subscription tokens are long-lived (days), so the old
        ``last_refresh`` + fixed-TTL heuristic forced a needless refresh on
        every cold start (which rotates the refresh_token — risky if the
        write-back ever fails). Only fall back to the heuristic when the
        token can't be decoded.
        """
        token = (bundle.get("tokens") or {}).get("access_token") or ""
        exp = _jwt_exp(token)
        if exp is not None:
            return (exp - time.time()) < REFRESH_SAFETY_SECONDS

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
            # Always re-read from disk: the file is tiny and local, and the
            # codex CLI (or a fresh `codex login`) on the host may have
            # rotated the tokens since we last looked. Cheaper than serving a
            # stale token and eating a 401 round-trip.
            bundle = self._load_bundle()
            if self._expires_soon(bundle):
                bundle = await self._refresh(bundle)
            self._cached_bundle = bundle

        tokens = bundle.get("tokens") or {}
        access_token = tokens.get("access_token")
        if not access_token:
            raise RuntimeError(
                f"ChatGPT OAuth bundle at {self.auth_path} has no access_token "
                f"after refresh."
            )
        # The codex backend authorizes per ChatGPT account, not just the
        # bearer — the account-id header is required (see extra_headers()).
        self._account_id = tokens.get("account_id")
        base_url = os.getenv(API_BASE_ENV) or DEFAULT_API_BASE
        return access_token, base_url

    # ── upstream quirks (ChatGPT codex backend) ───────────────────────────
    def extra_headers(self) -> dict[str, str]:
        """Headers the ChatGPT codex backend requires/expects.

        ``chatgpt-account-id`` is mandatory — without it the backend 401s
        even with a valid bearer. The others mirror the codex CLI so we look
        like a known-good client.
        """
        headers = {
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
            "session_id": self._session_id,
        }
        if self._account_id:
            headers["chatgpt-account-id"] = self._account_id
        return headers

    def prepare_request(self, responses_body: dict) -> dict:
        """Adapt the translated Responses body to the codex backend's quirks.

        - strip params it rejects with 400 ``temperature``/``max_output_tokens``
        - force ``stream``/``store`` (it requires stream=true, store=false)
        - guarantee a non-empty ``instructions`` (it requires the field)
        """
        for key in _UNSUPPORTED_PARAMS:
            responses_body.pop(key, None)
        responses_body["stream"] = True
        responses_body["store"] = False
        if not responses_body.get("instructions"):
            responses_body["instructions"] = _FALLBACK_INSTRUCTIONS
        responses_body["prompt_cache_key"] = _prompt_cache_key(responses_body)
        # Give the reasoning model an actual reasoning budget. Without this
        # the backend runs at effort "none" and behaves like a much weaker
        # model. translate.py may already have set `reasoning` from an
        # inbound Anthropic `thinking` block — only default when it didn't.
        if "reasoning" not in responses_body:
            effort = (os.getenv(REASONING_EFFORT_ENV) or "").strip().lower()
            if effort not in _VALID_EFFORT:
                effort = DEFAULT_REASONING_EFFORT
            responses_body["reasoning"] = {"effort": effort}
        # Request human-readable reasoning summaries so the backend streams
        # response.reasoning_summary_text.delta events (→ Anthropic thinking
        # blocks in the UI). Without `summary` the backend returns only the
        # opaque encrypted reasoning item — a signature but no visible text,
        # exactly the "reasoning is silent" symptom. Matches codex CLI's
        # `model_reasoning_summary = auto`.
        responses_body["reasoning"].setdefault("summary", "auto")
        return responses_body
