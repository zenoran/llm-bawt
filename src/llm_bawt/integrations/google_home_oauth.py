"""Single-client Google Home linking: opaque tokens, existing browser SSO only.

Not a general OAuth provider. No passwords, agent execution or public admin API.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
from sqlalchemy import Column, Float, MetaData, String, Table, Text, delete, select, update
from sqlalchemy.exc import IntegrityError

from llm_bawt.utils.schema import SchemaBootstrapGuard


class ProbeUnavailable(Exception):
    """Disabled or unconfigured integration / unavailable identity service."""


class ProbeUnauthorized(Exception):
    """Invalid browser session or access token."""


class OAuthError(Exception):
    def __init__(self, error: str):
        self.error = error
        super().__init__(error)


@dataclass(frozen=True)
class ProbeSettings:
    enabled: bool = False
    project_id: str = ""
    client_id: str = ""
    allowed_subject: str = ""
    allowed_email: str = ""
    required_scope: str = "probe.capture"
    public_origin: str = ""
    sso_auth_url: str = ""
    sso_signin_url: str = ""

    def validate(self):
        origin = urlsplit(self.public_origin)
        signin = urlsplit(self.sso_signin_url)
        auth = urlsplit(self.sso_auth_url)
        if (self.enabled is not True
                or not re.fullmatch(r"[a-z][a-z0-9-]{4,62}", self.project_id)
                or not self.client_id or not self.allowed_subject
                or not self.allowed_email or "@" not in self.allowed_email
                or not self.required_scope or len(self.required_scope.split()) != 1
                or origin.scheme != "https" or not origin.hostname
                or origin.path or origin.query or origin.fragment or origin.username or origin.password
                or signin.scheme != "https" or not signin.hostname
                or signin.query or signin.fragment or signin.username or signin.password
                or auth.scheme not in ("http", "https") or not auth.hostname
                or auth.query or auth.fragment or auth.username or auth.password):
            raise ProbeUnavailable()

    @property
    def redirects(self) -> tuple[str, str]:
        return tuple(f"https://{host}/r/{self.project_id}" for host in (
            "oauth-redirect.googleusercontent.com", "oauth-redirect-sandbox.googleusercontent.com",
        ))

    def authorization(self, params: dict[str, str]) -> dict[str, str]:
        self.validate()
        if params.get("client_id") != self.client_id or params.get("redirect_uri") not in self.redirects:
            raise OAuthError("invalid_request")
        if params.get("response_type") != "code":
            raise OAuthError("unsupported_response_type")
        if not params.get("state") or len(params["state"]) > 2048:
            raise OAuthError("invalid_request")
        if params.get("scope", self.required_scope) != self.required_scope:
            raise OAuthError("invalid_scope")
        # Unsupported PKCE must not be silently accepted/downgraded.
        if "code_challenge" in params or "code_challenge_method" in params:
            raise OAuthError("invalid_request")
        return {"redirect_uri": params["redirect_uri"], "state": params["state"]}


_metadata = MetaData()
_links = Table(
    "google_home_probe_links", _metadata,
    Column("link_key", String(64), primary_key=True),
    Column("generation", String(64), nullable=False),
)
_tokens = Table(
    "google_home_probe_tokens", _metadata,
    Column("digest", String(64), primary_key=True),
    Column("link_key", String(64), nullable=False, index=True),
    Column("generation", String(64), nullable=False),
    Column("kind", String(16), nullable=False),
    Column("expires_at", Float, nullable=False),
    Column("data", Text, nullable=False),
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class ProbeOAuthStore:
    """Hash-only token storage; per-link locking serializes refresh and unlink.

    Revocation changes generation and deletes all pending/issued credentials.
    Refresh tokens are stable until unlink, matching Google's linking contract.
    """
    _guard = SchemaBootstrapGuard()

    def __init__(self, engine, settings: ProbeSettings):
        self.engine = engine
        self.settings = settings
        self.link_key = _digest(f"{settings.client_id}\0{settings.allowed_subject}\0{settings.required_scope}")
        self._guard.run(engine, "google-home-oauth-v1", lambda conn: _metadata.create_all(conn))
        with engine.connect() as conn:
            exists = conn.execute(select(_links.c.link_key).where(_links.c.link_key == self.link_key)).first()
        if not exists:
            try:
                with engine.begin() as conn:
                    conn.execute(_links.insert().values(link_key=self.link_key, generation=secrets.token_hex(32)))
            except IntegrityError:
                pass  # A concurrent first request initialized the same link.

    def _lock(self, conn) -> str:
        return conn.execute(select(_links.c.generation).where(
            _links.c.link_key == self.link_key,
        ).with_for_update()).scalar_one()

    def _issue(self, conn, generation: str, kind: str, ttl: int, data: dict | None = None) -> str:
        token = secrets.token_urlsafe(32)
        conn.execute(_tokens.insert().values(
            digest=_digest(token), link_key=self.link_key, generation=generation, kind=kind,
            expires_at=time.time() + ttl if ttl else 0, data=json.dumps(data or {}),
        ))
        return token

    def _find(self, conn, token: str, kind: str, generation: str, consume: bool = False) -> dict:
        if not token or len(token) > 512:
            raise OAuthError("invalid_grant")
        predicate = (
            (_tokens.c.digest == _digest(token)) & (_tokens.c.link_key == self.link_key)
            & (_tokens.c.generation == generation) & (_tokens.c.kind == kind)
            & ((_tokens.c.expires_at == 0) | (_tokens.c.expires_at > time.time()))
        )
        stmt = delete(_tokens).where(predicate).returning(_tokens.c.data) if consume else select(_tokens.c.data).where(predicate)
        data = conn.execute(stmt).scalar_one_or_none()
        if data is None:
            raise OAuthError("invalid_grant")
        return json.loads(data)

    def _prune(self, conn):
        conn.execute(delete(_tokens).where(
            (_tokens.c.link_key == self.link_key) & (_tokens.c.expires_at != 0)
            & (_tokens.c.expires_at <= time.time()),
        ))

    def consent(self, params: dict[str, str]) -> str:
        data = self.settings.authorization(params)
        with self.engine.begin() as conn:
            generation = self._lock(conn)
            self._prune(conn)
            return self._issue(conn, generation, "consent", 600, data)

    def approve(self, consent: str, allow: bool) -> dict[str, str]:
        with self.engine.begin() as conn:
            generation = self._lock(conn)
            data = self._find(conn, consent, "consent", generation, consume=True)
            if allow:
                data["code"] = self._issue(conn, generation, "code", 300, {"redirect_uri": data["redirect_uri"]})
            else:
                data["error"] = "access_denied"
            return data

    def exchange(self, params: dict[str, str]) -> dict:
        grant = params.get("grant_type")
        if grant not in ("authorization_code", "refresh_token"):
            raise OAuthError("unsupported_grant_type")
        with self.engine.begin() as conn:
            generation = self._lock(conn)
            self._prune(conn)
            if grant == "authorization_code":
                data = self._find(conn, params.get("code", ""), "code", generation, consume=True)
                if params.get("redirect_uri") != data["redirect_uri"]:
                    raise OAuthError("invalid_grant")
                refresh = self._issue(conn, generation, "refresh", 0)
            else:
                self._find(conn, params.get("refresh_token", ""), "refresh", generation)
                refresh = None
            result = {"token_type": "Bearer", "access_token": self._issue(conn, generation, "access", 3600),
                      "expires_in": 3600, "scope": self.settings.required_scope}
            if refresh:
                result["refresh_token"] = refresh
            return result

    def authenticate(self, authorization: str) -> str:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token or any(c.isspace() for c in token):
            raise ProbeUnauthorized()
        try:
            with self.engine.begin() as conn:
                generation = self._lock(conn)
                self._find(conn, token, "access", generation)
        except OAuthError as exc:
            raise ProbeUnauthorized() from exc
        return self.settings.allowed_subject

    def revoke(self, authorization: str):
        # Recheck the bearer under the same lock as revocation. An old DISCONNECT
        # must not revoke a newly linked generation after a concurrent unlink.
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer":
            raise ProbeUnauthorized()
        with self.engine.begin() as conn:
            generation = self._lock(conn)
            try:
                self._find(conn, token, "access", generation)
            except OAuthError as exc:
                raise ProbeUnauthorized() from exc
            conn.execute(update(_links).where(_links.c.link_key == self.link_key).values(generation=secrets.token_hex(32)))
            conn.execute(delete(_tokens).where(_tokens.c.link_key == self.link_key))


class ProbeAuthenticator:
    def __init__(self, settings: ProbeSettings, client_secret: str, store: ProbeOAuthStore):
        self.settings = settings
        self.client_secret = client_secret
        self.store = store

    def authenticate_client(self, params: dict[str, str]):
        if (params.get("client_id") != self.settings.client_id
                or not hmac.compare_digest(params.get("client_secret", "").encode(), self.client_secret.encode())):
            raise OAuthError("invalid_client")

    async def authenticate(self, authorization: str, http=None) -> str:
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(self.store.authenticate, authorization)

    async def browser_identity(self, cookie: str, http: httpx.AsyncClient) -> str:
        # Only cookies are sent to the fixed operator-configured SSO endpoint.
        # Browser-supplied email, Authorization and X-Auth-* headers are ignored.
        if not cookie or len(cookie) > 16384:
            raise ProbeUnauthorized()
        try:
            response = await http.get(self.settings.sso_auth_url, headers={"Cookie": cookie},
                                      timeout=10, follow_redirects=False)
        except httpx.HTTPError as exc:
            raise ProbeUnavailable() from exc
        if response.status_code in (401, 403):
            raise ProbeUnauthorized()
        if response.status_code not in (200, 202):
            raise ProbeUnavailable()
        email = response.headers.get("x-auth-request-email", "")
        if not email or email.casefold() != self.settings.allowed_email.casefold():
            raise ProbeUnauthorized()
        return self.settings.allowed_subject
