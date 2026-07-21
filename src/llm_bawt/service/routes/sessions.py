"""Thread (session) management routes — TASK-284.

Exposes the durable `sessions` registry as a user-scoped thread API for a
threaded UI: list/read/create/activate over the one-active-per-(bot,user)
invariant enforced in the DB (idx_sessions_one_active_per_bot_user).

All routes resolve an effective `(bot_id, user_id)` and NEVER cross the user
boundary — a session that belongs to a different user is treated as absent
(404), so users sharing a bot cannot see, read, activate, or hydrate each
other's threads.

The lifecycle primitives (rotate/activate/get_or_create) live in
`PostgreSQLShortTermManager` and are surfaced here through the async
`MemoryStorage` wrappers; this module owns only HTTP shape + user scoping.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...mcp_server.storage import get_storage
from ..dependencies import get_effective_bot_id, get_service
from ..logging import get_service_logger

router = APIRouter()
log = get_service_logger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SessionInfo(BaseModel):
    id: str
    bot_id: str
    user_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    status: str
    session_metadata: dict | None = None


class SessionListResponse(BaseModel):
    bot_id: str
    user_id: str
    sessions: list[SessionInfo]
    total_count: int


class SessionMessage(BaseModel):
    id: str | None = None
    role: str
    content: str
    timestamp: float | None = None
    session_id: str | None = None


class SessionTranscriptResponse(BaseModel):
    session_id: str
    bot_id: str
    messages: list[SessionMessage]
    total_count: int


class CreateSessionResponse(BaseModel):
    session_id: str
    bot_id: str
    user_id: str
    status: str = "active"
    rotated: bool = Field(
        default=True,
        description="True — creating a thread rotates (closes) the prior active one.",
    )


class ActivateSessionResponse(BaseModel):
    session_id: str
    bot_id: str
    user_id: str
    activated: bool
    provider_session: str | None = Field(
        default=None,
        description=(
            "Agent bots only (TASK-284 step 15): 'resumed' when the thread's "
            "stored provider session id was restored for the next turn, "
            "'reseed' when none was stored so the provider will cold-start "
            "and re-seed from this thread's context. None for chat bots."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_user(user: str | None) -> str:
    """Resolve the effective user id or 400 (mirrors routes/config.py)."""
    service = get_service()
    user_id = (user or getattr(service.config, "DEFAULT_USER", "") or "").strip()
    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="user is required (no DEFAULT_USER configured)",
        )
    return user_id


def _to_info(row: dict) -> SessionInfo:
    return SessionInfo(
        id=row["id"],
        bot_id=row.get("bot_id", ""),
        user_id=row.get("user_id"),
        started_at=str(row["started_at"]) if row.get("started_at") is not None else None,
        ended_at=str(row["ended_at"]) if row.get("ended_at") is not None else None,
        status=row.get("status", ""),
        session_metadata=row.get("session_metadata"),
    )


async def _owned_session_or_404(session_id: str, bot_id: str, user_id: str) -> dict:
    """Fetch a session and enforce (bot, user) ownership; 404 otherwise."""
    storage = get_storage()
    row = await storage.get_session(session_id, bot_id=bot_id)
    if not row or row.get("bot_id") != bot_id or row.get("user_id") != user_id:
        # Cross-user / cross-bot / missing all look identical to the caller.
        raise HTTPException(status_code=404, detail="Session not found")
    return row


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("/v1/sessions", response_model=SessionListResponse, tags=["Sessions"])
async def list_sessions(
    bot_id: str = Query(None, description="Bot slug; defaults to service default bot"),
    user: str | None = Query(None, description="User id; defaults to DEFAULT_USER"),
    status: str | None = Query(None, description="Filter by status: active | completed"),
    limit: int = Query(50, ge=1, le=200),
):
    """List a user's threads for a bot, newest first."""
    effective_bot = get_effective_bot_id(bot_id)
    user_id = _resolve_user(user)
    rows = await get_storage().list_sessions(
        bot_id=effective_bot, user_id=user_id, status=status, limit=limit
    )
    return SessionListResponse(
        bot_id=effective_bot,
        user_id=user_id,
        sessions=[_to_info(r) for r in rows],
        total_count=len(rows),
    )


@router.get("/v1/sessions/active", response_model=SessionInfo, tags=["Sessions"])
async def get_active_session(
    bot_id: str = Query(None, description="Bot slug; defaults to service default bot"),
    user: str | None = Query(None, description="User id; defaults to DEFAULT_USER"),
):
    """Return the current active thread for (bot, user)."""
    effective_bot = get_effective_bot_id(bot_id)
    user_id = _resolve_user(user)
    row = await get_storage().get_active_session(bot_id=effective_bot, user_id=user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No active session")
    return _to_info(row)


@router.get("/v1/sessions/{session_id}", response_model=SessionInfo, tags=["Sessions"])
async def get_session(
    session_id: str,
    bot_id: str = Query(None, description="Bot slug; defaults to service default bot"),
    user: str | None = Query(None, description="User id; defaults to DEFAULT_USER"),
):
    """Return one thread's metadata (user-scoped)."""
    effective_bot = get_effective_bot_id(bot_id)
    user_id = _resolve_user(user)
    row = await _owned_session_or_404(session_id, effective_bot, user_id)
    return _to_info(row)


@router.get(
    "/v1/sessions/{session_id}/messages",
    response_model=SessionTranscriptResponse,
    tags=["Sessions"],
)
async def get_session_messages(
    session_id: str,
    bot_id: str = Query(None, description="Bot slug; defaults to service default bot"),
    user: str | None = Query(None, description="User id; defaults to DEFAULT_USER"),
    limit: int = Query(500, ge=1, le=2000),
):
    """Return a thread's raw transcript (user-scoped)."""
    effective_bot = get_effective_bot_id(bot_id)
    user_id = _resolve_user(user)
    await _owned_session_or_404(session_id, effective_bot, user_id)
    rows = await get_storage().get_messages(
        bot_id=effective_bot, session_id=session_id, limit=limit
    )
    return SessionTranscriptResponse(
        session_id=session_id,
        bot_id=effective_bot,
        messages=[
            SessionMessage(
                id=str(r["id"]) if r.get("id") is not None else None,
                role=r.get("role", "?"),
                content=r.get("content", ""),
                timestamp=r.get("timestamp"),
                session_id=r.get("session_id"),
            )
            for r in rows
        ],
        total_count=len(rows),
    )


@router.post("/v1/sessions", response_model=CreateSessionResponse, tags=["Sessions"])
async def create_session(
    bot_id: str = Query(None, description="Bot slug; defaults to service default bot"),
    user: str | None = Query(None, description="User id; defaults to DEFAULT_USER"),
):
    """Open a fresh thread for (bot, user), rotating (closing) the prior active
    one. Non-destructive: old rows are preserved under the closed session.
    """
    effective_bot = get_effective_bot_id(bot_id)
    user_id = _resolve_user(user)
    new_id = await get_storage().rotate_session(bot_id=effective_bot, user_id=user_id)
    # TASK-284 step 15: for agent bots a fresh thread must not resume the OLD
    # provider transcript — clear the stored provider session so the bridge
    # cold-starts and re-seeds ("reseed"). No-op for chat bots.
    _coordinate_agent_provider_on_activate(
        effective_bot, {"id": new_id, "session_metadata": {}}
    )
    return CreateSessionResponse(
        session_id=new_id, bot_id=effective_bot, user_id=user_id
    )


@router.post(
    "/v1/sessions/{session_id}/activate",
    response_model=ActivateSessionResponse,
    tags=["Sessions"],
)
async def activate_session(
    session_id: str,
    bot_id: str = Query(None, description="Bot slug; defaults to service default bot"),
    user: str | None = Query(None, description="User id; defaults to DEFAULT_USER"),
):
    """Switch the active thread to an existing one (user-scoped)."""
    effective_bot = get_effective_bot_id(bot_id)
    user_id = _resolve_user(user)
    # Ownership check first so a cross-user id can't be activated (404, not 403,
    # to avoid confirming the session exists for another user).
    row = await _owned_session_or_404(session_id, effective_bot, user_id)
    ok = await get_storage().activate_session(
        session_id, bot_id=effective_bot, user_id=user_id
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    provider_session = _coordinate_agent_provider_on_activate(effective_bot, row)
    return ActivateSessionResponse(
        session_id=session_id,
        bot_id=effective_bot,
        user_id=user_id,
        activated=True,
        provider_session=provider_session,
    )


def _coordinate_agent_provider_on_activate(bot_id: str, session_row: dict) -> str | None:
    """TASK-284 step 15: point an agent bot's provider at the activated thread.

    Provider hydration stays bridge-owned; this only sets WHICH provider
    session the bridge resumes next turn:

    - thread metadata carries ``provider_session_id`` (mirrored when the
      bridge persisted it) → restore it into ``agent_backend_config.
      session_key`` so the SDK resumes that transcript → returns "resumed";
    - no stored provider id → clear ``session_key`` so the bridge cold-starts
      and re-seeds from this thread's context via the shared assembler →
      returns "reseed".

    Applies only to claude-code/codex (openclaw's session_key is a routing
    key, never per-thread). Best-effort: any failure leaves the profile
    untouched and returns None — the DB-side activation already succeeded.
    """
    try:
        service = get_service()
        from ...bots import BotManager, invalidate_bots_cache

        bot = BotManager(service.config).get_bot(bot_id)
        backend = (getattr(bot, "agent_backend", None) or "").strip() if bot else ""
        if backend not in ("claude-code", "codex"):
            return None

        meta = session_row.get("session_metadata") or {}
        provider_sid = str(meta.get("provider_session_id") or "").strip()

        from ..dependencies import get_bot_profile_store

        store = get_bot_profile_store(service.config)
        profile = store.get(bot_id)
        if profile is None:
            return None
        bc = dict(profile.agent_backend_config or {})
        if provider_sid and ":" not in provider_sid:
            bc["session_key"] = provider_sid
            if meta.get("provider_session_model"):
                bc["session_model"] = meta["provider_session_model"]
            outcome = "resumed"
        else:
            bc.pop("session_key", None)
            outcome = "reseed"
        profile.agent_backend_config = bc
        store.upsert(profile)
        # Cached Bot/instance state must see the new session_key (the bridge
        # reads it live over HTTP, but the app-side seed decision reads the
        # cached bot).
        invalidate_bots_cache()
        invalidate = getattr(service, "invalidate_bot_instances", None)
        if callable(invalidate):
            invalidate(bot_id)
        log.info(
            "Agent thread activate: bot=%s thread=%s provider_session=%s",
            bot_id, session_row.get("id"), outcome,
        )
        return outcome
    except Exception as e:
        log.warning("Agent provider coordination failed for %s: %s", bot_id, e)
        return None
