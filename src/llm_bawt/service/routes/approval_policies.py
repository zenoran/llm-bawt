"""Approval-gated tool policy admin + resolve routes (TASK-293).

Source of truth for the feature. CRUD over the policy rules, the compiled
bundle bridges fetch, the audit list of gated requests, and the resolve
endpoint a user hits to approve/deny a pending request (which grants the bridge
a one-shot allow and returns a continuation prompt the client dispatches).
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..dependencies import get_service, get_tool_approval_policy_store
from ...approval_policies import REQ_APPROVED, REQ_DENIED

log = logging.getLogger(__name__)
router = APIRouter()


def _store():
    service = get_service()
    store = get_tool_approval_policy_store(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Approval policy database unavailable")
    return store


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class PolicyUpsert(BaseModel):
    enabled: bool | None = None
    backend_scope: str | None = None
    tool_name: str | None = None
    matcher_type: str | None = None
    pattern: str | None = None
    field: str | None = None
    action: str | None = None
    severity: str | None = None
    category: str | None = None
    approval_prompt: str | None = None
    order: int | None = None

    def writable(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ResolveRequest(BaseModel):
    decision: str = Field(..., description="'approve' or 'deny'")
    bot_id: str = Field("", description="Bot slug (for tab fanout)")
    user_id: str = Field("nick", description="User id (for tab fanout)")
    resolved_by: str | None = None


# ---------------------------------------------------------------------------
# Policy CRUD
# ---------------------------------------------------------------------------

@router.get("/v1/tool-approval-policies", tags=["Approval Policies"])
def list_policies():
    store = _store()
    rows = store.list_all()
    return {"policies": [r.to_api() for r in rows], "total": len(rows)}


@router.get("/v1/tool-approval-policies/bundle", tags=["Approval Policies"])
def get_bundle(etag: str | None = None):
    """Compiled bundle a bridge fetches. If ``etag`` matches, returns
    ``{unchanged: true}`` so the bridge can skip re-parsing."""
    store = _store()
    bundle = store.compile_bundle()
    if etag and etag == bundle.etag:
        return {"unchanged": True, "etag": bundle.etag, "version": bundle.version}
    return bundle.to_dict()


@router.post("/v1/tool-approval-policies", tags=["Approval Policies"], status_code=201)
def create_policy(body: PolicyUpsert):
    store = _store()
    row = store.create(body.writable())
    return row.to_api()


@router.post("/v1/tool-approval-policies/seed-defaults", tags=["Approval Policies"])
def seed_defaults():
    store = _store()
    seeded = store.seed_defaults()
    return {"seeded": seeded, "total": len(store.list_all())}


@router.get("/v1/tool-approval-policies/{policy_id}", tags=["Approval Policies"])
def get_policy(policy_id: str):
    store = _store()
    row = store.get(policy_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
    return row.to_api()


@router.patch("/v1/tool-approval-policies/{policy_id}", tags=["Approval Policies"])
def update_policy(policy_id: str, body: PolicyUpsert):
    store = _store()
    row = store.update(policy_id, body.writable())
    if row is None:
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
    return row.to_api()


@router.delete("/v1/tool-approval-policies/{policy_id}", tags=["Approval Policies"])
def delete_policy(policy_id: str):
    store = _store()
    if not store.delete(policy_id):
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
    return {"success": True, "id": policy_id}


@router.post("/v1/admin/reload-tool-approval-policies", tags=["Admin"])
async def reload_policies():
    """Force every bridge to drop its cached bundle now (no restart).

    CRUD edits propagate within the bridge cache TTL on their own; call this to
    make a change take effect immediately (the admin UI calls it after saves).
    """
    store = _store()
    bundle = store.compile_bundle()
    sub = _subscriber()
    if sub is not None:
        try:
            await sub.publish_approval_reload()
        except Exception:  # noqa: BLE001
            log.warning("reload publish failed", exc_info=True)
    return {"status": "reloaded", "etag": bundle.etag, "policies": len(bundle.policies)}


# ---------------------------------------------------------------------------
# Request audit + resolve
# ---------------------------------------------------------------------------

@router.get("/v1/tool-approval-requests", tags=["Approval Policies"])
def list_requests(status: str | None = None, bot_id: str | None = None, limit: int = 50):
    store = _store()
    rows = store.list_requests(status=status, bot_id=bot_id, limit=min(max(limit, 1), 200))
    return {"requests": [r.to_api() for r in rows], "total": len(rows)}


@router.post("/v1/chat/approvals/{request_id}/resolve", tags=["Approval Policies"])
async def resolve_approval(request_id: str, body: ResolveRequest):
    """Approve or deny a pending gated tool call.

    On approve: record it, grant the bridge a one-shot allow keyed by the
    request's grant_key, and return a continuation prompt the client dispatches
    so the model re-issues the now-allowed call. On deny: record it and return a
    prompt telling the model it was refused. Idempotent on already-resolved.
    """
    store = _store()
    row = store.get_request(request_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No approval request id={request_id}")

    decision = (body.decision or "").strip().lower()
    if decision not in ("approve", "approved", "allow", "deny", "denied", "reject"):
        raise HTTPException(status_code=400, detail="decision must be 'approve' or 'deny'")
    approved = decision in ("approve", "approved", "allow")

    subject = row.subject or ""
    bot_id = body.bot_id or row.bot_id
    user_id = body.user_id or row.user_id

    if row.status != "pending":
        # Idempotent replay — return the same continuation the first resolve did.
        prompt = _continuation_prompt(row.status == REQ_APPROVED, subject, row.tool_name)
        return {
            "ok": True, "detail": "already_resolved", "status": row.status,
            "request_id": request_id, "bot_id": bot_id,
            "continuation_prompt": prompt, "parent_turn_id": row.turn_id,
            "already_resolved": True,
        }

    new_status = REQ_APPROVED if approved else REQ_DENIED
    updated = store.resolve_request(
        request_id, status=new_status, resolved_by=body.resolved_by,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Request disappeared during resolve")

    subscriber = _subscriber()
    if approved and subscriber is not None:
        # Grant the bridge a one-shot allow BEFORE the client dispatches the
        # continuation turn, so the re-issued tool call sails through.
        try:
            await subscriber.send_approval_grant(
                session_key=row.session_key or "main",
                grant_key=row.grant_key,
                backend=row.backend,
                request_id=request_id,
            )
        except Exception:  # noqa: BLE001
            log.exception("Failed to send approval.grant for %s", request_id)

    await _fanout_resolved(subscriber, bot_id, user_id, request_id, row.turn_id, new_status)

    prompt = _continuation_prompt(approved, subject, row.tool_name)
    log.info(
        "Approval %s: id=%s bot=%s subject=%r — client will dispatch continuation",
        new_status, request_id, bot_id, subject[:80],
    )
    return {
        "ok": True, "detail": new_status, "status": new_status,
        "request_id": request_id, "bot_id": bot_id,
        "continuation_prompt": prompt, "parent_turn_id": row.turn_id,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subscriber():
    try:
        from ...agent_backends.agent_bridge import get_agent_subscriber

        return get_agent_subscriber()
    except Exception:  # noqa: BLE001
        return None


def _continuation_prompt(approved: bool, subject: str, tool_name: str) -> str:
    shown = subject if len(subject) <= 400 else subject[:397] + "…"
    if approved:
        return (
            f"[The user APPROVED the {tool_name} action you requested. Re-issue "
            f"exactly this call now and continue:\n\n{shown}]"
        )
    return (
        f"[The user DENIED the {tool_name} action you requested:\n\n{shown}\n\n"
        f"Do not attempt it again. Continue without it, or explain what you need.]"
    )


async def _fanout_resolved(subscriber, bot_id, user_id, request_id, turn_id, status) -> None:
    """Fan out an approval_resolved unified event so every tab clears its card."""
    if subscriber is None:
        return
    try:
        await subscriber._redis.xadd(
            f"events:{bot_id}:{user_id}",
            {"payload": json.dumps({
                "_type": "approval_resolved",
                "bot_id": bot_id,
                "user_id": user_id,
                "request_id": request_id,
                "turn_id": turn_id,
                "status": status,
                "ts": time.time(),
            }, ensure_ascii=False, default=str)},
            maxlen=5000,
            approximate=True,
        )
    except Exception:  # noqa: BLE001
        log.debug("failed to publish approval_resolved for %s", request_id, exc_info=True)
