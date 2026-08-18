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
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..approval_continuations import (
    build_continuation_prompt,
    build_respond_prompt,
    spawn_continuation,
)
from ..dependencies import (
    get_service,
    get_tool_approval_policy_store,
    get_turn_log_store,
)
from ...approval_policies import (
    EXEC_FAILED,
    EXEC_SKIPPED,
    EXEC_SUCCEEDED,
    KIND_MCP,
    REQ_APPROVED,
    REQ_CANCELLED,
    REQ_DENIED,
    REQ_RESPONDED,
)

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
    decision: str = Field(..., description="'approve', 'deny', 'cancel', or 'respond'")
    bot_id: str = Field("", description="Bot slug (for tab fanout)")
    user_id: str = Field("nick", description="User id (for tab fanout)")
    resolved_by: str | None = None
    # 'respond' only: the user's own guidance sent to the agent instead of the
    # canned refusal. Optional — empty falls back to a neutral "not run" note.
    message: str = Field("", description="Custom continuation text for 'respond'")
    # When True (default), the SERVER dispatches the continuation turn so the bot
    # is reliably told "approved, run exactly this" regardless of which surface
    # resolved the request (admin page, API, script — not just the chat card).
    # A client that dispatches its own continuation must pass False to avoid a
    # double turn.
    dispatch_continuation: bool = Field(
        False, description="Server dispatches the continuation turn (opt-in; the chat card dispatches client-side)",
    )


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


def _decode_stored_args(row) -> dict[str, Any]:
    try:
        value = json.loads(row.tool_arguments_json or "{}")
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError("stored MCP tool arguments are malformed") from error
    if not isinstance(value, dict):
        raise RuntimeError("stored MCP tool arguments are not an object")
    return value


def _normalize_mcp_result(result: Any) -> Any:
    """Convert FastMCP content blocks/results into a durable JSON value."""
    if isinstance(result, dict):
        return result
    if isinstance(result, (str, int, float, bool)) or result is None:
        return result
    if isinstance(result, (list, tuple)):
        normalized: list[Any] = []
        for block in result:
            text_value = getattr(block, "text", None)
            if isinstance(text_value, str):
                try:
                    normalized.append(json.loads(text_value))
                except json.JSONDecodeError:
                    normalized.append(text_value)
                continue
            model_dump = getattr(block, "model_dump", None)
            normalized.append(model_dump() if callable(model_dump) else str(block))
        return normalized[0] if len(normalized) == 1 else normalized
    model_dump = getattr(result, "model_dump", None)
    return model_dump() if callable(model_dump) else str(result)


def _stored_mcp_resolution(row) -> dict[str, Any]:
    result: Any = None
    if row.result_json:
        try:
            result = json.loads(row.result_json)
        except json.JSONDecodeError:
            result = row.result_json
    return {
        "ok": row.execution_state in (EXEC_SUCCEEDED, EXEC_SKIPPED),
        "detail": "already_resolved",
        "status": row.status,
        "request_id": row.id,
        "bot_id": row.bot_id,
        "parent_turn_id": row.turn_id,
        "already_resolved": True,
        "request_kind": KIND_MCP,
        "execution_state": row.execution_state,
        "result": result,
        "result_is_error": bool(row.result_is_error),
        "continuation_prompt": None,
        "server_dispatched": bool(row.continuation_capable),
        "continuation_status": row.continuation_state,
    }


async def _resolve_mcp_request(store, row, *, outcome: str, message: str, resolved_by: str | None):
    """Resolve + exactly-once execute one MCP-kind approval request."""
    new_status = {
        "approve": REQ_APPROVED,
        "deny": REQ_DENIED,
        "cancel": REQ_CANCELLED,
        "respond": REQ_RESPONDED,
    }[outcome]
    updated = store.resolve_request(row.id, status=new_status, resolved_by=resolved_by)
    if updated is None:
        raise HTTPException(status_code=404, detail="Request disappeared during resolve")

    if outcome != "approve":
        if outcome == "cancel":
            payload = {"status": "cancelled", "message": "The MCP call was cancelled."}
        elif outcome == "respond":
            payload = {
                "status": "responded",
                "message": message or "The user chose not to run the MCP call.",
            }
        else:
            payload = {"status": "denied", "message": "The MCP call was denied."}
        completed = store.complete_mcp_execution(
            row.id,
            result_json=json.dumps(payload, ensure_ascii=False),
            is_error=outcome != "cancel",
            skipped=True,
        )
    else:
        claimed = store.claim_mcp_execution(row.id)
        if claimed is None:
            current = store.get_request(row.id)
            return _stored_mcp_resolution(current or updated)
        try:
            from ...mcp_server.registry import mcp

            stored_args = _decode_stored_args(claimed)
            # Approved ops calls derive job idempotency from the durable approval
            # request. A stale execution lease may be reclaimed after a crash,
            # but the operation service then returns the already-created job.
            trusted_overrides = (
                {"idempotency_key": claimed.id}
                if claimed.tool_name == "ops_run" else None
            )
            result = await mcp.call_approved_tool(
                claimed.tool_name,
                stored_args,
                expected_invocation_hash=claimed.invocation_hash or "",
                trusted_argument_overrides=trusted_overrides,
            )
            payload = _normalize_mcp_result(result)
            completed = store.complete_mcp_execution(
                row.id,
                result_json=json.dumps(payload, ensure_ascii=False, default=str),
                is_error=False,
            )
        except Exception as error:  # noqa: BLE001
            log.exception("Approved MCP execution failed id=%s", row.id)
            payload = {"status": "failed", "error": str(error)}
            completed = store.complete_mcp_execution(
                row.id,
                result_json=json.dumps(payload, ensure_ascii=False),
                is_error=True,
                error=str(error),
            )
    if completed is None:
        raise HTTPException(status_code=500, detail="Could not persist MCP execution result")
    completed = store.enqueue_continuation(row.id) or completed
    response = _stored_mcp_resolution(completed)
    response["detail"] = completed.status
    response["already_resolved"] = False
    return response


@router.get("/v1/tool-approval-requests", tags=["Approval Policies"])
def list_requests(status: str | None = None, bot_id: str | None = None, limit: int = 50):
    store = _store()
    rows = store.list_requests(status=status, bot_id=bot_id, limit=min(max(limit, 1), 200))
    return {"requests": [r.to_api() for r in rows], "total": len(rows)}


@router.post("/v1/chat/approvals/{request_id}/resolve", tags=["Approval Policies"])
async def resolve_approval(request_id: str, body: ResolveRequest):
    """Approve, deny, cancel, or respond to a pending gated tool call.

    On approve: record it, grant the bridge a one-shot allow keyed by the
    request's grant_key, and return a continuation prompt the client dispatches
    so the model re-issues the now-allowed call. On deny: record it and return a
    prompt telling the model it was refused. On cancel: record it and return a
    null continuation — the request is silently dropped without warning the
    agent (no grant, no token-costing acknowledgement). On respond: the tool is
    NOT run (no grant), but the user's own ``message`` becomes the continuation
    instead of the canned refusal — for correcting false-positive gates with
    bespoke guidance. Idempotent on already-resolved.
    """
    store = _store()
    row = store.get_request(request_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No approval request id={request_id}")

    decision = (body.decision or "").strip().lower()
    if decision in ("approve", "approved", "allow"):
        outcome = "approve"
    elif decision in ("deny", "denied", "reject"):
        outcome = "deny"
    elif decision in ("cancel", "cancelled", "canceled", "abort", "dismiss"):
        outcome = "cancel"
    elif decision in ("respond", "reply", "guide", "message"):
        outcome = "respond"
    else:
        raise HTTPException(
            status_code=400,
            detail="decision must be 'approve', 'deny', 'cancel', or 'respond'",
        )
    approved = outcome == "approve"

    subject = row.subject or ""
    bot_id = body.bot_id or row.bot_id
    user_id = body.user_id or row.user_id
    message = (body.message or "").strip()

    if row.request_kind == KIND_MCP:
        # MCP approvals are server-owned: execute the exact stored invocation
        # once, persist its actual result, and never ask the model to re-issue.
        if row.status != "pending":
            return _stored_mcp_resolution(row)
        result = await _resolve_mcp_request(
            store,
            row,
            outcome=outcome,
            message=message,
            resolved_by=body.resolved_by,
        )
        await _fanout_resolved(
            _subscriber(), bot_id, user_id, request_id, row.turn_id, result["status"]
        )
        return result

    if row.status != "pending":
        # Idempotent replay — return the same continuation the first resolve did.
        # Cancelled requests never carried a continuation, so replay returns none.
        if row.status == REQ_CANCELLED:
            prompt = None
        elif row.status == REQ_RESPONDED:
            prompt = build_respond_prompt(message, subject, row.tool_name)
        else:
            prompt = build_continuation_prompt(row.status == REQ_APPROVED, subject, row.tool_name)
        return {
            "ok": True, "detail": "already_resolved", "status": row.status,
            "request_id": request_id, "bot_id": bot_id,
            "continuation_prompt": prompt, "parent_turn_id": row.turn_id,
            "already_resolved": True,
        }

    new_status = {
        "approve": REQ_APPROVED,
        "deny": REQ_DENIED,
        "cancel": REQ_CANCELLED,
        "respond": REQ_RESPONDED,
    }[outcome]
    updated = store.resolve_request(
        request_id, status=new_status, resolved_by=body.resolved_by,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Request disappeared during resolve")

    # TASK-305: stamp the tool_call_record so the approval card survives reload.
    try:
        turn_log_store = get_turn_log_store()
        turn_log_store.set_approval_status(
            tool_use_id=request_id,  # approval request id == gated call's tool_use_id
            approval_request_id=request_id,
            approval_status=new_status,
        )
    except Exception:
        log.debug("Could not stamp tool_call_record approval status for %s", request_id, exc_info=True)

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

    # Cancel is silent: no continuation turn, so the agent is never told and
    # spends no tokens acknowledging. Respond sends the user's own guidance.
    # Approve/deny return the canned prompt the client dispatches to resume
    # (approve) or close out (deny) the turn.
    if outcome == "cancel":
        prompt = None
    elif outcome == "respond":
        prompt = build_respond_prompt(message, subject, row.tool_name)
    else:
        prompt = build_continuation_prompt(approved, subject, row.tool_name)
    # Dispatch the continuation SERVER-SIDE (default) so the bot is reliably
    # messaged regardless of which surface resolved this. Skipped for cancel
    # (silent by design) and when the caller opts to dispatch it itself.
    server_dispatched = False
    if prompt and body.dispatch_continuation:
        spawn_continuation(
            bot_id=bot_id,
            user_id=user_id,
            prompt=prompt,
            parent_turn_id=row.turn_id,
            # Only an approve carries a one-shot grant that must settle first.
            grant_settle_s=0.75 if approved else 0.0,
        )
        server_dispatched = True
    log.info(
        "Approval %s: id=%s bot=%s subject=%r — %s",
        new_status, request_id, bot_id, subject[:80],
        "silent cancel (no continuation)" if outcome == "cancel"
        else "server dispatched continuation" if server_dispatched
        else "client will dispatch continuation",
    )
    return {
        "ok": True, "detail": new_status, "status": new_status,
        "request_id": request_id, "bot_id": bot_id,
        "continuation_prompt": prompt, "parent_turn_id": row.turn_id,
        "cancelled": outcome == "cancel",
        "server_dispatched": server_dispatched,
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
