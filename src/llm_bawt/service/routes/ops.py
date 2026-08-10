"""HTTP admin surface for the ops catalog + job ledger (TASK-639).

CRUD for :class:`OpsOperation`, the seed bootstrap, and read-only job
listing / status endpoints. Also exposes a direct-dispatch endpoint so the
BawtHub operations page can trigger a job without going through the MCP
approval flow (still policy-gated at the HTTP layer per operator setup).

BawtHub can proxy every route here via its existing ``/api/chat/proxy/v1/*``
path — no new frontend server prefix is required.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..dependencies import get_ops_service, get_ops_store, get_service

log = logging.getLogger(__name__)
router = APIRouter()


def _store():
    service = get_service()
    store = get_ops_store(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Ops store database unavailable")
    return store


def _service():
    service = get_service()
    ops = get_ops_service(service.config)
    return ops


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class OperationUpsert(BaseModel):
    slug: str | None = None
    title: str | None = None
    description: str | None = None
    enabled: bool | None = None
    executor_kind: str | None = None
    target_host: str | None = None
    run_as_user: str | None = None
    working_directory: str | None = None
    command_script: str | None = None
    args_schema_json: str | None = None
    args_defaults_json: str | None = None
    timeout_seconds: int | None = None
    start_delay_seconds: int | None = None
    max_output_bytes: int | None = None
    max_concurrent: int | None = None
    risk_level: str | None = None
    category: str | None = None
    approval_prompt_prefix: str | None = None

    def writable(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class DispatchRequest(BaseModel):
    operation: str = Field(..., description="Operation slug")
    args: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None


# ---------------------------------------------------------------------------
# Operation CRUD
# ---------------------------------------------------------------------------

@router.get("/v1/ops/operations", tags=["Ops"])
def list_operations(include_disabled: bool = False, include_soft_deleted: bool = False):
    store = _store()
    rows = store.list_operations(
        include_disabled=include_disabled,
        include_soft_deleted=include_soft_deleted,
    )
    return {"operations": [r.to_api(include_script=False) for r in rows], "total": len(rows)}


@router.get("/v1/ops/operations/{slug}", tags=["Ops"])
def get_operation(slug: str, include_script: bool = True):
    store = _store()
    row = store.get_operation_by_slug(slug)
    if row is None:
        raise HTTPException(status_code=404, detail=f"operation not found: {slug}")
    return row.to_api(include_script=include_script)


@router.post("/v1/ops/operations", tags=["Ops"], status_code=201)
def create_operation(body: OperationUpsert):
    store = _store()
    try:
        row = store.create_operation(body.writable(), actor="api")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return row.to_api(include_script=True)


@router.patch("/v1/ops/operations/{slug}", tags=["Ops"])
def update_operation(slug: str, body: OperationUpsert):
    store = _store()
    try:
        row = store.update_operation(slug, body.writable(), actor="api")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"operation not found: {slug}")
    return row.to_api(include_script=True)


@router.post("/v1/ops/operations/{slug}/soft-delete", tags=["Ops"])
def soft_delete_operation(slug: str):
    store = _store()
    if not store.soft_delete_operation(slug, actor="api"):
        raise HTTPException(status_code=404, detail=f"operation not found: {slug}")
    return {"ok": True, "slug": slug}


@router.post("/v1/ops/operations/{slug}/enable", tags=["Ops"])
def enable_operation(slug: str, body: dict[str, Any] | None = None):
    enabled = True if body is None else bool(body.get("enabled", True))
    store = _store()
    row = store.update_operation(slug, {"enabled": enabled}, actor="api")
    if row is None:
        raise HTTPException(status_code=404, detail=f"operation not found: {slug}")
    return row.to_api(include_script=False)


@router.post("/v1/ops/seed-defaults", tags=["Ops"])
def seed_defaults():
    """Insert every canonical seed row that isn't already present.

    Never overwrites operator edits — a slug that already exists is skipped
    whether or not the seed dict has drifted.
    """
    from ...ops.seeds import seed_all

    store = _store()
    inserted, skipped = seed_all(store)
    return {"inserted": inserted, "skipped": skipped, "total_inserted": len(inserted)}


# ---------------------------------------------------------------------------
# Job listing + status
# ---------------------------------------------------------------------------

@router.get("/v1/ops/jobs", tags=["Ops"])
def list_jobs(operation: str | None = None, state: str | None = None, limit: int = 50):
    store = _store()
    rows = store.list_jobs(
        operation_slug=operation, state=state, limit=min(max(limit, 1), 200),
    )
    return {"jobs": [r.to_api(include_output=False) for r in rows], "total": len(rows)}


@router.get("/v1/ops/jobs/{job_id}", tags=["Ops"])
def get_job(job_id: str, output_tail_bytes: int | None = None):
    ops = _service()
    result = ops.get_job_status(
        job_id,
        output_tail_bytes=output_tail_bytes,
        reconcile_if_active=True,
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return result


# ---------------------------------------------------------------------------
# Direct dispatch (HTTP surface for the operator UI)
# ---------------------------------------------------------------------------

@router.post("/v1/ops/jobs", tags=["Ops"], status_code=202)
def dispatch_job(body: DispatchRequest):
    """Directly dispatch a job via HTTP.

    Same semantics as the ``ops_run`` MCP tool — validation + idempotency
    + executor dispatch — but callable from BawtHub's operations page or
    from a terminal (``curl``). Approval gating for this endpoint stays a
    frontend concern; the store enforces disabled/soft-deleted status.
    """
    from ...ops.service import OpsDispatchError

    ops = _service()
    idem = (body.idempotency_key or "").strip()
    if not idem:
        import hashlib
        import json as _json
        canonical = _json.dumps(
            {"op": body.operation, "args": body.args or {}},
            ensure_ascii=False, sort_keys=True,
        )
        idem = "http-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    try:
        result = ops.dispatch_job(
            operation_slug=body.operation,
            args=body.args or {},
            idempotency_key=idem,
        )
    except OpsDispatchError as exc:
        status = {
            "operation_not_found": 404,
            "operation_disabled": 409,
            "args_invalid": 400,
            "executor_unavailable": 503,
            "dispatch_failed": 502,
            "executor_kind_unknown": 500,
        }.get(exc.code, 500)
        raise HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)}) from exc
    return result
