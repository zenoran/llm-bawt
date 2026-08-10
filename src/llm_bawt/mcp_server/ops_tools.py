"""Generic ops MCP tools (TASK-639).

Three tools, deliberately narrow so the agent surface stays stable as the
catalog grows:

* :func:`ops_list_operations` — discover what operations exist right now.
* :func:`ops_run` — invoke one by slug with validated args. Never accepts a
  raw command string; the executable command comes from the DB row.
* :func:`ops_job_status` — poll a running/terminal job.

These tools do NOT own their own approval gating — the approval-aware
FastMCP interceptor (Slice E) evaluates a policy against every call before
the function runs. Until the interceptor is wired the tools remain
approval-agnostic: the operator can gate the same three tools via a
classic bridge-hook policy on ``ops_run``/``ops_job_status`` names.
"""

from __future__ import annotations

import logging
from typing import Any

from .registry import mcp

logger = logging.getLogger(__name__)


def _get_ops_service():
    """Lazy accessor — the app service is constructed after the MCP module
    is imported, so we resolve on each call.
    """
    from ..service.dependencies import get_ops_service, get_service

    svc = get_service()
    return get_ops_service(svc.config)


# ---------------------------------------------------------------------------
# ops_list_operations
# ---------------------------------------------------------------------------

@mcp.tool(name="ops_list_operations")
async def ops_list_operations(include_disabled: bool = False) -> dict:
    """List every operation the agent can invoke via ``ops_run``.

    Args:
        include_disabled: If True, include disabled rows (operator view).

    Returns:
        Dict with keys:
          - ``operations``: list of ``{slug, title, description, risk_level,
            category, target_host, timeout_seconds, args_schema,
            args_defaults}``
          - ``total``: count in the list
    """
    ops = _get_ops_service()
    rows = ops.list_operations_for_agent(include_disabled=include_disabled)
    return {"operations": rows, "total": len(rows)}


# ---------------------------------------------------------------------------
# ops_run
# ---------------------------------------------------------------------------

@mcp.tool(name="ops_run")
async def ops_run(
    operation: str,
    args: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
) -> dict:
    """Run a catalogued operation.

    The agent supplies a slug + validated args — the executable command
    itself lives in the DB and is authored by the operator. Unknown args
    are rejected before any script runs.

    Args:
        operation: Operation slug (e.g. ``"llm-bawt.restart-app"``). Get
            valid slugs from ``ops_list_operations``.
        args: Arguments matching the operation's declared JSON Schema.
        idempotency_key: Optional stable key. A retry with the same key
            returns the pre-existing job without re-dispatching.

    Returns:
        ``{job_id, operation, state, submitted_at, host_unit_name}`` on
        success. On failure raises with a machine-readable ``code`` in the
        error message: ``operation_not_found`` | ``operation_disabled`` |
        ``args_invalid`` | ``executor_unavailable`` | ``dispatch_failed``.
    """
    from ..ops.service import OpsDispatchError

    if not operation or not isinstance(operation, str):
        raise ValueError("operation slug is required")
    ops = _get_ops_service()

    idem = (idempotency_key or "").strip()
    if not idem:
        # A caller with no explicit key still gets deduplicated within a
        # short window via the (operation, sorted-args) subject. This is a
        # weaker guarantee than an approval-request-id, and the caller
        # should prefer supplying an explicit key.
        import hashlib
        import json as _json
        canonical = _json.dumps(
            {"op": operation, "args": args or {}},
            ensure_ascii=False, sort_keys=True,
        )
        idem = "auto-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    try:
        result = ops.dispatch_job(
            operation_slug=operation,
            args=args or {},
            idempotency_key=idem,
        )
    except OpsDispatchError as exc:
        # Surface a structured error the agent can parse. MCP tools
        # signal errors by raising; the caller sees a tool_result with
        # is_error=True and the message text.
        raise RuntimeError(f"[{exc.code}] {exc}") from exc
    return {
        "job_id": result["id"],
        "operation": result["operation"],
        "state": result["state"],
        "submitted_at": result["submitted_at"],
        "host_unit_name": result.get("host_unit_name"),
        "idempotency_key": idem,
    }


# ---------------------------------------------------------------------------
# ops_job_status
# ---------------------------------------------------------------------------

@mcp.tool(name="ops_job_status")
async def ops_job_status(
    job_id: str,
    output_tail_bytes: int | None = None,
) -> dict:
    """Poll a job. Reconciles the host status file into the DB on the way
    if the job is still active.

    Args:
        job_id: Job id from a prior ``ops_run`` call.
        output_tail_bytes: When non-null, include the last N bytes of the
            job's combined stdout/stderr in the response. Default: omit.

    Returns:
        Full job metadata dict including ``state``, ``exit_code``,
        ``terminal`` flag, ``started_at`` / ``finished_at``, and
        ``output_tail`` when requested.
    """
    if not job_id:
        raise ValueError("job_id is required")
    ops = _get_ops_service()
    result = ops.get_job_status(
        job_id,
        output_tail_bytes=output_tail_bytes,
        reconcile_if_active=True,
    )
    if result is None:
        raise LookupError(f"no job with id {job_id!r}")
    return result


__all__ = ["ops_list_operations", "ops_run", "ops_job_status"]
