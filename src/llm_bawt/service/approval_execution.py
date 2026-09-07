"""Execution lease lifetime for approved MCP invocations."""
from __future__ import annotations

import asyncio


class ApprovalExecutionLeaseLost(RuntimeError):
    """Local authority expired; outcome needs recovery, not a success/failure guess."""


async def run_with_execution_lease(store, row, invocation):
    """Renew while executing; a lost lease cancels local work, never authorizes replay.

    Cancellation cannot retract an already-started external side effect. The
    persisted RUNNING claim therefore remains for uncertain/idempotent recovery.
    """
    async def renew():
        while True:
            await asyncio.sleep(30)
            try:
                renewed = await asyncio.to_thread(
                    store.renew_mcp_execution_claim, row.id, row.execution_claim_token,
                )
            except Exception as exc:
                raise ApprovalExecutionLeaseLost("Could not renew approved MCP execution lease") from exc
            if not renewed:
                raise ApprovalExecutionLeaseLost("Approved MCP execution lease lost")

    worker = asyncio.ensure_future(invocation)
    heartbeat = asyncio.create_task(renew())
    try:
        done, _ = await asyncio.wait((worker, heartbeat), return_when=asyncio.FIRST_COMPLETED)
        if heartbeat in done:
            await heartbeat
        return await worker
    finally:
        for task in (worker, heartbeat):
            task.cancel()
        await asyncio.gather(worker, heartbeat, return_exceptions=True)
