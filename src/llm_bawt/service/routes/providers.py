"""Provider-connect routes — /v1/providers.

Drives the reusable "connect a provider" flow (device OAuth / api key) used by
the first-run wizard and settings UI. Phase 1 ships GitHub (device flow) + the
internal git-credential endpoint that feeds the tenant bridge's git helper.

Adapter network calls are sync (httpx); we run them in a threadpool so the async
event loop isn't blocked.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from ..dependencies import get_service
from ..providers.base import AUTH_CLI_OAUTH, AUTH_DEVICE_OAUTH
from ..providers.github import GitHubAdapter, GitHubConfigError
from ..providers.registry import all_adapters, get_adapter

logger = logging.getLogger(__name__)

router = APIRouter()


class PollRequest(BaseModel):
    device_code: str


class CliLoginCompleteRequest(BaseModel):
    session_id: str
    code: str


class SelectReposRequest(BaseModel):
    installation_id: int
    repos: list[str] | None = None


def _adapter_or_404(provider_id: str):
    service = get_service()
    adapter = get_adapter(service.config, provider_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{provider_id}'")
    return adapter


@router.get("/v1/providers")
async def list_providers():
    service = get_service()
    return {"providers": [a.descriptor() for a in all_adapters(service.config)]}


@router.get("/v1/providers/{provider_id}")
async def get_provider(provider_id: str):
    return _adapter_or_404(provider_id).descriptor()


@router.post("/v1/providers/{provider_id}/connect/start")
async def connect_start(provider_id: str):
    adapter = _adapter_or_404(provider_id)
    if not adapter.supports(AUTH_DEVICE_OAUTH):
        raise HTTPException(status_code=400, detail=f"{provider_id} has no device flow")
    try:
        start = await run_in_threadpool(adapter.start_device_flow)
    except GitHubConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.warning("device flow start failed for %s: %s", provider_id, e)
        raise HTTPException(status_code=502, detail=f"provider error: {e}")
    return {
        "user_code": start.user_code,
        "verification_uri": start.verification_uri,
        "device_code": start.device_code,
        "interval": start.interval,
        "expires_in": start.expires_in,
    }


@router.post("/v1/providers/{provider_id}/connect/poll")
async def connect_poll(provider_id: str, body: PollRequest):
    adapter = _adapter_or_404(provider_id)
    if not adapter.supports(AUTH_DEVICE_OAUTH):
        raise HTTPException(status_code=400, detail=f"{provider_id} has no device flow")
    try:
        result = await run_in_threadpool(adapter.poll_device_flow, body.device_code)
    except GitHubConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.warning("device flow poll failed for %s: %s", provider_id, e)
        raise HTTPException(status_code=502, detail=f"provider error: {e}")
    out = {"status": result.status, "detail": result.detail}
    if result.record is not None:
        out["connection"] = result.record.public()
    return out


# --- CLI-driven OAuth (e.g. Claude subscription login) ----------------------
@router.post("/v1/providers/{provider_id}/login/start")
async def cli_login_start(provider_id: str):
    """Spawn the provider's login CLI under a PTY and return the authorize URL.

    The live process is held server-side keyed by ``session_id`` until the user
    submits the pasted code via ``/login/complete``.
    """
    adapter = _adapter_or_404(provider_id)
    if not adapter.supports(AUTH_CLI_OAUTH):
        raise HTTPException(status_code=400, detail=f"{provider_id} has no cli login")
    try:
        start = await run_in_threadpool(adapter.start_cli_login)
    except Exception as e:  # noqa: BLE001
        logger.warning("cli login start failed for %s: %s", provider_id, e)
        raise HTTPException(status_code=502, detail=f"login error: {e}")
    return {
        "session_id": start.session_id,
        "verification_uri": start.verification_uri,
        "instructions": start.instructions,
    }


@router.post("/v1/providers/{provider_id}/login/complete")
async def cli_login_complete(provider_id: str, body: CliLoginCompleteRequest):
    """Submit the pasted code into the live login session; persist on success."""
    adapter = _adapter_or_404(provider_id)
    if not adapter.supports(AUTH_CLI_OAUTH):
        raise HTTPException(status_code=400, detail=f"{provider_id} has no cli login")
    result = await run_in_threadpool(
        adapter.complete_cli_login, body.session_id, body.code
    )
    if result.status != "connected":
        raise HTTPException(status_code=400, detail=result.detail or "login failed")
    out = {"status": result.status, "detail": result.detail}
    if result.record is not None:
        out["connection"] = result.record.public()
    return out


@router.delete("/v1/providers/{provider_id}")
async def disconnect(provider_id: str):
    adapter = _adapter_or_404(provider_id)
    ok = await run_in_threadpool(adapter.disconnect)
    return {"disconnected": ok}


# --- GitHub-specific --------------------------------------------------------
@router.get("/v1/providers/github/installations")
async def github_installations():
    service = get_service()
    adapter = get_adapter(service.config, "github")
    if not isinstance(adapter, GitHubAdapter):
        raise HTTPException(status_code=404, detail="github not available")
    return {"installations": await run_in_threadpool(adapter.list_installations)}


@router.post("/v1/providers/github/repos")
async def github_select_repos(body: SelectReposRequest):
    service = get_service()
    adapter = get_adapter(service.config, "github")
    if not isinstance(adapter, GitHubAdapter):
        raise HTTPException(status_code=404, detail="github not available")
    try:
        record = await run_in_threadpool(
            adapter.select_installation, body.installation_id, body.repos
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))
    return {"connection": record.public()}


@router.get("/v1/providers/github/git-credential")
async def github_git_credential(request: Request):
    """Internal: mint a fresh installation token in git-credential format.

    The tenant bridge configures ``credential.helper`` to curl this endpoint.
    App :8642 is not published in the prod tenant (only frontend :3000 is), so
    this is reachable only on the internal docker network. If
    ``BRIDGE_GIT_CREDENTIAL_TOKEN`` is set, we additionally require it via the
    ``X-Bridge-Token`` header for defense in depth.
    """
    expected = os.getenv("BRIDGE_GIT_CREDENTIAL_TOKEN")
    if expected and request.headers.get("X-Bridge-Token") != expected:
        raise HTTPException(status_code=401, detail="bad bridge token")

    service = get_service()
    adapter = get_adapter(service.config, "github")
    if not isinstance(adapter, GitHubAdapter):
        raise HTTPException(status_code=404, detail="github not available")
    try:
        token = await run_in_threadpool(adapter.mint_installation_token)
    except GitHubConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.warning("git-credential mint failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

    # git credential helper protocol: key=value lines on stdout.
    body = f"username=x-access-token\npassword={token}\n"
    return Response(content=body, media_type="text/plain")
