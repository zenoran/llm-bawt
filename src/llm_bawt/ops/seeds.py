"""Canonical operation catalog seeds (TASK-639).

Every row here is inserted per-slug via
:meth:`OpsStore.seed_operation_if_missing` — an existing row with the same
slug is preserved as-is, never overwritten. Operators own the catalog after
first insert; seeds only bootstrap or add net-new slugs.

Seeds ship **disabled** so nothing dangerous is invocable until an operator
explicitly enables the row in the BawtHub UI. This is a deliberate safety
choice — the TASK-639 spec explicitly warns against "a safety feature that
ships disabled protects nothing", but that guidance applies to *approval
policies*, not to the executable ops catalog itself. Restart-your-own-bridge
scripts should require a human enable.

Argument schemas use ``additionalProperties: false`` so unknown args are
rejected before any script runs.
"""

from __future__ import annotations

import json
from typing import Any

from .models import (
    EXECUTOR_NOHUP_SSH,
    RISK_CRITICAL,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ECHO_HOST = "nick@172.18.0.1"
_LLM_BAWT_DIR = "/home/nick/dev/llm-bawt"
_BAWTHUB_DIR = "/home/nick/dev/bawthub"


def _schema(props: dict[str, Any] | None = None, required: list[str] | None = None) -> str:
    body: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": props or {},
    }
    if required:
        body["required"] = required
    return json.dumps(body, ensure_ascii=False)


def _no_args_schema() -> str:
    return _schema({})


def _defaults(d: dict[str, Any] | None = None) -> str:
    return json.dumps(d or {}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# llm-bawt service restarts
# ---------------------------------------------------------------------------

_LLM_BAWT_SEEDS: list[dict[str, Any]] = [
    {
        "slug": "llm-bawt.restart-app",
        "title": "Restart llm-bawt app",
        "description": (
            "Reload Python source in the `app` container. Safe for `.py` edits — "
            "does not rebuild image. Cuts SSE mid-turn, so use ``start_delay_seconds`` "
            "when calling from an active chat."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_NOHUP_SSH,
        "target_host": _ECHO_HOST,
        "working_directory": _LLM_BAWT_DIR,
        "command_script": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "cd \"$OPS_WORKING_DIR\"\n"
            "docker compose restart app\n"
        ),
        "args_schema_json": _no_args_schema(),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 120,
        "start_delay_seconds": 5,
        "risk_level": RISK_MEDIUM,
        "category": "restart",
        "approval_prompt_prefix": "Restart llm-bawt app container",
    },
    {
        "slug": "llm-bawt.restart-aux",
        "title": "Restart llm-bawt aux service",
        "description": (
            "Restart a non-bridge llm-bawt service (crawl4ai, playwright-mcp, "
            "local-model-bridge). Bridges + redis are excluded — those need "
            "explicit dedicated ops."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_NOHUP_SSH,
        "target_host": _ECHO_HOST,
        "working_directory": _LLM_BAWT_DIR,
        "command_script": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "cd \"$OPS_WORKING_DIR\"\n"
            "case \"$OPS_ARG_SERVICE\" in\n"
            "  crawl4ai|playwright-mcp|local-model-bridge) ;;\n"
            "  *) echo \"forbidden service: $OPS_ARG_SERVICE\" >&2; exit 2 ;;\n"
            "esac\n"
            "docker compose restart \"$OPS_ARG_SERVICE\"\n"
        ),
        "args_schema_json": _schema(
            {
                "service": {
                    "type": "string",
                    "enum": ["crawl4ai", "playwright-mcp", "local-model-bridge"],
                }
            },
            required=["service"],
        ),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 180,
        "start_delay_seconds": 5,
        "risk_level": RISK_LOW,
        "category": "restart",
        "approval_prompt_prefix": "Restart llm-bawt aux service",
    },
    {
        "slug": "llm-bawt.restart-bridge",
        "title": "Restart an agent bridge",
        "description": (
            "Restart claude-code-bridge, openclaw-bridge, or codex-bridge. HIGH "
            "risk: kills active agent sessions hosted by that bridge. Requires "
            "explicit operator approval each time."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_NOHUP_SSH,
        "target_host": _ECHO_HOST,
        "working_directory": _LLM_BAWT_DIR,
        "command_script": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "cd \"$OPS_WORKING_DIR\"\n"
            "case \"$OPS_ARG_BRIDGE\" in\n"
            "  claude-code-bridge|openclaw-bridge|codex-bridge) ;;\n"
            "  *) echo \"forbidden bridge: $OPS_ARG_BRIDGE\" >&2; exit 2 ;;\n"
            "esac\n"
            "docker compose restart \"$OPS_ARG_BRIDGE\"\n"
        ),
        "args_schema_json": _schema(
            {
                "bridge": {
                    "type": "string",
                    "enum": ["claude-code-bridge", "openclaw-bridge", "codex-bridge"],
                }
            },
            required=["bridge"],
        ),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 180,
        "start_delay_seconds": 10,
        "risk_level": RISK_HIGH,
        "category": "restart",
        "approval_prompt_prefix": "Restart bridge (KILLS ACTIVE AGENTS)",
    },
    {
        "slug": "llm-bawt.restart-redis",
        "title": "Restart Redis",
        "description": (
            "Restart the shared Redis instance. CRITICAL risk: cuts the agent "
            "bridge event bus. Every in-flight agent turn dies. Requires "
            "explicit operator approval each time."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_NOHUP_SSH,
        "target_host": _ECHO_HOST,
        "working_directory": _LLM_BAWT_DIR,
        "command_script": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "cd \"$OPS_WORKING_DIR\"\n"
            "docker compose restart redis\n"
        ),
        "args_schema_json": _no_args_schema(),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 180,
        "start_delay_seconds": 10,
        "risk_level": RISK_CRITICAL,
        "category": "restart",
        "approval_prompt_prefix": "Restart Redis (KILLS ALL AGENTS)",
    },
]


# ---------------------------------------------------------------------------
# BawtHub restarts + prod rebuild
# ---------------------------------------------------------------------------

_BAWTHUB_SEEDS: list[dict[str, Any]] = [
    {
        "slug": "bawthub.restart-backend",
        "title": "Restart bawthub backend",
        "description": "Restart the bawthub Python voice/agent backend container.",
        "enabled": False,
        "executor_kind": EXECUTOR_NOHUP_SSH,
        "target_host": _ECHO_HOST,
        "working_directory": _BAWTHUB_DIR,
        "command_script": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "cd \"$OPS_WORKING_DIR\"\n"
            "docker compose restart backend\n"
        ),
        "args_schema_json": _no_args_schema(),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 120,
        "start_delay_seconds": 5,
        "risk_level": RISK_MEDIUM,
        "category": "restart",
        "approval_prompt_prefix": "Restart bawthub backend",
    },
    {
        "slug": "bawthub.restart-frontend",
        "title": "Restart bawthub frontend (HMR)",
        "description": "Restart the bawthub Vite/Hono HMR container.",
        "enabled": False,
        "executor_kind": EXECUTOR_NOHUP_SSH,
        "target_host": _ECHO_HOST,
        "working_directory": _BAWTHUB_DIR,
        "command_script": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "cd \"$OPS_WORKING_DIR\"\n"
            "docker compose restart frontend\n"
        ),
        "args_schema_json": _no_args_schema(),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 120,
        "start_delay_seconds": 0,
        "risk_level": RISK_LOW,
        "category": "restart",
        "approval_prompt_prefix": "Restart bawthub frontend HMR",
    },
    {
        "slug": "bawthub.rebuild-prod",
        "title": "Rebuild bawthub prod image",
        "description": (
            "Run `make rebuild-prod` in the bawthub repo — builds fresh prod "
            "images, tags a release, and rolls the prod container. HIGH risk: "
            "a broken build kills the public site."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_NOHUP_SSH,
        "target_host": _ECHO_HOST,
        "working_directory": _BAWTHUB_DIR,
        "command_script": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "cd \"$OPS_WORKING_DIR\"\n"
            "make rebuild-prod\n"
        ),
        "args_schema_json": _no_args_schema(),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 900,
        "start_delay_seconds": 0,
        "risk_level": RISK_HIGH,
        "category": "deploy",
        "approval_prompt_prefix": "Rebuild + roll bawthub prod",
    },
]


# ---------------------------------------------------------------------------
# Standalone container restarts (things not managed by a compose file the
# app happens to own — the ones on Unraid via docker socket over SSH)
# ---------------------------------------------------------------------------

_UNRAID_CONTAINER_SEEDS: list[dict[str, Any]] = [
    {
        "slug": "container.restart",
        "title": "Restart an allowlisted container",
        "description": (
            "Restart a specific container by name. Only the allowlist below is "
            "runnable — every other name is rejected inside the script."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_NOHUP_SSH,
        "target_host": _ECHO_HOST,
        "working_directory": "/home/nick",
        "command_script": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "case \"$OPS_ARG_CONTAINER\" in\n"
            "  bawthub-frontend-1|NginxProxyManager|bawthub-nocodb-1|"
            "llm-bawt-local-model-bridge|bawthub-public-site) ;;\n"
            "  *) echo \"forbidden container: $OPS_ARG_CONTAINER\" >&2; exit 2 ;;\n"
            "esac\n"
            "docker restart \"$OPS_ARG_CONTAINER\"\n"
        ),
        "args_schema_json": _schema(
            {
                "container": {
                    "type": "string",
                    "enum": [
                        "bawthub-frontend-1",
                        "NginxProxyManager",
                        "bawthub-nocodb-1",
                        "llm-bawt-local-model-bridge",
                        "bawthub-public-site",
                    ],
                }
            },
            required=["container"],
        ),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 120,
        "start_delay_seconds": 0,
        "risk_level": RISK_MEDIUM,
        "category": "restart",
        "approval_prompt_prefix": "Restart container",
    },
]


# ---------------------------------------------------------------------------
# Public seed set
# ---------------------------------------------------------------------------

SEEDS: list[dict[str, Any]] = [
    *_LLM_BAWT_SEEDS,
    *_BAWTHUB_SEEDS,
    *_UNRAID_CONTAINER_SEEDS,
]


def seed_all(store) -> tuple[list[str], list[str]]:
    """Insert every seed row that doesn't already exist.

    Returns ``(inserted_slugs, skipped_slugs)``. Preserves operator edits —
    any slug already present is skipped even if the seed dict differs.
    """
    inserted: list[str] = []
    skipped: list[str] = []
    for seed in SEEDS:
        row = store.seed_operation_if_missing(seed)
        if row is None:
            skipped.append(seed["slug"])
        else:
            inserted.append(seed["slug"])
    return inserted, skipped


__all__ = ["SEEDS", "seed_all"]
