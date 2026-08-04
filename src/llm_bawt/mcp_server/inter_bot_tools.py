"""MCP tools for immediate and durable inter-bot communication."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import httpx

from .server import _get_storage, mcp

logger = logging.getLogger(__name__)
_APP_BASE_URL = "http://localhost:8642"


def _compat_hook(name: str, default):
    """Honor legacy tests/importers that monkeypatch helpers on server.py."""
    from . import server

    candidate = getattr(server, name, default)
    return candidate if candidate is not default else default
_BOT_SEND_WAIT_SETTING = "bot_send_wait_seconds"
_bot_send_settings_resolver = None


def _bot_send_wait_ceiling_seconds() -> float:
    global _bot_send_settings_resolver
    try:
        if _bot_send_settings_resolver is None:
            from llm_bawt.runtime_settings import RuntimeSettingsResolver
            _bot_send_settings_resolver = RuntimeSettingsResolver(
                config=_get_storage().config, bot=None,
            )
        value = float(_bot_send_settings_resolver.resolve(
            _BOT_SEND_WAIT_SETTING, fallback=300,
        ))
        return value if value > 0 else 300.0
    except Exception:
        logger.warning("Could not resolve %s; using 300s", _BOT_SEND_WAIT_SETTING)
        return 300.0


async def _dispatch_bot_message(
    payload: dict,
    target_bot_id: str,
    sender_bot_id: str,
    timeout_seconds: float,
) -> dict:
    """Perform the normal chat-completion call for an immediate bot message."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_APP_BASE_URL}/v1/chat/completions",
                json=payload,
                headers=(
                    {"X-LLM-Bawt-Inter-Bot-Sender": sender_bot_id}
                    if sender_bot_id != "unknown"
                    else None
                ),
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            result = response.json()
        if "choices" in result and result["choices"]:
            content = result["choices"][0]["message"]["content"] or ""
            return {
                "success": True,
                "content": content,
                "bot_id": target_bot_id,
                "sender": sender_bot_id,
                "response_model": result.get("model"),
            }
        return {
            "success": False,
            "error": f"Invalid response format: {result}",
            "content": "",
            "bot_id": target_bot_id,
            "sender": sender_bot_id,
        }
    except httpx.TimeoutException as exc:
        logger.warning(
            "Inter-bot send to %s timed out after %.1fs (request still in flight server-side)",
            target_bot_id, timeout_seconds,
        )
        return {
            "success": False,
            "error": "timeout",
            "error_detail": str(exc),
            "in_flight": True,
            "warning": (
                f"Target bot did not respond within {timeout_seconds:.0f}s. "
                "The request is likely still being processed server-side. "
                "DO NOT RETRY — that will cause the target bot to receive the message twice."
            ),
            "content": "",
            "bot_id": target_bot_id,
            "sender": sender_bot_id,
        }
    except Exception as exc:
        logger.error("Inter-bot communication failed: %s", exc)
        return {
            "success": False,
            "error": str(exc) or exc.__class__.__name__,
            "content": "",
            "bot_id": target_bot_id,
            "sender": sender_bot_id,
        }


async def _check_bot_in_turn(target_bot_id: str) -> dict | None:
    """Return active-turn data, failing open for legacy immediate sends."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{_APP_BASE_URL}/v1/bots/{target_bot_id}/in-turn", timeout=5.0
            )
            response.raise_for_status()
            data = response.json()
        return data if data.get("in_turn") else None
    except Exception as exc:
        logger.warning("in-turn check for %s failed (allowing send): %s", target_bot_id, exc)
        return None


async def _enqueue_durable(
    *,
    prefer_steer: bool,
    target_bot_id: str,
    message: str,
    sender_bot_id: str,
    max_tokens: int | None,
    temperature: float,
    timeout_seconds: float,
    idempotency_key: str | None,
    project_id: str | None,
    task_id: str | None,
    message_kind: str | None,
    metadata: dict[str, Any] | None,
    session_policy: str | None,
    reset_session_before_delivery: bool | None,
    retain_history: bool | None,
    reset_reason: str | None,
) -> dict:
    body = {
        "target_bot_id": target_bot_id,
        "message": message,
        "sender_bot_id": sender_bot_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout_seconds": timeout_seconds,
        "idempotency_key": idempotency_key,
        "project_id": project_id,
        "task_id": task_id,
        "message_kind": message_kind,
        "metadata": metadata or {},
        "prefer_steer": prefer_steer,
        "session_policy": session_policy,
        "reset_session_before_delivery": reset_session_before_delivery,
        "retain_history": retain_history,
        "reset_reason": reset_reason,
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_APP_BASE_URL}/v1/inter-bot-deliveries",
                json=body,
                timeout=15.0,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            body = exc.response.json()
        except Exception:
            body = None
        if isinstance(body, dict) and body.get("delivery_id"):
            return {**body, "status_code": exc.response.status_code}
        detail: Any = body.get("detail") if isinstance(body, dict) else exc.response.text
        return {
            "success": False,
            "queued": False,
            "delivery": "steer_or_idle" if prefer_steer else "when_idle",
            "bot_id": target_bot_id,
            "sender": sender_bot_id,
            "error": str(detail),
            "status_code": exc.response.status_code,
        }
    except Exception as exc:
        return {
            "success": False,
            "queued": False,
            "delivery": "steer_or_idle" if prefer_steer else "when_idle",
            "bot_id": target_bot_id,
            "sender": sender_bot_id,
            "error": str(exc) or exc.__class__.__name__,
        }


@mcp.tool(name="bots_send_message")
async def send_message_to_bot(
    target_bot_id: str,
    message: str,
    sender_bot_id: str = "unknown",
    max_tokens: int | None = None,
    temperature: float = 0.7,
    fire_and_forget: bool | None = None,
    timeout_seconds: float = 300.0,
    force: bool = False,
    wait_for_reply: bool = False,
    queue_if_busy: bool = False,
    delivery: str | None = None,
    idempotency_key: str | None = None,
    project_id: str | None = None,
    task_id: str | None = None,
    message_kind: str | None = None,
    metadata: dict[str, Any] | None = None,
    session_policy: str | None = None,
    reset_session_before_delivery: bool | None = None,
    retain_history: bool | None = None,
    reset_reason: str | None = None,
) -> dict:
    """Send a message to another bot without creating concurrent agent turns.

    Asynchronous sends are durable by default and return a stable ``delivery_id``.
    If the target has an active steer-capable Claude Code turn, the message is
    persisted and steers that exact run in place. If steering is not ready yet,
    delivery retries in FIFO order. If the target is idle—or the backend cannot
    steer—the delivery starts exactly one normal turn when safe.

    Pass ``delivery="when_idle"`` (or ``queue_if_busy=True``) to explicitly skip
    steering and wait for a fresh idle turn. ``wait_for_reply=True`` remains the
    bounded synchronous compatibility mode. ``force=True`` is accepted for
    compatibility but never permits agent concurrency; asynchronous force sends
    use the same steer-or-safe-deliver contract.
    """
    normalized_delivery = (delivery or "").strip().lower().replace("-", "_")
    if delivery and normalized_delivery not in {
        "when_idle", "queued", "queue_if_busy", "immediate", "steer_or_idle"
    }:
        return {"success": False, "error": f"Unknown delivery mode '{delivery}'"}

    do_wait = wait_for_reply or (fire_and_forget is False)
    if not do_wait:
        prefer_steer = not (
            queue_if_busy
            or normalized_delivery in {"when_idle", "queued", "queue_if_busy"}
        )
        return await _enqueue_durable(
            prefer_steer=prefer_steer,
            target_bot_id=target_bot_id,
            message=message,
            sender_bot_id=sender_bot_id,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=max(timeout_seconds, 1800.0),
            idempotency_key=idempotency_key,
            project_id=project_id,
            task_id=task_id,
            message_kind=message_kind,
            metadata=metadata,
            session_policy=session_policy,
            reset_session_before_delivery=reset_session_before_delivery,
            retain_history=retain_history,
            reset_reason=reset_reason,
        )

    # Waited calls preserve bounded inline response behavior. They never use
    # force to create a second active agent turn.
    force = False
    active = await _compat_hook("_check_bot_in_turn", _check_bot_in_turn)(target_bot_id)
    if active is not None:
        return {
            "success": False,
            "sent": False,
            "in_turn": True,
            "bot_id": target_bot_id,
            "sender": sender_bot_id,
            "content": "",
            "turn_id": active.get("turn_id"),
            "turn_status": active.get("status"),
            "note": (
                f"Agent '{target_bot_id}' is in turn — waited send not started. "
                "Use the default asynchronous mode to durably steer or safely queue it."
            ),
        }

    formatted = message
    if sender_bot_id != "unknown":
        formatted = f"Message from bot '{sender_bot_id}': {message}"
    payload = {
        "messages": [{"role": "user", "content": formatted}],
        "bot_id": target_bot_id,
        "user_message_id": str(uuid.uuid4()),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "extract_memory": False,
        "augment_memory": True,
        "stream": False,
    }

    ceiling = _compat_hook(
        "_bot_send_wait_ceiling_seconds", _bot_send_wait_ceiling_seconds
    )()
    effective_timeout = min(timeout_seconds, ceiling)
    result = await _compat_hook(
        "_dispatch_bot_message", _dispatch_bot_message
    )(payload, target_bot_id, sender_bot_id, effective_timeout)
    if timeout_seconds > ceiling:
        result.update({
            "timeout_clamped": True,
            "requested_timeout_seconds": timeout_seconds,
            "effective_timeout_seconds": effective_timeout,
        })
    return result


@mcp.tool(name="bots_delivery_get")
async def get_delivery(delivery_id: str) -> dict:
    """Inspect one durable inter-bot delivery by stable delivery ID."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{_APP_BASE_URL}/v1/inter-bot-deliveries/{delivery_id}", timeout=10.0
        )
        if response.status_code == 404:
            return {"delivery_id": delivery_id, "error": "not found"}
        response.raise_for_status()
        return response.json()


@mcp.tool(name="bots_deliveries_list")
async def list_deliveries(
    sender_bot_id: str | None = None,
    target_bot_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> dict:
    """List durable inter-bot deliveries and lifecycle/error state."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{_APP_BASE_URL}/v1/inter-bot-deliveries",
            params={
                key: value for key, value in {
                    "sender_bot_id": sender_bot_id,
                    "target_bot_id": target_bot_id,
                    "status": status,
                    "limit": min(max(limit, 1), 200),
                }.items() if value is not None
            },
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()


@mcp.tool(name="bots_delivery_cancel")
async def cancel_delivery(delivery_id: str) -> dict:
    """Cancel a durable delivery while it is still QUEUED."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{_APP_BASE_URL}/v1/inter-bot-deliveries/{delivery_id}/cancel",
            timeout=10.0,
        )
        if response.status_code in (404, 409):
            try:
                detail = response.json().get("detail")
            except Exception:
                detail = response.text
            return {"delivery_id": delivery_id, "error": detail, "status_code": response.status_code}
        response.raise_for_status()
        return response.json()


@mcp.tool(name="agent_context_health")
async def agent_context_health(
    bot_id: str,
    user_id: str | None = None,
) -> dict:
    """Inspect resident context estimate, headroom, thresholds, and capabilities."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{_APP_BASE_URL}/v1/agent-context/health",
            params={k: v for k, v in {"bot_id": bot_id, "user": user_id}.items() if v is not None},
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()


@mcp.tool(name="agent_context_reset")
async def agent_context_reset(
    bot_id: str,
    session_policy: str,
    reason: str = "agent-requested context maintenance",
    user_id: str | None = None,
) -> dict:
    """Safely reset an idle agent session without deleting durable history."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{_APP_BASE_URL}/v1/agent-context/reset",
            json={
                "bot_id": bot_id,
                "user": user_id,
                "session_policy": session_policy,
                "reason": reason,
            },
            timeout=15.0,
        )
        if response.status_code in (409, 422):
            return {"success": False, "status_code": response.status_code, "error": response.json().get("detail")}
        response.raise_for_status()
        return response.json()


@mcp.tool(name="agent_context_compact")
async def agent_context_compact(
    bot_id: str,
    idempotency_key: str,
    sender_bot_id: str = "unknown",
) -> dict:
    """Queue one durable Claude /compact maintenance turn when the bot is idle."""
    key = (idempotency_key or "").strip()
    if not key:
        return {"success": False, "error": "idempotency_key is required"}
    health = await agent_context_health(bot_id)
    if not (health.get("capabilities") or {}).get("compact"):
        return {
            "success": False,
            "error": f"Backend {health.get('backend')!r} does not support compact",
        }
    return await _enqueue_durable(
        prefer_steer=False,
        target_bot_id=bot_id,
        message="/compact",
        sender_bot_id=sender_bot_id,
        max_tokens=None,
        temperature=0.0,
        timeout_seconds=1800.0,
        idempotency_key=key,
        project_id=None,
        task_id=None,
        message_kind="CONTEXT_MAINTENANCE",
        metadata={"context_action": "compact"},
        session_policy="continue",
        reset_session_before_delivery=None,
        retain_history=None,
        reset_reason=None,
    )


@mcp.tool(name="bots_list_available")
async def list_available_bots() -> list[dict]:
    """List bots available as inter-bot message targets."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{_APP_BASE_URL}/v1/bots", timeout=10.0)
            response.raise_for_status()
            data = response.json()
        bots = data.get("data", []) if isinstance(data, dict) else data
        return [
            {
                "slug": bot.get("slug", "unknown"),
                "name": bot.get("name", bot.get("slug", "Unknown")),
                "bot_type": bot.get("bot_type", "chat"),
                "description": bot.get("description", ""),
                "default_model": bot.get("default_model", ""),
                "agent_backend": bot.get("agent_backend"),
            }
            for bot in bots if isinstance(bot, dict)
        ]
    except Exception as exc:
        logger.error("Failed to list available bots: %s", exc)
        return []
