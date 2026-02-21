"""Turn log retrieval routes."""

import json

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import get_service
from ..schemas import ToolCallEvent, ToolCallEventsResponse, TurnLogDetail, TurnLogListItem, TurnLogListResponse
from ..tool_call_events import extract_trigger_message, message_id_matches, parse_message_filters
from ..turn_logs import TurnLogStore

router = APIRouter()


def _parse_json(value: str | None):
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


@router.get("/v1/turn-logs", response_model=TurnLogListResponse, tags=["Debug"])
async def list_turn_logs(
    bot_id: str | None = Query(None, description="Filter by bot ID"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    model: str | None = Query(None, description="Filter by model alias"),
    request_id: str | None = Query(None, description="Filter by request ID"),
    status: str | None = Query(None, description="Filter by status"),
    stream: bool | None = Query(None, description="Filter by streaming mode"),
    has_tools: bool | None = Query(None, description="Filter by presence of tool calls"),
    since_hours: int = Query(168, ge=1, le=168, description="Only include turns from recent N hours"),
    limit: int = Query(100, ge=1, le=200, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """List persisted turn logs (24h TTL by default)."""
    service = get_service()
    store = TurnLogStore(service.config, ttl_hours=168)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")

    rows, total_count = store.list_turns(
        bot_id=bot_id,
        user_id=user_id,
        model=model,
        request_id=request_id,
        status=status,
        stream=stream,
        has_tools=has_tools,
        since_hours=since_hours,
        limit=limit,
        offset=offset,
    )

    items = []
    for row in rows:
        tool_calls = _parse_json(row.tool_calls_json) or []
        response_text = row.response_text or ""
        items.append(
            TurnLogListItem(
                id=row.id,
                created_at=row.created_at,
                request_id=row.request_id,
                path=row.path,
                stream=row.stream,
                model=row.model,
                bot_id=row.bot_id,
                user_id=row.user_id,
                status=row.status,
                latency_ms=row.latency_ms,
                user_prompt=row.user_prompt,
                response_preview=response_text[:300] if response_text else None,
                tool_call_count=len(tool_calls) if isinstance(tool_calls, list) else 0,
                error_text=row.error_text,
            )
        )

    return TurnLogListResponse(
        turns=items,
        total_count=total_count,
        filters={
            "bot_id": bot_id,
            "user_id": user_id,
            "model": model,
            "request_id": request_id,
            "status": status,
            "stream": stream,
            "has_tools": has_tools,
            "since_hours": since_hours,
            "limit": limit,
            "offset": offset,
        },
    )


@router.get("/v1/turn-logs/{turn_id}", response_model=TurnLogDetail, tags=["Debug"])
async def get_turn_log(turn_id: str):
    """Get one persisted turn log by ID."""
    service = get_service()
    store = TurnLogStore(service.config, ttl_hours=168)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")

    row = store.get_turn(turn_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Turn log not found")

    parsed_request = _parse_json(row.request_json)
    parsed_tools = _parse_json(row.tool_calls_json) or []
    if not isinstance(parsed_tools, list):
        parsed_tools = [{"raw": parsed_tools}]

    return TurnLogDetail(
        id=row.id,
        created_at=row.created_at,
        request_id=row.request_id,
        path=row.path,
        stream=row.stream,
        model=row.model,
        bot_id=row.bot_id,
        user_id=row.user_id,
        status=row.status,
        latency_ms=row.latency_ms,
        user_prompt=row.user_prompt,
        request=parsed_request,
        response=row.response_text,
        tool_calls=parsed_tools,
        error_text=row.error_text,
    )


@router.get("/v1/tool-calls", response_model=ToolCallEventsResponse, tags=["Debug"])
async def get_tool_call_events(
    bot_id: str | None = Query(None, description="Filter by bot ID"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    message_id: str | None = Query(None, description="Single message ID to match"),
    message_ids: list[str] | None = Query(None, description="Message IDs (repeat param or CSV)"),
    since_hours: int = Query(168, ge=1, le=168, description="Only include turns from recent N hours"),
    limit: int = Query(200, ge=1, le=1000, description="Max tool-call events to return"),
):
    """List tool-call events keyed by trigger message ID for history annotation."""
    service = get_service()
    store = TurnLogStore(service.config, ttl_hours=168)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")

    target_ids = parse_message_filters(message_id, message_ids)
    rows, _ = store.list_turns(
        bot_id=bot_id,
        user_id=user_id,
        has_tools=True,
        since_hours=since_hours,
        limit=min(limit, 1000),
        offset=0,
    )

    events: list[ToolCallEvent] = []
    for row in rows:
        parsed_tools = _parse_json(row.tool_calls_json) or []
        if not isinstance(parsed_tools, list) or not parsed_tools:
            continue

        parsed_request = _parse_json(row.request_json)
        trigger = extract_trigger_message(parsed_request)
        if trigger is None:
            continue

        trigger_id, trigger_role, trigger_timestamp = trigger
        if not message_id_matches(trigger_id, target_ids):
            continue

        events.append(
            ToolCallEvent(
                turn_id=row.id,
                created_at=row.created_at,
                request_id=row.request_id,
                model=row.model,
                bot_id=row.bot_id,
                user_id=row.user_id,
                message_id=trigger_id,
                message_role=trigger_role,
                message_timestamp=trigger_timestamp,
                tool_call_count=len(parsed_tools),
                tool_calls=parsed_tools,
            )
        )
        if len(events) >= limit:
            break

    events.sort(key=lambda event: event.created_at)
    return ToolCallEventsResponse(
        events=events,
        total_count=len(events),
        filters={
            "bot_id": bot_id,
            "user_id": user_id,
            "message_id": message_id,
            "message_ids": sorted(target_ids),
            "since_hours": since_hours,
            "limit": limit,
        },
    )
