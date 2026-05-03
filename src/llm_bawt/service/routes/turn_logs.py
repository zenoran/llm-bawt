"""Turn log retrieval routes."""

import json

from fastapi import APIRouter, HTTPException, Query
from sqlmodel import Session, select

from ..dependencies import get_service
from ..schemas import ToolCallEvent, ToolCallEventsResponse, TurnLogDetail, TurnLogListItem, TurnLogListResponse
from ..tool_call_events import extract_trigger_message, message_id_matches, parse_message_filters
from ..turn_logs import ToolCallRecord, TurnLogStore

router = APIRouter()


def _parse_json(value: str | None):
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def _parse_token_usage(value: str | None) -> dict | None:
    """Decode token_usage_json from a turn-log row, returning None on bad JSON.

    The column may be missing on rows persisted before this field was added —
    `getattr(row, "token_usage_json", None)` already handles that case; here we
    just guard against malformed JSON so a single bad row can't 500 the route.
    """
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _live_tool_calls(store: TurnLogStore, turn_id: str) -> list[dict]:
    """Read realtime tool-call rows from `tool_call_records` for a turn.

    Tool events are persisted incrementally as they fire (via Redis sink),
    while `turn_logs.tool_calls_json` is only written at turn finalize.
    During an in-flight stream this is the only source of partial activity.
    """
    if store.engine is None or not turn_id:
        return []
    try:
        with Session(store.engine) as session:
            rows = list(
                session.exec(
                    select(ToolCallRecord)
                    .where(ToolCallRecord.turn_id == turn_id)
                    .order_by(ToolCallRecord.created_at)
                ).all()
            )
    except Exception:
        return []

    out: list[dict] = []
    for row in rows:
        args = _parse_json(row.arguments_json) or {}
        if not isinstance(args, dict):
            args = {"raw": args}
        # In-flight calls (no ended_at) MUST have result=None so the frontend
        # renders "running…".  Empty string would be treated as completed.
        is_finished = row.ended_at is not None or (row.result_text is not None and row.result_text != "")
        result_value = row.result_text if is_finished else None
        out.append({
            "iteration": row.iteration or 1,
            "index": len(out) + 1,
            "tool": row.tool_name,
            "name": row.tool_name,
            "arguments": args,
            "parameters": args,
            "result": result_value,
            "status": "completed" if is_finished else "pending",
            "call_id": row.call_id,
            "started_at": row.started_at,
            "ended_at": row.ended_at,
            "duration_ms": row.duration_ms,
        })
    return out


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
        if not isinstance(tool_calls, list) or not tool_calls:
            # tool_calls_json is filled at finalize. Streaming/in-flight turns
            # have realtime rows in tool_call_records — use those for the count
            # so the UI sees an accurate tool count immediately.
            tool_calls = _live_tool_calls(store, row.id)
        response_text = row.response_text or ""
        response_preview = response_text[:300] if response_text else None
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
                response_preview=response_preview,
                response_chars=len(response_text),
                response_preview_truncated=bool(response_text and len(response_text) > len(response_preview or "")),
                tool_call_count=len(tool_calls) if isinstance(tool_calls, list) else 0,
                error_text=row.error_text,
                animation=getattr(row, "animation", None),
                agent_session_key=getattr(row, "agent_session_key", None),
                agent_request_id=getattr(row, "agent_request_id", None),
                trigger_message_id=getattr(row, "trigger_message_id", None),
                token_usage=_parse_token_usage(getattr(row, "token_usage_json", None)),
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
    if not parsed_tools:
        # Fallback to realtime rows for in-flight turns.
        parsed_tools = _live_tool_calls(store, row.id)

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
        animation=getattr(row, "animation", None),
        agent_session_key=getattr(row, "agent_session_key", None),
        agent_request_id=getattr(row, "agent_request_id", None),
        token_usage=_parse_token_usage(getattr(row, "token_usage_json", None)),
    )


@router.get("/v1/tool-calls", response_model=ToolCallEventsResponse, tags=["Debug"])
async def get_tool_call_events(
    bot_id: str | None = Query(None, description="Filter by bot ID"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    message_id: str | None = Query(None, description="Single message ID to match"),
    message_ids: list[str] | None = Query(None, description="Message IDs (repeat param or CSV)"),
    after: float | None = Query(None, description="Only include turns created after this unix timestamp"),
    before: float | None = Query(None, description="Only include turns created before this unix timestamp"),
    since_hours: int = Query(168, ge=1, le=168, description="Only include turns from recent N hours (ignored if after/before set)"),
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
        # Don't filter by has_tools at the DB level: streaming/in-flight turns
        # have empty tool_calls_json but live rows in tool_call_records.  We
        # filter rows that produce zero events in the loop below.
        has_tools=None,
        trigger_message_ids=target_ids or None,
        after=after,
        before=before,
        since_hours=since_hours,
        limit=min(limit, 1000),
        offset=0,
    )

    events: list[ToolCallEvent] = []
    for row in rows:
        parsed_tools = _parse_json(row.tool_calls_json) or []
        if not isinstance(parsed_tools, list) or not parsed_tools:
            # Fallback to realtime tool_call_records for in-flight turns so
            # the UI can render tool activity before the turn finalizes.
            parsed_tools = _live_tool_calls(store, row.id)
        if not parsed_tools:
            continue

        trigger_id = row.trigger_message_id
        trigger_role = "user"
        trigger_timestamp: float | None = None
        if not trigger_id:
            parsed_request = _parse_json(row.request_json)
            trigger = extract_trigger_message(parsed_request)
            if trigger is not None:
                trigger_id, trigger_role, trigger_timestamp = trigger
                if target_ids and not message_id_matches(trigger_id, target_ids):
                    continue
            else:
                # Agent-backend turns store trigger_message_id directly.
                # If missing (legacy rows), fall back to the turn's own ID
                # so tool calls are still returned — the frontend maps them
                # via turn_id prefix matching.
                trigger_id = row.id
                trigger_timestamp = row.created_at.timestamp() if row.created_at else None

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
            "after": after,
            "before": before,
            "limit": limit,
        },
    )
