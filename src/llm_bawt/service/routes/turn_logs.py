"""Turn log retrieval routes."""

import json

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func
from sqlmodel import Session, select

from ..dependencies import get_service
from ..schemas import (
    RecentBotTurn,
    RecentByBotsResponse,
    ToolCallEvent,
    ToolCallEventsResponse,
    TurnLogDetail,
    TurnLogListItem,
    TurnLogListResponse,
)
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
            "text_offset": row.text_offset,
            "is_error": row.is_error,
            # TASK-344: SDK ids so sub-agent nesting survives reload.
            "tool_use_id": row.tool_use_id,
            "parent_tool_use_id": row.parent_tool_use_id,
        })
    return out


def _enrich_calls_with_timing(calls: list[dict], records: list[ToolCallRecord]) -> None:
    """Stamp each call with real per-call ``started_at``/``ended_at``/``duration_ms``.

    Sourced from ``tool_call_records``, which are written *incrementally* as
    tool_start/tool_end events fire (via the Redis event sink) — so their
    timestamps survive mid-stream reloads and terminated turns, unlike a
    finalize-time write. Without this every call in a turn inherits the
    event-level ``created_at`` (the turn start) and the UI renders N cards
    with one identical timestamp.

    Calls with no matching record (legacy rows, non-agent turns) are left
    untouched; the frontend then falls back to the event-level created_at.
    """
    if not records:
        return

    by_call_id = {r.call_id: r for r in records if r.call_id}
    by_iter_name: dict[tuple, list[ToolCallRecord]] = {}
    for rec in records:
        by_iter_name.setdefault((rec.iteration or 1, rec.tool_name), []).append(rec)
    cursors: dict[tuple, int] = {}

    def find(call: dict) -> ToolCallRecord | None:
        cid = call.get("call_id")
        if cid and cid in by_call_id:
            return by_call_id[cid]
        key = (call.get("iteration", 1), call.get("name") or call.get("tool"))
        bucket = by_iter_name.get(key)
        if not bucket:
            return None
        i = cursors.get(key, 0)
        if i < len(bucket):
            cursors[key] = i + 1
            return bucket[i]
        return None

    for call in calls:
        rec = find(call)
        if rec is None:
            continue
        if rec.started_at is not None:
            call["started_at"] = rec.started_at
        if rec.ended_at is not None:
            call["ended_at"] = rec.ended_at
        if rec.duration_ms is not None:
            call["duration_ms"] = rec.duration_ms


def _live_tool_call_counts(store: TurnLogStore, turn_ids: list[str]) -> dict[str, int]:
    """Count realtime tool-call rows for multiple turns in one query."""
    if store.engine is None or not turn_ids:
        return {}
    try:
        with Session(store.engine) as session:
            rows = session.exec(
                select(ToolCallRecord.turn_id, func.count())
                .where(ToolCallRecord.turn_id.in_(turn_ids))
                .group_by(ToolCallRecord.turn_id)
            ).all()
    except Exception:
        return {}

    counts: dict[str, int] = {}
    for turn_id, count in rows:
        if turn_id:
            counts[str(turn_id)] = int(count or 0)
    return counts


_TAIL_MAX = 1500
_TAIL_MIN = 200


def _response_tail(response_text: str) -> str | None:
    """Extract the bot's final output from a (potentially long) response.

    Strategy (in priority order):
    1. If the response is short enough, return it whole.
    2. Find the last markdown heading (## or #) — the final summary section.
       If the section is too short (< _TAIL_MIN), include the preceding heading
       section too so the reader gets enough context.
    3. Fallback: last _TAIL_MAX chars trimmed to the nearest paragraph break.
    """
    if not response_text:
        return None
    text = response_text.rstrip()
    if len(text) <= _TAIL_MAX:
        return text

    # Strategy 2: find the last markdown heading
    # Search from the end for lines starting with # or ##
    lines = text.split("\n")
    last_heading_idx = -1
    prev_heading_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].lstrip()
        if stripped.startswith("#"):
            if last_heading_idx == -1:
                last_heading_idx = i
            elif prev_heading_idx == -1:
                prev_heading_idx = i
                break

    if last_heading_idx >= 0:
        section = "\n".join(lines[last_heading_idx:]).strip()
        # If too short, pull in the previous heading section
        if len(section) < _TAIL_MIN and prev_heading_idx >= 0:
            section = "\n".join(lines[prev_heading_idx:]).strip()
        if section and len(section) <= _TAIL_MAX:
            return section

    # Strategy 3: fallback — last _TAIL_MAX chars, trimmed to paragraph break
    tail = text[-_TAIL_MAX:]
    pp = tail.find("\n\n")
    if 0 < pp < len(tail) - _TAIL_MIN:
        tail = tail[pp + 2:]
    return tail


@router.get("/v1/turn-logs", response_model=TurnLogListResponse, tags=["Debug"])
def list_turn_logs(
    bot_id: str | None = Query(None, description="Filter by bot ID"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    model: str | None = Query(None, description="Filter by model alias"),
    request_id: str | None = Query(None, description="Filter by request ID"),
    status: str | None = Query(None, description="Filter by status"),
    active_only: bool = Query(False, description="Only in-progress turns (ended_at IS NULL), path-agnostic"),
    stream: bool | None = Query(None, description="Filter by streaming mode"),
    has_tools: bool | None = Query(None, description="Filter by presence of tool calls"),
    since_hours: int = Query(168, ge=1, le=168, description="Only include turns from recent N hours"),
    limit: int = Query(100, ge=1, le=200, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """List persisted turn logs (24h TTL by default)."""
    service = get_service()
    store = TurnLogStore(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")

    rows, total_count = store.list_turns(
        bot_id=bot_id,
        user_id=user_id,
        model=model,
        request_id=request_id,
        status=status,
        active_only=active_only,
        stream=stream,
        has_tools=has_tools,
        since_hours=since_hours,
        limit=limit,
        offset=offset,
    )

    parsed_tool_calls_by_turn: dict[str, list] = {}
    turns_needing_live_counts: list[str] = []
    for row in rows:
        tool_calls = _parse_json(row.tool_calls_json) or []
        if isinstance(tool_calls, list) and tool_calls:
            parsed_tool_calls_by_turn[row.id] = tool_calls
        else:
            parsed_tool_calls_by_turn[row.id] = []
            turns_needing_live_counts.append(row.id)

    live_tool_counts = _live_tool_call_counts(store, turns_needing_live_counts)

    # TASK-269: embed the persisted question for turns that ended on one, so the
    # UI can render a QuestionMessage straight from history hydration.
    pq_store = getattr(service, "_pending_question_store", None)
    questions_by_turn: dict[str, dict] = {}
    if pq_store is not None:
        for row in rows:
            if getattr(row, "end_reason", None) != "question":
                continue
            qrow = None
            qid = getattr(row, "question_id", None)
            if qid:
                qrow = pq_store.get(qid)
            if qrow is None:
                qrow = pq_store.get_by_turn(row.id)
            if qrow is not None:
                questions_by_turn[row.id] = pq_store.row_to_dict(qrow)

    items = []
    for row in rows:
        tool_calls = parsed_tool_calls_by_turn.get(row.id, [])
        tool_call_count = len(tool_calls) if tool_calls else live_tool_counts.get(row.id, 0)
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
                response_tail=_response_tail(response_text),
                response_chars=len(response_text),
                response_preview_truncated=bool(response_text and len(response_text) > len(response_preview or "")),
                tool_call_count=tool_call_count,
                error_text=row.error_text,
                animation=getattr(row, "animation", None),
                agent_session_key=getattr(row, "agent_session_key", None),
                agent_request_id=getattr(row, "agent_request_id", None),
                trigger_message_id=getattr(row, "trigger_message_id", None),
                token_usage=_parse_token_usage(getattr(row, "token_usage_json", None)),
                end_reason=getattr(row, "end_reason", None),
                question_id=getattr(row, "question_id", None),
                parent_turn_id=getattr(row, "parent_turn_id", None),
                ended_at=getattr(row, "ended_at", None),
                question=questions_by_turn.get(row.id),
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
            "active_only": active_only,
            "stream": stream,
            "has_tools": has_tools,
            "since_hours": since_hours,
            "limit": limit,
            "offset": offset,
        },
    )


@router.get(
    "/v1/turn-logs/recent-by-bot",
    response_model=RecentByBotsResponse,
    tags=["Debug", "History"],
)
def recent_turns_by_bot(
    user_id: str = Query(..., description="User to scope by (required)"),
    bot_ids: str | None = Query(
        None,
        description=(
            "Comma-separated bot slugs to restrict to. Omit to return the latest "
            "turn for every bot this user has talked to in the window."
        ),
    ),
    since_hours: int = Query(
        168, ge=1, le=168, description="Only consider turns from the recent N hours (max 7d)."
    ),
    preview_chars: int = Query(
        180,
        ge=20,
        le=600,
        description="Trim user_prompt / response_text to this many characters.",
    ),
):
    """Return one "most recent turn" summary per bot in a single round trip.

    Replaces the per-bot dashboard fan-out (one ``/v1/history`` call per bot)
    with a single ``DISTINCT ON (bot_id)`` query. Useful for "who did I talk
    to recently, and what about?" UIs where each tile needs:

      - last activity timestamp + latency
      - last user prompt + assistant response preview
      - tool call count for the turn (incl. in-flight)
      - model used + token usage

    Bots in ``bot_ids`` that have no turn in the window are simply absent
    from the response — callers detect "no activity" by checking which slugs
    are missing from the returned ``turns`` list.
    """
    service = get_service()
    store = TurnLogStore(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")

    bot_list: list[str] | None = None
    if bot_ids:
        bot_list = [b.strip() for b in bot_ids.split(",") if b.strip()]
        # Empty after parsing → treat as "no filter" instead of "match nothing".
        if not bot_list:
            bot_list = None

    rows = store.recent_turns_by_bot(
        user_id=user_id,
        bot_ids=bot_list,
        since_hours=since_hours,
    )

    # In-flight turns haven't written tool_calls_json yet — fall back to the
    # live tool_call_records table so the count reflects what's running NOW.
    turns_needing_live: list[str] = []
    parsed_tools_by_turn: dict[str, list] = {}
    for row in rows:
        parsed = _parse_json(row.tool_calls_json) or []
        if isinstance(parsed, list) and parsed:
            parsed_tools_by_turn[row.id] = parsed
        else:
            parsed_tools_by_turn[row.id] = []
            turns_needing_live.append(row.id)

    live_counts = _live_tool_call_counts(store, turns_needing_live)

    items: list[RecentBotTurn] = []
    for row in rows:
        finalized_tools = parsed_tools_by_turn.get(row.id, [])
        tool_call_count = (
            len(finalized_tools) if finalized_tools else live_counts.get(row.id, 0)
        )
        user_prompt = (row.user_prompt or "")[:preview_chars] or None
        response_text = row.response_text or ""
        response_preview = response_text[:preview_chars] if response_text else None
        items.append(
            RecentBotTurn(
                bot_id=row.bot_id or "",
                turn_id=row.id,
                created_at=row.created_at,
                model=row.model,
                status=row.status,
                latency_ms=row.latency_ms,
                user_prompt_preview=user_prompt,
                response_preview=response_preview,
                response_tail=_response_tail(response_text),
                response_chars=len(response_text),
                response_preview_truncated=bool(
                    response_text and response_preview and len(response_text) > len(response_preview)
                ),
                tool_call_count=tool_call_count,
                token_usage=_parse_token_usage(getattr(row, "token_usage_json", None)),
                trigger_message_id=getattr(row, "trigger_message_id", None),
            )
        )

    # Sort newest-first so the caller doesn't have to.
    items.sort(key=lambda r: r.created_at, reverse=True)
    return RecentByBotsResponse(turns=items)


@router.get("/v1/bots/{bot_id}/in-turn", tags=["Bots"])
def bot_in_turn(
    bot_id: str,
    within_seconds: int = Query(
        1800, ge=1, le=86400,
        description="Only count in-flight turns started within this window (zombie guard).",
    ),
):
    """Report whether a bot is currently in a turn (path-agnostic).

    ``in_turn`` is true iff the bot has a turn-log row with ``ended_at IS NULL``
    inside the window — the same single source of truth for streaming and
    non-streaming, local and agent-backend turns.
    """
    service = get_service()
    store = TurnLogStore(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")

    turn = store.active_turn_for_bot(bot_id, within_seconds=within_seconds)
    if turn is None:
        return {"bot_id": bot_id, "in_turn": False}
    return {
        "bot_id": bot_id,
        "in_turn": True,
        "turn_id": turn.id,
        "status": turn.status,
        "model": turn.model,
        "started_at": turn.created_at.isoformat() if turn.created_at else None,
    }


@router.get("/v1/turn-logs/{turn_id}", response_model=TurnLogDetail, tags=["Debug"])
def get_turn_log(turn_id: str):
    """Get one persisted turn log by ID."""
    service = get_service()
    store = TurnLogStore(service.config)
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
        reasoning=getattr(row, "reasoning", None),
        tool_calls=parsed_tools,
        error_text=row.error_text,
        animation=getattr(row, "animation", None),
        agent_session_key=getattr(row, "agent_session_key", None),
        agent_request_id=getattr(row, "agent_request_id", None),
        token_usage=_parse_token_usage(getattr(row, "token_usage_json", None)),
        end_reason=getattr(row, "end_reason", None),
        question_id=getattr(row, "question_id", None),
        parent_turn_id=getattr(row, "parent_turn_id", None),
        question=_question_for_turn(service, row),
    )


def _question_for_turn(service, row) -> dict | None:
    """Embedded question dict for a turn that ended on one (TASK-269)."""
    if getattr(row, "end_reason", None) != "question":
        return None
    pq_store = getattr(service, "_pending_question_store", None)
    if pq_store is None:
        return None
    qrow = None
    qid = getattr(row, "question_id", None)
    if qid:
        qrow = pq_store.get(qid)
    if qrow is None:
        qrow = pq_store.get_by_turn(row.id)
    return pq_store.row_to_dict(qrow) if qrow is not None else None


@router.get("/v1/tool-calls", response_model=ToolCallEventsResponse, tags=["Debug"])
def get_tool_call_events(
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
    store = TurnLogStore(service.config)
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

    # Batch-fetch real per-call timing for every turn in one query so each
    # tool call can carry its source event timestamp (started_at/ended_at)
    # instead of all sharing the turn-level created_at.
    records_by_turn: dict[str, list[ToolCallRecord]] = {}
    turn_ids = [r.id for r in rows if r.id]
    if store.engine is not None and turn_ids:
        try:
            with Session(store.engine) as session:
                rec_rows = list(
                    session.exec(
                        select(ToolCallRecord).where(ToolCallRecord.turn_id.in_(turn_ids))
                    ).all()
                )
            for rec in rec_rows:
                if rec.turn_id:
                    records_by_turn.setdefault(rec.turn_id, []).append(rec)
        except Exception:
            pass

    events: list[ToolCallEvent] = []
    for row in rows:
        parsed_tools = _parse_json(row.tool_calls_json) or []
        if not isinstance(parsed_tools, list) or not parsed_tools:
            # Fallback to realtime tool_call_records for in-flight turns so
            # the UI can render tool activity before the turn finalizes.
            parsed_tools = _live_tool_calls(store, row.id)
        if not parsed_tools:
            continue

        # Attach real per-call timestamps (incremental source records).
        _enrich_calls_with_timing(parsed_tools, records_by_turn.get(row.id, []))

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
