"""History and summarization routes."""

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import get_service
from ..logging import get_service_logger
from ..schemas import (
    DeleteSummaryResponse,
    HistoryClearResponse,
    HistoryMessage,
    HistoryResponse,
    HistorySearchAllMessage,
    HistorySearchAllResponse,
    HistorySearchResponse,
    ListSummariesResponse,
    SummarizableSession,
    SummarizePreviewResponse,
    SummarizeResponse,
    SummaryInfo,
)

router = APIRouter()
log = get_service_logger(__name__)


def _fetch_attachments_via_shared_engine(
    config,
    bot_id: str,
    message_ids: list[str],
) -> dict[str, list[dict]]:
    """Direct-SQL read of ``{bot}_messages.attachments`` for server-mode.

    The route normally calls into ``PostgreSQLMemoryBackend.get_attachments_for_message_ids``
    via the embedded short-term manager, but MCP-server-mode deployments
    have no backend handle. This helper bypasses MCP and reads through
    the process-wide shared engine instead — same data, one round-trip,
    no new RPC. Returns the same ``{message_id: [refs]}`` shape.
    """
    from sqlalchemy import text
    from ...media.assets import _build_engine
    from ...memory.postgresql import _sanitize_table_name

    engine = _build_engine(config)
    if engine is None:
        return {mid: [] for mid in message_ids}

    table = f"{_sanitize_table_name(bot_id)}_messages"
    sql = text(f"SELECT id, attachments FROM {table} WHERE id = ANY(:ids)")
    result: dict[str, list[dict]] = {mid: [] for mid in message_ids}
    with engine.connect() as conn:
        rows = conn.execute(sql, {"ids": list(message_ids)}).mappings().all()
        for row in rows:
            refs = row.get("attachments") or []
            if isinstance(refs, list):
                result[row["id"]] = list(refs)
    return result


def _hydrate_attachments_for_page(
    service,
    bot_id: str,
    page_messages: list[dict],
) -> dict[str, list[dict]]:
    """Resolve per-message attachments for the given page (TASK-226).

    Returns ``{message_id: [resolved_attachment_dicts]}``. Messages with no
    media map to ``[]`` so callers can index unconditionally. The route
    drops orphan ``asset_id`` refs silently (already logged inside the
    serializer) — partial DB deletes never take the whole page down.

    Two DB round-trips per page regardless of message count:

    1. ``SELECT id, attachments FROM {bot}_messages WHERE id = ANY(...)``
    2. ``SELECT * FROM media_assets WHERE id = ANY(...)`` (only if the
       page references any assets at all).
    """
    message_ids = [str(m.get("id")) for m in page_messages if m.get("id")]
    if not message_ids:
        return {}

    # In embedded mode the short-term manager wraps a
    # PostgreSQLMemoryBackend we can call directly. In MCP-server mode
    # the manager has no ``_backend`` handle, so we fall back to a thin
    # direct SQL read against ``{bot}_messages.attachments`` using the
    # process-wide shared engine — same DB, just bypassing the MCP RPC
    # layer for this one read-only lookup. Both paths return the same
    # ``{message_id: [refs]}`` shape so downstream enrichment is unchanged.
    try:
        client = service.get_memory_client(bot_id)
        if not client:
            return {mid: [] for mid in message_ids}
        manager = client.get_short_term_manager()
        backend = getattr(manager, "_backend", None)

        if backend is not None:
            raw_refs_by_msg = backend.get_attachments_for_message_ids(message_ids)
        else:
            raw_refs_by_msg = _fetch_attachments_via_shared_engine(
                service.config, bot_id, message_ids
            )
    except Exception as e:
        log.warning("Failed to load attachment refs for history page: %s", e)
        return {mid: [] for mid in message_ids}

    # Run the cross-row enrichment through the canonical serializer so
    # the shape stays in lockstep with /v1/uploads and the chat-streaming
    # persistence path (TASK-225). The serializer mutates the wrapper
    # dicts in place; we use throwaway shells then extract.
    from ...media.assets import MediaAssetStore
    from ...media.serializers import enrich_attachments_for_messages

    shells: list[dict] = [
        {"_mid": mid, "attachments": raw_refs_by_msg.get(mid, [])}
        for mid in message_ids
    ]
    try:
        asset_store = MediaAssetStore(service.config)
        enrich_attachments_for_messages(shells, asset_store)
    except Exception as e:
        log.warning("Attachment enrichment failed: %s", e)
        return {mid: [] for mid in message_ids}

    return {s["_mid"]: s.get("attachments") or [] for s in shells}


def _fetch_reasoning_via_shared_engine(
    config,
    bot_id: str,
    message_ids: list[str],
) -> dict[str, str | None]:
    """Direct-SQL read of ``{bot}_messages.reasoning`` for server-mode (TASK-301).

    Mirror of ``_fetch_attachments_via_shared_engine`` for the reasoning column —
    used when the short-term manager has no embedded backend handle. Returns
    ``{message_id: reasoning_or_None}``.
    """
    from sqlalchemy import text
    from ...media.assets import _build_engine
    from ...memory.postgresql import _sanitize_table_name

    engine = _build_engine(config)
    if engine is None:
        return {mid: None for mid in message_ids}

    table = f"{_sanitize_table_name(bot_id)}_messages"
    sql = text(f"SELECT id, reasoning FROM {table} WHERE id = ANY(:ids)")
    result: dict[str, str | None] = {mid: None for mid in message_ids}
    with engine.connect() as conn:
        rows = conn.execute(sql, {"ids": list(message_ids)}).mappings().all()
        for row in rows:
            result[row["id"]] = row.get("reasoning")
    return result


def _hydrate_reasoning_for_page(
    service,
    bot_id: str,
    page_messages: list[dict],
) -> dict[str, str | None]:
    """Resolve per-message reasoning for the given page (TASK-301).

    Returns ``{message_id: reasoning_or_None}``. One focused read of a column
    the canonical ``get_messages`` path drops so reasoning never re-enters LLM
    context; the chat UI uses it to restore the collapsed "Thought process" lane
    on reload. Embedded mode reads via the backend; server mode falls back to the
    shared engine — same shape either way.
    """
    message_ids = [str(m.get("id")) for m in page_messages if m.get("id")]
    if not message_ids:
        return {}

    try:
        client = service.get_memory_client(bot_id)
        if not client:
            return {mid: None for mid in message_ids}
        manager = client.get_short_term_manager()
        backend = getattr(manager, "_backend", None)
        if backend is not None:
            return backend.get_reasoning_for_message_ids(message_ids)
        return _fetch_reasoning_via_shared_engine(service.config, bot_id, message_ids)
    except Exception as e:
        log.warning("Failed to load reasoning for history page: %s", e)
        return {mid: None for mid in message_ids}


# Hard ceiling for summarization prompts regardless of what the model advertises.
# Large advertised windows (e.g. 2M) are rarely practical for dense output tasks.
_MAX_SUMMARIZATION_PROMPT_TOKENS = 512_000


def _resolve_summarization_limits(service, model_alias: str | None) -> tuple[int, int]:
    """Return (max_prompt_tokens, max_chunk_tokens) tuned for the active model."""
    context_window = int(service.config.get_model_context_window(model_alias) or 0)

    if context_window <= 0:
        # Conservative fallback for unknown models.
        return (6000, 4000)

    response_budget = 512
    safety_margin = max(2048, int(context_window * 0.05))
    max_prompt_tokens = max(6000, context_window - response_budget - safety_margin)

    # Clamp to practical ceiling.
    max_prompt_tokens = min(max_prompt_tokens, _MAX_SUMMARIZATION_PROMPT_TOKENS)

    # Disable chunking for very large context models.
    if context_window >= 200_000:
        max_chunk_tokens = 0
    else:
        max_chunk_tokens = max(4000, min(64_000, int(max_prompt_tokens * 0.7)))

    return (max_prompt_tokens, max_chunk_tokens)


def _invalidate_bot_history_cache(service, bot_id: str) -> None:
    """Invalidate in-memory history views for a bot after summary writes."""
    for (_model, cached_bot_id, _user), llm_bawt in service._llm_bawt_cache.items():
        if cached_bot_id != bot_id:
            continue
        invalidate = getattr(llm_bawt, "invalidate_history_cache", None)
        if callable(invalidate):
            invalidate()


def _build_summary_callable(service, bot_id: str, user_id: str = "system", model: str | None = None):
    """Create a summarization callable using the service's model resolution path."""
    from ...memory.summarization import (
        _extract_json_object,
        compress_structured_summary_text,
        format_session_for_summarization,
        is_summary_low_quality,
    )
    from ...models.message import Message
    from ...prompt_registry import PromptResolver

    requested_model = (
        model
        or getattr(service.config, "MAINTENANCE_MODEL", None)
        or getattr(service.config, "SUMMARIZATION_MODEL", None)
    )
    model_alias = None
    client = None
    max_prompt_tokens = 6000
    max_chunk_tokens = 4000
    quality_retry_enabled = bool(getattr(service.config, "SUMMARIZATION_QUALITY_RETRY", False))

    try:
        model_alias, _ = service._resolve_request_model(
            requested_model,
            bot_id,
            local_mode=False,
        )
        llm_bawt = service._get_llm_bawt(
            model_alias=model_alias,
            bot_id=bot_id,
            user_id=user_id,
            local_mode=False,
        )
        client = llm_bawt.client
        max_prompt_tokens, max_chunk_tokens = _resolve_summarization_limits(service, model_alias)
        log.info("History summarization route using model: %s", model_alias)
        log.info(
            "History summarization limits: prompt<=%s tokens, chunk<=%s tokens",
            max_prompt_tokens,
            "disabled" if max_chunk_tokens <= 0 else max_chunk_tokens,
        )
    except Exception as e:
        log.error(f"Failed to resolve/load model for history summarization route: {e}")
        client = None

    prompt_resolver = PromptResolver(service.config)

    def summarize_with_loaded_client(session) -> str | None:
        if not client:
            return None

        conversation_text = format_session_for_summarization(session)
        prompt = prompt_resolver.render(
            key="history.summarization.single",
            variables={"messages": conversation_text},
        )
        estimated_tokens = len(prompt) // 4
        log.debug(
            "Per-session summarization: %s messages, ~%s estimated prompt tokens",
            session.message_count,
            f"{estimated_tokens:,}",
        )
        if estimated_tokens > max_prompt_tokens:
            log.warning(
                "Session too large (%s estimated tokens > %s), needs chunking",
                estimated_tokens,
                max_prompt_tokens,
            )
            return None

        try:
            messages = [
                Message(
                    role="system",
                    content="You are a helpful assistant that summarizes conversations concisely.",
                ),
                Message(role="user", content=prompt),
            ]
            response = client.query(
                messages=messages,
                max_tokens=520,
                temperature=0.3,
                plaintext_output=True,
                stream=False,
            )
            if not response:
                return None
            error_indicators = ["Error:", "exception occurred", "exceed context window", "tokens exceed"]
            if any(indicator.lower() in response.lower() for indicator in error_indicators):
                log.error(f"LLM returned error: {response[:100]}...")
                return None
            response_text = compress_structured_summary_text(response.strip(), source_session=session)
            if quality_retry_enabled and is_summary_low_quality(response_text, source_session=session):
                revision_prompt = (
                    "Your previous summary was too vague. Rewrite with concrete specifics from source material only. "
                    "Explicitly name entities, actions taken, tools/searches used, and unresolved items. "
                    "Keep required sections including Key Details.\n\n"
                    f"Original conversation:\n{conversation_text}\n\n"
                    f"Previous summary:\n{response_text}"
                )
                revised = client.query(
                    messages=[
                        Message(
                            role="system",
                            content="You are a precise summarizer. Avoid generic phrasing and preserve concrete facts.",
                        ),
                        Message(role="user", content=revision_prompt),
                    ],
                    max_tokens=620,
                    temperature=0.1,
                    plaintext_output=True,
                    stream=False,
                )
                if revised and not is_summary_low_quality(revised, source_session=session):
                    return compress_structured_summary_text(revised.strip(), source_session=session)
            return response_text
        except Exception as e:
            log.error(f"LLM summarization failed: {e}")
            return None

    def summarize_batch_with_loaded_client(sessions) -> dict[int, str] | None:
        if not client or not sessions:
            return None

        try:
            session_lines: list[str] = []
            for idx, session in enumerate(sessions, start=1):
                conversation_text = format_session_for_summarization(session)
                session_lines.append(
                    "\n".join(
                        [
                            f"### SESSION {idx}",
                            f"SESSION_INDEX: {idx}",
                            f"MESSAGE_COUNT: {session.message_count}",
                            "CONVERSATION:",
                            conversation_text,
                        ]
                    )
                )

            sessions_blob = "\n\n".join(session_lines)
            prompt = prompt_resolver.render(
                key="history.summarization.batch",
                variables={"sessions_blob": sessions_blob},
            )
            estimated_tokens = len(prompt) // 4
            log.info(
                "Batch summarization: %s sessions, ~%s estimated prompt tokens (limit %s)",
                len(sessions),
                f"{estimated_tokens:,}",
                f"{max_prompt_tokens:,}",
            )
            if estimated_tokens > max_prompt_tokens:
                log.warning(
                    "Batch too large (%s estimated tokens > %s), using per-session fallback",
                    estimated_tokens,
                    max_prompt_tokens,
                )
                return None

            model_max_tokens = int(service.config.get_model_max_tokens(model_alias) or 4096)
            requested_output_tokens = max(900, min(model_max_tokens, 320 * len(sessions)))

            # xAI OpenAI-compatible endpoint is typically most reliable with json_object mode.
            response_format = {"type": "json_object"}

            messages = [
                Message(
                    role="system",
                    content="You are a helpful assistant that summarizes conversations with strict JSON output.",
                ),
                Message(role="user", content=prompt),
            ]
            response = client.query(
                messages=messages,
                max_tokens=requested_output_tokens,
                temperature=0.2,
                plaintext_output=True,
                stream=False,
                response_format=response_format,
            )
            if not response:
                return None

            payload = _extract_json_object(response)
            if not payload:
                log.error("Batch summarization returned non-JSON output; retrying without response_format")
                retry_response = client.query(
                    messages=messages,
                    max_tokens=requested_output_tokens,
                    temperature=0.1,
                    plaintext_output=True,
                    stream=False,
                )
                payload = _extract_json_object(retry_response or "")
                if not payload:
                    preview = (retry_response or response or "")[:200].replace("\n", " ")
                    log.error("Batch summarization still non-JSON after retry: %s", preview)
                    return None

            summaries = payload.get("summaries")
            if not isinstance(summaries, list):
                log.error("Batch summarization JSON missing 'summaries' list")
                return None

            mapped: dict[int, str] = {}
            for item in summaries:
                if not isinstance(item, dict):
                    continue
                try:
                    session_index = int(item.get("session_index"))
                except (TypeError, ValueError):
                    continue
                if session_index < 1 or session_index > len(sessions):
                    continue

                summary = str(item.get("summary", "")).strip()
                key_details = str(item.get("key_details", "")).strip()
                intent = str(item.get("intent", "")).strip()
                tone = str(item.get("tone", "")).strip()
                open_loops = str(item.get("open_loops", "")).strip()
                cross_links = str(item.get("cross_session_links", "")).strip()
                if cross_links:
                    open_loops = f"{open_loops} Cross-session links: {cross_links}".strip()

                mapped[session_index] = compress_structured_summary_text(
                    "\n".join(
                        [
                            f"Summary: {summary}".rstrip(),
                            f"Key Details: {key_details}".rstrip(),
                            f"Intent: {intent}".rstrip(),
                            f"Tone: {tone}".rstrip(),
                            f"Open Loops: {open_loops}".rstrip(),
                        ]
                    ).strip(),
                    source_session=sessions[session_index - 1],
                )

            # Drop vague batch items so they get heuristic fallback instead.
            pre_filter_count = len(mapped)
            mapped = {
                idx: txt
                for idx, txt in mapped.items()
                if not is_summary_low_quality(txt, source_session=sessions[idx - 1] if idx <= len(sessions) else None)
            }
            if pre_filter_count != len(mapped):
                log.info(
                    "Batch quality filter: %s/%s sessions passed (%s filtered)",
                    len(mapped),
                    pre_filter_count,
                    pre_filter_count - len(mapped),
                )

            return mapped if mapped else None
        except Exception as e:
            log.error(f"Batch LLM summarization failed: {e}")
            return None

    return summarize_with_loaded_client, summarize_batch_with_loaded_client, max_chunk_tokens


def _resolve_cursor(
    raw: str,
    visible_messages: list[dict],
) -> float | None:
    """Resolve a `before`/`after` cursor to a unix-timestamp cutoff.

    Accepts:
      1) a numeric unix timestamp (``"1717891234.567"``)
      2) an ISO-8601 timestamp (``"2026-06-06T14:00:00Z"``)
      3) a message ID — looked up in ``visible_messages`` for its timestamp

    Returns ``None`` if the raw string is empty after stripping. Raises
    ``HTTPException(400)`` only for the message-ID branch when the ID
    isn't found; numeric/ISO failures fall through silently because the
    caller may have legitimately passed empty/garbage.
    """
    from datetime import datetime

    trimmed = raw.strip()
    if not trimmed:
        return None

    # 1) Numeric unix timestamp
    try:
        return float(trimmed)
    except ValueError:
        pass

    # 2) ISO timestamp
    iso_candidate = trimmed.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_candidate).timestamp()
    except ValueError:
        pass

    # 3) Message ID cursor
    cursor_msg = next(
        (m for m in visible_messages if str(m.get("id") or "") == trimmed),
        None,
    )
    if cursor_msg is None:
        raise HTTPException(status_code=400, detail="Invalid cursor")
    return float(cursor_msg.get("timestamp") or 0.0)


def _load_sorted_visible_messages(
    service,
    effective_bot_id: str,
) -> list[dict]:
    """Pull all visible messages for a bot, sorted chronologically.

    Shared by the per-bot history routes (``/v1/history``,
    ``/v1/history/around``). Filters out ``system`` / ``summary`` rows
    so summaries — which have their own surface — don't bleed into chat
    history paging.
    """
    client = service.get_memory_client(effective_bot_id)
    if not client:
        raise HTTPException(status_code=503, detail="Memory service unavailable")

    messages = client.get_messages(since_seconds=None)
    visible = [m for m in messages if m.get("role") not in ("system", "summary")]
    visible.sort(key=lambda m: (float(m.get("timestamp") or 0.0), str(m.get("id") or "")))
    return visible


def _filter_messages_by_time_range(
    messages: list[dict],
    *,
    since: float | None = None,
    until: float | None = None,
) -> list[dict]:
    """Apply inclusive timestamp bounds to a sorted message list."""
    if since is None and until is None:
        return messages

    filtered: list[dict] = []
    for msg in messages:
        ts = float(msg.get("timestamp") or 0.0)
        if since is not None and ts < since:
            continue
        if until is not None and ts > until:
            continue
        filtered.append(msg)
    return filtered


def _search_visible_messages(
    messages: list[dict],
    query: str,
) -> list[dict]:
    """Case-insensitive substring search over visible message content."""
    query_lower = query.lower()
    if not query_lower:
        return messages
    return [
        msg for msg in messages
        if query_lower in str(msg.get("content", "")).lower()
    ]


def _build_history_search_response(
    service,
    effective_bot_id: str,
    query: str,
    matches: list[dict],
    page_messages: list[dict],
    *,
    has_older: bool,
) -> HistorySearchResponse:
    """Hydrate attachments and assemble the paginated search response."""
    attachments_by_id = _hydrate_attachments_for_page(
        service, effective_bot_id, page_messages
    )
    reasoning_by_id = _hydrate_reasoning_for_page(
        service, effective_bot_id, page_messages
    )
    history_messages = [
        HistoryMessage(
            id=msg.get("id"),
            role=msg.get("role", ""),
            content=msg.get("content", ""),
            timestamp=msg.get("timestamp", 0.0),
            attachments=attachments_by_id.get(str(msg.get("id") or ""), []),
            reasoning=reasoning_by_id.get(str(msg.get("id") or "")),
        )
        for msg in page_messages
    ]

    oldest_timestamp = history_messages[0].timestamp if history_messages else None
    newest_timestamp = history_messages[-1].timestamp if history_messages else None

    return HistorySearchResponse(
        bot_id=effective_bot_id,
        query=query,
        messages=history_messages,
        total_count=len(matches),
        has_more=has_older,
        has_older=has_older,
        oldest_timestamp=oldest_timestamp,
        newest_timestamp=newest_timestamp,
    )


def _page_messages_before_cursor(
    messages: list[dict],
    *,
    before_ts: float | None,
    limit: int,
) -> tuple[list[dict], bool]:
    """Return the newest page of messages older than an optional cursor."""
    candidate_messages = [
        m for m in messages
        if before_ts is None or float(m.get("timestamp") or 0.0) < before_ts
    ]
    if limit > 0 and len(candidate_messages) > limit:
        return candidate_messages[-limit:], True
    return candidate_messages, False


def _resolve_cursor_against_messages(
    before: str | None,
    messages: list[dict],
) -> float | None:
    """Resolve an optional cursor against an already-filtered message set."""
    if not before:
        return None
    return _resolve_cursor(before, messages)


def _slice_history_search_matches(
    matches: list[dict],
    *,
    before_ts: float | None,
    limit: int,
) -> tuple[list[dict], bool]:
    """Return a paged history-search slice plus whether older matches remain."""
    return _page_messages_before_cursor(matches, before_ts=before_ts, limit=limit)


def _search_history_messages(
    service,
    effective_bot_id: str,
    query: str,
    *,
    limit: int,
    before: str | None,
    since: float | None,
    until: float | None,
) -> HistorySearchResponse:
    """Run per-bot history search with time-bounded, cursor-based paging."""
    visible_messages = _load_sorted_visible_messages(service, effective_bot_id)
    bounded_messages = _filter_messages_by_time_range(
        visible_messages,
        since=since,
        until=until,
    )
    matches = _search_visible_messages(bounded_messages, query)
    before_ts = _resolve_cursor_against_messages(before, matches)
    page_messages, has_older = _slice_history_search_matches(
        matches,
        before_ts=before_ts,
        limit=limit,
    )
    return _build_history_search_response(
        service,
        effective_bot_id,
        query,
        matches,
        page_messages,
        has_older=has_older,
    )


def _search_history_messages_legacy(
    service,
    effective_bot_id: str,
    query: str,
    *,
    limit: int,
) -> HistorySearchResponse:
    """Compatibility wrapper for callers expecting the old search behavior."""
    return _search_history_messages(
        service,
        effective_bot_id,
        query,
        limit=limit,
        before=None,
        since=None,
        until=None,
    )


def _search_history_time_bounds_description() -> str:
    """Human-readable docs fragment for time-bounded per-bot history search."""
    return "Inclusive unix-second timestamp bounds for constraining search results."


def _search_history_before_description() -> str:
    """Human-readable docs fragment for paged per-bot history search."""
    return "Cursor for older search-result pages (ISO timestamp, unix timestamp, or message ID)."


def _search_history_limit_description() -> str:
    """Human-readable docs fragment for search page size."""
    return "Maximum number of matching messages to return in this page."


def _search_history_query_description() -> str:
    """Human-readable docs fragment for the per-bot search query."""
    return "Case-insensitive substring query over visible bot history."


def _search_history_bot_description() -> str:
    """Human-readable docs fragment for per-bot search bot selection."""
    return "Bot ID (uses default if not specified)."


def _search_history_docstring() -> str:
    """Return the route docstring text for per-bot history search."""
    return "Search conversation history for a bot with cursor paging and optional time bounds."


def _search_history_error_log_message(query: str, error: Exception) -> str:
    """Format the per-bot history search error log."""
    return f"Failed to search history for query {query!r}: {error}"


def _resolve_search_limit(limit: int) -> int:
    """Normalize per-bot history search page size."""
    return max(0, limit)


def _resolve_search_bounds(
    since: float | None,
    until: float | None,
) -> tuple[float | None, float | None]:
    """Validate and normalize per-bot search time bounds."""
    if since is not None and until is not None and since > until:
        raise HTTPException(status_code=400, detail="`since` cannot be greater than `until`")
    return since, until


def _resolve_search_query(query: str) -> str:
    """Validate and normalize the incoming per-bot search query."""
    trimmed = query.strip()
    if not trimmed:
        raise HTTPException(status_code=400, detail="`query` cannot be empty")
    return trimmed


def _resolve_effective_bot_id(service, bot_id: str | None) -> str:
    """Apply default bot fallback shared by history routes."""
    return bot_id or service._default_bot


def _search_history_route_impl(
    service,
    query: str,
    bot_id: str | None,
    limit: int,
    before: str | None,
    since: float | None,
    until: float | None,
) -> HistorySearchResponse:
    """Shared implementation for the per-bot search route."""
    effective_bot_id = _resolve_effective_bot_id(service, bot_id)
    resolved_query = _resolve_search_query(query)
    resolved_limit = _resolve_search_limit(limit)
    resolved_since, resolved_until = _resolve_search_bounds(since, until)
    return _search_history_messages(
        service,
        effective_bot_id,
        resolved_query,
        limit=resolved_limit,
        before=before,
        since=resolved_since,
        until=resolved_until,
    )


def _handle_history_search_exception(query: str, error: Exception) -> None:
    """Log and raise a consistent per-bot search failure."""
    log.error(_search_history_error_log_message(query, error))
    raise HTTPException(status_code=500, detail=str(error))


def _run_history_search_route(
    query: str,
    bot_id: str | None,
    limit: int,
    before: str | None,
    since: float | None,
    until: float | None,
) -> HistorySearchResponse:
    """Thin wrapper so the route body stays compact and readable."""
    service = get_service()
    return _search_history_route_impl(
        service,
        query,
        bot_id,
        limit,
        before,
        since,
        until,
    )


def _history_search_route_doc() -> str:
    """Provide stable docs text without repeating long literals inline."""
    return _search_history_docstring()


def _history_search_limit_query() -> str:
    """Stable docs helper for per-bot search page size."""
    return _search_history_limit_description()


def _history_search_before_query() -> str:
    """Stable docs helper for per-bot search paging cursor."""
    return _search_history_before_description()


def _history_search_since_query() -> str:
    """Stable docs helper for lower time bound."""
    return _search_history_time_bounds_description()


def _history_search_until_query() -> str:
    """Stable docs helper for upper time bound."""
    return _search_history_time_bounds_description()


def _history_search_bot_query() -> str:
    """Stable docs helper for bot selection."""
    return _search_history_bot_description()


def _history_search_query_query() -> str:
    """Stable docs helper for query description."""
    return _search_history_query_description()


def _history_search_route(
    query: str,
    bot_id: str | None,
    limit: int,
    before: str | None,
    since: float | None,
    until: float | None,
) -> HistorySearchResponse:
    """Call the actual shared search implementation."""
    return _run_history_search_route(query, bot_id, limit, before, since, until)


def _history_search_legacy(
    query: str,
    bot_id: str | None,
    limit: int,
) -> HistorySearchResponse:
    """Legacy hook kept for any future callers needing pre-paging semantics."""
    service = get_service()
    effective_bot_id = _resolve_effective_bot_id(service, bot_id)
    resolved_query = _resolve_search_query(query)
    resolved_limit = _resolve_search_limit(limit)
    return _search_history_messages_legacy(
        service,
        effective_bot_id,
        resolved_query,
        limit=resolved_limit,
    )


def _history_search_route_body(
    query: str,
    bot_id: str | None,
    limit: int,
    before: str | None,
    since: float | None,
    until: float | None,
) -> HistorySearchResponse:
    """Route-body indirection so docs/logic stay separated."""
    return _history_search_route(query, bot_id, limit, before, since, until)


def _history_search_route_error(query: str, error: Exception) -> None:
    """Dedicated error path for the route body."""
    _handle_history_search_exception(query, error)


def _history_search_route_execute(
    query: str,
    bot_id: str | None,
    limit: int,
    before: str | None,
    since: float | None,
    until: float | None,
) -> HistorySearchResponse:
    """Route executor shared by POST surface(s)."""
    return _history_search_route_body(query, bot_id, limit, before, since, until)


def _load_all_messages_via_sql(
    service,
    bot_id: str,
) -> list[dict] | None:
    """Direct-SQL read of the entire ``{bot}_messages`` table.

    Bypasses :meth:`PostgreSQLShortTermManager.get_messages`'s
    summarization filter so deep-link routes (``/v1/history/around``) can
    locate ANY message that exists in the table — including messages whose
    content has been folded into a summary and is therefore hidden from
    the live chat tail. The summarization filter is correct for
    "build a prompt" / "show the live conversation" but wrong for
    "land me on this specific message that an upstream surface (Spotlight
    Search, an external link) already found and referenced."

    Mirrors the data-access pattern that powers
    ``mcp_server.storage.search_all_messages``, so search hits and
    deep-link landings see the same set of rows.

    Returns ``None`` on backend unavailability so the caller can surface
    a 503 with its own message rather than letting the exception bubble.
    """
    from sqlalchemy import text
    from ...media.assets import _build_engine
    from ...memory.postgresql import _sanitize_table_name

    engine = _build_engine(service.config)
    if engine is None:
        return None

    table = f"{_sanitize_table_name(bot_id)}_messages"
    sql = text(
        f"""
        SELECT id, role, content, timestamp
        FROM {table}
        WHERE role NOT IN ('system', 'summary')
        ORDER BY timestamp ASC, id ASC
        """
    )

    try:
        with engine.connect() as conn:
            rows = conn.execute(sql).mappings().all()
    except Exception as e:
        log.warning(f"_load_all_messages_via_sql failed for {bot_id}: {e}")
        return None

    return [
        {
            "id": str(row["id"] or ""),
            "role": str(row["role"] or ""),
            "content": str(row["content"] or ""),
            "timestamp": float(row["timestamp"] or 0.0),
        }
        for row in rows
    ]


def _build_history_response(
    service,
    effective_bot_id: str,
    visible_messages: list[dict],
    page_messages: list[dict],
    candidate_count: int | None,
    *,
    has_older: bool | None = None,
    has_newer: bool | None = None,
    anchor_id: str | None = None,
) -> HistoryResponse:
    """Hydrate attachments and assemble a HistoryResponse for a slice.

    Centralises the attachment hydration + boundary-flag work so the three
    history endpoints (legacy ``/v1/history`` backward, the new ``/v1/history``
    forward, and ``/v1/history/around``) all produce identically-shaped
    responses without copy-pasted glue.
    """
    attachments_by_id = _hydrate_attachments_for_page(
        service, effective_bot_id, page_messages
    )
    reasoning_by_id = _hydrate_reasoning_for_page(
        service, effective_bot_id, page_messages
    )
    history_messages = [
        HistoryMessage(
            id=msg.get("id"),
            role=msg.get("role", ""),
            content=msg.get("content", ""),
            timestamp=msg.get("timestamp", 0.0),
            attachments=attachments_by_id.get(str(msg.get("id") or ""), []),
            reasoning=reasoning_by_id.get(str(msg.get("id") or "")),
        )
        for msg in page_messages
    ]

    oldest_timestamp = history_messages[0].timestamp if history_messages else None
    newest_timestamp = history_messages[-1].timestamp if history_messages else None

    # If callers passed has_older / has_newer explicitly, trust them. Otherwise
    # infer from the candidate vs page sizes (legacy single-direction path).
    resolved_has_older = (
        has_older
        if has_older is not None
        else bool(candidate_count is not None and candidate_count > len(page_messages))
    )
    resolved_has_newer = has_newer if has_newer is not None else False

    return HistoryResponse(
        bot_id=effective_bot_id,
        messages=history_messages,
        total_count=len(history_messages),
        has_more=resolved_has_older,
        has_older=resolved_has_older,
        has_newer=resolved_has_newer,
        oldest_timestamp=oldest_timestamp,
        newest_timestamp=newest_timestamp,
        anchor_id=anchor_id,
    )


@router.get("/v1/history", response_model=HistoryResponse, tags=["History"])
def get_history(
    bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
    limit: int = Query(50, description="Maximum number of messages to return"),
    before: str | None = Query(
        None,
        description="Cursor for older history pages (ISO timestamp, unix timestamp, or message ID)",
    ),
    after: str | None = Query(
        None,
        description=(
            "Cursor for forward pagination — returns the OLDEST `limit` messages strictly newer "
            "than this cursor. Used after a deep-link landing (`/v1/history/around`) when the "
            "user scrolls down past the loaded window. Mutually exclusive with `before`."
        ),
    ),
):
    """Get conversation history for a bot.

    Two-direction pagination:

    - ``?before=<cursor>`` (default direction) — load older messages. Takes
      the NEWEST ``limit`` messages strictly older than the cursor.
    - ``?after=<cursor>`` — load newer messages. Takes the OLDEST ``limit``
      messages strictly newer than the cursor. Used to extend a deep-link
      window forward as the user scrolls down past it.

    Cursors accept unix timestamps, ISO-8601 strings, or message IDs.
    Passing both ``before`` and ``after`` is rejected (400).
    """
    if before and after:
        raise HTTPException(
            status_code=400,
            detail="`before` and `after` are mutually exclusive",
        )

    service = get_service()
    effective_bot_id = bot_id or service._default_bot

    try:
        visible_messages = _load_sorted_visible_messages(service, effective_bot_id)

        before_ts = _resolve_cursor(before, visible_messages) if before else None
        after_ts = _resolve_cursor(after, visible_messages) if after else None

        if after_ts is not None:
            # Forward page: oldest `limit` messages strictly newer than the cursor.
            candidate_messages = [
                m for m in visible_messages
                if float(m.get("timestamp") or 0.0) > after_ts
            ]
            page_messages = candidate_messages[:limit] if limit > 0 else candidate_messages
            # In the forward direction `has_more` semantically means
            # "more *newer* messages exist beyond the returned page".
            has_older = False  # forward queries don't tell us about the older side
            has_newer = bool(candidate_messages and len(candidate_messages) > len(page_messages))
        else:
            # Backward page (default): newest `limit` strictly older than the cursor.
            if before_ts is not None:
                candidate_messages = [
                    m for m in visible_messages
                    if float(m.get("timestamp") or 0.0) < before_ts
                ]
            else:
                candidate_messages = visible_messages
            page_messages = (
                candidate_messages[-limit:]
                if limit > 0 and len(candidate_messages) > limit
                else candidate_messages
            )
            has_older = len(candidate_messages) > len(page_messages)
            has_newer = False  # legacy direction never reports the newer side

        return _build_history_response(
            service,
            effective_bot_id,
            visible_messages,
            page_messages,
            candidate_count=None,
            has_older=has_older,
            has_newer=has_newer,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/history/around", response_model=HistoryResponse, tags=["History"])
def get_history_around(
    bot_id: str = Query(..., description="Bot ID"),
    message_id: str = Query(..., description="Anchor message ID — window is centered on this row"),
    before: int = Query(30, ge=0, le=200, description="Number of older messages to include"),
    after: int = Query(10, ge=0, le=200, description="Number of newer messages to include"),
):
    """Return a window of messages around an anchor.

    Powers deep-link entry into chat surfaces (``/chat/<bot>?message=<id>``).
    Returns the anchor row plus ``before`` older + ``after`` newer rows,
    with ``has_older`` / ``has_newer`` flags so the frontend knows whether
    further pagination in either direction is possible.

    The returned ``oldest_timestamp`` and ``newest_timestamp`` are valid
    cursors for ``/v1/history?before=...`` and ``/v1/history?after=...``
    respectively, so continued scrolling stays on the standard paging
    surface — no separate "extend window" endpoint needed.
    """
    service = get_service()

    try:
        visible_messages = _load_all_messages_via_sql(service, bot_id)
        if visible_messages is None:
            raise HTTPException(status_code=503, detail="Memory service unavailable")
        target_idx = next(
            (i for i, m in enumerate(visible_messages) if str(m.get("id") or "") == message_id),
            -1,
        )
        if target_idx < 0:
            raise HTTPException(status_code=404, detail=f"Message {message_id!r} not found")

        start = max(0, target_idx - before)
        end = min(len(visible_messages), target_idx + after + 1)
        page_messages = visible_messages[start:end]

        return _build_history_response(
            service,
            bot_id,
            visible_messages,
            page_messages,
            candidate_count=None,
            has_older=start > 0,
            has_newer=end < len(visible_messages),
            anchor_id=message_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to load history window around {message_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/history/search", response_model=HistorySearchResponse, tags=["History"])
def search_history(
    query: str = Query(..., description="Case-insensitive substring query over visible bot history."),
    bot_id: str = Query(None, description="Bot ID (uses default if not specified)."),
    limit: int = Query(50, description="Maximum number of matching messages to return in this page."),
    before: str | None = Query(
        None,
        description="Cursor for older search-result pages (ISO timestamp, unix timestamp, or message ID).",
    ),
    since: float | None = Query(
        None,
        description="Inclusive unix-second lower bound for constraining search results.",
    ),
    until: float | None = Query(
        None,
        description="Inclusive unix-second upper bound for constraining search results.",
    ),
):
    """Search conversation history for a bot with cursor paging and optional time bounds."""
    try:
        return _run_history_search_route(
            query=query,
            bot_id=bot_id,
            limit=limit,
            before=before,
            since=since,
            until=until,
        )
    except HTTPException:
        raise
    except Exception as e:
        _handle_history_search_exception(query, e)
        raise  # unreachable, keeps type-checkers happy


@router.get("/v1/history/search", response_model=HistorySearchResponse, tags=["History"])
def search_history_get(
    query: str = Query(..., description="Case-insensitive substring query over visible bot history."),
    bot_id: str = Query(None, description="Bot ID (uses default if not specified)."),
    limit: int = Query(50, description="Maximum number of matching messages to return in this page."),
    before: str | None = Query(
        None,
        description="Cursor for older search-result pages (ISO timestamp, unix timestamp, or message ID).",
    ),
    since: float | None = Query(
        None,
        description="Inclusive unix-second lower bound for constraining search results.",
    ),
    until: float | None = Query(
        None,
        description="Inclusive unix-second upper bound for constraining search results.",
    ),
):
    """GET variant of per-bot history search for proxy routes that prefer query params."""
    try:
        return _run_history_search_route(
            query=query,
            bot_id=bot_id,
            limit=limit,
            before=before,
            since=since,
            until=until,
        )
    except HTTPException:
        raise
    except Exception as e:
        _handle_history_search_exception(query, e)
        raise  # unreachable, keeps type-checkers happy


@router.post(
    "/v1/history/search_all",
    response_model=HistorySearchAllResponse,
    tags=["History"],
)
async def search_all_history(
    query: str = Query(..., description="Search query (FTS keywords or substring/literal)"),
    limit: int = Query(50, description="Maximum total results across all bots"),
    role_filter: str | None = Query(
        None,
        description="Only include messages with this role (user|assistant). System messages are always excluded.",
    ),
    mode: str = Query(
        "fts",
        description=(
            "Search mode. 'fts' (default) uses Postgres full-text search "
            "with the english config — stems + drops stop-words + ranks by "
            "lexeme density. Best for natural-language queries. 'trgm' uses "
            "the pg_trgm extension for substring/fuzzy matching against the "
            "literal content — best for IDs, file paths, hyphenated tokens, "
            "or anything where FTS tokenization would discard signal."
        ),
    ),
    sort_by: str = Query(
        "relevance",
        description=(
            "Result ordering. 'relevance' (default) sorts by rank (ts_rank "
            "for FTS, similarity for trgm) then timestamp; lets the densest "
            "match win even if it's old. 'recent' sorts by timestamp only — "
            "best when the query is a common token and you want the latest "
            "hit (e.g. '/new' in a chat history)."
        ),
    ),
    since: str | None = Query(
        None,
        description=(
            "Lower-bound (inclusive). Accepts ISO date ('2026-06-01'), "
            "ISO datetime ('2026-06-01T14:30:00'), or Unix seconds (1782857172.0). "
            "Omit for unbounded history."
        ),
    ),
    until: str | None = Query(
        None,
        description=(
            "Upper-bound (inclusive). Same formats as ``since``. Omit for now."
        ),
    ),
):
    """Cross-bot message search.

    Two modes share this route:

    * ``mode=fts`` (default) — Postgres FTS via
      :meth:`llm_bawt.mcp_server.storage.MemoryStorage.search_all_messages`.
      UNION ALL across every ``{bot}_messages`` table, ranked by
      ``ts_rank``. Good for concept-level queries; bad for IDs because
      ``build_fts_query`` strips digits and hyphens before building the
      tsquery (so ``TASK-241`` becomes ``task``, matching every "task"
      message ever).
    * ``mode=trgm`` — pg_trgm substring/fuzzy match via
      :meth:`MemoryStorage.search_all_messages_trgm`. Same UNION ALL
      shape but with ``content ILIKE`` filter + ``similarity()`` rank,
      backed by per-table ``gin_trgm_ops`` GIN indexes. Matches the
      literal query string verbatim — what the user expects for
      ``TASK-241`` and friends.

    Returns ranked rows with bot attribution so a global search UI can
    render "who said this and when" without N+1 fans-out. First
    HTTP-exposed endpoint of the Spotlight Search project's Messages
    provider.
    """
    from llm_bawt.mcp_server.server import _parse_timestamp
    from llm_bawt.mcp_server.storage import get_storage

    try:
        since_ts = _parse_timestamp(since)
        until_ts = _parse_timestamp(until)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        storage = get_storage()
        if mode == "trgm":
            rows = await storage.search_all_messages_trgm(
                query=query,
                n_results=limit,
                role_filter=role_filter,
                sort_by=sort_by,
                since=since_ts,
                until=until_ts,
            )
        else:
            # Default / legacy / explicit "fts" all route to the original
            # ts_rank-backed search so existing callers (and the MCP
            # surface, which is FTS-only) keep working unchanged.
            rows = await storage.search_all_messages(
                query=query,
                n_results=limit,
                role_filter=role_filter,
                sort_by=sort_by,
                since=since_ts,
                until=until_ts,
            )
        messages = [
            HistorySearchAllMessage(
                id=str(row.get("id", "")),
                role=str(row.get("role", "")),
                content=str(row.get("content", "")),
                timestamp=float(row.get("timestamp", 0.0) or 0.0),
                bot_id=str(row.get("source", "")),
                rank=float(row.get("rank", 0.0) or 0.0),
            )
            for row in rows
        ]
        # The storage methods now stamp the unbounded total onto every row
        # via a window COUNT(*). Surface it as `total_count` so the UI can
        # render "showing N of M". Empty result → 0. Read from the first
        # row since the value is the same on all rows.
        unbounded_total = int(rows[0].get("total", 0)) if rows else 0
        return HistorySearchAllResponse(
            query=query,
            messages=messages,
            total_count=unbounded_total,
        )
    except Exception as e:
        log.error(f"Failed cross-bot history search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/v1/history", response_model=HistoryClearResponse, tags=["History"])
def clear_history(
    bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
):
    """Clear conversation history for a bot."""
    service = get_service()

    effective_bot_id = bot_id or service._default_bot

    try:
        # Try the fast path: clear directly via DB backend (no model loading needed)
        cleared = _clear_history_direct(service.config, effective_bot_id)

        if not cleared:
            # Fallback: use the full LLMBawt instance
            model_alias = list(service._available_models)[0] if service._available_models else None
            if not model_alias:
                raise HTTPException(status_code=500, detail="No models available")
            llm_bawt = service._get_llm_bawt(model_alias, effective_bot_id, service.config.DEFAULT_USER)
            llm_bawt.history_manager.clear_history()

        # Evict any cached LLMBawt instances for this bot
        stale_keys = [k for k in service._llm_bawt_cache if k[1] == effective_bot_id]
        for k in stale_keys:
            del service._llm_bawt_cache[k]

        return HistoryClearResponse(
            success=True,
            message=f"History cleared for bot '{effective_bot_id}'"
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to clear history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _clear_history_direct(config, bot_id: str) -> bool:
    """Clear history directly via PostgreSQL without loading a model.

    Returns True if cleared successfully, False if DB is unavailable.

    NOTE on connection pooling (TASK-202): ``PostgreSQLMemoryBackend``
    instances now share a process-wide engine (see
    ``llm_bawt.memory.postgresql._get_shared_memory_engine``), so do NOT
    call ``backend.engine.dispose()`` here — that would tear down the
    pool used by every other bot.
    """
    try:
        from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend
        backend = PostgreSQLMemoryBackend(config, bot_id=bot_id)
        # Clear messages table
        from sqlalchemy.orm import Session as SASession
        from sqlalchemy import delete as sa_delete
        with SASession(backend.engine) as session:
            session.execute(sa_delete(backend.messages_table))
            session.commit()
        # Clear memories table
        backend.clear()
        log.info("History cleared directly for bot '%s'", bot_id)
        return True
    except Exception as e:
        log.warning("Direct history clear failed for '%s': %s", bot_id, e)
        return False


@router.delete("/v1/history/{message_id}", response_model=HistoryClearResponse, tags=["History"])
def delete_message(
    message_id: str,
    bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
):
    """Delete (forget) a single message by its ID.

    Moves the message to the bot's ``*_forgotten_messages`` archive table so it
    stops appearing in history but stays recoverable. Backs the chat UI delete
    button (``DELETE /api/chat/history/{id}``); ``message_id`` is the DB UUID the
    ``GET /v1/history`` response carries, and a leading prefix (>= 8 chars) also
    matches. Returns 404 when no such message exists for the bot.

    NOTE: single-segment path, so it does not collide with the more specific
    ``DELETE /v1/history/summary/{summary_id}`` route (two segments) or the
    no-param ``DELETE /v1/history`` clear-all route.
    """
    service = get_service()
    effective_bot_id = bot_id or service._default_bot

    try:
        from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend
        backend = PostgreSQLMemoryBackend(service.config, bot_id=effective_bot_id)
        forgotten = backend.ignore_message_by_id(message_id)

        if not forgotten:
            raise HTTPException(
                status_code=404,
                detail=f"Message '{message_id}' not found for bot '{effective_bot_id}'",
            )

        # Evict cached LLMBawt instances so their in-memory history_manager
        # doesn't keep serving the just-forgotten message on the next turn.
        stale_keys = [k for k in service._llm_bawt_cache if k[1] == effective_bot_id]
        for k in stale_keys:
            del service._llm_bawt_cache[k]

        return HistoryClearResponse(
            success=True,
            message=f"Message '{message_id}' deleted for bot '{effective_bot_id}'",
            deleted_count=1,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to delete message {message_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/v1/history/summarize/preview", response_model=SummarizePreviewResponse, tags=["History"])
def preview_summarizable_sessions(
    bot_id: str = Query(None, description="Bot ID"),
):
    """Preview sessions that would be summarized (dry run)."""
    from datetime import datetime
    from ...memory.summarization import HistorySummarizer

    service = get_service()
    effective_bot_id = bot_id or service._default_bot

    try:
        # Compute context budget for budget-driven eligibility
        ctx_window = int(service.config.get_model_context_window(None) or 0)
        max_output = int(service.config.get_model_max_tokens(None) or 4096)
        max_context_tokens = max(0, ctx_window - max_output) if ctx_window > 0 else 0

        summarizer = HistorySummarizer(
            service.config,
            effective_bot_id,
            max_context_tokens=max_context_tokens,
        )
        sessions = summarizer.preview_summarizable_sessions()

        session_infos = []
        total_messages = 0

        for session in sessions:
            total_messages += session.message_count

            # Get first and last user messages for preview
            user_msgs = [m for m in session.messages if m.get("role") == "user"]
            first_msg = user_msgs[0].get("content", "")[:100] if user_msgs else ""
            last_msg = user_msgs[-1].get("content", "")[:100] if len(user_msgs) > 1 else first_msg

            session_infos.append(SummarizableSession(
                start_timestamp=session.start_timestamp,
                end_timestamp=session.end_timestamp,
                start_time=datetime.fromtimestamp(session.start_timestamp).strftime("%Y-%m-%d %H:%M"),
                end_time=datetime.fromtimestamp(session.end_timestamp).strftime("%H:%M"),
                message_count=session.message_count,
                first_message=first_msg,
                last_message=last_msg,
            ))

        return SummarizePreviewResponse(
            bot_id=effective_bot_id,
            sessions=session_infos,
            total_messages=total_messages,
        )
    except Exception as e:
        log.error(f"Failed to preview summarizable sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/history/summarize", response_model=SummarizeResponse, tags=["History"])
def summarize_history(
    bot_id: str = Query(None, description="Bot ID"),
    use_heuristic: bool = Query(False, description="Fall back to heuristic if LLM fails"),
    model: str | None = Query(None, description="Optional model alias override"),
):
    """Summarize eligible history sessions."""
    from ...memory.summarization import HistorySummarizer

    service = get_service()
    effective_bot_id = bot_id or service._default_bot

    try:
        summarize_with_loaded_client, summarize_batch_with_loaded_client, max_chunk_tokens = _build_summary_callable(
            service=service,
            bot_id=effective_bot_id,
            user_id="system",
            model=model,
        )
        # Compute context budget from model
        resolved_model = model or getattr(service, '_default_model', None)
        max_prompt_tokens, max_chunk_tokens_resolved = _resolve_summarization_limits(service, resolved_model)
        if max_chunk_tokens == 0:
            max_chunk_tokens = max_chunk_tokens_resolved
        ctx_window = int(service.config.get_model_context_window(resolved_model) or 0)
        max_output = int(service.config.get_model_max_tokens(resolved_model) or 4096)
        max_context_tokens = max(0, ctx_window - max_output) if ctx_window > 0 else 0

        summarizer = HistorySummarizer(
            service.config,
            effective_bot_id,
            summarize_fn=summarize_with_loaded_client,
            summarize_batch_fn=summarize_batch_with_loaded_client,
            max_context_tokens=max_context_tokens,
        )
        result = summarizer.summarize_eligible_sessions(
            use_heuristic_fallback=use_heuristic,
            max_tokens_per_chunk=max_chunk_tokens,
        )
        _invalidate_bot_history_cache(service, effective_bot_id)

        return SummarizeResponse(
            success=True,
            sessions_summarized=result.get("sessions_summarized", 0),
            messages_summarized=result.get("messages_summarized", 0),
            sessions_targeted=result.get("sessions_targeted"),
            summaries_replaced=result.get("summaries_replaced"),
            errors=result.get("errors", []),
        )
    except Exception as e:
        log.error(f"Failed to summarize history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/history/summarize/rebuild", response_model=SummarizeResponse, tags=["History"])
def rebuild_history_summaries(
    bot_id: str = Query(None, description="Bot ID"),
    sessions: int = Query(5, ge=0, le=2000, description="How many recent eligible sessions to rebuild (0 = all eligible)"),
    use_heuristic: bool = Query(False, description="Fall back to heuristic if LLM fails"),
    start_ts: float | None = Query(None, description="Optional Unix start timestamp filter"),
    end_ts: float | None = Query(None, description="Optional Unix end timestamp filter"),
    purge_existing: bool = Query(False, description="Delete existing historical summaries in range before rebuild"),
    model: str | None = Query(None, description="Optional model alias override"),
):
    """Rebuild summaries for the last N eligible sessions from original history."""
    from ...memory.summarization import HistorySummarizer

    service = get_service()
    effective_bot_id = bot_id or service._default_bot

    try:
        summarize_with_loaded_client, summarize_batch_with_loaded_client, max_chunk_tokens = _build_summary_callable(
            service=service,
            bot_id=effective_bot_id,
            user_id="system",
            model=model,
        )
        # Compute context budget from model
        resolved_model = model or getattr(service, '_default_model', None)
        ctx_window = int(service.config.get_model_context_window(resolved_model) or 0)
        max_output = int(service.config.get_model_max_tokens(resolved_model) or 4096)
        max_context_tokens = max(0, ctx_window - max_output) if ctx_window > 0 else 0

        summarizer = HistorySummarizer(
            service.config,
            effective_bot_id,
            summarize_fn=summarize_with_loaded_client,
            summarize_batch_fn=summarize_batch_with_loaded_client,
            max_context_tokens=max_context_tokens,
        )
        result = summarizer.rebuild_recent_sessions(
            session_limit=sessions,
            use_heuristic_fallback=use_heuristic,
            max_tokens_per_chunk=max_chunk_tokens,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            purge_existing=purge_existing,
        )
        _invalidate_bot_history_cache(service, effective_bot_id)
        return SummarizeResponse(
            success=True,
            sessions_summarized=result.get("sessions_summarized", 0),
            messages_summarized=result.get("messages_summarized", 0),
            sessions_targeted=result.get("sessions_targeted"),
            summaries_replaced=result.get("summaries_replaced"),
            summaries_purged=result.get("summaries_purged"),
            errors=result.get("errors", []),
        )
    except Exception as e:
        log.error(f"Failed to rebuild history summaries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/v1/history/summaries", response_model=ListSummariesResponse, tags=["History"])
def list_summaries(
    bot_id: str = Query(None, description="Bot ID"),
):
    """List existing history summaries."""
    from datetime import datetime
    from ...memory.summarization import HistorySummarizer

    service = get_service()
    effective_bot_id = bot_id or service._default_bot

    try:
        summarizer = HistorySummarizer(service.config, effective_bot_id)
        summaries = summarizer.list_summaries()

        summary_infos = []
        for summ in summaries:
            start_ts = summ.get("session_start")
            end_ts = summ.get("session_end")

            summary_infos.append(SummaryInfo(
                id=summ.get("id", ""),
                content=summ.get("content", ""),
                timestamp=summ.get("timestamp", 0),
                session_start_time=datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M") if start_ts else None,
                session_end_time=datetime.fromtimestamp(end_ts).strftime("%H:%M") if end_ts else None,
                message_count=summ.get("message_count", 0),
                method=summ.get("method", "unknown"),
            ))

        return ListSummariesResponse(
            bot_id=effective_bot_id,
            summaries=summary_infos,
            total_count=len(summary_infos),
        )
    except Exception as e:
        log.error(f"Failed to list summaries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/v1/history/summary/{summary_id}", response_model=DeleteSummaryResponse, tags=["History"])
def delete_summary(
    summary_id: str,
    bot_id: str = Query(None, description="Bot ID"),
):
    """Delete a summary and restore the original messages."""
    from ...memory.summarization import HistorySummarizer

    service = get_service()
    effective_bot_id = bot_id or service._default_bot

    try:
        summarizer = HistorySummarizer(service.config, effective_bot_id)
        result = summarizer.delete_summary(summary_id)

        if result.get("success"):
            return DeleteSummaryResponse(
                success=True,
                summary_id=result.get("summary_id"),
                messages_restored=result.get("messages_restored", 0),
            )
        else:
            return DeleteSummaryResponse(
                success=False,
                detail=result.get("error", "Failed to delete summary"),
            )
    except Exception as e:
        log.error(f"Failed to delete summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
