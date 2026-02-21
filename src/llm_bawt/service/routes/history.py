"""History and summarization routes."""

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import get_service
from ..logging import get_service_logger
from ..schemas import (
    DeleteSummaryResponse,
    HistoryClearResponse,
    HistoryMessage,
    HistoryResponse,
    HistorySearchResponse,
    ListSummariesResponse,
    SummarizableSession,
    SummarizePreviewResponse,
    SummarizeResponse,
    SummaryInfo,
)

router = APIRouter()
log = get_service_logger(__name__)


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
        BATCH_SUMMARIZATION_PROMPT,
        SUMMARIZATION_PROMPT,
        _extract_json_object,
        compress_structured_summary_text,
        format_session_for_summarization,
        is_summary_low_quality,
    )
    from ...models.message import Message

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

    def summarize_with_loaded_client(session) -> str | None:
        if not client:
            return None

        conversation_text = format_session_for_summarization(session)
        prompt = SUMMARIZATION_PROMPT.format(messages=conversation_text)
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
            prompt = BATCH_SUMMARIZATION_PROMPT.format(sessions_blob=sessions_blob)
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
            if model_alias and "grok-4-fast" in model_alias and model_max_tokens < 16384:
                model_max_tokens = 16384
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


@router.get("/v1/history", response_model=HistoryResponse, tags=["History"])
async def get_history(
    bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
    limit: int = Query(50, description="Maximum number of messages to return"),
    before: str | None = Query(
        None,
        description="Cursor for older history pages (ISO timestamp, unix timestamp, or message ID)",
    ),
):
    """Get conversation history for a bot."""
    from datetime import datetime

    service = get_service()
    
    effective_bot_id = bot_id or service._default_bot
    
    try:
        # Use memory client to get messages from database
        client = service.get_memory_client(effective_bot_id)
        if not client:
            raise HTTPException(status_code=503, detail="Memory service unavailable")
        
        # Get all messages then paginate in-memory (includes both embedded and MCP server modes).
        messages = client.get_messages(since_seconds=None)

        # Filter out system messages and normalize chronological ordering.
        visible_messages = [m for m in messages if m.get("role") != "system"]
        visible_messages.sort(key=lambda m: (float(m.get("timestamp") or 0.0), str(m.get("id") or "")))

        # Resolve `before` cursor to timestamp cutoff.
        before_ts: float | None = None
        if before:
            raw = before.strip()
            if raw:
                # 1) Numeric unix timestamp
                try:
                    before_ts = float(raw)
                except ValueError:
                    before_ts = None

                # 2) ISO timestamp
                if before_ts is None:
                    iso_candidate = raw.replace("Z", "+00:00")
                    try:
                        before_ts = datetime.fromisoformat(iso_candidate).timestamp()
                    except ValueError:
                        before_ts = None

                # 3) Message ID cursor
                if before_ts is None:
                    cursor_msg = next((m for m in visible_messages if str(m.get("id") or "") == raw), None)
                    if cursor_msg is None:
                        raise HTTPException(status_code=400, detail="Invalid 'before' cursor")
                    before_ts = float(cursor_msg.get("timestamp") or 0.0)

        if before_ts is not None:
            candidate_messages = [m for m in visible_messages if float(m.get("timestamp") or 0.0) < before_ts]
        else:
            candidate_messages = visible_messages

        # Take newest `limit` from the candidate set, while returning chronological order.
        if limit > 0 and len(candidate_messages) > limit:
            page_messages = candidate_messages[-limit:]
        else:
            page_messages = candidate_messages

        history_messages = [
            HistoryMessage(
                id=msg.get("id"),
                role=msg.get("role", ""),
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp", 0.0)
            )
            for msg in page_messages
        ]

        oldest_timestamp = history_messages[0].timestamp if history_messages else None
        has_more = len(candidate_messages) > len(page_messages)
        
        return HistoryResponse(
            bot_id=effective_bot_id,
            messages=history_messages,
            total_count=len(history_messages),
            has_more=has_more,
            oldest_timestamp=oldest_timestamp,
        )
    except Exception as e:
        log.error(f"Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/history/search", response_model=HistorySearchResponse, tags=["History"])
async def search_history(
    query: str = Query(..., description="Search query"),
    bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
    limit: int = Query(50, description="Maximum number of messages to return"),
):
    """Search conversation history for a bot."""
    service = get_service()
    
    effective_bot_id = bot_id or service._default_bot
    
    try:
        client = service.get_memory_client(effective_bot_id)
        if not client:
            raise HTTPException(status_code=503, detail="Memory service unavailable")
        
        # Get all messages and filter by query
        messages = client.get_messages(since_seconds=None)
        query_lower = query.lower()
        
        matching = [
            msg for msg in messages
            if msg.get("role") != "system" and query_lower in msg.get("content", "").lower()
        ]
        
        # Apply limit
        if limit > 0 and len(matching) > limit:
            matching = matching[-limit:]
        
        history_messages = [
            HistoryMessage(
                id=msg.get("id"),
                role=msg.get("role", ""),
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp", 0.0)
            )
            for msg in matching
        ]
        
        return HistorySearchResponse(
            bot_id=effective_bot_id,
            query=query,
            messages=history_messages,
            total_count=len(history_messages)
        )
    except Exception as e:
        log.error(f"Failed to search history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/v1/history", response_model=HistoryClearResponse, tags=["History"])
async def clear_history(
    bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
):
    """Clear conversation history for a bot."""
    service = get_service()
    
    effective_bot_id = bot_id or service._default_bot
    
    try:
        model_alias = list(service._available_models)[0] if service._available_models else None
        if not model_alias:
            raise HTTPException(status_code=500, detail="No models available")
        
        llm_bawt = service._get_llm_bawt(model_alias, effective_bot_id, service.config.DEFAULT_USER)
        llm_bawt.history_manager.clear_history()
        
        # Also remove from cache to force fresh state
        cache_key = (model_alias, effective_bot_id, service.config.DEFAULT_USER)
        if cache_key in service._llm_bawt_cache:
            del service._llm_bawt_cache[cache_key]
        
        return HistoryClearResponse(
            success=True,
            message=f"History cleared for bot '{effective_bot_id}'"
        )
    except Exception as e:
        log.error(f"Failed to clear history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/v1/history/summarize/preview", response_model=SummarizePreviewResponse, tags=["History"])
async def preview_summarizable_sessions(
    bot_id: str = Query(None, description="Bot ID"),
):
    """Preview sessions that would be summarized (dry run)."""
    from datetime import datetime
    from ...memory.summarization import HistorySummarizer

    service = get_service()
    effective_bot_id = bot_id or service._default_bot

    try:
        summarizer = HistorySummarizer(service.config, effective_bot_id)
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
async def summarize_history(
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
        summarizer = HistorySummarizer(
            service.config, 
            effective_bot_id,
            summarize_fn=summarize_with_loaded_client,
            summarize_batch_fn=summarize_batch_with_loaded_client,
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
async def rebuild_history_summaries(
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
        summarizer = HistorySummarizer(
            service.config,
            effective_bot_id,
            summarize_fn=summarize_with_loaded_client,
            summarize_batch_fn=summarize_batch_with_loaded_client,
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
async def list_summaries(
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
async def delete_summary(
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
