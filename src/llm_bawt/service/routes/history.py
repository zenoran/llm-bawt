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

@router.get("/v1/history", response_model=HistoryResponse, tags=["History"])
async def get_history(
    bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
    limit: int = Query(50, description="Maximum number of messages to return"),
):
    """Get conversation history for a bot."""
    service = get_service()
    
    effective_bot_id = bot_id or service._default_bot
    
    try:
        # Use memory client to get messages from database
        client = service.get_memory_client(effective_bot_id)
        if not client:
            raise HTTPException(status_code=503, detail="Memory service unavailable")
        
        # Get messages directly from database (no time filter)
        messages = client.get_messages(since_seconds=None)  # Get all messages
        
        # Apply limit (from most recent)
        if limit > 0 and len(messages) > limit:
            messages = messages[-limit:]
        
        history_messages = [
            HistoryMessage(
                id=msg.get("id"),
                role=msg.get("role", ""),
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp", 0.0)
            )
            for msg in messages
            if msg.get("role") != "system"  # Don't include system messages
        ]
        
        return HistoryResponse(
            bot_id=effective_bot_id,
            messages=history_messages,
            total_count=len(history_messages)
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
    from ..memory.summarization import HistorySummarizer

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
):
    """Summarize eligible history sessions."""
    from ..memory.summarization import HistorySummarizer, format_session_for_summarization, SUMMARIZATION_PROMPT
    from ..models.message import Message

    service = get_service()
    effective_bot_id = bot_id or service._default_bot

    # Create a summarize function that uses the loaded client directly
    # This avoids the HTTP self-call deadlock
    def summarize_with_loaded_client(session) -> str | None:
        """Summarize using the already-loaded LLM client."""
        if not service._client_cache:
            log.warning("No model loaded for summarization")
            return None
        
        # Get the loaded client
        model_alias = list(service._client_cache.keys())[0]
        client = service._client_cache[model_alias]
        
        # Build the prompt
        conversation_text = format_session_for_summarization(session)
        prompt = SUMMARIZATION_PROMPT.format(messages=conversation_text)
        
        # Check for token limits (rough estimate: 4 chars per token)
        estimated_tokens = len(prompt) // 4
        if estimated_tokens > 6000:  # Leave room for response
            log.warning(f"Session too large ({estimated_tokens} estimated tokens), needs chunking")
            return None  # Will trigger chunked summarization
        
        try:
            messages = [
                Message(role="system", content="You are a helpful assistant that summarizes conversations concisely."),
                Message(role="user", content=prompt),
            ]
            response = client.query(
                messages=messages,
                max_tokens=200,
                temperature=0.3,
                plaintext_output=True,  # No rich formatting for background tasks
                stream=False,  # Don't stream for summarization
            )
            
            # Check for error responses in content
            if response:
                error_indicators = ["Error:", "exception occurred", "exceed context window", "tokens exceed"]
                for indicator in error_indicators:
                    if indicator.lower() in response.lower():
                        log.error(f"LLM returned error: {response[:100]}...")
                        return None
            
            return response.strip() if response else None
        except Exception as e:
            log.error(f"LLM summarization failed: {e}")
            return None

    try:
        summarizer = HistorySummarizer(
            service.config, 
            effective_bot_id,
            summarize_fn=summarize_with_loaded_client,
        )
        result = summarizer.summarize_eligible_sessions(use_heuristic_fallback=use_heuristic)

        return SummarizeResponse(
            success=True,
            sessions_summarized=result.get("sessions_summarized", 0),
            messages_summarized=result.get("messages_summarized", 0),
            errors=result.get("errors", []),
        )
    except Exception as e:
        log.error(f"Failed to summarize history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/v1/history/summaries", response_model=ListSummariesResponse, tags=["History"])
async def list_summaries(
    bot_id: str = Query(None, description="Bot ID"),
):
    """List existing history summaries."""
    from datetime import datetime
    from ..memory.summarization import HistorySummarizer

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
    from ..memory.summarization import HistorySummarizer

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
