"""History summarization for llm-bawt.

This module provides session detection and summarization functionality
to condense older conversation history, allowing more context to fit
in the LLM window.

Sessions are detected by timestamp gaps (default 1 hour). Sessions older
than HISTORY_DURATION (default 30 min) are eligible for summarization.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .summarization import Session

logger = logging.getLogger(__name__)

# Summarization prompt template
SUMMARIZATION_PROMPT = """Summarize this conversation session concisely. Focus on:
1. Key topics discussed
2. Decisions made or actions taken
3. Important user information shared
4. Unresolved questions

Keep under 100 words. Write in third person. Be specific about what was discussed.

Conversation:
{messages}

Summary (include specific topics and any decisions made):"""


@dataclass
class Session:
    """A detected conversation session."""
    start_timestamp: float
    end_timestamp: float
    messages: list[dict]
    message_ids: list[str]

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def duration_seconds(self) -> float:
        return self.end_timestamp - self.start_timestamp

    @property
    def start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.start_timestamp)

    @property
    def end_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.end_timestamp)


def detect_sessions(
    messages: list[dict],
    session_gap_seconds: int = 3600,
) -> list[Session]:
    """Detect conversation sessions by splitting on timestamp gaps.

    Args:
        messages: List of message dicts with 'id', 'role', 'content', 'timestamp'
        session_gap_seconds: Gap between messages to consider them in different sessions

    Returns:
        List of Session objects, sorted by start timestamp
    """
    if not messages:
        return []

    # Sort by timestamp
    sorted_msgs = sorted(messages, key=lambda m: m.get("timestamp", 0))

    sessions: list[Session] = []
    current_session_msgs: list[dict] = []
    current_session_ids: list[str] = []
    session_start: float | None = None
    last_timestamp: float | None = None

    for msg in sorted_msgs:
        timestamp = msg.get("timestamp", 0)
        msg_id = msg.get("id", "")

        # Skip summary rows - they shouldn't be counted in session detection
        if msg.get("role") == "summary":
            continue

        # Check if this starts a new session
        if last_timestamp is not None and (timestamp - last_timestamp) > session_gap_seconds:
            # Save the current session
            if current_session_msgs and session_start is not None:
                sessions.append(Session(
                    start_timestamp=session_start,
                    end_timestamp=last_timestamp,
                    messages=current_session_msgs,
                    message_ids=current_session_ids,
                ))
            # Start a new session
            current_session_msgs = []
            current_session_ids = []
            session_start = None

        # Add message to current session
        current_session_msgs.append(msg)
        current_session_ids.append(msg_id)
        if session_start is None:
            session_start = timestamp
        last_timestamp = timestamp

    # Don't forget the last session
    if current_session_msgs and session_start is not None and last_timestamp is not None:
        sessions.append(Session(
            start_timestamp=session_start,
            end_timestamp=last_timestamp,
            messages=current_session_msgs,
            message_ids=current_session_ids,
        ))

    return sessions


def find_summarizable_sessions(
    sessions: list[Session],
    history_duration_seconds: int = 1800,  # 30 minutes
    min_messages: int = 4,
) -> list[Session]:
    """Filter sessions to those eligible for summarization.

    A session is eligible if:
    - It ended more than history_duration_seconds ago
    - It has at least min_messages messages
    
    Note: We don't need to track summarized messages because they are
    moved to the forgotten table, so they won't appear in sessions.

    Args:
        sessions: List of Session objects
        history_duration_seconds: Sessions must be older than this to summarize
        min_messages: Minimum messages in a session to summarize

    Returns:
        List of eligible Session objects
    """
    cutoff = time.time() - history_duration_seconds

    eligible = []
    for session in sessions:
        # Must be older than the history window
        if session.end_timestamp >= cutoff:
            continue

        # Must have enough messages
        if session.message_count < min_messages:
            continue

        eligible.append(session)

    return eligible


def format_session_for_summarization(session: Session) -> str:
    """Format a session's messages as text for the summarization prompt."""
    lines = []
    for msg in session.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        elif role == "system":
            # Skip system messages in summarization
            continue
        else:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token average)."""
    return len(text) // 4


def split_session_into_chunks(
    session: Session,
    max_tokens_per_chunk: int = 4000,  # Leave room for prompt template and response
) -> list[Session]:
    """Split a large session into smaller chunks that fit within token limits.
    
    Each chunk maintains chronological order and is treated as a sub-session
    that will be summarized independently.
    
    Args:
        session: The session to split
        max_tokens_per_chunk: Maximum tokens per chunk (conservative estimate)
        
    Returns:
        List of Session objects (sub-sessions), or [session] if no split needed
    """
    # Format to estimate size
    full_text = format_session_for_summarization(session)
    total_tokens = estimate_tokens(full_text)
    
    # If it fits, return as-is
    if total_tokens <= max_tokens_per_chunk:
        return [session]
    
    # Split into chunks
    chunks = []
    current_msgs = []
    current_ids = []
    current_tokens = 0
    
    for msg, msg_id in zip(session.messages, session.message_ids):
        # Estimate this message's tokens
        role = msg.get("role", "user")
        content = msg.get("content", "")
        msg_text = f"{role}: {content}"
        msg_tokens = estimate_tokens(msg_text)
        
        # If adding this message would exceed limit, start new chunk
        if current_tokens + msg_tokens > max_tokens_per_chunk and current_msgs:
            # Create chunk from current messages
            chunks.append(Session(
                start_timestamp=current_msgs[0].get("timestamp", 0),
                end_timestamp=current_msgs[-1].get("timestamp", 0),
                messages=current_msgs.copy(),
                message_ids=current_ids.copy(),
            ))
            current_msgs = []
            current_ids = []
            current_tokens = 0
        
        current_msgs.append(msg)
        current_ids.append(msg_id)
        current_tokens += msg_tokens
    
    # Don't forget the last chunk
    if current_msgs:
        chunks.append(Session(
            start_timestamp=current_msgs[0].get("timestamp", 0),
            end_timestamp=current_msgs[-1].get("timestamp", 0),
            messages=current_msgs.copy(),
            message_ids=current_ids.copy(),
        ))
    
    logger.info(f"Split session with {session.message_count} messages into {len(chunks)} chunks")
    return chunks


def summarize_session_with_llm(
    session: Session,
    service_url: str = "http://127.0.0.1:8642",
    timeout: float = 120.0,  # Increased from 30s - small models need more time
) -> str | None:
    """Summarize a session using the LLM service.

    Calls /v1/llm/complete which uses the already-loaded model in the service.

    Args:
        session: The session to summarize
        service_url: Base URL of the llm-service
        timeout: Request timeout in seconds

    Returns:
        Summary text if successful, None on failure
    """
    conversation_text = format_session_for_summarization(session)
    prompt = SUMMARIZATION_PROMPT.format(messages=conversation_text)
    
    # Log the prompt size for debugging
    logger.debug(f"Summarization prompt: {len(prompt)} chars for {session.message_count} messages")

    try:
        response = httpx.post(
            f"{service_url}/v1/llm/complete",
            json={
                "prompt": prompt,
                "system": "You are a helpful assistant that summarizes conversations concisely.",
                "max_tokens": 200,
                "temperature": 0.3,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("content", "").strip()
        
        # Check for error responses that got saved as content
        error_indicators = [
            "Error:",
            "exception occurred",
            "exceed context window",
            "tokens exceed",
            "CUDA error",
            "out of memory",
        ]
        for indicator in error_indicators:
            if indicator.lower() in content.lower():
                logger.error(f"LLM returned error in content: {content[:100]}...")
                return None
        
        return content
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            logger.warning("LLM service has no model loaded")
        else:
            logger.error(f"LLM service error: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.TimeoutException:
        logger.error(f"LLM service timed out after {timeout}s - model may be slow or overloaded")
        return None
    except httpx.RequestError as e:
        logger.error(f"LLM service connection error: {e}")
        return None


def summarize_session_heuristic(session: Session) -> str:
    """Generate a heuristic summary without LLM.

    This is a fallback when the LLM service is unavailable.
    Extracts key information from user messages.
    """
    user_messages = [
        msg.get("content", "").strip()
        for msg in session.messages
        if msg.get("role") == "user"
    ]

    if not user_messages:
        return f"Conversation session with {session.message_count} messages."

    # Take first and last user messages to show topic range
    first_msg = user_messages[0][:100].strip()
    last_msg = user_messages[-1][:100].strip() if len(user_messages) > 1 else ""

    # Build summary with date for time context
    start_time = session.start_datetime.strftime("%Y-%m-%d")
    start_time_full = session.start_datetime.strftime("%Y-%m-%d %H:%M")
    msg_count = session.message_count

    summary_parts = [f"On {start_time}: Conversation about"]
    
    # Include topic hints from first message
    first_topic = first_msg[:80] + "..." if len(first_msg) > 80 else first_msg
    summary_parts.append(f"\"{first_topic}\"")
    
    if last_msg and last_msg != first_msg:
        last_topic = last_msg[:80] + "..." if len(last_msg) > 80 else last_msg
        summary_parts.append(f"and later \"{last_topic}\"")
    
    summary_parts.append(f"({msg_count} messages total)")

    return " ".join(summary_parts)


class HistorySummarizer:
    """Manages history summarization for a bot.

    Usage:
        summarizer = HistorySummarizer(config, bot_id="nova")

        # Dry run to preview
        preview = summarizer.preview_summarizable_sessions()

        # Actually summarize
        results = summarizer.summarize_eligible_sessions()
    """

    def __init__(
        self,
        config: Any,
        bot_id: str = "nova",
        service_url: str | None = None,
        summarize_fn: "Callable[[Session], str | None] | None" = None,
    ):
        self.config = config
        self.bot_id = bot_id

        # Get service URL from config or use default
        host = getattr(config, "SERVICE_HOST", "127.0.0.1")
        port = getattr(config, "SERVICE_PORT", 8642)
        self.service_url = service_url or f"http://{host}:{port}"
        
        # Custom summarization function (provided by API to avoid HTTP self-call)
        self._summarize_fn = summarize_fn

        # Get summarization settings from config
        self.session_gap_seconds = getattr(config, "SUMMARIZATION_SESSION_GAP_SECONDS", 3600)
        self.min_messages = getattr(config, "SUMMARIZATION_MIN_MESSAGES", 4)
        self.history_duration = getattr(config, "HISTORY_DURATION", 1800)

        # Initialize the PostgreSQL backend
        from .postgresql import PostgreSQLMemoryBackend
        self._backend = PostgreSQLMemoryBackend(config, bot_id)

    def _get_all_messages(self) -> list[dict]:
        """Fetch all messages from the database."""
        with self._backend.engine.connect() as conn:
            from sqlalchemy import text
            sql = text(f"""
                SELECT id, role, content, timestamp, summarized, summary_metadata
                FROM {self._backend._messages_table_name}
                ORDER BY timestamp ASC
            """)
            rows = conn.execute(sql).fetchall()
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                    "summarized": row.summarized,
                    "summary_metadata": row.summary_metadata,
                }
                for row in rows
            ]

    def detect_all_sessions(self) -> list[Session]:
        """Detect all conversation sessions."""
        messages = self._get_all_messages()
        return detect_sessions(messages, self.session_gap_seconds)

    def preview_summarizable_sessions(self) -> list[Session]:
        """Preview sessions that would be summarized (dry run).
        
        Since summarized messages are moved to the forgotten table,
        we only see unsummarized messages here. No need to track
        which messages have been summarized - they simply don't exist
        in the messages table anymore.
        """
        messages = self._get_all_messages()
        sessions = detect_sessions(messages, self.session_gap_seconds)
        # Filter to only non-summary messages for session detection
        regular_messages = [m for m in messages if m.get("role") != "summary"]
        sessions = detect_sessions(regular_messages, self.session_gap_seconds)
        return find_summarizable_sessions(
            sessions,
            history_duration_seconds=self.history_duration,
            min_messages=self.min_messages,
        )

    def summarize_session(
        self,
        session: Session,
        use_heuristic_fallback: bool = True,
    ) -> dict:
        """Summarize a single session and store the result.

        Args:
            session: The session to summarize
            use_heuristic_fallback: If True, use heuristic if LLM fails

        Returns:
            Dict with 'success', 'summary_id', 'summary_text', 'method'
        """
        # Try LLM first - use custom function if provided, otherwise HTTP
        if self._summarize_fn:
            # Use provided function (avoids HTTP self-call when run in API)
            summary_text = self._summarize_fn(session)
        else:
            # Use HTTP endpoint (for CLI usage)
            summary_text = summarize_session_with_llm(session, self.service_url)
        method = "llm"

        if not summary_text and use_heuristic_fallback:
            summary_text = summarize_session_heuristic(session)
            method = "heuristic"

        if not summary_text:
            return {"success": False, "error": "Failed to generate summary"}

        # Store the summary
        summary_id = str(uuid.uuid4())
        summary_metadata = {
            "session_start": session.start_timestamp,
            "session_end": session.end_timestamp,
            "message_ids": session.message_ids,
            "message_count": session.message_count,
            "summarization_method": method,
            "created_at": time.time(),
        }

        with self._backend.engine.connect() as conn:
            from sqlalchemy import text

            # Move original messages to the forgotten table (preserves them for restoration)
            if session.message_ids:
                # Copy messages to forgotten table
                copy_sql = text(f"""
                    INSERT INTO {self._backend._forgotten_table_name}
                    (id, role, content, timestamp, session_id, processed, created_at, forgotten_at)
                    SELECT id, role, content, timestamp, session_id, processed, created_at, CURRENT_TIMESTAMP
                    FROM {self._backend._messages_table_name}
                    WHERE id = ANY(:ids)
                    ON CONFLICT (id) DO NOTHING
                """)
                conn.execute(copy_sql, {"ids": session.message_ids})
                
                # Delete original messages
                delete_sql = text(f"""
                    DELETE FROM {self._backend._messages_table_name}
                    WHERE id = ANY(:ids)
                """)
                conn.execute(delete_sql, {"ids": session.message_ids})

            # Insert summary as a message with role='summary'
            # Timestamp is session end time for natural ordering
            insert_sql = text(f"""
                INSERT INTO {self._backend._messages_table_name}
                (id, role, content, timestamp, summary_metadata, created_at)
                VALUES (:id, 'summary', :content, :timestamp, :metadata, CURRENT_TIMESTAMP)
            """)
            conn.execute(insert_sql, {
                "id": summary_id,
                "content": summary_text,
                "timestamp": session.end_timestamp,
                "metadata": json.dumps(summary_metadata),
            })

            conn.commit()

        logger.info(f"Created summary {summary_id[:8]} for session with {session.message_count} messages ({method})")

        return {
            "success": True,
            "summary_id": summary_id,
            "summary_text": summary_text,
            "method": method,
            "message_count": session.message_count,
        }

    def summarize_eligible_sessions(
        self,
        use_heuristic_fallback: bool = True,
        max_tokens_per_chunk: int = 4000,
    ) -> dict:
        """Summarize all eligible sessions.
        
        Large sessions are automatically split into chunks that fit within
        the model's context window.

        Returns:
            Dict with 'sessions_summarized', 'messages_summarized', 'results', 'errors'
        """
        eligible = self.preview_summarizable_sessions()

        if not eligible:
            return {
                "sessions_summarized": 0,
                "messages_summarized": 0,
                "results": [],
                "errors": [],
            }

        results = []
        errors = []
        total_messages = 0

        for session in eligible:
            try:
                # Check if session needs to be chunked
                chunks = split_session_into_chunks(session, max_tokens_per_chunk)
                
                for i, chunk in enumerate(chunks):
                    chunk_label = f" (chunk {i+1}/{len(chunks)})" if len(chunks) > 1 else ""
                    try:
                        result = self.summarize_session(chunk, use_heuristic_fallback)
                        if result.get("success"):
                            results.append(result)
                            total_messages += chunk.message_count
                            logger.info(f"Summarized{chunk_label}: {chunk.message_count} messages")
                        else:
                            error_msg = result.get("error", "Unknown error")
                            errors.append(f"Session{chunk_label}: {error_msg}")
                            logger.error(f"Failed to summarize{chunk_label}: {error_msg}")
                    except Exception as e:
                        logger.error(f"Error summarizing session{chunk_label}: {e}")
                        errors.append(f"Session{chunk_label}: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing session: {e}")
                errors.append(str(e))

        return {
            "sessions_summarized": len(results),
            "messages_summarized": total_messages,
            "results": results,
            "errors": errors,
        }

    def list_summaries(self) -> list[dict]:
        """List all existing summaries."""
        with self._backend.engine.connect() as conn:
            from sqlalchemy import text
            sql = text(f"""
                SELECT id, content, timestamp, summary_metadata
                FROM {self._backend._messages_table_name}
                WHERE role = 'summary'
                ORDER BY timestamp DESC
            """)
            rows = conn.execute(sql).fetchall()

            summaries = []
            for row in rows:
                metadata = row.summary_metadata or {}
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                summaries.append({
                    "id": row.id,
                    "content": row.content,
                    "timestamp": row.timestamp,
                    "session_start": metadata.get("session_start"),
                    "session_end": metadata.get("session_end"),
                    "message_count": metadata.get("message_count", 0),
                    "method": metadata.get("summarization_method", "unknown"),
                })

            return summaries

    def delete_summary(self, summary_id: str) -> dict:
        """Delete a summary and unmark the original messages.

        Args:
            summary_id: Full or prefix of summary UUID

        Returns:
            Dict with 'success', 'messages_restored'
        """
        with self._backend.engine.connect() as conn:
            from sqlalchemy import text

            # Find the summary (support prefix matching)
            if len(summary_id) < 36:
                find_sql = text(f"""
                    SELECT id, summary_metadata
                    FROM {self._backend._messages_table_name}
                    WHERE role = 'summary' AND id LIKE :pattern
                    LIMIT 1
                """)
                row = conn.execute(find_sql, {"pattern": f"{summary_id}%"}).fetchone()
            else:
                find_sql = text(f"""
                    SELECT id, summary_metadata
                    FROM {self._backend._messages_table_name}
                    WHERE role = 'summary' AND id = :id
                """)
                row = conn.execute(find_sql, {"id": summary_id}).fetchone()

            if not row:
                return {"success": False, "error": f"Summary '{summary_id}' not found"}

            full_id = row.id
            metadata = row.summary_metadata or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            message_ids = metadata.get("message_ids", [])

            # Unmark the original messages
            if message_ids:
                update_sql = text(f"""
                    UPDATE {self._backend._messages_table_name}
                    SET summarized = FALSE
                    WHERE id = ANY(:ids)
                """)
                conn.execute(update_sql, {"ids": message_ids})

            # Delete the summary
            delete_sql = text(f"""
                DELETE FROM {self._backend._messages_table_name}
                WHERE id = :id
            """)
            conn.execute(delete_sql, {"id": full_id})

            conn.commit()

        logger.info(f"Deleted summary {full_id[:8]}, restored {len(message_ids)} messages")

        return {
            "success": True,
            "summary_id": full_id,
            "messages_restored": len(message_ids),
        }

    def get_summary_count(self) -> int:
        """Get the count of existing summaries."""
        with self._backend.engine.connect() as conn:
            from sqlalchemy import text
            sql = text(f"""
                SELECT COUNT(*) FROM {self._backend._messages_table_name}
                WHERE role = 'summary'
            """)
            return conn.execute(sql).scalar() or 0
