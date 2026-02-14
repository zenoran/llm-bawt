"""History summarization for llm-bawt.

This module provides session detection and summarization functionality
to condense older conversation history, allowing more context to fit
in the LLM window.

Sessions are detected by timestamp gaps (default 1 hour). Sessions older
than HISTORY_DURATION (default 30 min) are eligible for summarization.
Summaries are non-destructive: raw messages remain in the messages table.
"""

import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .summarization import Session

logger = logging.getLogger(__name__)

# Summarization prompt template
SUMMARIZATION_PROMPT = """Summarize this conversation session for future continuation.

Requirements:
0. Be strictly faithful to source material. Do not sanitize, soften, editorialize, or introduce bias. Do not omit concrete facts due to style preferences.
1. Capture key topics, decisions, and unresolved threads.
2. Capture user intent/goals (what they were trying to accomplish).
3. Capture emotional tone/mood trajectory and communication style cues.
4. Keep factual continuity anchors (names, constraints, preferences stated in-session).
5. Preserve critical implementation details when present (files, commands, config values, errors, fixes, decisions).
6. Do not add facts not present in the conversation.

Output format (plain text, exactly these sections):
Summary:
Key Details:
Intent:
Tone:
Open Loops:

Target detail level: moderately detailed (about 180-420 words total across all sections).
Prefer retaining concrete specifics over brevity when trade-offs are needed.

Conversation:
{messages}
"""


BATCH_SUMMARIZATION_PROMPT = """You are summarizing multiple conversation sessions together.

Goal:
1. Produce one concise structured summary for EACH session.
2. Use cross-session awareness to enrich each summary with recurring patterns/themes.
3. Do not invent facts.
4. Be strictly faithful to source material. Do not sanitize, soften, editorialize, or introduce bias.

Return STRICT JSON only (no markdown, no prose):
{{
    "global_themes": ["theme 1", "theme 2"],
    "summaries": [
        {{
            "session_index": 1,
            "summary": "...",
            "key_details": "...",
            "intent": "...",
            "tone": "...",
            "open_loops": "...",
            "cross_session_links": "..."
        }}
    ]
}}

Constraints:
- Keep each session summary moderately detailed (~180-420 words worth of content across fields).
- "cross_session_links" should mention recurring user goals/patterns if present; otherwise empty string.
- session_index must match input ordering exactly.

Sessions:
{sessions_blob}
"""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from model output."""
    if not text:
        return None

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0:
        return None

    # If no closing brace is present, attempt best-effort brace completion.
    if end <= start:
        snippet = candidate[start:]
        open_braces = snippet.count("{")
        close_braces = snippet.count("}")
        open_brackets = snippet.count("[")
        close_brackets = snippet.count("]")
        if open_brackets > close_brackets:
            snippet += "]" * (open_brackets - close_brackets)
        if open_braces > close_braces:
            snippet += "}" * (open_braces - close_braces)
        try:
            parsed = json.loads(snippet)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    snippet = candidate[start : end + 1]
    try:
        parsed = json.loads(snippet)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def extract_summary_sections(text: str) -> dict[str, str]:
    """Extract structured fields from summary text."""
    sections = {
        "summary": "",
        "key_details": "",
        "intent": "",
        "tone": "",
        "open_loops": "",
    }
    if not text:
        return sections

    current_key: str | None = None
    buffers: dict[str, list[str]] = {k: [] for k in sections}
    key_map = {
        "summary:": "summary",
        "key details:": "key_details",
        "key_details:": "key_details",
        "intent:": "intent",
        "tone:": "tone",
        "open loops:": "open_loops",
        "open_loops:": "open_loops",
    }

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_key:
                buffers[current_key].append("")
            continue
        normalized = line.lower()
        matched = False
        for prefix, key in key_map.items():
            if normalized.startswith(prefix):
                current_key = key
                value = line[len(prefix):].strip()
                if value:
                    buffers[key].append(value)
                matched = True
                break
        if matched:
            continue
        if current_key:
            buffers[current_key].append(line)
        else:
            buffers["summary"].append(line)

    for key, lines in buffers.items():
        value = "\n".join(lines).strip()
        sections[key] = value

    return sections


def normalize_structured_summary_text(text: str) -> str:
    """Normalize summary text into consistent sectioned format."""
    data = extract_summary_sections(text)
    return "\n".join(
        [
            f"Summary: {data['summary']}".rstrip(),
            f"Key Details: {data['key_details']}".rstrip(),
            f"Intent: {data['intent']}".rstrip(),
            f"Tone: {data['tone']}".rstrip(),
            f"Open Loops: {data['open_loops']}".rstrip(),
        ]
    ).strip()


def is_summary_low_quality(text: str, source_session: "Session | None" = None) -> bool:
    """Heuristic gate for vague/underspecified summaries.

    Reject summaries that are too short or dominated by generic wording.
    When *source_session* is provided, thresholds are scaled to the actual
    conversation weight so trivial sessions (e.g. "hi" / "testing") don't
    trigger expensive per-session retry cascades for content that genuinely
    has nothing more to say.
    """
    sections = extract_summary_sections(text)
    summary = (sections.get("summary") or "").strip()
    key_details = (sections.get("key_details") or "").strip()
    intent = (sections.get("intent") or "").strip()
    open_loops = (sections.get("open_loops") or "").strip()

    total_len = len(summary) + len(key_details) + len(intent) + len(open_loops)

    # For sessions with very little real content, accept shorter summaries.
    source_chars = 0
    if source_session is not None:
        source_chars = sum(len(m.get("content", "")) for m in source_session.messages)

    # Trivial sessions (< 200 chars of source content) → accept anything non-empty.
    if source_chars > 0 and source_chars < 200:
        return total_len < 20

    # Light sessions (< 600 chars) → lenient thresholds.
    if source_chars > 0 and source_chars < 600:
        return total_len < 80

    if total_len < 260:
        return True

    if len(key_details) < 120:
        return True

    generic_patterns = [
        r"\bgame discussion\b",
        r"\btesting\b",
        r"\bplayful\b",
        r"\bvent\/motivation\b",
        r"\bmisc\b",
        r"\bvarious topics\b",
    ]
    merged = f"{summary}\n{key_details}\n{intent}\n{open_loops}".lower()
    if any(re.search(pat, merged) for pat in generic_patterns):
        return True

    detail_signals = 0
    if re.search(r"\b(error|exception|command|config|file|path|api|model|session|timestamp|search)\b", merged):
        detail_signals += 1
    if re.search(r"\b\d{2,}\b", merged):
        detail_signals += 1
    if re.search(r"[`'\"]", merged):
        detail_signals += 1
    if detail_signals == 0:
        return True

    return False


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
    skip_low_signal: bool = True,
    min_user_messages: int = 2,
    min_total_content_chars: int = 160,
    min_meaningful_turns: int = 3,
) -> list[Session]:
    """Filter sessions to those eligible for summarization.

    A session is eligible if:
    - It ended more than history_duration_seconds ago
    - It has at least min_messages messages
    
    Args:
        sessions: List of Session objects
        history_duration_seconds: Sessions must be older than this to summarize
        min_messages: Minimum messages in a session to summarize
        skip_low_signal: If True, skip short/test/greeting-only sessions
        min_user_messages: Minimum user messages required for a meaningful session
        min_total_content_chars: Minimum total non-whitespace chars across user+assistant turns
        min_meaningful_turns: Minimum count of substantial user/assistant turns

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

        if skip_low_signal and is_low_signal_session(
            session,
            min_user_messages=min_user_messages,
            min_total_content_chars=min_total_content_chars,
            min_meaningful_turns=min_meaningful_turns,
        ):
            continue

        eligible.append(session)

    return eligible


def is_low_signal_session(
    session: Session,
    min_user_messages: int = 2,
    min_total_content_chars: int = 160,
    min_meaningful_turns: int = 3,
) -> bool:
    """Heuristic filter for low-value sessions (greetings/tests/noise)."""
    user_messages = [
        (msg.get("content", "") or "").strip().lower()
        for msg in session.messages
        if msg.get("role") == "user"
    ]
    assistant_messages = [
        (msg.get("content", "") or "").strip()
        for msg in session.messages
        if msg.get("role") == "assistant"
    ]

    if len(user_messages) < min_user_messages:
        return True

    total_chars = sum(len(m.strip()) for m in user_messages) + sum(len(m.strip()) for m in assistant_messages)
    if total_chars < min_total_content_chars:
        return True

    meaningful_turns = 0
    for msg in session.messages:
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = (msg.get("content", "") or "").strip()
        if len(content) >= 20:
            meaningful_turns += 1
    if meaningful_turns < min_meaningful_turns:
        return True

    noise_patterns = {
        "hi", "hey", "hello", "yo", "sup", "test", "testing", "ok", "okay", "k", "lol", "lmao", "...",
    }
    short_noise_count = 0
    for user_text in user_messages:
        normalized = re.sub(r"[^a-z0-9\s]+", "", user_text).strip()
        compact = re.sub(r"\s+", " ", normalized)
        if compact in noise_patterns or len(compact) <= 3:
            short_noise_count += 1

    if user_messages and short_noise_count / len(user_messages) >= 0.8:
        return True

    return False


def estimate_session_token_savings(session: Session) -> int:
    """Estimate token savings from replacing a session with a summary.

    This is a heuristic for prioritization, not an exact tokenizer measurement.
    """
    raw_tokens = estimate_tokens(format_session_for_summarization(session))
    # Typical summary target is short but non-trivial; clamp to a practical range.
    expected_summary_tokens = min(200, max(48, raw_tokens // 5))
    return max(raw_tokens - expected_summary_tokens, 0)


def prioritize_summarizable_sessions(sessions: list[Session]) -> list[Session]:
    """Prioritize sessions by token savings impact (largest first)."""
    return sorted(
        sessions,
        key=lambda s: (-estimate_session_token_savings(s), s.start_timestamp),
    )


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
    if max_tokens_per_chunk <= 0:
        return [session]

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
                "max_tokens": 320,
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
    Produces the same structured sections as the LLM prompt contract.
    """
    def clean(text: str) -> str:
        return " ".join((text or "").strip().split())

    def clip(text: str, limit: int = 140) -> str:
        text = clean(text)
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    user_messages = [clean(msg.get("content", "")) for msg in session.messages if msg.get("role") == "user"]
    assistant_messages = [clean(msg.get("content", "")) for msg in session.messages if msg.get("role") == "assistant"]
    user_messages = [m for m in user_messages if m]
    assistant_messages = [m for m in assistant_messages if m]

    start_label = session.start_datetime.strftime("%Y-%m-%d %H:%M")
    end_label = session.end_datetime.strftime("%H:%M")
    duration_min = max(int(session.duration_seconds // 60), 0)
    header = f"Session on {start_label}-{end_label} ({duration_min}m, {session.message_count} msgs)."

    if not user_messages:
        return (
            f"Summary: {header}\n"
            "Intent: No clear user intent detected.\n"
            "Tone: Neutral.\n"
            "Open Loops: None identified."
        )

    first_user = clip(user_messages[0], 100)
    last_user = clip(user_messages[-1], 100)

    thread = f"Primary thread started with: \"{first_user}\"."
    if last_user != first_user:
        thread += f" Ended around: \"{last_user}\"."

    intent_candidates = [
        m for m in reversed(user_messages)
        if "?" in m or any(k in m.lower() for k in ("need", "want", "can you", "please", "should", "help"))
    ]
    intent_text = clip(intent_candidates[0], 120) if intent_candidates else clip(last_user, 120)

    all_user_text = " ".join(user_messages).lower()
    frustrated_markers = ("wtf", "fuck", "frustrat", "annoy", "upset", "angry", "bullshit")
    warm_markers = ("thanks", "thank you", "good call", "appreciate", "cool", "sounds good")
    if any(marker in all_user_text for marker in frustrated_markers):
        tone = "Tense/frustrated at points; direct language."
    elif any(marker in all_user_text for marker in warm_markers):
        tone = "Mostly cooperative and positive."
    else:
        tone = "Neutral-pragmatic."

    recent_questions = [clip(m, 120) for m in user_messages[-8:] if "?" in m]
    if recent_questions:
        open_loops = "; ".join(recent_questions[-2:])
    elif assistant_messages:
        open_loops = f"Confirm outcome/follow-up from assistant's latest guidance: \"{clip(assistant_messages[-1], 120)}\"."
    else:
        open_loops = "None explicit."

    return (
        f"Summary: {header} {thread}\n"
        f"Intent: {intent_text}\n"
        f"Tone: {tone}\n"
        f"Open Loops: {open_loops}"
    )


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
        summarize_batch_fn: "Callable[[list[Session]], dict[int, str] | None] | None" = None,
        settings_getter: "Callable[[str, Any], Any] | None" = None,
    ):
        self.config = config
        self.bot_id = bot_id
        self._settings_getter = settings_getter

        # Get service URL from config or use default
        host = getattr(config, "SERVICE_HOST", "127.0.0.1")
        port = getattr(config, "SERVICE_PORT", 8642)
        self.service_url = service_url or f"http://{host}:{port}"

        # Custom summarization function (provided by API to avoid HTTP self-call)
        self._summarize_fn = summarize_fn
        self._summarize_batch_fn = summarize_batch_fn

        # Get summarization settings — resolved via runtime settings if available
        self.session_gap_seconds = self._setting(
            "summarization_session_gap_seconds",
            getattr(config, "SUMMARIZATION_SESSION_GAP_SECONDS", 3600),
        )
        self.min_messages = self._setting(
            "summarization_min_messages",
            getattr(config, "SUMMARIZATION_MIN_MESSAGES", 4),
        )
        self.skip_low_signal = bool(
            self._setting(
                "summarization_skip_low_signal",
                getattr(config, "SUMMARIZATION_SKIP_LOW_SIGNAL", True),
            )
        )
        self.min_user_messages_for_summary = int(
            self._setting(
                "summarization_min_user_messages",
                getattr(config, "SUMMARIZATION_MIN_USER_MESSAGES", 2),
            )
        )
        self.min_content_chars_for_summary = int(
            self._setting(
                "summarization_min_content_chars",
                getattr(config, "SUMMARIZATION_MIN_CONTENT_CHARS", 160),
            )
        )
        self.min_meaningful_turns_for_summary = int(
            self._setting(
                "summarization_min_meaningful_turns",
                getattr(config, "SUMMARIZATION_MIN_MEANINGFUL_TURNS", 3),
            )
        )
        self.history_duration = self._setting(
            "history_duration_seconds",
            getattr(config, "HISTORY_DURATION_SECONDS", 1800),
        )

        # Initialize the PostgreSQL backend
        from .postgresql import PostgreSQLMemoryBackend
        self._backend = PostgreSQLMemoryBackend(config, bot_id)

    def _setting(self, key: str, fallback: Any) -> Any:
        """Resolve a setting through the runtime settings getter if available."""
        if self._settings_getter:
            return self._settings_getter(key, fallback)
        return fallback

    def _get_all_messages(self) -> list[dict]:
        """Fetch all messages from the database."""
        with self._backend.engine.connect() as conn:
            from sqlalchemy import text
            sql = text(f"""
                SELECT id, role, content, timestamp, summarized, recalled_history, summary_metadata
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
                    "recalled_history": row.recalled_history,
                    "summary_metadata": row.summary_metadata,
                }
                for row in rows
            ]

    def _find_existing_summary_ids(self, conn: Any, session: Session) -> list[str]:
        """Find historical summaries that already cover this exact message set."""
        from sqlalchemy import text

        sql = text(
            f"""
            SELECT id, summary_metadata
            FROM {self._backend._messages_table_name}
            WHERE role = 'summary'
              AND timestamp >= :start_ts - 1
              AND timestamp <= :end_ts + 1
            ORDER BY timestamp DESC
            """
        )
        rows = conn.execute(
            sql,
            {"start_ts": session.start_timestamp, "end_ts": session.end_timestamp},
        ).fetchall()

        matching_ids: list[str] = []
        for row in rows:
            meta = row.summary_metadata or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    continue

            if not isinstance(meta, dict):
                continue

            if meta.get("summary_type") != "historical":
                continue

            existing_ids = meta.get("message_ids", [])
            if isinstance(existing_ids, list) and existing_ids == session.message_ids:
                matching_ids.append(row.id)

        return matching_ids

    def _find_replaceable_summary_ids(self, conn: Any, session: Session) -> list[str]:
        """Find historical summary rows that should be replaced for this session/chunk.

        This is intentionally broader than exact message-id matching so rebuild mode can
        replace legacy summaries and previously unchunked summaries that overlap the
        same source messages.
        """
        from sqlalchemy import text

        sql = text(
            f"""
            SELECT id, summary_metadata, timestamp
            FROM {self._backend._messages_table_name}
            WHERE role = 'summary'
              AND timestamp >= :start_ts - 1
              AND timestamp <= :end_ts + 1
            ORDER BY timestamp DESC
            """
        )
        rows = conn.execute(
            sql,
            {"start_ts": session.start_timestamp, "end_ts": session.end_timestamp},
        ).fetchall()

        chunk_ids = set(session.message_ids)
        replace_ids: list[str] = []

        for row in rows:
            meta = row.summary_metadata or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}

            if not isinstance(meta, dict):
                meta = {}

            # Never touch active-session rolling summaries here.
            if meta.get("summary_type") == "active_session":
                continue

            existing_ids = meta.get("message_ids", [])
            if isinstance(existing_ids, list) and existing_ids:
                # Exact match or any overlap with current source messages.
                if existing_ids == session.message_ids or bool(chunk_ids.intersection(existing_ids)):
                    replace_ids.append(row.id)
                    continue

            # Legacy/metadata-light fallback: overlapping session time window.
            try:
                existing_start = float(meta.get("session_start"))
                existing_end = float(meta.get("session_end"))
                if not (existing_end < session.start_timestamp or existing_start > session.end_timestamp):
                    replace_ids.append(row.id)
                    continue
            except (TypeError, ValueError):
                pass

            # Last-resort legacy fallback: non-active summary row in the timestamp window.
            replace_ids.append(row.id)

        return list(dict.fromkeys(replace_ids))

    def detect_all_sessions(self) -> list[Session]:
        """Detect all conversation sessions."""
        messages = self._get_all_messages()
        return detect_sessions(messages, self.session_gap_seconds)

    def preview_summarizable_sessions(self) -> list[Session]:
        """Preview sessions that would be summarized (dry run).
        """
        messages = self._get_all_messages()
        # Only detect sessions on raw, non-recalled, unsummarized messages.
        regular_messages = [
            m
            for m in messages
            if m.get("role") != "summary"
            and not bool(m.get("recalled_history"))
            and not bool(m.get("summarized"))
        ]
        sessions = detect_sessions(regular_messages, self.session_gap_seconds)
        eligible = find_summarizable_sessions(
            sessions,
            history_duration_seconds=self.history_duration,
            min_messages=self.min_messages,
            skip_low_signal=self.skip_low_signal,
            min_user_messages=self.min_user_messages_for_summary,
            min_total_content_chars=self.min_content_chars_for_summary,
            min_meaningful_turns=self.min_meaningful_turns_for_summary,
        )
        return prioritize_summarizable_sessions(eligible)

    def summarize_session(
        self,
        session: Session,
        use_heuristic_fallback: bool = True,
        replace_existing: bool = False,
        summary_text_override: str | None = None,
        method_override: str | None = None,
    ) -> dict:
        """Summarize a single session and store the result.

        Args:
            session: The session to summarize
            use_heuristic_fallback: If True, use heuristic if LLM fails

        Returns:
            Dict with 'success', 'summary_id', 'summary_text', 'method'
        """
        # Try LLM first - use custom function if provided, otherwise HTTP
        if summary_text_override is not None:
            summary_text = summary_text_override
            method = method_override or "llm"
        else:
            if self._summarize_fn:
                # Use provided function (avoids HTTP self-call when run in API)
                summary_text = self._summarize_fn(session)
            else:
                # Use HTTP endpoint (for CLI usage)
                summary_text = summarize_session_with_llm(session, self.service_url)
            method = method_override or "llm"

        if not summary_text and use_heuristic_fallback:
            summary_text = summarize_session_heuristic(session)
            method = "heuristic"

        if not summary_text:
            return {"success": False, "error": "Failed to generate summary"}

        normalized_text = normalize_structured_summary_text(summary_text)
        if not normalized_text.startswith("[HISTORICAL SUMMARY]"):
            normalized_text = f"[HISTORICAL SUMMARY]\n{normalized_text.strip()}"
        sections = extract_summary_sections(normalized_text)

        # Store the summary
        summary_id = str(uuid.uuid4())
        summary_metadata = {
            "session_start": session.start_timestamp,
            "session_end": session.end_timestamp,
            "message_ids": session.message_ids,
            "message_count": session.message_count,
            "summarization_method": method,
            "created_at": time.time(),
            "summary_type": "historical",
            "summary_sections": sections,
            "intent": sections.get("intent", ""),
            "tone": sections.get("tone", ""),
            "open_loops": sections.get("open_loops", ""),
        }

        with self._backend.engine.connect() as conn:
            from sqlalchemy import text

            existing_summary_ids = self._find_existing_summary_ids(conn, session)
            if existing_summary_ids and not replace_existing:
                return {
                    "success": True,
                    "summary_id": existing_summary_ids[0],
                    "summary_text": None,
                    "method": "existing",
                    "message_count": session.message_count,
                    "created": False,
                    "replaced_existing": 0,
                }
            replaced_existing = 0
            if replace_existing:
                replace_ids = self._find_replaceable_summary_ids(conn, session)
                if replace_ids:
                    delete_sql = text(
                        f"""
                        DELETE FROM {self._backend._messages_table_name}
                        WHERE id = ANY(:ids)
                        """
                    )
                    conn.execute(delete_sql, {"ids": replace_ids})
                    replaced_existing = len(replace_ids)

            # Mark source messages as summarized, but keep them in place.
            if session.message_ids:
                update_sql = text(
                    f"""
                    UPDATE {self._backend._messages_table_name}
                    SET summarized = TRUE
                    WHERE id = ANY(:ids)
                    """
                )
                conn.execute(update_sql, {"ids": session.message_ids})

            # Insert summary as a message with role='summary'
            # Timestamp is session end time for natural ordering
            insert_sql = text(f"""
                INSERT INTO {self._backend._messages_table_name}
                (id, role, content, timestamp, summary_metadata, created_at)
                VALUES (:id, 'summary', :content, :timestamp, :metadata, CURRENT_TIMESTAMP)
            """)
            conn.execute(insert_sql, {
                "id": summary_id,
                "content": normalized_text,
                "timestamp": session.end_timestamp,
                "metadata": json.dumps(summary_metadata),
            })

            conn.commit()

        logger.info(f"Created summary {summary_id[:8]} for session with {session.message_count} messages ({method})")

        return {
            "success": True,
            "summary_id": summary_id,
            "summary_text": normalized_text,
            "method": method,
            "message_count": session.message_count,
            "created": True,
            "replaced_existing": replaced_existing,
            "session_start": session.start_timestamp,
            "session_end": session.end_timestamp,
        }

    def _summarize_sessions_batch(
        self,
        sessions: list[Session],
        use_heuristic_fallback: bool,
        replace_existing: bool,
    ) -> tuple[list[dict], list[str], int, int, int]:
        """Summarize multiple sessions in one LLM call when batch callable is available.

        Returns:
            (results, errors, created_count, total_messages, replaced_count)
        """
        if not sessions or not self._summarize_batch_fn:
            return ([], [], 0, 0, 0)

        batch_size = int(self._setting("summarization_batch_size", 5) or 5)
        batch_size = max(1, batch_size)

        # Build micro-batch chunks.
        micro_batches: list[tuple[int, list[Session]]] = []
        for start in range(0, len(sessions), batch_size):
            micro_batches.append((start, sessions[start : start + batch_size]))

        # Run micro-batches concurrently — each is an independent API call.
        concurrency = int(self._setting("summarization_batch_concurrency", 3) or 3)
        concurrency = max(1, min(concurrency, len(micro_batches)))

        full_summary_map: dict[int, str] = {}

        def _run_micro_batch(item: tuple[int, list[Session]]) -> dict[int, str]:
            start, chunk = item
            partial = self._summarize_batch_fn(chunk)
            result: dict[int, str] = {}
            if partial:
                for local_idx, summary_text in partial.items():
                    global_idx = start + int(local_idx)
                    if 1 <= global_idx <= len(sessions):
                        result[global_idx] = summary_text
            return result

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_run_micro_batch, mb) for mb in micro_batches]
            for fut in as_completed(futures):
                try:
                    full_summary_map.update(fut.result())
                except Exception as exc:
                    logger.error("Micro-batch failed: %s", exc)

        if not full_summary_map:
            return ([], [], 0, 0, 0)

        results: list[dict] = []
        errors: list[str] = []
        created_count = 0
        total_messages = 0
        replaced_count = 0
        covered_indices: set[int] = set()

        for idx, session in enumerate(sessions, start=1):
            summary_text = full_summary_map.get(idx)
            method = "llm-batch"
            if not summary_text and use_heuristic_fallback:
                summary_text = summarize_session_heuristic(session)
                method = "heuristic"

            if not summary_text:
                errors.append(f"Session {idx}: missing summary in batch output")
                continue

            covered_indices.add(idx)

            result = self.summarize_session(
                session,
                use_heuristic_fallback=use_heuristic_fallback,
                replace_existing=replace_existing,
                summary_text_override=summary_text,
                method_override=method,
            )
            if result.get("success"):
                results.append(result)
                created_count += 1
                total_messages += session.message_count
                replaced_count += int(result.get("replaced_existing", 0) or 0)
            else:
                errors.append(f"Session {idx}: {result.get('error', 'Unknown error')}")

        # Fallback: sessions the batch missed get heuristic summaries instead of
        # expensive per-session LLM calls.  This avoids N extra API round-trips
        # for sessions the model chose to skip or that failed quality checks.
        missing_indices = [i for i in range(1, len(sessions) + 1) if i not in covered_indices]
        if missing_indices:
            logger.info(
                "Batch output missing %s/%s sessions; using heuristic for indices: %s",
                len(missing_indices),
                len(sessions),
                missing_indices,
            )
            # Remove placeholder missing-summary errors now that we'll handle them.
            errors = [e for e in errors if "missing summary in batch output" not in e]

            generated_fallback_texts: dict[int, str | None] = {}
            for index in missing_indices:
                session = sessions[index - 1]
                generated_fallback_texts[index] = summarize_session_heuristic(session)

            for idx in missing_indices:
                session = sessions[idx - 1]
                text = generated_fallback_texts.get(idx)
                if not text:
                    errors.append(f"Session {idx}: Missing in batch and fallback generation failed")
                    continue

                fallback_result = self.summarize_session(
                    session,
                    use_heuristic_fallback=use_heuristic_fallback,
                    replace_existing=replace_existing,
                    summary_text_override=text,
                    method_override="heuristic-fallback",
                )
                if fallback_result.get("success"):
                    results.append(fallback_result)
                    created_count += 1
                    total_messages += session.message_count
                    replaced_count += int(fallback_result.get("replaced_existing", 0) or 0)
                else:
                    errors.append(f"Session {idx}: {fallback_result.get('error', 'Fallback store failed')}")

        return (results, errors, created_count, total_messages, replaced_count)

    def _purge_historical_summaries(
        self,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
    ) -> int:
        """Delete historical summaries in optional time range.

        Active-session summaries are never deleted by this helper.
        """
        from sqlalchemy import text

        with self._backend.engine.connect() as conn:
            sql = text(
                f"""
                SELECT id, summary_metadata, timestamp
                FROM {self._backend._messages_table_name}
                WHERE role = 'summary'
                ORDER BY timestamp ASC
                """
            )
            rows = conn.execute(sql).fetchall()

            to_delete: list[str] = []
            for row in rows:
                ts = float(row.timestamp or 0)
                if start_timestamp is not None and ts < start_timestamp:
                    continue
                if end_timestamp is not None and ts > end_timestamp:
                    continue

                meta = row.summary_metadata or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                if isinstance(meta, dict) and meta.get("summary_type") == "active_session":
                    continue

                to_delete.append(row.id)

            if not to_delete:
                return 0

            delete_sql = text(
                f"""
                DELETE FROM {self._backend._messages_table_name}
                WHERE id = ANY(:ids)
                """
            )
            conn.execute(delete_sql, {"ids": to_delete})
            conn.commit()
            return len(to_delete)

    def _find_active_session(self) -> Session | None:
        """Find the current in-progress session (newest, unsummarized raw messages)."""
        messages = self._get_all_messages()
        regular_messages = [
            m
            for m in messages
            if m.get("role") != "summary"
            and not bool(m.get("recalled_history"))
            and not bool(m.get("summarized"))
        ]
        sessions = detect_sessions(regular_messages, self.session_gap_seconds)
        if not sessions:
            return None
        candidate = sessions[-1]
        if candidate.message_count < self.min_messages:
            return None
        cutoff = time.time() - self.history_duration
        if candidate.end_timestamp < cutoff:
            return None
        return candidate

    def _find_active_summary_id(self, conn: Any, session: Session) -> str | None:
        """Find existing active-session summary row for this session."""
        for summary_id in self._iter_active_summary_ids(conn):
            meta = self._summary_metadata_by_id(conn, summary_id)
            if not isinstance(meta, dict):
                continue
            try:
                existing_start = float(meta.get("session_start"))
            except (TypeError, ValueError):
                continue
            if abs(existing_start - session.start_timestamp) < 1:
                return summary_id
        return None

    def _iter_active_summary_ids(self, conn: Any) -> list[str]:
        """Return active-session summary ids for this bot."""
        from sqlalchemy import text

        sql = text(
            f"""
            SELECT id, summary_metadata
            FROM {self._backend._messages_table_name}
            WHERE role = 'summary'
            ORDER BY created_at DESC
            LIMIT 200
            """
        )
        rows = conn.execute(sql).fetchall()
        active_ids: list[str] = []
        for row in rows:
            meta = row.summary_metadata or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    continue
            if not isinstance(meta, dict):
                continue
            if meta.get("summary_type") != "active_session":
                continue
            active_ids.append(row.id)
        return active_ids

    def _summary_metadata_by_id(self, conn: Any, summary_id: str) -> dict[str, Any] | None:
        from sqlalchemy import text

        sql = text(
            f"""
            SELECT summary_metadata
            FROM {self._backend._messages_table_name}
            WHERE id = :id
            LIMIT 1
            """
        )
        row = conn.execute(sql, {"id": summary_id}).fetchone()
        if not row:
            return None
        meta = row.summary_metadata or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                return None
        return meta if isinstance(meta, dict) else None

    def upsert_active_session_summary(self, use_heuristic_fallback: bool = True) -> dict:
        """Create or update rolling summary for the current active session."""
        session = self._find_active_session()
        if not session:
            with self._backend.engine.connect() as conn:
                from sqlalchemy import text

                active_ids = self._iter_active_summary_ids(conn)
                if active_ids:
                    delete_sql = text(
                        f"""
                        DELETE FROM {self._backend._messages_table_name}
                        WHERE id = ANY(:ids)
                        """
                    )
                    conn.execute(delete_sql, {"ids": active_ids})
                    conn.commit()
            return {
                "active_summary_updated": False,
                "reason": "no_active_session",
                "active_summary_cleared": len(active_ids) if 'active_ids' in locals() else 0,
            }

        if self._summarize_fn:
            summary_text = self._summarize_fn(session)
        else:
            summary_text = summarize_session_with_llm(session, self.service_url)
        method = "llm"

        if not summary_text and use_heuristic_fallback:
            summary_text = summarize_session_heuristic(session)
            method = "heuristic"

        if not summary_text:
            return {"active_summary_updated": False, "reason": "summary_failed"}

        normalized_text = normalize_structured_summary_text(summary_text)
        normalized_text = f"[ACTIVE SESSION SUMMARY]\n{normalized_text.strip()}"
        sections = extract_summary_sections(normalized_text)
        summary_metadata = {
            "summary_type": "active_session",
            "session_start": session.start_timestamp,
            "session_end": session.end_timestamp,
            "message_count": session.message_count,
            "message_ids": session.message_ids,
            "summarization_method": method,
            "updated_at": time.time(),
            "summary_sections": sections,
            "intent": sections.get("intent", ""),
            "tone": sections.get("tone", ""),
            "open_loops": sections.get("open_loops", ""),
        }

        with self._backend.engine.connect() as conn:
            from sqlalchemy import text

            summary_id = self._find_active_summary_id(conn, session)
            if summary_id:
                update_sql = text(
                    f"""
                    UPDATE {self._backend._messages_table_name}
                    SET content = :content,
                        timestamp = :timestamp,
                        summary_metadata = :metadata
                    WHERE id = :id
                    """
                )
                conn.execute(
                    update_sql,
                    {
                        "id": summary_id,
                        "content": normalized_text,
                        "timestamp": session.end_timestamp,
                        "metadata": json.dumps(summary_metadata),
                    },
                )
                created = False
            else:
                summary_id = str(uuid.uuid4())
                insert_sql = text(
                    f"""
                    INSERT INTO {self._backend._messages_table_name}
                    (id, role, content, timestamp, summary_metadata, created_at)
                    VALUES (:id, 'summary', :content, :timestamp, :metadata, CURRENT_TIMESTAMP)
                    """
                )
                conn.execute(
                    insert_sql,
                    {
                        "id": summary_id,
                        "content": normalized_text,
                        "timestamp": session.end_timestamp,
                        "metadata": json.dumps(summary_metadata),
                    },
                )
                created = True
            conn.commit()

        logger.info(
            "Active session summary %s for %s messages",
            "created" if created else "updated",
            session.message_count,
        )
        return {
            "active_summary_updated": True,
            "active_summary_id": summary_id,
            "active_summary_created": created,
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
            active_result = self.upsert_active_session_summary(use_heuristic_fallback=use_heuristic_fallback)
            return {
                "sessions_summarized": 0,
                "messages_summarized": 0,
                "results": [],
                "errors": [],
                "active_summary": active_result,
            }

        results = []
        errors = []
        total_messages = 0
        created_count = 0
        skipped_existing_count = 0

        if self._summarize_batch_fn and max_tokens_per_chunk <= 0 and len(eligible) > 1:
            batch_results, batch_errors, batch_created, batch_messages, _ = self._summarize_sessions_batch(
                sessions=eligible,
                use_heuristic_fallback=use_heuristic_fallback,
                replace_existing=False,
            )
            if batch_created > 0 or batch_errors:
                results = batch_results
                errors = batch_errors
                created_count = batch_created
                total_messages = batch_messages
            else:
                logger.warning("Batch summarization returned no output; falling back to per-session mode")
        if created_count == 0 and not results and not errors:
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
                                if result.get("created", True):
                                    created_count += 1
                                    total_messages += chunk.message_count
                                    logger.info(f"Summarized{chunk_label}: {chunk.message_count} messages")
                                else:
                                    skipped_existing_count += 1
                                    logger.debug(f"Skipped already summarized session{chunk_label}")
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

        active_result = self.upsert_active_session_summary(use_heuristic_fallback=use_heuristic_fallback)

        return {
            "sessions_summarized": created_count,
            "messages_summarized": total_messages,
            "sessions_skipped_existing": skipped_existing_count,
            "results": results,
            "errors": errors,
            "active_summary": active_result,
        }

    def rebuild_recent_sessions(
        self,
        session_limit: int = 5,
        use_heuristic_fallback: bool = True,
        max_tokens_per_chunk: int = 4000,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
        purge_existing: bool = False,
    ) -> dict:
        """Recalculate summaries for the most recent eligible historical sessions.

        Unlike normal summarization, this uses raw history regardless of existing
        summarized flags and replaces existing summaries for the targeted sessions.
        """
        messages = self._get_all_messages()
        regular_messages = [
            m
            for m in messages
            if m.get("role") != "summary"
            and not bool(m.get("recalled_history"))
        ]
        sessions = detect_sessions(regular_messages, self.session_gap_seconds)
        eligible = find_summarizable_sessions(
            sessions,
            history_duration_seconds=self.history_duration,
            min_messages=self.min_messages,
            skip_low_signal=self.skip_low_signal,
            min_user_messages=self.min_user_messages_for_summary,
            min_total_content_chars=self.min_content_chars_for_summary,
            min_meaningful_turns=self.min_meaningful_turns_for_summary,
        )
        eligible = prioritize_summarizable_sessions(eligible)

        if start_timestamp is not None or end_timestamp is not None:
            eligible = [
                s
                for s in eligible
                if (start_timestamp is None or s.end_timestamp >= start_timestamp)
                and (end_timestamp is None or s.start_timestamp <= end_timestamp)
            ]

        if session_limit > 0:
            eligible = eligible[-session_limit:]

        purged_count = 0
        if purge_existing:
            purged_count = self._purge_historical_summaries(
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )

        if not eligible:
            active_result = self.upsert_active_session_summary(use_heuristic_fallback=use_heuristic_fallback)
            return {
                "sessions_targeted": 0,
                "sessions_summarized": 0,
                "messages_summarized": 0,
                "summaries_replaced": 0,
                "results": [],
                "errors": [],
                "active_summary": active_result,
                "summaries_purged": purged_count,
            }

        results = []
        errors = []
        total_messages = 0
        created_count = 0
        replaced_count = 0

        if self._summarize_batch_fn and max_tokens_per_chunk <= 0 and len(eligible) > 1:
            batch_results, batch_errors, batch_created, batch_messages, batch_replaced = self._summarize_sessions_batch(
                sessions=eligible,
                use_heuristic_fallback=use_heuristic_fallback,
                replace_existing=True,
            )
            if batch_created > 0 or batch_errors:
                results = batch_results
                errors = batch_errors
                created_count = batch_created
                total_messages = batch_messages
                replaced_count = batch_replaced
            else:
                logger.warning("Batch rebuild returned no output; falling back to per-session mode")

        if created_count == 0 and not results and not errors:
            for session in eligible:
                try:
                    chunks = split_session_into_chunks(session, max_tokens_per_chunk)
                    for i, chunk in enumerate(chunks):
                        chunk_label = f" (chunk {i+1}/{len(chunks)})" if len(chunks) > 1 else ""
                        result = self.summarize_session(
                            chunk,
                            use_heuristic_fallback=use_heuristic_fallback,
                            replace_existing=True,
                        )
                        if result.get("success"):
                            results.append(result)
                            created_count += 1
                            total_messages += chunk.message_count
                            replaced_count += int(result.get("replaced_existing", 0) or 0)
                            logger.info(
                                "Rebuilt summary%s: %s messages (%s replaced)",
                                chunk_label,
                                chunk.message_count,
                                result.get("replaced_existing", 0),
                            )
                        else:
                            error_msg = result.get("error", "Unknown error")
                            errors.append(f"Session{chunk_label}: {error_msg}")
                            logger.error(f"Failed to rebuild{chunk_label}: {error_msg}")
                except Exception as e:
                    logger.error(f"Error rebuilding session: {e}")
                    errors.append(str(e))

        active_result = self.upsert_active_session_summary(use_heuristic_fallback=use_heuristic_fallback)
        return {
            "sessions_targeted": len(eligible),
            "sessions_summarized": created_count,
            "messages_summarized": total_messages,
            "summaries_replaced": replaced_count,
            "results": results,
            "errors": errors,
            "active_summary": active_result,
            "summaries_purged": purged_count,
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
            rows_unmarked = 0
            if message_ids:
                update_sql = text(f"""
                    UPDATE {self._backend._messages_table_name}
                    SET summarized = FALSE
                    WHERE id = ANY(:ids)
                """)
                update_result = conn.execute(update_sql, {"ids": message_ids})
                rows_unmarked = int(update_result.rowcount or 0)

            # Delete the summary
            delete_sql = text(f"""
                DELETE FROM {self._backend._messages_table_name}
                WHERE id = :id
            """)
            conn.execute(delete_sql, {"id": full_id})

            conn.commit()

        logger.info(f"Deleted summary {full_id[:8]}, unmarked {rows_unmarked} source messages")

        return {
            "success": True,
            "summary_id": full_id,
            "messages_restored": rows_unmarked,
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
