"""Storage wrapper for llm-bawt MCP server.

Thin async wrapper around the existing PostgreSQLMemoryBackend that:
1. Provides async-compatible interface for FastMCP tools
2. Handles connection lifecycle
3. Exposes simplified CRUD operations

The actual implementation delegates to the battle-tested postgresql.py backend.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend
from llm_bawt.memory.embeddings import generate_embedding
from llm_bawt.utils.config import Config

if TYPE_CHECKING:
    from llm_bawt.memory.postgresql import PostgreSQLShortTermManager

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """A stored memory/fact."""
    id: str
    content: str
    tags: list[str] = field(default_factory=lambda: ["misc"])
    importance: float = 0.5
    source_message_ids: list[str] = field(default_factory=list)
    access_count: int = 0
    created_at: datetime | None = None
    last_accessed: datetime | None = None
    relevance: float | None = None  # Set during search results
    intent: str | None = None
    stakes: str | None = None
    emotional_charge: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "tags": self.tags,
            "importance": self.importance,
            "source_message_ids": self.source_message_ids,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "relevance": self.relevance,
            "intent": self.intent,
            "stakes": self.stakes,
            "emotional_charge": self.emotional_charge,
        }


@dataclass
class Message:
    """A conversation message."""
    id: str
    role: str
    content: str
    timestamp: float
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
        }


class MemoryStorage:
    """Async-friendly wrapper around PostgreSQLMemoryBackend.
    
    Provides the storage layer for MCP memory tools.
    Each bot_id gets isolated tables automatically.
    """

    _backends: dict[str, PostgreSQLMemoryBackend] = {}
    _short_term_managers: dict[str, "PostgreSQLShortTermManager"] = {}

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def _get_backend(self, bot_id: str) -> PostgreSQLMemoryBackend:
        """Get or create a backend for the given bot."""
        if bot_id not in self._backends:
            self._backends[bot_id] = PostgreSQLMemoryBackend(
                config=self.config,
                bot_id=bot_id,
            )
        return self._backends[bot_id]

    def get_backend(self, bot_id: str) -> PostgreSQLMemoryBackend:
        """Public access to get or create a backend for the given bot.
        
        Used by MemoryClient for raw backend access.
        """
        return self._get_backend(bot_id)

    def get_short_term_manager(self, bot_id: str) -> "PostgreSQLShortTermManager":
        """Get a short-term memory manager for conversation history.
        
        The short-term manager handles session messages for HistoryManager.
        """
        if bot_id not in self._short_term_managers:
            from llm_bawt.memory.postgresql import PostgreSQLShortTermManager
            self._short_term_managers[bot_id] = PostgreSQLShortTermManager(
                config=self.config,
                bot_id=bot_id,
            )
        return self._short_term_managers[bot_id]

    # =========================================================================
    # Memory Operations
    # =========================================================================

    async def store_memory(
        self,
        content: str,
        bot_id: str = "default",
        tags: list[str] | None = None,
        importance: float = 0.5,
        source_message_ids: list[str] | None = None,
    ) -> Memory:
        """Store a new memory with auto-generated embedding."""
        memory_id = str(uuid.uuid4())
        tags = tags or ["misc"]
        
        backend = self._get_backend(bot_id)
        
        # Generate embedding
        embedding = generate_embedding(content, self.config.MEMORY_EMBEDDING_MODEL, verbose=self.config.VERBOSE)
        
        # Store via backend (handles dedup, indexing, etc.)
        backend.add_memory(
            memory_id=memory_id,
            content=content,
            tags=tags,
            importance=importance,
            source_message_ids=source_message_ids or [],
            embedding=embedding,
        )
        
        return Memory(
            id=memory_id,
            content=content,
            tags=tags,
            importance=importance,
            source_message_ids=source_message_ids or [],
            created_at=datetime.now(timezone.utc),
        )

    async def search_memories(
        self,
        query: str,
        bot_id: str = "default",
        n_results: int = 10,
        min_relevance: float = 0.0,
        tags: list[str] | None = None,
    ) -> list[Memory]:
        """Search memories using semantic similarity."""
        backend = self._get_backend(bot_id)
        
        # Generate query embedding
        query_embedding = generate_embedding(query, self.config.MEMORY_EMBEDDING_MODEL, verbose=self.config.VERBOSE)
        
        if query_embedding:
            results = backend.search_memories_by_embedding(
                embedding=query_embedding,
                n_results=n_results,
                min_importance=min_relevance,
                tags=tags,
            )
        else:
            # Fallback to text search
            results = backend.search_memories_by_text(
                query=query,
                n_results=n_results,
                min_importance=min_relevance,
                tags=tags,
            )
        
        memories = []
        for r in results:
            # Filter by min_relevance (similarity score)
            relevance = r.get("similarity") or r.get("relevance", 0.0)
            if relevance < min_relevance:
                continue
            
            memories.append(Memory(
                id=r["id"],
                content=r["content"],
                tags=r.get("tags", ["misc"]),
                importance=r.get("importance", 0.5),
                source_message_ids=r.get("source_message_ids", []),
                access_count=r.get("access_count", 0),
                created_at=r.get("created_at"),
                last_accessed=r.get("last_accessed"),
                relevance=relevance,
                intent=r.get("intent"),
                stakes=r.get("stakes"),
                emotional_charge=r.get("emotional_charge"),
            ))
        
        return memories

    async def get_memory(self, memory_id: str, bot_id: str = "default") -> Memory | None:
        """Get a specific memory by ID."""
        backend = self._get_backend(bot_id)
        
        # Use the backend's method if available, otherwise use raw SQL
        result = None
        if hasattr(backend, 'get_memory_by_id'):
            result = backend.get_memory_by_id(memory_id)  # type: ignore
        else:
            # Fallback: raw SQL lookup
            from sqlalchemy import text
            try:
                with backend.engine.connect() as conn:
                    sql = text(f"""
                        SELECT id, content, tags, importance, source_message_ids
                        FROM {backend._memories_table_name}
                        WHERE id = :id
                    """)
                    row = conn.execute(sql, {"id": memory_id}).fetchone()
                    if row:
                        result = {
                            "id": row[0],
                            "content": row[1],
                            "tags": row[2] if row[2] else ["misc"],
                            "importance": row[3] if row[3] else 0.5,
                            "source_message_ids": row[4] if row[4] else [],
                        }
            except Exception as e:
                logger.error(f"Failed to get memory {memory_id}: {e}")
        
        if result:
            return Memory(
                id=result["id"],
                content=result["content"],
                tags=result.get("tags", ["misc"]),
                importance=result.get("importance", 0.5),
                source_message_ids=result.get("source_message_ids", []),
            )
        return None

    async def update_memory(
        self,
        memory_id: str,
        bot_id: str = "default",
        content: str | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
    ) -> Memory | None:
        """Update an existing memory."""
        backend = self._get_backend(bot_id)
        
        # Get current memory
        current = await self.get_memory(memory_id, bot_id)
        if not current:
            return None
        
        # Update with new values
        new_content = content if content is not None else current.content
        new_tags = tags if tags is not None else current.tags
        new_importance = importance if importance is not None else current.importance
        
        # Re-generate embedding if content changed
        embedding = None
        if content is not None:
            embedding = generate_embedding(new_content, self.config.MEMORY_EMBEDDING_MODEL, verbose=self.config.VERBOSE)
        
        backend.add_memory(
            memory_id=memory_id,
            content=new_content,
            tags=new_tags,
            importance=new_importance,
            embedding=embedding,
        )
        
        return Memory(
            id=memory_id,
            content=new_content,
            tags=new_tags,
            importance=new_importance,
        )

    async def delete_memory(
        self,
        memory_id: str,
        bot_id: str = "default",
    ) -> bool:
        """Delete a memory."""
        backend = self._get_backend(bot_id)
        
        try:
            if hasattr(backend, 'delete_memory'):
                return backend.delete_memory(memory_id)
            else:
                # Fallback: use raw SQL
                from sqlalchemy import text
                with backend.engine.connect() as conn:
                    conn.execute(text(f"DELETE FROM {backend._memories_table_name} WHERE id = :id"), {"id": memory_id})
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    # =========================================================================
    # Message Operations
    # =========================================================================

    async def add_message(
        self,
        role: str,
        content: str,
        bot_id: str = "default",
        session_id: str | None = None,
        timestamp: float | None = None,
        message_id: str | None = None,
        attachments: list[dict] | None = None,
        reasoning: str | None = None,
        user_id: str | None = None,
    ) -> Message:
        """Add a message to conversation history.

        ``message_id`` allows the client to supply a stable UUID (e.g. the
        frontend-generated user-message UUID) so downstream joins on
        ``trigger_message_id`` work without remapping.

        ``attachments`` (TASK-222) is the tiny JSONB ref list persisted on the
        message row's ``attachments`` column. ``None`` leaves the column at
        its default ``[]``.

        ``session_id`` / ``user_id`` (TASK-284): this is the single live write
        chokepoint for both embedded and server mode, so it owns thread
        attribution. When the caller doesn't supply ``session_id`` (the normal
        case), the active thread for ``(bot_id, user_id)`` is resolved from the
        DB and stamped on the row. Resolution is best-effort: a failure logs and
        leaves ``session_id`` NULL rather than dropping the message.
        """
        provided = (str(message_id).strip() if message_id else "") or None
        message_id_final = provided or str(uuid.uuid4())
        ts = timestamp if timestamp is not None else time.time()

        # TASK-284: stamp the live thread's session_id on every insert. Resolve
        # from the sessions table (DB-derived source of truth) so all workers /
        # cached instances converge on the same active thread per (bot, user).
        if session_id is None:
            try:
                manager = self.get_short_term_manager(bot_id)
                session_id = manager.get_or_create_active_session(
                    bot_id=bot_id, user_id=user_id
                )
            except Exception as e:
                logger.warning(
                    "TASK-284 session resolution failed for bot=%s user=%s: %s",
                    bot_id, user_id, e,
                )

        backend = self._get_backend(bot_id)
        backend.add_message(
            message_id=message_id_final,
            role=role,
            content=content,
            timestamp=ts,
            session_id=session_id,
            attachments=attachments,
            reasoning=reasoning,
        )

        return Message(
            id=message_id_final,
            role=role,
            content=content,
            timestamp=ts,
            session_id=session_id,
        )

    async def get_recent_messages(
        self,
        bot_id: str = "default",
        max_messages: int = 20,
        max_age_seconds: int = 3600,
    ) -> list[Message]:
        """Get recent conversation messages."""
        backend = self._get_backend(bot_id)
        
        # Get recent messages
        cutoff = time.time() - max_age_seconds
        
        try:
            from sqlalchemy import text
            with backend.engine.connect() as conn:
                sql = text(f"""
                    SELECT id, role, content, timestamp, session_id
                    FROM {backend._messages_table_name}
                    WHERE timestamp >= :cutoff
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                rows = conn.execute(sql, {"cutoff": cutoff, "limit": max_messages}).fetchall()
                
                # Reverse to chronological order
                return [
                    Message(
                        id=row.id,
                        role=row.role,
                        content=row.content,
                        timestamp=row.timestamp,
                        session_id=row.session_id,
                    )
                    for row in reversed(rows)
                ]
        except Exception as e:
            logger.error(f"Failed to get recent messages: {e}")
            return []

    async def get_unprocessed_messages(
        self,
        bot_id: str = "default",
        limit: int = 100,
    ) -> list[Message]:
        """Get messages that haven't been processed for memory extraction."""
        backend = self._get_backend(bot_id)
        results = backend.get_unprocessed_messages(limit=limit)
        
        return [
            Message(
                id=r["id"],
                role=r["role"],
                content=r["content"],
                timestamp=r["timestamp"],
                session_id=r.get("session_id"),
            )
            for r in results
        ]

    async def mark_messages_processed(
        self,
        message_ids: list[str],
        bot_id: str = "default",
    ) -> None:
        """Mark messages as processed for memory extraction."""
        backend = self._get_backend(bot_id)
        backend.mark_messages_processed(message_ids)

    # =========================================================================
    # Admin / Maintenance Operations (used by service UX endpoints)
    # =========================================================================

    async def stats(self, bot_id: str = "default") -> dict[str, Any]:
        """Return memory/message statistics."""
        backend = self._get_backend(bot_id)
        return backend.stats()

    async def get_high_importance_memories(
        self,
        bot_id: str = "default",
        n_results: int = 20,
        min_importance: float = 0.0,
    ) -> list[dict[str, Any]]:
        backend = self._get_backend(bot_id)
        return backend.get_high_importance_memories(n_results=n_results, min_importance=min_importance)

    async def list_recent_memories(self, bot_id: str = "default", n: int = 10) -> list[dict[str, Any]]:
        backend = self._get_backend(bot_id)
        if hasattr(backend, "list_recent"):
            return backend.list_recent(n=n)
        return []

    async def preview_recent_messages(self, bot_id: str = "default", count: int = 10) -> list[dict[str, Any]]:
        backend = self._get_backend(bot_id)
        return backend.preview_recent_messages(count)

    async def preview_messages_since_minutes(self, bot_id: str = "default", minutes: int = 60) -> list[dict[str, Any]]:
        backend = self._get_backend(bot_id)
        return backend.preview_messages_since_minutes(minutes)

    async def preview_ignored_messages(self, bot_id: str = "default") -> list[dict[str, Any]]:
        backend = self._get_backend(bot_id)
        return backend.preview_ignored_messages()

    async def ignore_recent_messages(self, bot_id: str = "default", count: int = 10) -> int:
        backend = self._get_backend(bot_id)
        return backend.ignore_recent_messages(count)

    async def ignore_messages_since_minutes(self, bot_id: str = "default", minutes: int = 60) -> int:
        backend = self._get_backend(bot_id)
        return backend.ignore_messages_since_minutes(minutes)

    async def get_message_by_id(self, bot_id: str = "default", message_id: str = "") -> dict | None:
        backend = self._get_backend(bot_id)
        return backend.get_message_by_id(message_id)

    async def ignore_message_by_id(self, bot_id: str = "default", message_id: str = "") -> bool:
        backend = self._get_backend(bot_id)
        return backend.ignore_message_by_id(message_id)

    async def restore_ignored_messages(self, bot_id: str = "default") -> int:
        backend = self._get_backend(bot_id)
        return backend.restore_ignored_messages()

    async def get_messages_for_summary(self, bot_id: str = "default", summary_id: str = "") -> list[dict[str, Any]]:
        manager = self.get_short_term_manager(bot_id)
        rows = manager.get_messages_for_summary(summary_id)
        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "id": getattr(row, "db_id", None),
                    "role": getattr(row, "role", ""),
                    "content": getattr(row, "content", ""),
                    "timestamp": getattr(row, "timestamp", 0.0),
                }
            )
        return results

    async def mark_messages_recalled(self, bot_id: str = "default", message_ids: list[str] | None = None) -> int:
        manager = self.get_short_term_manager(bot_id)
        return manager.mark_messages_recalled(message_ids or [])

    async def delete_memories_by_source_message_ids(self, bot_id: str = "default", message_ids: list[str] | None = None) -> int:
        backend = self._get_backend(bot_id)
        return backend.delete_memories_by_source_message_ids(message_ids or [])

    async def regenerate_embeddings(self, bot_id: str = "default", batch_size: int = 50) -> dict[str, Any]:
        backend = self._get_backend(bot_id)
        return backend.regenerate_embeddings(batch_size=batch_size)

    async def consolidate_memories(
        self,
        bot_id: str = "default",
        dry_run: bool = True,
        similarity_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run consolidation using the existing backend implementation.

        Note: This may use a heuristic merge when no local LLM is available.
        """
        from llm_bawt.memory.consolidation import MemoryConsolidator, get_local_llm_client

        backend = self._get_backend(bot_id)
        llm_client = None if dry_run else get_local_llm_client(self.config)

        threshold = similarity_threshold or getattr(self.config, "MEMORY_CONSOLIDATION_THRESHOLD", 0.92)
        consolidator = MemoryConsolidator(
            backend=backend,
            llm_client=llm_client,
            similarity_threshold=threshold,
            config=self.config,
        )

        result = consolidator.consolidate(dry_run=dry_run)
        # Result is a dataclass-like object; make it JSON-friendly.
        return {
            "clusters_found": result.clusters_found,
            "clusters_merged": result.clusters_merged,
            "memories_consolidated": result.memories_consolidated,
            "new_memories_created": result.new_memories_created,
            "errors": result.errors,
            "dry_run": dry_run,
            "similarity_threshold": threshold,
        }

    async def update_memory_meaning(
        self,
        bot_id: str = "default",
        memory_id: str = "",
        intent: str | None = None,
        stakes: str | None = None,
        emotional_charge: float | None = None,
        recurrence_keywords: list[str] | None = None,
        updated_tags: list[str] | None = None,
    ) -> bool:
        """Update meaning metadata on an existing memory."""
        if not memory_id:
            return False
        from llm_bawt.memory.maintenance import update_memory_meaning

        backend = self._get_backend(bot_id)
        return bool(
            update_memory_meaning(
                backend=backend,
                memory_id=memory_id,
                intent=intent,
                stakes=stakes,
                emotional_charge=emotional_charge,
                recurrence_keywords=recurrence_keywords,
                updated_tags=updated_tags,
            )
        )

    async def run_maintenance(
        self,
        bot_id: str = "default",
        run_consolidation: bool = True,
        run_recurrence_detection: bool = True,
        run_decay_pruning: bool = False,
        run_orphan_cleanup: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run unified memory maintenance inside the memory service."""
        from llm_bawt.memory.maintenance import MemoryMaintenance

        backend = self._get_backend(bot_id)
        maint = MemoryMaintenance(
            backend=backend,
            llm_client=None,
            config=self.config,
        )
        result = maint.run(
            run_consolidation=run_consolidation,
            run_recurrence_detection=run_recurrence_detection,
            run_decay_pruning=run_decay_pruning,
            run_orphan_cleanup=run_orphan_cleanup,
            dry_run=dry_run,
        )
        return result.to_dict()

    # =========================================================================
    # History / Short-term Message Operations (DB-backed, exposed via MCP tools)
    # =========================================================================

    async def get_messages(
        self,
        bot_id: str = "default",
        since_seconds: int | None = None,
        limit: int | None = None,
        raw: bool = False,
    ) -> list[dict[str, Any]]:
        """Get messages for building a context window.

        Default path (``raw=False``) uses the short-term manager's
        summary-aware retrieval, so older session summaries stand in for raw
        history that has aged out of the recent window. This is the chatbot
        short-term/long-term memory model — DO NOT change its behaviour.

        When ``raw=True`` the summary-aware path is bypassed and the messages
        table is queried directly: every real bubble is returned regardless of
        its ``summarized`` flag, and ``role='summary'`` husks are excluded.
        Callers like self_recap / self_tail want the actual transcript the user
        sees in the app, not a summary of a summary.
        """
        if raw:
            return self._get_messages_raw(
                bot_id, since_seconds=since_seconds, limit=limit
            )

        manager = self.get_short_term_manager(bot_id)

        try:
            messages = manager.get_messages(since_minutes=since_seconds)
            if limit is not None and limit >= 0:
                messages = messages[-limit:]

            return [
                {
                    "id": msg.db_id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "session_id": None,
                }
                for msg in messages
            ]
        except Exception as e:
            logger.error("Failed to get messages: %s", e)
            return []

    def _get_messages_raw(
        self,
        bot_id: str = "default",
        since_seconds: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Direct-table read of real message bubbles.

        Bypasses the summary-aware manager entirely. Returns every row whose
        role is NOT ``summary`` (so prior recaps/session summaries are excluded)
        and ignores the ``summarized`` flag (so bubbles already rolled into a
        summary are still returned). Ordered oldest-first; ``limit`` keeps the
        most recent N.
        """
        from sqlalchemy import text

        backend = self._get_backend(bot_id)
        clauses = ["role <> 'summary'"]
        params: dict[str, Any] = {}
        if since_seconds is not None and since_seconds >= 0:
            params["cutoff"] = time.time() - since_seconds
            clauses.append("timestamp >= :cutoff")
        where = " AND ".join(clauses)
        sql = f"""
            SELECT id, role, content, timestamp, session_id
            FROM {backend._messages_table_name}
            WHERE {where}
            ORDER BY timestamp ASC
        """
        try:
            with backend.engine.connect() as conn:
                rows = conn.execute(text(sql), params).fetchall()
            out = [
                {
                    "id": r.id,
                    "role": r.role,
                    "content": r.content,
                    "timestamp": r.timestamp,
                    "session_id": r.session_id,
                }
                for r in rows
            ]
            if limit is not None and limit >= 0:
                out = out[-limit:]
            return out
        except Exception as e:
            logger.error("Failed to get raw messages: %s", e)
            return []

    async def clear_messages(self, bot_id: str = "default") -> int:
        """Delete all messages for a bot."""
        backend = self._get_backend(bot_id)
        from sqlalchemy import text

        try:
            with backend.engine.connect() as conn:
                res = conn.execute(text(f"DELETE FROM {backend._messages_table_name}"))
                conn.commit()
                return int(getattr(res, "rowcount", 0) or 0)
        except Exception as e:
            logger.error("Failed to clear messages: %s", e)
            return 0

    async def remove_last_message_if_partial(self, bot_id: str = "default", role: str = "assistant") -> bool:
        """Remove the most recent message if it matches role."""
        backend = self._get_backend(bot_id)
        from sqlalchemy import text

        try:
            with backend.engine.connect() as conn:
                row = conn.execute(
                    text(
                        f"""
                        SELECT id, role
                        FROM {backend._messages_table_name}
                        ORDER BY timestamp DESC
                        LIMIT 1
                        """
                    )
                ).fetchone()
                if not row or row.role != role:
                    return False
                conn.execute(
                    text(f"DELETE FROM {backend._messages_table_name} WHERE id = :id"),
                    {"id": row.id},
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error("Failed to remove last message: %s", e)
            return False

    async def store_recap_summary(
        self,
        bot_id: str,
        content: str,
        window_start: float,
        window_end: float,
        message_ids: list[str],
        model: str,
    ) -> str | None:
        """Persist a self-recap as a ``role='summary'`` row.

        Unlike the compaction summarizer (``HistorySummarizer.summarize_session``)
        this deliberately does NOT mark the source messages ``summarized=TRUE``.
        A recap is a handoff artifact layered on top of history, not a
        compaction that should evict raw messages from the live context window.
        Mirrors the proven INSERT in ``summarization.py`` (json.dumps into the
        ``summary_metadata`` JSON column, no ``::jsonb`` cast needed).
        """
        import json

        from sqlalchemy import text

        backend = self._get_backend(bot_id)
        summary_id = str(uuid.uuid4())
        metadata = {
            "summary_type": "self_recap",
            "summarization_method": "self-recap",
            "model": model,
            "session_start": window_start,
            "session_end": window_end,
            "message_ids": message_ids,
            "message_count": len(message_ids),
            "created_at": time.time(),
        }
        try:
            with backend.engine.connect() as conn:
                conn.execute(
                    text(
                        f"""
                        INSERT INTO {backend._messages_table_name}
                        (id, role, content, timestamp, summary_metadata, created_at)
                        VALUES (:id, 'summary', :content, :ts, :meta, CURRENT_TIMESTAMP)
                        """
                    ),
                    {
                        "id": summary_id,
                        "content": content,
                        "ts": window_end,
                        "meta": json.dumps(metadata),
                    },
                )
                conn.commit()
            logger.info(
                "Stored self-recap summary %s for bot=%s (%d source messages)",
                summary_id[:8], bot_id, len(message_ids),
            )
            return summary_id
        except Exception as e:
            logger.error("Failed to store recap summary: %s", e)
            return None

    # =========================================================================
    # Sessions (shared `sessions` table)
    # =========================================================================

    async def close_session(
        self,
        session_id: str,
        bot_id: str = "default",
    ) -> bool:
        """Mark a session row completed (sets ended_at + status='completed').

        Returns True iff a row was updated. Idempotent: closing an already-
        closed session returns False without error.
        """
        manager = self.get_short_term_manager(bot_id)
        return manager.close_session(session_id)

    async def get_session(
        self,
        session_id: str,
        bot_id: str = "default",
    ) -> dict | None:
        """Return a session row by id (or None). The `sessions` table is
        shared, so `bot_id` only selects which engine to use — any bot
        works.
        """
        manager = self.get_short_term_manager(bot_id)
        return manager.get_session(session_id)

    async def list_sessions(
        self,
        bot_id: str = "default",
        since: float | str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List sessions for a bot, newest first.

        Pass `bot_id=""` to query across all bots.
        """
        # Use a default manager engine, but pass the requested bot through
        # to the query (manager is bot-scoped, the table is not).
        engine_manager = self.get_short_term_manager(bot_id or "default")
        return engine_manager.list_sessions(
            bot_id=bot_id,
            since=since,
            status=status,
            limit=limit,
        )

    async def get_active_session(
        self,
        bot_id: str = "default",
    ) -> dict | None:
        """Return the most-recent active session for a bot, or None."""
        manager = self.get_short_term_manager(bot_id)
        return manager.get_active_session(bot_id=bot_id)


    # =========================================================================
    # Cross-bot / Source Discovery & Search
    # =========================================================================

    def _discover_tables(self, suffix: str) -> list[tuple[str, str]]:
        """Discover all bot tables matching a naming convention.

        Args:
            suffix: Table name suffix, e.g. ``_messages`` or ``_memories``.

        Returns:
            List of ``(bot_id, table_name)`` tuples.
        """
        from sqlalchemy import text

        backend = self._get_backend("default")
        try:
            with backend.engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_name LIKE :pattern
                    ORDER BY table_name
                """), {"pattern": f"%\\{suffix}"}).fetchall()

                results: list[tuple[str, str]] = []
                for (table_name,) in rows:
                    bot_id = table_name.removesuffix(suffix)
                    if bot_id:
                        results.append((bot_id, table_name))
                return results
        except Exception as e:
            logger.error("Failed to discover %s tables: %s", suffix, e)
            return []

    def _global_search_excluded_bot_ids(self) -> set[str]:
        """Return sanitized bot IDs opted out of aggregate search."""
        try:
            from llm_bawt.bots import BotManager
            from llm_bawt.memory.postgresql import _sanitize_table_name

            return {
                _sanitize_table_name(bot.slug)
                for bot in BotManager(self.config).list_bots()
                if getattr(bot, "include_in_global_search", True) is False
            }
        except Exception as e:
            logger.warning("Failed to load global search bot policy: %s", e)
            return set()

    def _discover_global_search_tables(self, suffix: str) -> list[tuple[str, str]]:
        """Discover tables that are eligible for aggregate search."""
        tables = self._discover_tables(suffix)
        if not tables:
            return []

        excluded = self._global_search_excluded_bot_ids()
        if not excluded:
            return tables

        return [
            (bot_id, table_name)
            for bot_id, table_name in tables
            if bot_id not in excluded
        ]

    async def list_memory_sources(self) -> list[dict[str, Any]]:
        """Discover all available memory sources (bot_ids with memory tables).

        Queries PostgreSQL for tables matching the ``*_memories`` naming
        convention and returns basic stats for each source.
        """
        from sqlalchemy import text

        backend = self._get_backend("default")
        tables = self._discover_global_search_tables("_memories")
        if not tables:
            return []

        try:
            sources: list[dict[str, Any]] = []
            with backend.engine.connect() as conn:
                for bot_id, table_name in tables:
                    count_row = conn.execute(
                        text(f"SELECT COUNT(*) FROM {table_name}")  # noqa: S608
                    ).fetchone()
                    memory_count = count_row[0] if count_row else 0
                    sources.append({
                        "source": bot_id,
                        "memory_count": memory_count,
                    })
            return sources
        except Exception as e:
            logger.error("Failed to list memory sources: %s", e)
            return []

    async def search_all_messages(
        self,
        query: str,
        n_results: int = 10,
        role_filter: str | None = None,
        sort_by: str = "relevance",
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict[str, Any]]:
        """Full-text search across ALL bots' message histories.

        Builds a UNION ALL query across every ``*_messages`` table so a single
        question like *"who was working on the stop button?"* returns ranked
        results from all bots in one call.

        Parameters drive the Spotlight modal's facet controls:

        * ``sort_by`` — ``"relevance"`` (default; rank then timestamp) or
          ``"recent"`` (timestamp only, ignores rank). The Spotlight UI
          exposes both for cases where the most-relevant doc is two months
          old and the user actually wants the latest match.
        * ``since`` / ``until`` — Unix seconds. Inclusive bounds; either can
          be omitted. Used by the time-range chips (24h / 7d / 30d / all).

        Each returned row also carries a ``total`` field (window COUNT(*)
        over the unbounded result set) so the UI can render "showing N of M
        matches" without a second roundtrip. ``total`` is identical on
        every row; callers can read it from the first row and ignore the
        rest. Empty result → no rows → caller should treat as ``total=0``.
        """
        from sqlalchemy import text
        from llm_bawt.memory.postgresql import build_fts_query

        or_query = build_fts_query(query)
        if not or_query:
            return []

        tables = self._discover_global_search_tables("_messages")
        if not tables:
            return []

        # Compose conditional WHERE fragments once. They reference bind
        # params resolved per-execution below.
        time_lower = "AND timestamp >= :since" if since is not None else ""
        time_upper = "AND timestamp <= :until" if until is not None else ""
        role_clause = "AND role = :role_filter" if role_filter else ""

        # Build UNION ALL across all message tables
        sub_selects: list[str] = []
        for bot_id, table_name in tables:
            # bot_id is embedded as a literal; table_name comes from information_schema
            sub_selects.append(f"""(
                SELECT id, role, content, timestamp,
                       '{bot_id}' AS source,
                       ts_rank(to_tsvector('english', content),
                               to_tsquery('english', :query)) AS rank
                FROM {table_name}
                WHERE to_tsvector('english', content) @@ to_tsquery('english', :query)
                  AND role != 'system'
                  {role_clause}
                  {time_lower}
                  {time_upper}
            )""")

        union_sql = "\nUNION ALL\n".join(sub_selects)
        # ``sort_by="recent"`` ignores rank entirely so the user can search
        # for a common word and still see the latest hit. Whitelist before
        # interpolation — sort_by is the only SQL fragment built from a
        # caller-supplied string, so the comparison is the safety net.
        order_clause = "ORDER BY timestamp DESC" if sort_by == "recent" else "ORDER BY rank DESC, timestamp DESC"
        # ``COUNT(*) OVER ()`` is the unbounded-total trick: each row in
        # the LIMIT slice carries the count of all rows that matched the
        # WHERE, so the caller knows "you're seeing 10 of 1497" without
        # a second roundtrip. Postgres has to materialize all matches to
        # count them — fine for tens of thousands, watch if we ever hit
        # very common short tokens.
        full_sql = text(f"""
            SELECT *, COUNT(*) OVER () AS total
            FROM ({union_sql}) AS u
            {order_clause}
            LIMIT :limit
        """)

        params: dict[str, Any] = {"query": or_query, "limit": n_results}
        if role_filter:
            params["role_filter"] = role_filter
        if since is not None:
            params["since"] = since
        if until is not None:
            params["until"] = until

        backend = self._get_backend("default")
        try:
            with backend.engine.connect() as conn:
                rows = conn.execute(full_sql, params).fetchall()
                return [
                    {
                        "id": row.id,
                        "role": row.role,
                        "content": row.content,
                        "timestamp": row.timestamp,
                        "source": row.source,
                        "rank": row.rank,
                        "total": int(row.total),
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error("search_all_messages failed: %s", e)
            return []

    async def search_all_messages_trgm(
        self,
        query: str,
        n_results: int = 10,
        role_filter: str | None = None,
        sort_by: str = "relevance",
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict[str, Any]]:
        """Cross-bot substring/fuzzy search via ``pg_trgm``.

        Companion to :meth:`search_all_messages` (which uses Postgres FTS via
        ``to_tsquery``). FTS is great for natural-language queries — it
        stems, drops stop-words, and ranks by lexeme density — but it
        actively breaks on identifiers: ``to_tsquery('english', 'TASK-241')``
        tokenizes to ``task`` (the digit half is dropped by our tokenizer
        and the english stemmer doesn't know what to do with ``241`` anyway),
        so a search for ``TASK-241`` returns every message that ever said
        the word "task." That's wrong for the case where the user is
        looking up an ID.

        This method goes the other way: treat ``content`` as a flat
        character sequence, find substring matches with ``ILIKE``, and rank
        by trigram similarity. Backed by a per-table
        ``USING gin (content gin_trgm_ops)`` index so the leading-wildcard
        ILIKE stays fast. No tokenizer, no stemming, no stop-words —
        ``TASK-241`` matches the literal substring or it doesn't.

        UI dispatches between the two via the ``mode`` param on the route:
        ``mode=fts`` (default) keeps prior behaviour; ``mode=trgm`` calls
        this method. The Spotlight modal exposes the choice as a toggle.
        """
        from sqlalchemy import text

        q = (query or "").strip()
        if not q:
            return []

        tables = self._discover_global_search_tables("_messages")
        if not tables:
            return []

        # Optional WHERE fragments. Same pattern as search_all_messages.
        time_lower = "AND timestamp >= :since" if since is not None else ""
        time_upper = "AND timestamp <= :until" if until is not None else ""
        role_clause = "AND role = :role_filter" if role_filter else ""

        # Build UNION ALL across all message tables. Each sub-select runs
        # the trigram-indexed ILIKE filter and computes a similarity score
        # for ranking. similarity() is per-row and doesn't use the index,
        # but it's cheap and only computed on the rows that already
        # survived the ILIKE filter.
        sub_selects: list[str] = []
        for bot_id, table_name in tables:
            sub_selects.append(f"""(
                SELECT id, role, content, timestamp,
                       '{bot_id}' AS source,
                       similarity(content, :query) AS rank
                FROM {table_name}
                WHERE content ILIKE :ilike
                  AND role != 'system'
                  {role_clause}
                  {time_lower}
                  {time_upper}
            )""")

        union_sql = "\nUNION ALL\n".join(sub_selects)
        # ``sort_by="recent"`` ignores similarity — useful for "find the
        # latest mention of this token" once the user knows the token
        # itself is common.
        order_clause = "ORDER BY timestamp DESC" if sort_by == "recent" else "ORDER BY rank DESC, timestamp DESC"
        # Window COUNT(*) for the unbounded total. See the FTS sibling
        # for rationale.
        full_sql = text(f"""
            SELECT *, COUNT(*) OVER () AS total
            FROM ({union_sql}) AS u
            {order_clause}
            LIMIT :limit
        """)

        # `ilike` carries the wildcarded form used by the WHERE clause.
        # `query` stays raw so similarity() compares against the user's
        # literal input — wildcards in the score input would skew the
        # trigram comparison.
        params: dict[str, Any] = {
            "query": q,
            "ilike": f"%{q}%",
            "limit": n_results,
        }
        if role_filter:
            params["role_filter"] = role_filter
        if since is not None:
            params["since"] = since
        if until is not None:
            params["until"] = until

        backend = self._get_backend("default")
        try:
            with backend.engine.connect() as conn:
                rows = conn.execute(full_sql, params).fetchall()
                return [
                    {
                        "id": row.id,
                        "role": row.role,
                        "content": row.content,
                        "timestamp": row.timestamp,
                        "source": row.source,
                        "rank": row.rank,
                        "total": int(row.total),
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error("search_all_messages_trgm failed: %s", e)
            return []

    async def search_all_memories(
        self,
        query: str,
        n_results: int = 10,
        min_relevance: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Semantic search across ALL bots' memory stores.

        Generates one embedding, runs the existing per-bot search (with
        temporal decay, access boost, diversity sampling), then merges and
        ranks across all sources.
        """
        from llm_bawt.memory.embeddings import generate_embedding

        query_embedding = generate_embedding(
            query, self.config.MEMORY_EMBEDDING_MODEL, verbose=self.config.VERBOSE,
        )
        if not query_embedding:
            return []

        tables = self._discover_global_search_tables("_memories")
        if not tables:
            return []

        all_results: list[dict[str, Any]] = []
        for bot_id, _table_name in tables:
            try:
                backend = self._get_backend(bot_id)
                results = backend.search_memories_by_embedding(
                    embedding=query_embedding,
                    n_results=n_results,
                    min_importance=0.0,
                )
                for r in results:
                    relevance = r.get("similarity") or r.get("relevance", 0.0)
                    if relevance < min_relevance:
                        continue
                    r["source"] = bot_id
                    r["relevance"] = relevance
                    all_results.append(r)
            except Exception as e:
                logger.warning("search_all_memories skipped %s: %s", bot_id, e)

        # Sort by effective_score (from the backend's scoring) then relevance
        all_results.sort(
            key=lambda r: r.get("effective_score", r.get("relevance", 0.0)),
            reverse=True,
        )
        return all_results[:n_results]


# Singleton instance
_storage: MemoryStorage | None = None


def get_storage() -> MemoryStorage:
    """Get the singleton storage instance."""
    global _storage
    if _storage is None:
        _storage = MemoryStorage()
    return _storage
