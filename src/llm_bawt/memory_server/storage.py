"""Storage wrapper for MCP memory server.

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
from datetime import datetime
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
            created_at=datetime.utcnow(),
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
    ) -> Message:
        """Add a message to conversation history."""
        message_id = str(uuid.uuid4())
        ts = timestamp if timestamp is not None else time.time()
        
        backend = self._get_backend(bot_id)
        backend.add_message(
            message_id=message_id,
            role=role,
            content=content,
            timestamp=ts,
            session_id=session_id,
        )
        
        return Message(
            id=message_id,
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
    ) -> list[dict[str, Any]]:
        """Get messages for building a context window."""
        backend = self._get_backend(bot_id)
        from sqlalchemy import text

        params: dict[str, Any] = {}
        where = ""
        if since_seconds is not None:
            cutoff = time.time() - since_seconds
            where = "WHERE timestamp >= :cutoff"
            params["cutoff"] = cutoff

        limit_sql = ""
        if limit is not None:
            limit_sql = "LIMIT :limit"
            params["limit"] = limit

        sql = text(
            f"""
            SELECT id, role, content, timestamp, session_id
            FROM {backend._messages_table_name}
            {where}
            ORDER BY timestamp ASC
            {limit_sql}
            """
        )

        try:
            with backend.engine.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                    "session_id": row.session_id,
                }
                for row in rows
            ]
        except Exception as e:
            logger.error("Failed to get messages: %s", e)
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


# Singleton instance
_storage: MemoryStorage | None = None


def get_storage() -> MemoryStorage:
    """Get the singleton storage instance."""
    global _storage
    if _storage is None:
        _storage = MemoryStorage()
    return _storage
