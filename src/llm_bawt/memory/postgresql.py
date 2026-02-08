"""PostgreSQL memory backend for llm-bawt with pgvector support.

This backend uses PostgreSQL with the pgvector extension for semantic similarity search.
Each bot gets its own isolated tables:
  - {bot_id}_messages: Permanent message storage (all messages)
  - {bot_id}_memories: Distilled, importance-weighted memories with embeddings

The separation allows:
  - Messages: Complete conversation history, never deleted
  - Memories: Curated facts extracted from conversations with importance scores
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Any, TYPE_CHECKING
from urllib.parse import quote_plus

from sqlalchemy import (
    Column, String, Text, Float, DateTime, Integer, Boolean, MetaData, Table,
    create_engine, select, delete, text, insert, update, JSON
)
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool

from .base import MemoryBackend

if TYPE_CHECKING:
    from ..models.message import Message as MessageModel

logger = logging.getLogger(__name__)


def _sanitize_table_name(bot_id: str) -> str:
    """Sanitize bot_id for use in table names.
    
    Only allows alphanumeric and underscore, lowercase.
    """
    sanitized = re.sub(r'[^a-z0-9_]', '', bot_id.lower())
    if not sanitized:
        sanitized = "default"
    return sanitized


# Shared metadata for table definitions
metadata = MetaData()

# Cache for table objects
_memory_table_cache: dict[str, Table] = {}
_message_table_cache: dict[str, Table] = {}


def get_message_table_pg(bot_id: str) -> Table:
    """Get or create a message Table for a specific bot (PostgreSQL version)."""
    table_name = f"{_sanitize_table_name(bot_id)}_messages"

    if table_name in _message_table_cache:
        return _message_table_cache[table_name]

    table = Table(
        table_name,
        metadata,
        Column("id", String(36), primary_key=True),  # UUID
        Column("role", String(20), nullable=False),
        Column("content", Text, nullable=False),
        Column("timestamp", Float, nullable=False),
        Column("session_id", String(36), nullable=True),  # For grouping conversations
        Column("processed", Boolean, default=False),  # Whether memory extraction has run
        Column("summarized", Boolean, default=False),  # Whether this message is included in a summary
        Column("recalled_history", Boolean, default=False),  # Whether this message was re-inserted via recall tool
        Column("summary_metadata", JSON, nullable=True),  # For role='summary' rows only
        Column("created_at", DateTime, default=datetime.utcnow),
        extend_existing=True
    )

    _message_table_cache[table_name] = table
    return table


# Cache for forgotten message tables
_forgotten_table_cache: dict[str, Table] = {}


def get_forgotten_table_pg(bot_id: str) -> Table:
    """Get or create a forgotten messages Table for a specific bot (PostgreSQL version).
    
    This table stores messages that have been 'forgotten' (soft-deleted).
    They can be restored later if needed.
    """
    table_name = f"{_sanitize_table_name(bot_id)}_forgotten_messages"
    
    if table_name in _forgotten_table_cache:
        return _forgotten_table_cache[table_name]
    
    table = Table(
        table_name,
        metadata,
        Column("id", String(36), primary_key=True),  # UUID
        Column("role", String(20), nullable=False),
        Column("content", Text, nullable=False),
        Column("timestamp", Float, nullable=False),
        Column("session_id", String(36), nullable=True),
        Column("processed", Boolean, default=False),
        Column("created_at", DateTime, default=datetime.utcnow),
        Column("forgotten_at", DateTime, default=datetime.utcnow),  # When it was forgotten
        extend_existing=True
    )
    
    _forgotten_table_cache[table_name] = table
    return table


def get_memory_table_pg(bot_id: str) -> Table:
    """Get or create a memory Table for a specific bot (PostgreSQL version)."""
    table_name = f"{_sanitize_table_name(bot_id)}_memories"
    
    if table_name in _memory_table_cache:
        return _memory_table_cache[table_name]
    
    table = Table(
        table_name,
        metadata,
        Column("id", String(36), primary_key=True),  # UUID
        Column("content", Text, nullable=False),
        Column("tags", JSON, nullable=False, default=["misc"]),
        Column("importance", Float, nullable=False, default=0.5),
        Column("source_message_ids", JSON, nullable=True),  # Array of message UUIDs
        Column("access_count", Integer, default=0),  # For reinforcement
        Column("last_accessed", DateTime, nullable=True),
        Column("intent", Text, nullable=True),
            Column("stakes", Text, nullable=True),
            Column("emotional_charge", Float, nullable=True),
            Column("recurrence_keywords", JSON, nullable=True),
            Column("meaning_updated_at", DateTime, nullable=True),
        Column("created_at", DateTime, default=datetime.utcnow),
        Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
        # embedding column added via raw SQL for pgvector
        extend_existing=True
    )
    
    _memory_table_cache[table_name] = table
    return table


class PostgreSQLMemoryBackend(MemoryBackend):
    """PostgreSQL-based memory backend with pgvector for semantic search.
    
    This is the new memory backend designed for:
    - Permanent message storage (all messages preserved)
    - Distilled memories with importance weighting
    - Semantic similarity search via pgvector
    - Source linking back to original messages
    
    Configuration (via environment variables or .env):
        LLM_BAWT_POSTGRES_HOST: Database host (default: localhost)
        LLM_BAWT_POSTGRES_PORT: Database port (default: 5432)
        LLM_BAWT_POSTGRES_USER: Database user
        LLM_BAWT_POSTGRES_PASSWORD: Database password
        LLM_BAWT_POSTGRES_DATABASE: Database name (default: llm_bawt)
    """
    
    # Embedding dimension (matches sentence-transformers all-MiniLM-L6-v2)
    EMBEDDING_DIM = 384
    
    def __init__(self, config: Any, bot_id: str = "nova", embedding_dim: int | None = None):
        super().__init__(config, bot_id=bot_id)
        
        # Get embedding settings from config
        self.embedding_model = getattr(config, 'MEMORY_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        if embedding_dim is None:
            embedding_dim = getattr(config, 'MEMORY_EMBEDDING_DIM', 384)
        
        # Get PostgreSQL connection settings from config
        host = getattr(config, 'POSTGRES_HOST', 'localhost')
        port = int(getattr(config, 'POSTGRES_PORT', 5432))
        user = getattr(config, 'POSTGRES_USER', 'llm_bawt')
        password = getattr(config, 'POSTGRES_PASSWORD', '')
        database = getattr(config, 'POSTGRES_DATABASE', 'llm_bawt')
        
        self.database = database
        self.bot_id_sanitized = _sanitize_table_name(bot_id)
        self._messages_table_name = f"{self.bot_id_sanitized}_messages"
        self._memories_table_name = f"{self.bot_id_sanitized}_memories"
        self._forgotten_table_name = f"{self.bot_id_sanitized}_forgotten_messages"
        self.embedding_dim = embedding_dim
        
        # Get table definitions
        self.messages_table = get_message_table_pg(bot_id)
        self.memories_table = get_memory_table_pg(bot_id)
        self.forgotten_table = get_forgotten_table_pg(bot_id)
        
        # Build connection URL for PostgreSQL
        encoded_password = quote_plus(password)
        connection_url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{database}"
        
        self.engine = create_engine(
            connection_url,
            echo=False,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
        )
        
        self._ensure_tables_exist()
        logger.debug(f"Connected to PostgreSQL at {host}:{port}/{database} (bot: {bot_id})")
    
    def _ensure_tables_exist(self) -> None:
        """Create the bot's tables if they don't exist."""
        with self.engine.connect() as conn:
            # Ensure pgvector extension is available
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            except Exception as e:
                logger.debug(f"pgvector extension check: {e}")
            
            # Create messages table
            messages_sql = text(f"""
                CREATE TABLE IF NOT EXISTS {self._messages_table_name} (
                    id VARCHAR(36) PRIMARY KEY,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DOUBLE PRECISION NOT NULL,
                    session_id VARCHAR(36),
                    processed BOOLEAN DEFAULT FALSE,
                    summarized BOOLEAN DEFAULT FALSE,
                    summary_metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create forgotten messages table (for soft-deleted messages)
            forgotten_sql = text(f"""
                CREATE TABLE IF NOT EXISTS {self._forgotten_table_name} (
                    id VARCHAR(36) PRIMARY KEY,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DOUBLE PRECISION NOT NULL,
                    session_id VARCHAR(36),
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP,
                    forgotten_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create memories table with vector column
            memories_sql = text(f"""
                CREATE TABLE IF NOT EXISTS {self._memories_table_name} (
                    id VARCHAR(36) PRIMARY KEY,
                    content TEXT NOT NULL,
                    tags JSONB NOT NULL DEFAULT '["misc"]'::jsonb,
                    importance REAL NOT NULL DEFAULT 0.5,
                    source_message_ids JSONB,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP,
                    intent TEXT,
                    stakes TEXT,
                    emotional_charge REAL,
                    recurrence_keywords JSONB,
                    meaning_updated_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding vector({self.embedding_dim}),
                    meaning_embedding vector({self.embedding_dim})
                )
            """)
            
            # Migration: drop superseded_by column (we now delete instead of marking)
            drop_superseded_sql = text(f"""
                ALTER TABLE {self._memories_table_name}
                DROP COLUMN IF EXISTS superseded_by
            """)

            add_tags_sql = text(f"""
                ALTER TABLE {self._memories_table_name}
                ADD COLUMN IF NOT EXISTS tags JSONB NOT NULL DEFAULT '["misc"]'::jsonb,
                ADD COLUMN IF NOT EXISTS intent TEXT,
                ADD COLUMN IF NOT EXISTS stakes TEXT,
                ADD COLUMN IF NOT EXISTS emotional_charge REAL,
                ADD COLUMN IF NOT EXISTS recurrence_keywords JSONB,
                ADD COLUMN IF NOT EXISTS meaning_updated_at TIMESTAMP,
                ADD COLUMN IF NOT EXISTS meaning_embedding vector({self.embedding_dim})
            """)

            # Migration: drop legacy memory_type column
            drop_memory_type_sql = text(f"""
                ALTER TABLE {self._memories_table_name}
                DROP COLUMN IF EXISTS memory_type
            """)

            # Migration: add summarization columns to messages table
            add_summarization_cols_sql = text(f"""
                ALTER TABLE {self._messages_table_name}
                ADD COLUMN IF NOT EXISTS summarized BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS summary_metadata JSONB
            """)

            # Migration: add recalled_history column to messages table
            add_recalled_history_sql = text(f"""
                ALTER TABLE {self._messages_table_name}
                ADD COLUMN IF NOT EXISTS recalled_history BOOLEAN DEFAULT FALSE
            """)
            
            try:
                conn.execute(messages_sql)
                conn.execute(forgotten_sql)
                conn.execute(memories_sql)
                # Run migrations for existing tables
                try:
                    conn.execute(add_tags_sql)
                except Exception:
                    pass
                try:
                    conn.execute(drop_memory_type_sql)
                except Exception:
                    pass  # Column may already be dropped
                try:
                    conn.execute(drop_superseded_sql)
                except Exception:
                    pass  # Column may already be dropped
                try:
                    conn.execute(add_summarization_cols_sql)
                except Exception:
                    pass  # Columns may already exist
                try:
                    conn.execute(add_recalled_history_sql)
                except Exception:
                    pass  # Column may already exist
                conn.commit()
                
                # Create indexes
                self._create_indexes(conn)
                
                logger.debug(f"Ensured tables exist for bot {self.bot_id}")
            except Exception as e:
                logger.error(f"Failed to create tables: {e}")
                raise
    
    def _create_indexes(self, conn) -> None:
        """Create indexes for efficient querying."""
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_{self._messages_table_name}_timestamp ON {self._messages_table_name}(timestamp)",
            f"CREATE INDEX IF NOT EXISTS idx_{self._messages_table_name}_session ON {self._messages_table_name}(session_id)",
            f"CREATE INDEX IF NOT EXISTS idx_{self._messages_table_name}_processed ON {self._messages_table_name}(processed)",
            f"CREATE INDEX IF NOT EXISTS idx_{self._memories_table_name}_importance ON {self._memories_table_name}(importance)",
            f"CREATE INDEX IF NOT EXISTS idx_{self._memories_table_name}_accessed ON {self._memories_table_name}(last_accessed)",
            f"CREATE INDEX IF NOT EXISTS idx_{self._memories_table_name}_tags_gin ON {self._memories_table_name} USING gin (tags)",
        ]
        
        # HNSW index for vector similarity (if we have embeddings)
        # This is the fastest index type for pgvector
        hnsw_index = f"""
            CREATE INDEX IF NOT EXISTS idx_{self._memories_table_name}_embedding 
            ON {self._memories_table_name} 
            USING hnsw (embedding vector_cosine_ops)
        """
        meaning_hnsw_index = f"""
            CREATE INDEX IF NOT EXISTS idx_{self._memories_table_name}_meaning_embedding 
            ON {self._memories_table_name} 
            USING hnsw (meaning_embedding vector_cosine_ops)
        """
        
        for idx_sql in indexes:
            try:
                conn.execute(text(idx_sql))
            except Exception as e:
                logger.debug(f"Index creation (may already exist): {e}")
        
        try:
            conn.execute(text(hnsw_index))
        except Exception as e:
            logger.debug(f"HNSW index creation (may already exist): {e}")
        try:
            conn.execute(text(meaning_hnsw_index))
        except Exception as e:
            logger.debug(f"Meaning HNSW index creation (may already exist): {e}")
        
        conn.commit()
    
    # =========================================================================
    # Message Storage (permanent conversation history)
    # =========================================================================
    
    def add_message(
        self,
        message_id: str,
        role: str,
        content: str,
        timestamp: float,
        session_id: str | None = None,
    ) -> None:
        """Add a message to permanent storage.
        
        Messages are NEVER deleted - they form the complete conversation history.
        """
        if not content or content.isspace():
            logger.warning(f"Skipping empty content for message ID: {message_id}")
            return
        
        with Session(self.engine) as session:
            try:
                # Check if exists (upsert)
                stmt = select(self.messages_table).where(
                    self.messages_table.c.id == message_id
                )
                existing = session.execute(stmt).first()
                
                if existing:
                    stmt = (
                        update(self.messages_table)
                        .where(self.messages_table.c.id == message_id)
                        .values(content=content, timestamp=timestamp)
                    )
                    session.execute(stmt)
                else:
                    stmt = insert(self.messages_table).values(
                        id=message_id,
                        role=role,
                        content=content,
                        timestamp=timestamp,
                        session_id=session_id,
                        processed=False,
                        created_at=datetime.utcnow(),
                    )
                    session.execute(stmt)
                
                session.commit()
                logger.debug(f"Added message {message_id} to {self._messages_table_name}")
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to add message {message_id}: {e}")
    
    def get_unprocessed_messages(self, limit: int = 100) -> list[dict]:
        """Get messages that haven't been processed for memory extraction."""
        with Session(self.engine) as session:
            stmt = (
                select(self.messages_table)
                .where(self.messages_table.c.processed == False)
                .order_by(self.messages_table.c.timestamp.asc())
                .limit(limit)
            )
            rows = session.execute(stmt).fetchall()
            
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
    
    def mark_messages_processed(self, message_ids: list[str]) -> None:
        """Mark messages as processed for memory extraction."""
        if not message_ids:
            return
        
        with Session(self.engine) as session:
            try:
                stmt = (
                    update(self.messages_table)
                    .where(self.messages_table.c.id.in_(message_ids))
                    .values(processed=True)
                )
                session.execute(stmt)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to mark messages processed: {e}")
    
    def get_messages_by_ids(self, message_ids: list[str]) -> list[dict]:
        """Retrieve messages by their IDs (for context retrieval)."""
        if not message_ids:
            return []
        
        with Session(self.engine) as session:
            stmt = (
                select(self.messages_table)
                .where(self.messages_table.c.id.in_(message_ids))
                .order_by(self.messages_table.c.timestamp.asc())
            )
            rows = session.execute(stmt).fetchall()
            
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
    
    # =========================================================================
    # Memory Storage (distilled, importance-weighted)
    # =========================================================================
    
    def add_memory(
        self,
        memory_id: str,
        content: str,
        tags: list[str] | None = None,
        importance: float = 0.5,
        source_message_ids: list[str] | None = None,
        embedding: list[float] | None = None,
        intent: str | None = None,
        stakes: str | None = None,
        emotional_charge: float | None = None,
        recurrence_keywords: list[str] | None = None,
        meaning_embedding: list[float] | None = None,
        meaning_updated_at: datetime | None = None,
    ) -> None:
        """Add a distilled memory to storage.
        
        If no embedding is provided, one will be generated automatically
        using the configured local embedding model.
        """
        if not content or content.isspace():
            logger.warning(f"Skipping empty content for memory ID: {memory_id}")
            return
        
        # Normalize tags
        tags = tags or ["misc"]
        tags = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
        if not tags:
            tags = ["misc"]
        
        # Generate embeddings if not provided
        if embedding is None:
            try:
                from .embeddings import generate_embedding
                embedding = generate_embedding(content, self.embedding_model, verbose=getattr(self.config, 'VERBOSE', False))
                if embedding:
                    logger.debug(f"Generated embedding for memory: {content[:50]}...")
            except Exception as e:
                logger.debug(f"Could not generate embedding: {e}")
        if meaning_embedding is None:
            try:
                from .embeddings import generate_embedding
                meaning_text_parts = [intent or "", stakes or "", "emotional" if emotional_charge else "", " ".join(recurrence_keywords or [])]
                meaning_text = " | ".join([p for p in meaning_text_parts if p])
                meaning_embedding = generate_embedding(meaning_text or content, self.embedding_model, verbose=getattr(self.config, 'VERBOSE', False))
            except Exception as e:
                logger.debug(f"Could not generate meaning embedding: {e}")
        if meaning_embedding:
            meaning_updated_at = meaning_updated_at or datetime.utcnow()
        
        with self.engine.connect() as conn:
            try:
                # Check for duplicate/similar content BEFORE inserting
                # Use embedding similarity if available, otherwise exact content match
                # Threshold from config (default 0.85) - catches semantic duplicates
                dedup_threshold = getattr(self.config, 'MEMORY_DEDUP_SIMILARITY', 0.85)
                
                if embedding:
                    # Vector similarity check - find most similar existing memory
                    dedup_sql = text(f"""
                        SELECT id, content, 1 - (embedding <=> :embedding) AS similarity
                        FROM {self._memories_table_name}
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> :embedding
                        LIMIT 1
                    """)
                    similar = conn.execute(dedup_sql, {
                        "embedding": f"[{','.join(str(x) for x in embedding)}]",
                    }).first()
                    
                    if similar and similar.similarity > dedup_threshold:
                        logger.debug(f"Skipping duplicate memory (similarity={similar.similarity:.2%} > {dedup_threshold:.2%}): '{content[:50]}...' similar to '{similar.content[:50]}...'")
                        return
                else:
                    # Exact content match fallback
                    exact_sql = text(f"""
                        SELECT id FROM {self._memories_table_name} 
                        WHERE LOWER(content) = LOWER(:content)
                        LIMIT 1
                    """)
                    exact_match = conn.execute(exact_sql, {"content": content}).first()
                    if exact_match:
                        logger.debug(f"Skipping exact duplicate memory: '{content[:50]}...'")
                        return
                
                # Check if this specific ID exists (for updates)
                check_sql = text(f"""
                    SELECT id FROM {self._memories_table_name} WHERE id = :id
                """)
                existing = conn.execute(check_sql, {"id": memory_id}).first()
                
                if existing:
                    # Update existing
                    update_sql = text(f"""
                        UPDATE {self._memories_table_name}
                        SET content = :content,
                            tags = CAST(:tags AS jsonb),
                            importance = :importance,
                            source_message_ids = :source_ids,
                            embedding = COALESCE(:embedding, embedding),
                            intent = :intent,
                            stakes = :stakes,
                            emotional_charge = :emotional_charge,
                            recurrence_keywords = CAST(:recurrence_keywords AS jsonb),
                            meaning_embedding = COALESCE(:meaning_embedding, meaning_embedding),
                            meaning_updated_at = COALESCE(:meaning_updated_at, meaning_updated_at),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = :id
                    """)
                    conn.execute(update_sql, {
                        "id": memory_id,
                        "content": content,
                        "tags": json.dumps(tags),
                        "importance": importance,
                        "source_ids": json.dumps(source_message_ids or []),
                        "embedding": f"[{','.join(str(x) for x in embedding)}]" if embedding else None,
                        "intent": intent,
                        "stakes": stakes,
                        "emotional_charge": emotional_charge,
                        "recurrence_keywords": json.dumps(recurrence_keywords or []),
                        "meaning_embedding": f"[{','.join(str(x) for x in meaning_embedding)}]" if meaning_embedding else None,
                        "meaning_updated_at": meaning_updated_at,
                    })
                else:
                    # Insert new
                    insert_sql = text(f"""
                        INSERT INTO {self._memories_table_name}
                        (id, content, tags, importance, source_message_ids, embedding,
                         intent, stakes, emotional_charge, recurrence_keywords, meaning_embedding, meaning_updated_at, created_at)
                        VALUES (:id, :content, CAST(:tags AS jsonb), :importance, :source_ids, :embedding,
                                :intent, :stakes, :emotional_charge, CAST(:recurrence_keywords AS jsonb), :meaning_embedding, :meaning_updated_at, CURRENT_TIMESTAMP)
                    """)
                    conn.execute(insert_sql, {
                        "id": memory_id,
                        "content": content,
                        "tags": json.dumps(tags),
                        "importance": importance,
                        "source_ids": json.dumps(source_message_ids or []),
                        "embedding": f"[{','.join(str(x) for x in embedding)}]" if embedding else None,
                        "intent": intent,
                        "stakes": stakes,
                        "emotional_charge": emotional_charge,
                        "recurrence_keywords": json.dumps(recurrence_keywords or []),
                        "meaning_embedding": f"[{','.join(str(x) for x in meaning_embedding)}]" if meaning_embedding else None,
                        "meaning_updated_at": meaning_updated_at,
                    })
                
                conn.commit()
                logger.debug(f"Added memory {memory_id} to {self._memories_table_name}")
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to add memory {memory_id}: {e}")
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory.
        
        Supports both full UUID and prefix matching (first 8 chars).
        """
        with Session(self.engine) as session:
            try:
                # If it's a short ID (prefix), use LIKE matching
                if len(memory_id) < 36:  # Full UUID is 36 chars
                    stmt = delete(self.memories_table).where(
                        self.memories_table.c.id.like(f"{memory_id}%")
                    )
                else:
                    stmt = delete(self.memories_table).where(
                        self.memories_table.c.id == memory_id
                    )
                result = session.execute(stmt)
                session.commit()
                return result.rowcount > 0
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to delete memory {memory_id}: {e}")
                return False

    def delete_memories_by_source_message_ids(self, message_ids: list[str]) -> int:
        """Delete all memories whose source_message_ids contain any of the given message IDs.
        
        Args:
            message_ids: List of message UUIDs to match against source_message_ids.
            
        Returns:
            Number of memories deleted.
        """
        if not message_ids:
            return 0
            
        with self.engine.connect() as conn:
            try:
                # Use JSONB containment: source_message_ids ?| array[...] checks if any element matches
                sql = text(f"""
                    DELETE FROM {self._memories_table_name}
                    WHERE source_message_ids ?| :ids
                    RETURNING id
                """)
                result = conn.execute(sql, {"ids": message_ids})
                deleted_ids = [row.id for row in result.fetchall()]
                conn.commit()
                
                if deleted_ids:
                    logger.debug(f"Deleted {len(deleted_ids)} memories associated with forgotten messages")
                return len(deleted_ids)
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to delete memories by source message IDs: {e}")
                return 0
    
    def update_memory_access(self, memory_id: str) -> None:
        """Update access tracking for a memory (reinforcement)."""
        with self.engine.connect() as conn:
            try:
                sql = text(f"""
                    UPDATE {self._memories_table_name}
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE id = :id
                """)
                conn.execute(sql, {"id": memory_id})
                conn.commit()
            except Exception as e:
                logger.error(f"Failed to update memory access: {e}")
    
    # =========================================================================
    # Search Methods
    # =========================================================================
    
    def search_memories_by_text(
        self,
        query: str,
        n_results: int = 5,
        min_importance: float = 0.0,
        tags: list[str] | None = None,
    ) -> list[dict]:
        """Search memories using PostgreSQL full-text search."""
        if not query or query.isspace():
            return []
        
        with self.engine.connect() as conn:
            try:
                # Build the query with optional filters
                tag_filter = ""
                if tags:
                    tag_list = ",".join(f"'" + t + "'" for t in tags)
                    tag_filter = f"AND tags ?| ARRAY[{tag_list}]"
                
                # Use PostgreSQL full-text search
                sql = text(f"""
                          SELECT id, content, tags, importance, source_message_ids,
                           access_count, last_accessed, created_at,
                           intent, stakes, emotional_charge,
                           ts_rank(to_tsvector('english', content), plainto_tsquery('english', :query)) AS rank
                    FROM {self._memories_table_name}
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :query)
                    AND importance >= :min_importance
                    {tag_filter}
                    ORDER BY rank DESC, importance DESC
                    LIMIT :limit
                """)
                
                rows = conn.execute(sql, {
                    "query": query,
                    "min_importance": min_importance,
                    "limit": n_results,
                }).fetchall()
                
                results = []
                for row in rows:
                    # Update access tracking
                    self.update_memory_access(row.id)
                    
                    row_tags = row.tags if isinstance(row.tags, list) else (json.loads(row.tags) if row.tags else ["misc"])
                    results.append({
                        "id": row.id,
                        "content": row.content,
                        "tags": row_tags,
                        "importance": row.importance,
                        "source_message_ids": row.source_message_ids or [],
                        "access_count": row.access_count,
                        "relevance": row.rank,
                        "intent": row.intent,
                        "stakes": row.stakes,
                        "emotional_charge": row.emotional_charge,
                    })
                
                return results
                
            except Exception as e:
                logger.error(f"Failed to search memories: {e}")
                return []
    
    def search_memories_by_embedding(
        self,
        embedding: list[float],
        n_results: int = 5,
        min_importance: float = 0.0,
        tags: list[str] | None = None,
        meaning_embedding: list[float] | None = None,
        meaning_weight: float | None = None,
    ) -> list[dict]:
        """Search memories using vector similarity with temporal decay and diversity.
        
        The effective score combines:
        - Semantic similarity (cosine distance)
        - Base importance
        - Temporal decay (memories fade over time)
        - Access boost (frequently accessed memories get reinforced)
        - Diversity sampling (avoid echo chambers by sampling across time/types)
        """
        if not embedding:
            return []
        
        # Get decay settings from config
        decay_enabled = getattr(self.config, 'MEMORY_DECAY_ENABLED', True)
        half_life_days = getattr(self.config, 'MEMORY_DECAY_HALF_LIFE_DAYS', 90.0)
        access_boost_factor = getattr(self.config, 'MEMORY_ACCESS_BOOST_FACTOR', 0.15)
        recency_weight = getattr(self.config, 'MEMORY_RECENCY_WEIGHT', 0.3)
        diversity_enabled = getattr(self.config, 'MEMORY_DIVERSITY_ENABLED', True)
        
        # Different decay rates per tag (multiplier on half_life)
        # Higher = slower decay (more persistent)
        tag_decay_multipliers = {
            'fact': 2.0,          # Core facts persist longer
            'professional': 1.5,  # Career info moderately persistent
            'preference': 0.8,    # Preferences change
            'health': 1.2,        # Health info somewhat persistent
            'relationship': 1.0,  # Relationships change at normal rate
            'event': 0.5,         # Events become less relevant quickly
            'plan': 0.3,          # Plans/goals are very temporal
        }
        
        with self.engine.connect() as conn:
            try:
                tag_filter = ""
                if tags:
                    tag_list = ",".join(f"'" + t + "'" for t in tags)
                    tag_filter = f"AND tags ?| ARRAY[{tag_list}]"
                
                if decay_enabled:
                    # Fetch more candidates for post-processing with decay + diversity
                    fetch_limit = n_results * 4 if diversity_enabled else n_results * 2
                    
                    sql = text(f"""
                           SELECT id, content, tags, importance, source_message_ids,
                               access_count, last_accessed, created_at,
                               intent, stakes, emotional_charge,
                               1 - (embedding <=> :embedding) AS similarity,
                               CASE WHEN :use_meaning THEN 1 - (meaning_embedding <=> :meaning_embedding) ELSE NULL END AS meaning_similarity,
                               EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) / 86400.0 AS age_days
                        FROM {self._memories_table_name}
                        WHERE embedding IS NOT NULL
                        AND importance >= :min_importance
                        {tag_filter}
                        ORDER BY embedding <=> :embedding
                        LIMIT :limit
                    """)
                else:
                    # Simple similarity-based search without decay
                    sql = text(f"""
                           SELECT id, content, tags, importance, source_message_ids,
                               access_count, last_accessed, created_at,
                               intent, stakes, emotional_charge,
                               1 - (embedding <=> :embedding) AS similarity,
                               CASE WHEN :use_meaning THEN 1 - (meaning_embedding <=> :meaning_embedding) ELSE NULL END AS meaning_similarity,
                               0 AS age_days
                        FROM {self._memories_table_name}
                        WHERE embedding IS NOT NULL
                        AND importance >= :min_importance
                        {tag_filter}
                        ORDER BY embedding <=> :embedding
                        LIMIT :limit
                    """)
                    fetch_limit = n_results
                
                rows = conn.execute(sql, {
                    "embedding": str(embedding),
                    "meaning_embedding": str(meaning_embedding) if meaning_embedding else str(embedding),
                    "use_meaning": bool(meaning_embedding),
                    "min_importance": min_importance,
                    "limit": fetch_limit,
                }).fetchall()
                
                if not rows:
                    return []
                
                # Calculate effective scores with decay
                import math
                scored_results = []
                
                meaning_w = meaning_weight if meaning_weight is not None else getattr(self.config, 'MEMORY_MEANING_WEIGHT', 0.3)
                meaning_w = max(0.0, min(1.0, meaning_w))
                for row in rows:
                    # Convert Decimal types to float for calculations
                    similarity = float(row.similarity) if row.similarity else 0.0
                    meaning_similarity = float(row.meaning_similarity) if meaning_embedding and row.meaning_similarity is not None else None
                    importance = float(row.importance) if row.importance else 0.5
                    age_days = float(row.age_days) if row.age_days else 0.0
                    access_count = int(row.access_count) if row.access_count else 0
                    
                    # Parse tags
                    row_tags = row.tags if isinstance(row.tags, list) else (json.loads(row.tags) if row.tags else ["misc"])
                    
                    # Get tag-specific half-life (use first tag for decay rate)
                    primary_tag = row_tags[0] if row_tags else "misc"
                    tag_multiplier = tag_decay_multipliers.get(primary_tag, 1.0)
                    effective_half_life = half_life_days * tag_multiplier
                    
                    # Temporal decay: exp(-age * ln(2) / half_life)
                    if decay_enabled and age_days > 0:
                        decay_factor = math.exp(-age_days * math.log(2) / effective_half_life)
                    else:
                        decay_factor = 1.0
                    
                    # Access boost: 1 + factor * log(access_count + 1)
                    access_boost = 1.0 + access_boost_factor * math.log(access_count + 1)
                    
                    # Combine semantic similarities
                    sim_combined = similarity
                    if meaning_similarity is not None:
                        sim_combined = (1 - meaning_w) * similarity + meaning_w * meaning_similarity
                    
                    # Combined score:
                    # similarity provides base relevance
                    # importance is the extracted importance
                    # decay_factor reduces old memories
                    # access_boost reinforces frequently used ones
                    # recency_weight balances recency vs base importance
                    base_score = sim_combined * importance
                    recency_score = sim_combined * decay_factor
                    effective_score = (
                        (1 - recency_weight) * base_score + 
                        recency_weight * recency_score
                    ) * access_boost
                    
                    scored_results.append({
                        "id": row.id,
                        "content": row.content,
                        "tags": row_tags,
                        "importance": importance,
                        "source_message_ids": row.source_message_ids or [],
                        "access_count": access_count,
                        "similarity": similarity,
                        "meaning_similarity": meaning_similarity,
                        "age_days": age_days,
                        "decay_factor": decay_factor,
                        "effective_score": effective_score,
                        "intent": row.intent,
                        "stakes": row.stakes,
                        "emotional_charge": float(row.emotional_charge) if row.emotional_charge else None,
                    })
                
                # Sort by effective score
                scored_results.sort(key=lambda x: x["effective_score"], reverse=True)
                
                # Apply diversity sampling if enabled
                if diversity_enabled and len(scored_results) > n_results:
                    results = self._diversity_sample(scored_results, n_results)
                else:
                    results = scored_results[:n_results]
                
                # Update access counts for retrieved memories
                for r in results:
                    self.update_memory_access(r["id"])
                
                return results
                
            except Exception as e:
                logger.error(f"Failed to search by embedding: {e}")
                return []
    
    def _diversity_sample(self, candidates: list[dict], n_results: int) -> list[dict]:
        """Sample diverse memories to avoid echo chambers.
        
        Strategy:
        1. Always include top-scoring result
        2. Ensure representation from different memory types
        3. Ensure representation from different time periods
        4. Fill remaining slots by score
        """
        if len(candidates) <= n_results:
            return candidates
        
        selected = []
        used_ids = set()
        
        # 1. Always take the top result
        selected.append(candidates[0])
        used_ids.add(candidates[0]["id"])
        
        # 2. Ensure tag diversity - try to get one from each represented tag set
        tags_seen = set(candidates[0].get("tags", ["misc"]))
        for candidate in candidates[1:]:
            if len(selected) >= n_results:
                break
            candidate_tags = set(candidate.get("tags", ["misc"]))
            if not candidate_tags.issubset(tags_seen) and candidate["id"] not in used_ids:
                selected.append(candidate)
                used_ids.add(candidate["id"])
                tags_seen.update(candidate_tags)
        
        # 3. Ensure temporal diversity - split into time buckets
        if len(selected) < n_results:
            # Recent (< 7 days), Medium (7-30 days), Old (30+ days)
            buckets = {"recent": [], "medium": [], "old": []}
            for candidate in candidates:
                if candidate["id"] in used_ids:
                    continue
                age = candidate.get("age_days", 0)
                if age < 7:
                    buckets["recent"].append(candidate)
                elif age < 30:
                    buckets["medium"].append(candidate)
                else:
                    buckets["old"].append(candidate)
            
            # Try to get one from each bucket we haven't covered
            for bucket_name in ["medium", "old", "recent"]:  # Prioritize less-recent
                if len(selected) >= n_results:
                    break
                bucket = buckets[bucket_name]
                if bucket:
                    candidate = bucket[0]
                    selected.append(candidate)
                    used_ids.add(candidate["id"])
        
        # 4. Fill remaining by score
        for candidate in candidates:
            if len(selected) >= n_results:
                break
            if candidate["id"] not in used_ids:
                selected.append(candidate)
                used_ids.add(candidate["id"])
        
        # Re-sort by effective_score for consistent ordering
        selected.sort(key=lambda x: x["effective_score"], reverse=True)
        
        return selected
    
    # =========================================================================
    # MemoryBackend Interface Implementation
    # =========================================================================
    
    def add(self, message_id: str, role: str, content: str, timestamp: float) -> None:
        """Add a message to storage (implements MemoryBackend interface).
        
        For backwards compatibility, this adds to the messages table.
        """
        self.add_message(message_id, role, content, timestamp)
    
    def search_messages_by_text(
        self,
        query: str,
        n_results: int = 5,
        exclude_recent_seconds: float = 5.0,
        role_filter: str | None = "user",
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict]:
        """Search raw messages using PostgreSQL full-text search.

        This is a fallback when no distilled memories exist yet.
        Uses OR logic so any matching word will return results.

        Args:
            query: Search query
            n_results: Max number of results
            exclude_recent_seconds: Exclude messages from the last N seconds to avoid
                                   finding the query message itself
            role_filter: Only include messages with this role (default: "user" to avoid
                        retrieving assistant hallucinations as facts). Set to None to
                        include all roles.
            since: Unix timestamp - only include messages after this time.
            until: Unix timestamp - only include messages before this time.
        """
        if not query or query.isspace():
            return []

        cutoff_time = time.time() - exclude_recent_seconds

        with self.engine.connect() as conn:
            try:
                # Use websearch_to_tsquery for better handling, but we need OR logic
                # Extract words and build an OR query manually
                # Filter out common words AND conversational meta-words that don't help find content
                stop_words = {
                    # Standard English stop words
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
                    'we', 'you', 'i', 'he', 'she', 'it', 'they', 'them', 'their', 'our',
                    'my', 'your', 'his', 'her', 'its', 'what', 'which', 'who', 'whom',
                    'this', 'that', 'these', 'those', 'am', 'not', 'no', 'yes', 'so',
                    'if', 'then', 'than', 'too', 'very', 'just', 'about', 'before', 'after',
                    # Conversational meta-words (common in memory queries but not content)
                    'remember', 'tell', 'told', 'said', 'say', 'know', 'think', 'thought',
                    'talk', 'talked', 'talking', 'conversation', 'conversations', 'discussed',
                    'discuss', 'discussion', 'mention', 'mentioned', 'anything', 'something',
                    'everything', 'nothing', 'past', 'previous', 'earlier', 'last', 'time',
                    'when', 'where', 'how', 'why', 'like', 'want', 'wanted', 'please',
                }

                # Extract meaningful words
                import re
                words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
                meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]

                if not meaningful_words:
                    # Fall back to simple search if no meaningful words
                    meaningful_words = [w for w in words if len(w) > 2][:3]

                if not meaningful_words:
                    return []

                # Build OR query: word1 | word2 | word3
                or_query = ' | '.join(meaningful_words)

                # Build filter clauses
                role_clause = "AND role = :role" if role_filter else ""
                since_clause = "AND timestamp >= :since" if since is not None else ""
                until_clause = "AND timestamp <= :until" if until is not None else ""

                sql = text(f"""
                    SELECT id, role, content, timestamp,
                           ts_rank(to_tsvector('english', content), to_tsquery('english', :query)) AS rank
                    FROM {self._messages_table_name}
                    WHERE to_tsvector('english', content) @@ to_tsquery('english', :query)
                    AND timestamp < :cutoff
                    {role_clause}
                    {since_clause}
                    {until_clause}
                    ORDER BY rank DESC, timestamp DESC
                    LIMIT :limit
                """)

                params: dict[str, Any] = {
                    "query": or_query,
                    "limit": n_results,
                    "cutoff": cutoff_time,
                }
                if role_filter:
                    params["role"] = role_filter
                if since is not None:
                    params["since"] = since
                if until is not None:
                    params["until"] = until

                rows = conn.execute(sql, params).fetchall()

                return [
                    {
                        "id": row.id,
                        "content": row.content,
                        "role": row.role,
                        "timestamp": row.timestamp,
                        "relevance": row.rank,
                    }
                    for row in rows
                ]

            except Exception as e:
                logger.error(f"Failed to search messages: {e}")
                return []
    
    def search(self, query: str, n_results: int = 5, min_relevance: float = 0.0) -> list[dict] | None:
        """Search for relevant memories (implements MemoryBackend interface).
        
        Search strategy:
        1. Text search on memories (fast, keyword matching) - if good matches found, use them
        2. Embedding search + high-importance blend - semantic search combined with important facts
        3. Raw messages fallback (last resort)
        
        For most queries, we blend embedding search results with high-importance core facts
        (like user's name) to ensure both relevance and foundational context.
        """
        # Minimum relevance threshold for text search to be considered "good"
        TEXT_SEARCH_MIN_RELEVANCE = 0.1
        
        def format_memory_result(r: dict) -> dict:
            return {
                "id": r["id"],
                "document": r["content"],
                "content": r["content"],  # Keep both for compatibility
                "metadata": {
                    "tags": r.get("tags", ["misc"]),
                    "importance": r["importance"],
                    "source_message_ids": r.get("source_message_ids", []),
                },
                "relevance": r.get("similarity", r.get("relevance", r["importance"])),
                # Meaning fields for structured context
                "intent": r.get("intent"),
                "stakes": r.get("stakes"),
                "emotional_charge": r.get("emotional_charge"),
                "tags": r.get("tags", ["misc"]),
                "importance": r.get("importance", 0.5),
            }
        
        # 1. Try text search on memories - but only accept high-quality matches
        text_results = self.search_memories_by_text(query, n_results, min_importance=min_relevance)
        if text_results:
            # Check if any result has good relevance (not just keyword matching noise)
            best_relevance = max(r.get("relevance", 0) for r in text_results)
            if best_relevance >= TEXT_SEARCH_MIN_RELEVANCE:
                logger.debug(f"Found {len(text_results)} memories via text search (best relevance: {best_relevance:.3f})")
                return [format_memory_result(r) for r in text_results]
            else:
                logger.debug(f"Text search results too weak (best: {best_relevance:.3f}), trying other methods")
        
        # 2. Blend embedding search with high-importance core facts
        combined_results = []
        seen_ids = set()
        
        # Get embedding search results (query-relevant)
        if self.embedding_model:
            try:
                from .embeddings import generate_embedding
                query_embedding = generate_embedding(query, self.embedding_model, verbose=getattr(self.config, 'VERBOSE', False))
                if query_embedding:
                    embedding_results = self.search_memories_by_embedding(query_embedding, n_results, min_importance=min_relevance)
                    if embedding_results:
                        logger.debug(f"Found {len(embedding_results)} memories via embedding search")
                        for r in embedding_results:
                            if r["id"] not in seen_ids:
                                combined_results.append(format_memory_result(r))
                                seen_ids.add(r["id"])
            except Exception as e:
                logger.debug(f"Embedding search failed: {e}")
        
        # Add high-importance core facts (limit to a few to avoid overwhelming)
        # These are foundational facts like name, profession that should always be included
        high_importance_results = self.get_high_importance_memories(3, min_importance=0.9)
        for r in high_importance_results:
            if r["id"] not in seen_ids:
                combined_results.append(format_memory_result(r))
                seen_ids.add(r["id"])
        
        if combined_results:
            logger.debug(f"Returning {len(combined_results)} blended results (embedding + core facts)")
            return combined_results[:n_results]
        
        # 3. Last resort: search raw messages
        message_results = self.search_messages_by_text(query, n_results, exclude_recent_seconds=10.0)
        
        if not message_results:
            return None
        
        logger.debug(f"Falling back to {len(message_results)} message results")
        
        # Format message results like memory results
        return [
            {
                "id": r["id"],
                "document": r["content"],
                "metadata": {
                    "role": r["role"],
                    "timestamp": r["timestamp"],
                },
                "relevance": r.get("relevance", 0.5),
            }
            for r in message_results
        ]
    
    def get_high_importance_memories(self, n_results: int = 10, min_importance: float = 0.7) -> list[dict]:
        """Get the most important memories, regardless of query matching.
        
        Useful as a fallback for general queries like "what do you know about me?"
        """
        with self.engine.connect() as conn:
            try:
                sql = text(f"""
                    SELECT id, content, tags, importance, source_message_ids,
                           access_count, last_accessed, created_at,
                           intent, stakes, emotional_charge
                    FROM {self._memories_table_name}
                    WHERE importance >= :min_importance
                    ORDER BY importance DESC, access_count DESC, created_at DESC
                    LIMIT :limit
                """)
                
                rows = conn.execute(sql, {
                    "min_importance": min_importance,
                    "limit": n_results,
                }).fetchall()
                
                return [
                    {
                        "id": row.id,
                        "content": row.content,
                        "tags": row.tags if isinstance(row.tags, list) else (json.loads(row.tags) if row.tags else ["misc"]),
                        "importance": row.importance,
                        "source_message_ids": row.source_message_ids or [],
                        "relevance": row.importance,  # Use importance as relevance for fallback
                        "intent": row.intent,
                        "stakes": row.stakes,
                        "emotional_charge": float(row.emotional_charge) if row.emotional_charge else None,
                    }
                    for row in rows
                ]
                
            except Exception as e:
                logger.error(f"Failed to get high-importance memories: {e}")
                return []
    
    def clear(self) -> bool:
        """Clear all memories (NOT messages - those are permanent)."""
        with Session(self.engine) as session:
            try:
                stmt = delete(self.memories_table)
                session.execute(stmt)
                session.commit()
                logger.debug(f"Cleared all memories from {self._memories_table_name}")
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to clear memories: {e}")
                return False
    
    def ignore_recent_messages(self, count: int) -> int:
        """Move the last N messages to the forgotten table.
        
        Returns the number of messages actually forgotten.
        """
        with self.engine.connect() as conn:
            try:
                # Get IDs of last N messages
                select_sql = text(f"""
                    SELECT id, role, content, timestamp, session_id, processed, created_at
                    FROM {self._messages_table_name}
                    ORDER BY timestamp DESC
                    LIMIT :count
                """)
                rows = conn.execute(select_sql, {"count": count}).fetchall()
                
                if not rows:
                    return 0
                
                ids_to_forget = [row.id for row in rows]
                
                # Insert into forgotten table
                for row in rows:
                    insert_sql = text(f"""
                        INSERT INTO {self._forgotten_table_name} 
                        (id, role, content, timestamp, session_id, processed, created_at, forgotten_at)
                        VALUES (:id, :role, :content, :timestamp, :session_id, :processed, :created_at, CURRENT_TIMESTAMP)
                        ON CONFLICT (id) DO NOTHING
                    """)
                    conn.execute(insert_sql, {
                        "id": row.id,
                        "role": row.role,
                        "content": row.content,
                        "timestamp": row.timestamp,
                        "session_id": row.session_id,
                        "processed": row.processed,
                        "created_at": row.created_at,
                    })
                
                # Delete from messages table
                delete_sql = text(f"""
                    DELETE FROM {self._messages_table_name}
                    WHERE id = ANY(:ids)
                """)
                conn.execute(delete_sql, {"ids": ids_to_forget})
                conn.commit()
                
                logger.debug(f"Forgot {len(ids_to_forget)} messages")
                return len(ids_to_forget)
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to forget messages: {e}")
                return 0
    
    def ignore_messages_since_minutes(self, minutes: int) -> int:
        """Move all messages from the last N minutes to the forgotten table.
        
        Returns the number of messages actually forgotten.
        """
        cutoff = time.time() - (minutes * 60)
        with self.engine.connect() as conn:
            try:
                # Get messages to forget
                select_sql = text(f"""
                    SELECT id, role, content, timestamp, session_id, processed, created_at
                    FROM {self._messages_table_name}
                    WHERE timestamp >= :cutoff
                    ORDER BY timestamp ASC
                """)
                rows = conn.execute(select_sql, {"cutoff": cutoff}).fetchall()
                
                if not rows:
                    return 0
                
                ids_to_forget = [row.id for row in rows]
                
                # Insert into forgotten table
                for row in rows:
                    insert_sql = text(f"""
                        INSERT INTO {self._forgotten_table_name} 
                        (id, role, content, timestamp, session_id, processed, created_at, forgotten_at)
                        VALUES (:id, :role, :content, :timestamp, :session_id, :processed, :created_at, CURRENT_TIMESTAMP)
                        ON CONFLICT (id) DO NOTHING
                    """)
                    conn.execute(insert_sql, {
                        "id": row.id,
                        "role": row.role,
                        "content": row.content,
                        "timestamp": row.timestamp,
                        "session_id": row.session_id,
                        "processed": row.processed,
                        "created_at": row.created_at,
                    })
                
                # Delete from messages table
                delete_sql = text(f"""
                    DELETE FROM {self._messages_table_name}
                    WHERE id = ANY(:ids)
                """)
                conn.execute(delete_sql, {"ids": ids_to_forget})
                conn.commit()
                
                logger.debug(f"Forgot {len(ids_to_forget)} messages from last {minutes} minutes")
                return len(ids_to_forget)
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to forget messages: {e}")
                return 0
    
    def get_message_by_id(self, message_id: str) -> dict | None:
        """Get a specific message by its ID.
        
        Supports both full UUID and prefix matching (first 8 chars).
        Returns the message dict or None if not found.
        """
        with self.engine.connect() as conn:
            try:
                # Find the message (support prefix matching)
                if len(message_id) < 36:
                    select_sql = text(f"""
                        SELECT id, role, content, timestamp, session_id, processed, created_at, summary_metadata
                        FROM {self._messages_table_name}
                        WHERE id LIKE :id_pattern
                        LIMIT 1
                    """)
                    row = conn.execute(select_sql, {"id_pattern": f"{message_id}%"}).fetchone()
                else:
                    select_sql = text(f"""
                        SELECT id, role, content, timestamp, session_id, processed, created_at, summary_metadata
                        FROM {self._messages_table_name}
                        WHERE id = :id
                    """)
                    row = conn.execute(select_sql, {"id": message_id}).fetchone()
                
                if not row:
                    return None
                
                return {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                    "session_id": row.session_id,
                    "processed": row.processed,
                    "created_at": str(row.created_at) if row.created_at else None,
                    "summary_metadata": row.summary_metadata,
                }
            except Exception as e:
                logger.error(f"Failed to get message {message_id}: {e}")
                return None
    
    def ignore_message_by_id(self, message_id: str) -> bool:
        """Move a specific message to the forgotten table by its ID.
        
        Supports both full UUID and prefix matching (first 8 chars).
        Returns True if a message was forgotten, False otherwise.
        """
        with self.engine.connect() as conn:
            try:
                # Find the message (support prefix matching)
                if len(message_id) < 36:
                    select_sql = text(f"""
                        SELECT id, role, content, timestamp, session_id, processed, created_at
                        FROM {self._messages_table_name}
                        WHERE id LIKE :id_pattern
                        LIMIT 1
                    """)
                    row = conn.execute(select_sql, {"id_pattern": f"{message_id}%"}).fetchone()
                else:
                    select_sql = text(f"""
                        SELECT id, role, content, timestamp, session_id, processed, created_at
                        FROM {self._messages_table_name}
                        WHERE id = :id
                    """)
                    row = conn.execute(select_sql, {"id": message_id}).fetchone()
                
                if not row:
                    logger.debug(f"Message {message_id} not found")
                    return False
                
                # Insert into forgotten table
                insert_sql = text(f"""
                    INSERT INTO {self._forgotten_table_name} 
                    (id, role, content, timestamp, session_id, processed, created_at, forgotten_at)
                    VALUES (:id, :role, :content, :timestamp, :session_id, :processed, :created_at, CURRENT_TIMESTAMP)
                    ON CONFLICT (id) DO NOTHING
                """)
                conn.execute(insert_sql, {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                    "session_id": row.session_id,
                    "processed": row.processed,
                    "created_at": row.created_at,
                })
                
                # Delete from messages table
                delete_sql = text(f"""
                    DELETE FROM {self._messages_table_name}
                    WHERE id = :id
                """)
                conn.execute(delete_sql, {"id": row.id})
                conn.commit()
                
                logger.debug(f"Forgot message {row.id}")
                return True
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to forget message {message_id}: {e}")
                return False
    
    def restore_ignored_messages(self) -> int:
        """Restore all forgotten messages back to the messages table.
        
        Returns the number of messages restored.
        """
        with self.engine.connect() as conn:
            try:
                # Get all forgotten messages
                select_sql = text(f"""
                    SELECT id, role, content, timestamp, session_id, processed, created_at
                    FROM {self._forgotten_table_name}
                    ORDER BY timestamp ASC
                """)
                rows = conn.execute(select_sql).fetchall()
                
                if not rows:
                    return 0
                
                ids_to_restore = [row.id for row in rows]
                
                # Insert back into messages table
                for row in rows:
                    insert_sql = text(f"""
                        INSERT INTO {self._messages_table_name} 
                        (id, role, content, timestamp, session_id, processed, created_at)
                        VALUES (:id, :role, :content, :timestamp, :session_id, :processed, :created_at)
                        ON CONFLICT (id) DO NOTHING
                    """)
                    conn.execute(insert_sql, {
                        "id": row.id,
                        "role": row.role,
                        "content": row.content,
                        "timestamp": row.timestamp,
                        "session_id": row.session_id,
                        "processed": row.processed,
                        "created_at": row.created_at,
                    })
                
                # Delete from forgotten table
                delete_sql = text(f"""
                    DELETE FROM {self._forgotten_table_name}
                    WHERE id = ANY(:ids)
                """)
                conn.execute(delete_sql, {"ids": ids_to_restore})
                conn.commit()
                
                logger.debug(f"Restored {len(ids_to_restore)} forgotten messages")
                return len(ids_to_restore)
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to restore messages: {e}")
                return 0
    
    def get_ignored_count(self) -> int:
        """Get the count of currently forgotten messages."""
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {self._forgotten_table_name}"))
            return result.scalar() or 0

    def preview_recent_messages(self, count: int) -> list[dict]:
        """Preview the last N messages (for confirmation before forget).
        
        Returns list of message dicts with id, role, content, timestamp.
        """
        with self.engine.connect() as conn:
            sql = text(f"""
                SELECT id, role, content, timestamp
                FROM {self._messages_table_name}
                ORDER BY timestamp DESC
                LIMIT :count
            """)
            rows = conn.execute(sql, {"count": count}).fetchall()
            
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                }
                for row in reversed(rows)  # Return in chronological order
            ]
    
    def preview_messages_since_minutes(self, minutes: int) -> list[dict]:
        """Preview messages from the last N minutes (for confirmation before forget).
        
        Returns list of message dicts with id, role, content, timestamp.
        """
        cutoff = time.time() - (minutes * 60)
        with self.engine.connect() as conn:
            sql = text(f"""
                SELECT id, role, content, timestamp
                FROM {self._messages_table_name}
                WHERE timestamp >= :cutoff
                ORDER BY timestamp ASC
            """)
            rows = conn.execute(sql, {"cutoff": cutoff}).fetchall()
            
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                }
                for row in rows
            ]
    
    def preview_ignored_messages(self) -> list[dict]:
        """Preview all currently forgotten messages (for confirmation before restore).
        
        Returns list of message dicts with id, role, content, timestamp.
        """
        with self.engine.connect() as conn:
            sql = text(f"""
                SELECT id, role, content, timestamp
                FROM {self._forgotten_table_name}
                ORDER BY timestamp ASC
            """)
            rows = conn.execute(sql).fetchall()
            
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                }
                for row in rows
            ]

    def list_recent(self, n: int = 10) -> list[dict]:
        """List the most recent memories."""
        with Session(self.engine) as session:
            stmt = (
                select(self.memories_table)
                .order_by(self.memories_table.c.created_at.desc())
                .limit(n)
            )
            rows = session.execute(stmt).fetchall()
            
            return [
                {
                    "id": row.id,
                    "document": row.content,
                    "metadata": {
                        "tags": row.tags if isinstance(row.tags, list) else (json.loads(row.tags) if row.tags else ["misc"]),
                        "importance": row.importance,
                        "source_message_ids": row.source_message_ids or [],
                    },
                }
                for row in rows
            ]
    
    def stats(self) -> dict:
        """Get statistics about memory storage."""
        with self.engine.connect() as conn:
            try:
                # Memory stats
                mem_stats_sql = text(f"""
                    SELECT 
                        COUNT(*) as total,
                        MIN(created_at) as oldest,
                        MAX(created_at) as newest,
                        AVG(importance) as avg_importance,
                        pg_total_relation_size('{self._memories_table_name}') as size_bytes
                    FROM {self._memories_table_name}
                """)
                mem_row = conn.execute(mem_stats_sql).first()
                
                # Message stats
                msg_stats_sql = text(f"""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN processed THEN 1 ELSE 0 END) as processed,
                        MIN(created_at) as oldest,
                        MAX(created_at) as newest
                    FROM {self._messages_table_name}
                """)
                msg_row = conn.execute(msg_stats_sql).first()
                
                # Forgotten message count
                forgotten_count = 0
                try:
                    forgotten_sql = text(f"SELECT COUNT(*) FROM {self._forgotten_table_name}")
                    forgotten_count = conn.execute(forgotten_sql).scalar() or 0
                except Exception:
                    pass  # Table may not exist yet
                
                return {
                    "memories": {
                        "total_count": mem_row.total if mem_row else 0,
                        "oldest_timestamp": mem_row.oldest.isoformat() if mem_row and mem_row.oldest else None,
                        "newest_timestamp": mem_row.newest.isoformat() if mem_row and mem_row.newest else None,
                        "avg_importance": float(mem_row.avg_importance) if mem_row and mem_row.avg_importance else 0.0,
                        "storage_size_bytes": mem_row.size_bytes if mem_row else 0,
                    },
                    "messages": {
                        "total_count": msg_row.total if msg_row else 0,
                        "processed_count": msg_row.processed if msg_row else 0,
                        "forgotten_count": forgotten_count,
                        "oldest_timestamp": msg_row.oldest.isoformat() if msg_row and msg_row.oldest else None,
                        "newest_timestamp": msg_row.newest.isoformat() if msg_row and msg_row.newest else None,
                    },
                }
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                return {"error": str(e)}
    
    def regenerate_embeddings(self, batch_size: int = 50) -> dict:
        """Regenerate embeddings for all memories that don't have them.
        
        This is useful after installing sentence-transformers or when
        switching embedding models. Also handles dimension migration.
        
        Args:
            batch_size: Number of memories to process at once
            
        Returns:
            Dict with counts of updated and failed memories
        """
        try:
            from .embeddings import generate_embeddings_batch, get_embedding_dimension
        except ImportError:
            return {"error": "sentence-transformers not installed", "updated": 0, "failed": 0}
        
        # Check if we need to alter the column dimension
        actual_dim = get_embedding_dimension(self.embedding_model)
        
        updated = 0
        failed = 0
        
        with self.engine.connect() as conn:
            # First, check and alter column dimension if needed
            try:
                check_dim_sql = text(f"""
                    SELECT atttypmod 
                    FROM pg_attribute 
                    WHERE attrelid = '{self._memories_table_name}'::regclass 
                    AND attname = 'embedding'
                """)
                result = conn.execute(check_dim_sql).first()
                if result:
                    current_dim = result[0]
                    if current_dim != actual_dim and current_dim > 0:
                        logger.debug(f"Migrating embedding dimension from {current_dim} to {actual_dim}")
                        # Drop index, alter column, recreate index
                        conn.execute(text(f"DROP INDEX IF EXISTS idx_{self._memories_table_name}_embedding"))
                        conn.execute(text(f"ALTER TABLE {self._memories_table_name} ALTER COLUMN embedding TYPE vector({actual_dim})"))
                        conn.execute(text(f"""
                            CREATE INDEX idx_{self._memories_table_name}_embedding 
                            ON {self._memories_table_name} 
                            USING hnsw (embedding vector_cosine_ops)
                        """))
                        # Clear existing embeddings since they're wrong dimension
                        conn.execute(text(f"UPDATE {self._memories_table_name} SET embedding = NULL"))
                        conn.commit()
            except Exception as e:
                logger.debug(f"Could not check/alter embedding dimension: {e}")
            
            # Get all memories without embeddings
            fetch_sql = text(f"""
                SELECT id, content 
                FROM {self._memories_table_name}
                WHERE embedding IS NULL
                ORDER BY importance DESC
                LIMIT :limit
            """)
            
            while True:
                rows = conn.execute(fetch_sql, {"limit": batch_size}).fetchall()
                if not rows:
                    break
                
                # Extract texts and IDs
                ids = [row.id for row in rows]
                texts = [row.content for row in rows]
                
                # Generate embeddings in batch
                embeddings = generate_embeddings_batch(texts, self.embedding_model)
                
                # Update each memory
                for mem_id, embedding in zip(ids, embeddings):
                    if embedding:
                        try:
                            update_sql = text(f"""
                                UPDATE {self._memories_table_name}
                                SET embedding = :embedding
                                WHERE id = :id
                            """)
                            conn.execute(update_sql, {
                                "id": mem_id,
                                "embedding": str(embedding),
                            })
                            updated += 1
                        except Exception as e:
                            logger.error(f"Failed to update embedding for {mem_id}: {e}")
                            failed += 1
                    else:
                        failed += 1
                
                conn.commit()
                logger.debug(f"Regenerated embeddings: {updated} updated, {failed} failed")
        
        return {"updated": updated, "failed": failed, "embedding_dim": actual_dim}
    
    # =========================================================================
    # Short-term Memory Manager (for session history)
    # =========================================================================
    
    @classmethod
    def get_short_term_manager(cls, config: Any, bot_id: str = "nova") -> "PostgreSQLShortTermManager":
        """Factory method to get a short-term memory manager."""
        return PostgreSQLShortTermManager(config, bot_id)


class PostgreSQLShortTermManager:
    """Manages short-term (session) conversation history using PostgreSQL.
    
    This stores messages permanently
    but provides session-scoped access for building context windows.
    """
    
    def __init__(self, config: Any, bot_id: str = "nova"):
        self.config = config
        self.bot_id = bot_id
        self._backend = PostgreSQLMemoryBackend(config, bot_id)
        self._current_session_id = str(uuid.uuid4())
    
    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._current_session_id
    
    def new_session(self) -> str:
        """Start a new session and return its ID."""
        self._current_session_id = str(uuid.uuid4())
        return self._current_session_id
    
    def add_message(self, role: str, content: str, timestamp: float | None = None) -> str:
        """Add a message to the current session.
        
        Returns the message ID.
        """
        message_id = str(uuid.uuid4())
        ts = timestamp or datetime.utcnow().timestamp()
        
        self._backend.add_message(
            message_id=message_id,
            role=role,
            content=content,
            timestamp=ts,
            session_id=self._current_session_id,
        )
        
        return message_id
    
    def get_session_history(self, limit: int = 50) -> list[dict]:
        """Get messages from the current session."""
        with Session(self._backend.engine) as session:
            stmt = (
                select(self._backend.messages_table)
                .where(self._backend.messages_table.c.session_id == self._current_session_id)
                .order_by(self._backend.messages_table.c.timestamp.desc())
                .limit(limit)
            )
            rows = session.execute(stmt).fetchall()
            
            # Return in chronological order
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                }
                for row in reversed(rows)
            ]
    
    def get_recent_messages(self, limit: int = 20) -> list[dict]:
        """Get recent messages regardless of session (for context building)."""
        with Session(self._backend.engine) as session:
            stmt = (
                select(self._backend.messages_table)
                .order_by(self._backend.messages_table.c.timestamp.desc())
                .limit(limit)
            )
            rows = session.execute(stmt).fetchall()
            
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                    "session_id": row.session_id,
                }
                for row in reversed(rows)
            ]
    
    def count(self) -> int:
        """Count messages in current session."""
        with Session(self._backend.engine) as session:
            from sqlalchemy import func
            stmt = (
                select(func.count())
                .select_from(self._backend.messages_table)
                .where(self._backend.messages_table.c.session_id == self._current_session_id)
            )
            return session.execute(stmt).scalar() or 0

    def get_messages(self, since_minutes: int | None = None, include_summaries: bool = True) -> list:
        """Get messages, optionally filtered by time.

        When include_summaries is True (default), applies smart filtering:
        - Recent messages (within since_minutes): included as-is
        - Older messages: skipped if summarized=TRUE, but summaries (role='summary') are included

        This allows summaries to condense older history while keeping recent context intact.

        Args:
            since_minutes: If provided, only return messages from the last N seconds
                          (note: despite the name, this is actually in seconds for
                          backward compatibility with HISTORY_DURATION).
            include_summaries: If True, include summary rows for older sessions

        Returns:
            List of Message objects.
        """
        from ..models.message import Message

        with self._backend.engine.connect() as conn:
            if since_minutes is not None and include_summaries:
                # Smart filtering: recent messages + older summaries
                cutoff = time.time() - since_minutes

                # Query: get recent messages OR summaries of older sessions
                # Exclude old messages that have been summarized
                sql = text(f"""
                    SELECT id, role, content, timestamp
                    FROM {self._backend._messages_table_name}
                    WHERE
                        -- Recent messages (within history window)
                        (timestamp >= :cutoff)
                        OR
                        -- Summaries of older sessions
                        (role = 'summary' AND timestamp < :cutoff)
                    ORDER BY timestamp ASC
                """)
                rows = conn.execute(sql, {"cutoff": cutoff}).fetchall()

                return [
                    Message(role=row.role, content=row.content, timestamp=row.timestamp, db_id=row.id)
                    for row in rows
                ]
            else:
                # Simple time-based filtering (original behavior)
                stmt = (
                    select(self._backend.messages_table)
                    .order_by(self._backend.messages_table.c.timestamp.asc())
                )

                if since_minutes is not None:
                    cutoff = time.time() - since_minutes
                    stmt = stmt.where(self._backend.messages_table.c.timestamp >= cutoff)

                with Session(self._backend.engine) as session:
                    rows = session.execute(stmt).fetchall()

                    return [
                        Message(role=row.role, content=row.content, timestamp=row.timestamp, db_id=row.id)
                        for row in rows
                    ]

    def clear(self) -> bool:
        """Clear all messages for this bot."""
        return self._backend.clear()

    def remove_last_message_if_partial(self, role: str) -> bool:
        """Remove the last message if it matches the specified role.
        
        Used for cleanup on error/interrupt.
        """
        with Session(self._backend.engine) as session:
            # Find the last message
            stmt = (
                select(self._backend.messages_table)
                .order_by(self._backend.messages_table.c.timestamp.desc())
                .limit(1)
            )
            row = session.execute(stmt).fetchone()
            
            if row and row.role == role:
                delete_stmt = (
                    self._backend.messages_table.delete()
                    .where(self._backend.messages_table.c.id == row.id)
                )
                session.execute(delete_stmt)
                session.commit()
                return True
            return False

    def get_messages_for_summary(self, summary_id: str) -> list:
        """Get the original raw messages that a summary covers.

        Finds the summary by ID, extracts its timestamp range from
        summary_metadata, and returns all user/assistant messages in
        that range.

        Args:
            summary_id: Database ID of the summary row.

        Returns:
            List of Message objects, or empty list if not found.
        """
        from ..models.message import Message

        with self._backend.engine.connect() as conn:
            # Get the summary row
            sql = text(f"""
                SELECT id, timestamp, summary_metadata
                FROM {self._backend._messages_table_name}
                WHERE id = :id AND role = 'summary'
            """)
            row = conn.execute(sql, {"id": summary_id}).fetchone()
            if not row:
                return []

            # Determine the time range from metadata or estimate from surrounding summaries
            meta = row.summary_metadata or {}
            start_ts = meta.get("start_timestamp", row.timestamp - 7200)
            end_ts = meta.get("end_timestamp", row.timestamp)

            # Fetch raw messages in this time range
            msgs_sql = text(f"""
                SELECT id, role, content, timestamp
                FROM {self._backend._messages_table_name}
                WHERE role IN ('user', 'assistant')
                  AND timestamp >= :start AND timestamp <= :end
                ORDER BY timestamp ASC
            """)
            rows = conn.execute(msgs_sql, {"start": start_ts, "end": end_ts}).fetchall()
            return [
                Message(role=r.role, content=r.content, timestamp=r.timestamp, db_id=r.id)
                for r in rows
            ]

    def mark_messages_recalled(self, message_ids: list[str]) -> int:
        """Mark messages as recalled (re-inserted from summary expansion).

        Args:
            message_ids: List of database IDs to mark.

        Returns:
            Number of rows updated.
        """
        if not message_ids:
            return 0

        with self._backend.engine.connect() as conn:
            sql = text(f"""
                UPDATE {self._backend._messages_table_name}
                SET recalled_history = TRUE
                WHERE id = ANY(:ids)
            """)
            result = conn.execute(sql, {"ids": message_ids})
            conn.commit()
            return result.rowcount
