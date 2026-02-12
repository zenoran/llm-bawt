"""FastMCP server exposing memory tools.

Run standalone:
    uv run python -m llm_bawt.memory_server
Or via entry point (after install):
    llm-memory
"""

from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from llm_bawt.shared.logging import LogConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "llm-memory",
    json_response=True,
    stateless_http=True,
)

# Suppress uvicorn access logs by setting log_level to WARNING
# We log our own human-friendly MCP operation summaries via ServiceLogger
mcp.settings.log_level = "WARNING"


# ---------------------------------------------------------------------------
# Storage accessor (lazy load to avoid import-time DB connection)
# ---------------------------------------------------------------------------

def _get_storage():
    from llm_bawt.memory_server.storage import get_storage
    return get_storage()


# ---------------------------------------------------------------------------
# Memory Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def store_memory(
    content: str,
    tags: list[str] | None = None,
    importance: float = 0.5,
    bot_id: str = "default",
    source_message_ids: list[str] | None = None,
) -> dict:
    """Store a new memory/fact in the knowledge base.

    Args:
        content: The memory content to store.
        tags: Categorization tags (identity, preference, etc.).
        importance: Importance score 0.0-1.0.
        bot_id: Bot namespace for isolation.
        source_message_ids: Message IDs this memory was derived from.

    Returns:
        Stored memory dict with generated id.
    """
    logger.debug("MCP tool invoked: tools/store_memory bot_id=%s", bot_id)
    storage = _get_storage()
    memory = await storage.store_memory(
        content=content,
        bot_id=bot_id,
        tags=tags,
        importance=importance,
        source_message_ids=source_message_ids,
    )
    return memory.to_dict()


@mcp.tool()
async def search_memories(
    query: str,
    bot_id: str = "default",
    n_results: int = 10,
    min_relevance: float = 0.0,
    tags: list[str] | None = None,
) -> list[dict]:
    """Search memories using semantic similarity.

    Args:
        query: Natural language search query.
        bot_id: Bot namespace for isolation.
        n_results: Maximum results to return.
        min_relevance: Minimum similarity threshold 0-1.
        tags: Optional tag filter.

    Returns:
        List of memory dicts with relevance scores.
    """
    logger.debug("MCP tool invoked: tools/search_memories bot_id=%s", bot_id)
    storage = _get_storage()
    memories = await storage.search_memories(
        query=query,
        bot_id=bot_id,
        n_results=n_results,
        min_relevance=min_relevance,
        tags=tags,
    )
    return [m.to_dict() for m in memories]


@mcp.tool()
async def get_recent_context(
    bot_id: str = "default",
    n_messages: int = 10,
    n_memories: int = 5,
    query: str | None = None,
) -> dict:
    """Get recent conversation context (messages + relevant memories).

    Args:
        bot_id: Bot namespace.
        user_id: User for profile context.
        max_messages: Maximum messages to return.
        max_age_seconds: Time window.

    Returns:
        Dict with keys: messages, memories.
    """
    logger.debug("MCP tool invoked: tools/get_recent_context bot_id=%s", bot_id)
    storage = _get_storage()
    messages = await storage.get_recent_messages(
        bot_id=bot_id,
        max_messages=n_messages,
        max_age_seconds=3600,
    )

    memories: list[dict] = []
    if n_memories > 0:
        search_query = query
        if not search_query and messages:
            search_query = messages[-1].content
        if search_query:
            mems = await storage.search_memories(
                query=search_query,
                bot_id=bot_id,
                n_results=n_memories,
                min_relevance=0.0,
            )
            memories = [m.to_dict() for m in mems]

    return {"messages": [m.to_dict() for m in messages], "memories": memories}


@mcp.tool()
async def add_message(
    role: str,
    content: str,
    bot_id: str = "default",
    session_id: str | None = None,
    timestamp: float | None = None,
) -> dict:
    """Add a message to conversation history.

    Args:
        role: user, assistant, or system.
        content: Message content.
        bot_id: Bot namespace.
        session_id: Optional session grouping.

    Returns:
        Stored message dict.
    """
    logger.debug("MCP tool invoked: tools/add_message bot_id=%s role=%s", bot_id, role)
    storage = _get_storage()
    message = await storage.add_message(
        role=role,
        content=content,
        bot_id=bot_id,
        session_id=session_id,
        timestamp=timestamp,
    )
    return message.to_dict()


@mcp.tool()
async def extract_facts(
    messages: list[dict],
    bot_id: str,  # Required - must be passed explicitly
    user_id: str,  # Required - must be passed explicitly
    store: bool = True,
    use_llm: bool = True,
) -> list[dict]:
    """Extract facts from conversation messages (LLM-based).

    Args:
        messages: List of message dicts with role/content.
        bot_id: Bot namespace (required).
        user_id: User ID for profile attribute extraction (required).
        store: Whether to persist extracted facts.
        use_llm: Whether to use LLM extraction (falls back to heuristics if False).

    Returns:
        List of extracted memory dicts.
    """
    if not bot_id:
        raise ValueError("bot_id is required for extract_facts")
    if not user_id:
        raise ValueError("user_id is required for extract_facts")
    from llm_bawt.memory_server.extraction import extract_facts_from_messages
    
    facts = await extract_facts_from_messages(
        messages=messages,
        use_llm=use_llm,
        user_id=user_id,
    )
    
    if not facts:
        return []
    
    logger.debug(f"Extracted {len(facts)} facts from {len(messages)} messages")
    
    # Store facts as memories if requested
    if store:
        storage = _get_storage()
        stored_facts = []
        for fact in facts:
            memory = await storage.store_memory(
                content=fact["content"],
                bot_id=bot_id,
                tags=fact.get("tags", ["misc"]),
                importance=fact.get("importance", 0.5),
                source_message_ids=fact.get("source_message_ids", []),
            )
            stored_facts.append(memory.to_dict())
        return stored_facts
    
    return facts


@mcp.tool()
async def update_memory(
    memory_id: str,
    bot_id: str = "default",
    content: str | None = None,
    tags: list[str] | None = None,
    importance: float | None = None,
) -> dict | None:
    """Update an existing memory.

    Args:
        memory_id: ID of memory to update.
        content: New content (optional).
        tags: New tags (optional).
        importance: New importance score (optional).
        bot_id: Bot namespace.

    Returns:
        Updated memory dict, or None if not found.
    """
    logger.debug("MCP tool invoked: tools/update_memory bot_id=%s memory_id=%s", bot_id, memory_id)
    storage = _get_storage()
    memory = await storage.update_memory(
        memory_id=memory_id,
        bot_id=bot_id,
        content=content,
        tags=tags,
        importance=importance,
    )
    return memory.to_dict() if memory else None


@mcp.tool()
async def delete_memory(
    memory_id: str,
    bot_id: str = "default",
) -> bool:
    """Delete a memory.

    Args:
        memory_id: ID of memory to delete.
        bot_id: Bot namespace.

    Returns:
        True if deleted successfully.
    """
    logger.debug("MCP tool invoked: tools/delete_memory bot_id=%s memory_id=%s", bot_id, memory_id)
    storage = _get_storage()
    result = await storage.delete_memory(
        memory_id=memory_id,
        bot_id=bot_id,
    )
    return result


@mcp.tool()
async def supersede_memory(
    old_memory_id: str,
    new_memory_id: str,
    bot_id: str = "default",
) -> bool:
    """Mark a memory as superseded by another (or DELETED)."""
    storage = _get_storage()
    backend = storage.get_backend(bot_id)
    if hasattr(backend, "supersede_memory"):
        return bool(backend.supersede_memory(old_memory_id, new_memory_id))  # type: ignore[attr-defined]
    return False


@mcp.tool()
async def list_recent(
    bot_id: str = "default",
    n: int = 50,
) -> list[dict]:
    """List recent memories (backend-native shape)."""
    storage = _get_storage()
    return await storage.list_recent_memories(bot_id=bot_id, n=n)


@mcp.tool()
async def stats(bot_id: str = "default") -> dict:
    """Get memory/message stats."""
    logger.debug("MCP tool invoked: tools/stats bot_id=%s", bot_id)
    storage = _get_storage()
    return await storage.stats(bot_id=bot_id)


@mcp.tool()
async def list_memories(
    bot_id: str = "default",
    limit: int = 20,
    min_importance: float = 0.0,
) -> list[dict]:
    """List memories ordered by importance."""
    storage = _get_storage()
    return await storage.get_high_importance_memories(
        bot_id=bot_id,
        n_results=limit,
        min_importance=min_importance,
    )


@mcp.tool()
async def preview_recent_messages(bot_id: str = "default", count: int = 10) -> list[dict]:
    storage = _get_storage()
    return await storage.preview_recent_messages(bot_id=bot_id, count=count)


@mcp.tool()
async def preview_messages_since_minutes(bot_id: str = "default", minutes: int = 60) -> list[dict]:
    storage = _get_storage()
    return await storage.preview_messages_since_minutes(bot_id=bot_id, minutes=minutes)


@mcp.tool()
async def preview_ignored_messages(bot_id: str = "default") -> list[dict]:
    storage = _get_storage()
    return await storage.preview_ignored_messages(bot_id=bot_id)


@mcp.tool()
async def ignore_recent_messages(bot_id: str = "default", count: int = 10) -> int:
    storage = _get_storage()
    return await storage.ignore_recent_messages(bot_id=bot_id, count=count)


@mcp.tool()
async def ignore_messages_since_minutes(bot_id: str = "default", minutes: int = 60) -> int:
    storage = _get_storage()
    return await storage.ignore_messages_since_minutes(bot_id=bot_id, minutes=minutes)


@mcp.tool()
async def get_message_by_id(bot_id: str = "default", message_id: str = "") -> dict | None:
    """Get a specific message by ID (supports prefix match)."""
    storage = _get_storage()
    return await storage.get_message_by_id(bot_id=bot_id, message_id=message_id)


@mcp.tool()
async def ignore_message_by_id(bot_id: str = "default", message_id: str = "") -> bool:
    """Move a specific message to the forgotten table by ID (soft delete)."""
    storage = _get_storage()
    return await storage.ignore_message_by_id(bot_id=bot_id, message_id=message_id)


@mcp.tool()
async def restore_ignored_messages(bot_id: str = "default") -> int:
    storage = _get_storage()
    return await storage.restore_ignored_messages(bot_id=bot_id)


@mcp.tool()
async def get_messages_for_summary(bot_id: str = "default", summary_id: str = "") -> list[dict]:
    """Get raw user/assistant messages referenced by a summary row."""
    storage = _get_storage()
    return await storage.get_messages_for_summary(bot_id=bot_id, summary_id=summary_id)


@mcp.tool()
async def mark_messages_recalled(bot_id: str = "default", message_ids: list[str] | None = None) -> int:
    """Mark messages as recalled from summary expansion."""
    storage = _get_storage()
    return await storage.mark_messages_recalled(bot_id=bot_id, message_ids=message_ids)


@mcp.tool()
async def delete_memories_by_source_message_ids(bot_id: str = "default", message_ids: list[str] | None = None) -> int:
    storage = _get_storage()
    return await storage.delete_memories_by_source_message_ids(bot_id=bot_id, message_ids=message_ids)


@mcp.tool()
async def regenerate_embeddings(bot_id: str = "default", batch_size: int = 50) -> dict:
    storage = _get_storage()
    return await storage.regenerate_embeddings(bot_id=bot_id, batch_size=batch_size)


@mcp.tool()
async def consolidate_memories(
    bot_id: str = "default",
    dry_run: bool = True,
    similarity_threshold: float | None = None,
) -> dict:
    storage = _get_storage()
    return await storage.consolidate_memories(
        bot_id=bot_id,
        dry_run=dry_run,
        similarity_threshold=similarity_threshold,
    )


@mcp.tool()
async def update_memory_meaning(
    bot_id: str = "default",
    memory_id: str = "",
    intent: str | None = None,
    stakes: str | None = None,
    emotional_charge: float | None = None,
    recurrence_keywords: list[str] | None = None,
    updated_tags: list[str] | None = None,
) -> bool:
    logger.debug("MCP tool invoked: tools/update_memory_meaning bot_id=%s memory_id=%s", bot_id, memory_id)
    storage = _get_storage()
    return await storage.update_memory_meaning(
        bot_id=bot_id,
        memory_id=memory_id,
        intent=intent,
        stakes=stakes,
        emotional_charge=emotional_charge,
        recurrence_keywords=recurrence_keywords,
        updated_tags=updated_tags,
    )


@mcp.tool()
async def run_maintenance(
    bot_id: str = "default",
    run_consolidation: bool = True,
    run_recurrence_detection: bool = True,
    run_decay_pruning: bool = False,
    run_orphan_cleanup: bool = False,
    dry_run: bool = False,
) -> dict:
    logger.debug("MCP tool invoked: tools/run_maintenance bot_id=%s", bot_id)
    storage = _get_storage()
    return await storage.run_maintenance(
        bot_id=bot_id,
        run_consolidation=run_consolidation,
        run_recurrence_detection=run_recurrence_detection,
        run_decay_pruning=run_decay_pruning,
        run_orphan_cleanup=run_orphan_cleanup,
        dry_run=dry_run,
    )


@mcp.tool()
async def get_messages(
    bot_id: str = "default",
    since_seconds: int | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Get messages for building context windows."""
    logger.debug("MCP tool invoked: tools/get_messages bot_id=%s", bot_id)
    storage = _get_storage()
    return await storage.get_messages(bot_id=bot_id, since_seconds=since_seconds, limit=limit)


@mcp.tool()
async def clear_messages(bot_id: str = "default") -> int:
    """Delete all messages for a bot."""
    logger.debug("MCP tool invoked: tools/clear_messages bot_id=%s", bot_id)
    storage = _get_storage()
    return await storage.clear_messages(bot_id=bot_id)


@mcp.tool()
async def remove_last_message_if_partial(bot_id: str = "default", role: str = "assistant") -> bool:
    logger.debug("MCP tool invoked: tools/remove_last_message_if_partial bot_id=%s role=%s", bot_id, role)
    storage = _get_storage()
    return await storage.remove_last_message_if_partial(bot_id=bot_id, role=role)


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------


def run_server(
    transport: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8001,
) -> None:
    """Run the MCP server.

    When called as CLI entry point, parses command-line arguments.
    When called programmatically, uses provided arguments.

    Args:
        transport: 'stdio' or 'http'. Defaults to 'http' for service mode.
        host: Bind host for HTTP transport.
        port: Bind port for HTTP transport.
    """
    import argparse

    # Parse CLI arguments when called as entry point
    parser = argparse.ArgumentParser(description="Run the MCP memory server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=None,
        help="Transport protocol: 'stdio' for local tools, 'http' for web services (default: http)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind host for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Bind port for HTTP transport (default: 8001)",
    )

    args, _ = parser.parse_known_args()

    # CLI args override function arguments
    final_transport = args.transport or transport or "http"
    final_host = args.host if args.host != "0.0.0.0" else host
    final_port = args.port if args.port != 8001 else port

    # Default to verbose logging for the memory server so MCP tool invocations
    # are visible when running standalone. Can be disabled via env.
    verbose_env = os.getenv("LLM_BAWT_MEMORY_SERVER_VERBOSE", "1").lower()
    debug_env = os.getenv("LLM_BAWT_MEMORY_SERVER_DEBUG", "0").lower()
    verbose = verbose_env not in {"0", "false", "no"}
    debug = debug_env in {"1", "true", "yes"}
    
    # Only configure logging if running standalone (not in-process within llm-service)
    # Check if we're in the main thread - if not, we're likely in a daemon thread
    import threading
    is_standalone = threading.current_thread() is threading.main_thread()
    
    if is_standalone:
        LogConfig.configure(verbose=verbose or debug, debug=debug)
    
    # Suppress uvicorn access logs - FastMCP starts uvicorn internally,
    # and we log our own MCP operation summaries
    if not debug:
        logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    # Only log startup message when running standalone
    if is_standalone:
        logger.info("Starting MCP server: transport=%s host=%s port=%s", final_transport, final_host, final_port)

    if final_transport == "stdio":
        mcp.run(transport="stdio")
    else:
        # Use streamable-http which exposes /mcp endpoint for JSON-RPC calls
        mcp.settings.host = final_host
        mcp.settings.port = final_port
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    run_server()
