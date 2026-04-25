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

# Suppress noisy MCP library session lifecycle logging
logging.getLogger("mcp.server").setLevel(logging.WARNING)
logging.getLogger("mcp.server.streamable_http").setLevel(logging.WARNING)

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from llm_bawt.shared.logging import LogConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP Server instance
# ---------------------------------------------------------------------------

# Allow localhost by default; add LAN hosts via LLM_BAWT_MCP_ALLOWED_HOSTS env var.
# The MCP library matches host:port patterns — use ":*" suffix to allow any port.
_allowed_hosts = [
    h.strip() for h in os.getenv(
        "LLM_BAWT_MCP_ALLOWED_HOSTS",
        "127.0.0.1:*,localhost:*",
    ).split(",")
]
_allowed_origins = [f"http://{h}" for h in _allowed_hosts]

mcp = FastMCP(
    "llm-memory",
    json_response=True,
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_allowed_hosts,
        allowed_origins=_allowed_origins,
    ),
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
async def list_memory_sources() -> list[dict]:
    """List available memory sources (bot namespaces that have stored memories).

    Use this to discover which bots have memories you can search with
    ``search_memory_source``.

    Returns:
        List of dicts with 'source' (bot_id) and 'memory_count'.
    """
    logger.debug("MCP tool invoked: tools/list_memory_sources")
    storage = _get_storage()
    return await storage.list_memory_sources()


@mcp.tool()
async def search_memory_source(
    source: str,
    query: str,
    n_results: int = 10,
    min_relevance: float = 0.0,
    tags: list[str] | None = None,
) -> list[dict]:
    """Search another bot's memories by source (read-only cross-bot search).

    This lets any bot look into a specific bot's memory store without
    modifying it.  Use ``list_memory_sources`` first to discover available
    sources.

    Args:
        source: The bot_id whose memories to search (e.g. "nova", "mira").
        query: Natural language search query.
        n_results: Maximum results to return.
        min_relevance: Minimum similarity threshold 0-1.
        tags: Optional tag filter.

    Returns:
        List of memory dicts with relevance scores.  Each dict also
        includes a 'source' key indicating which bot the memory belongs to.
    """
    logger.debug("MCP tool invoked: tools/search_memory_source source=%s", source)
    storage = _get_storage()
    memories = await storage.search_memories(
        query=query,
        bot_id=source,
        n_results=n_results,
        min_relevance=min_relevance,
        tags=tags,
    )
    results = []
    for m in memories:
        d = m.to_dict()
        d["source"] = source
        results.append(d)
    return results


@mcp.tool()
async def search_all_messages(
    query: str,
    n_results: int = 10,
    role_filter: str | None = None,
) -> list[dict]:
    """Full-text search across ALL bots' message histories at once.

    Use this when you need to find who was talking about a topic, without
    knowing which bot to search.  Much faster than searching each bot
    individually.

    Args:
        query: Search keywords or phrase.
        n_results: Maximum total results across all bots.
        role_filter: Only include messages with this role (user/assistant).
                     System messages are always excluded.

    Returns:
        List of message dicts with 'source' (bot_id), role, content,
        timestamp, and full-text rank.
    """
    logger.debug("MCP tool invoked: tools/search_all_messages")
    storage = _get_storage()
    return await storage.search_all_messages(
        query=query,
        n_results=n_results,
        role_filter=role_filter,
    )


@mcp.tool()
async def search_all_memories(
    query: str,
    n_results: int = 10,
    min_relevance: float = 0.0,
) -> list[dict]:
    """Semantic search across ALL bots' memory stores at once.

    Use this when you need to find remembered facts without knowing which
    bot stored the memory.

    Args:
        query: Natural language search query.
        n_results: Maximum total results across all bots.
        min_relevance: Minimum similarity threshold 0-1.

    Returns:
        List of memory dicts with 'source' (bot_id) and relevance scores.
    """
    logger.debug("MCP tool invoked: tools/search_all_memories")
    storage = _get_storage()
    return await storage.search_all_memories(
        query=query,
        n_results=n_results,
        min_relevance=min_relevance,
    )


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
    message_id: str | None = None,
) -> dict:
    """Add a message to conversation history.

    Args:
        role: user, assistant, or system.
        content: Message content.
        bot_id: Bot namespace.
        session_id: Optional session grouping.
        message_id: Optional client-supplied UUID for the message.

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
        message_id=message_id,
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
# Inter-Bot Communication Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def send_message_to_bot(
    target_bot_id: str,
    message: str,
    sender_bot_id: str = "unknown",
    max_tokens: int | None = None,
    temperature: float = 0.7,
) -> dict:
    """Send a message to another bot and get their response.

    This allows bots to communicate with each other by sending messages
    and receiving responses. The conversation is isolated to this single
    exchange and doesn't affect either bot's persistent memory unless
    they choose to store it.

    Args:
        target_bot_id: The bot slug to send the message to.
        message: The message content to send.
        sender_bot_id: The bot slug of the sender (for context).
        max_tokens: Maximum tokens for the response.
        temperature: Temperature for the response generation.

    Returns:
        Dict with keys: 'content' (response text), 'bot_id' (target),
        'sender' (sender_bot_id), and 'success' (bool).
    """
    logger.debug("MCP tool invoked: send_message_to_bot target=%s sender=%s", target_bot_id, sender_bot_id)

    try:
        import httpx
        import json

        # Add sender context to the message if provided
        formatted_message = message
        if sender_bot_id != "unknown":
            formatted_message = f"Message from bot '{sender_bot_id}': {message}"

        # Prepare the request payload
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": formatted_message
                }
            ],
            "bot_id": target_bot_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # Don't extract memory from inter-bot conversations by default
            # to avoid cross-contamination unless explicitly desired
            "extract_memory": False,
            "augment_memory": True,  # Allow target bot to use its own memory
            "stream": False,
        }

        # Make HTTP request to the chat completions API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8642/v1/chat/completions",
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()

        # Extract the response content
        if "choices" in result and result["choices"]:
            content = result["choices"][0]["message"]["content"] or ""
            return {
                "success": True,
                "content": content,
                "bot_id": target_bot_id,
                "sender": sender_bot_id,
                "response_model": result.get("model"),
            }
        else:
            return {
                "success": False,
                "error": f"Invalid response format: {result}",
                "content": "",
                "bot_id": target_bot_id,
                "sender": sender_bot_id,
            }

    except Exception as e:
        logger.error("Inter-bot communication failed: %s", str(e))
        return {
            "success": False,
            "error": str(e),
            "content": "",
            "bot_id": target_bot_id,
            "sender": sender_bot_id,
        }


@mcp.tool()
async def list_available_bots() -> list[dict]:
    """List all available bots that can receive messages.

    This helps bots discover what other bots are available for
    inter-bot communication via send_message_to_bot.

    Returns:
        List of bot info dicts with keys: 'slug', 'name', 'bot_type',
        'description', 'default_model', 'agent_backend'.
    """
    logger.debug("MCP tool invoked: list_available_bots")

    try:
        import httpx

        # Make HTTP request to the bots API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:8642/v1/bots",
                timeout=10.0
            )
            response.raise_for_status()
            bots_data = response.json()

        # Extract bot information
        result = []
        if isinstance(bots_data, list):
            for bot in bots_data:
                bot_info = {
                    "slug": bot.get("slug", "unknown"),
                    "name": bot.get("name", bot.get("slug", "Unknown")),
                    "bot_type": bot.get("bot_type", "chat"),
                    "description": bot.get("description", ""),
                    "default_model": bot.get("default_model", ""),
                    "agent_backend": bot.get("agent_backend"),
                }
                result.append(bot_info)

        return result

    except Exception as e:
        logger.error("Failed to list available bots: %s", str(e))
        return []


# ---------------------------------------------------------------------------
# Profile Tools
# ---------------------------------------------------------------------------


def _get_profile_manager():
    """Lazy-load ProfileManager."""
    from llm_bawt.profiles import ProfileManager
    from llm_bawt.utils.config import Config
    config = Config()
    return ProfileManager(config)


@mcp.tool()
async def profile(
    action: str,
    entity_type: str = "user",
    entity_id: str = "nick",
    category: str | None = None,
    key: str | None = None,
    value: str | None = None,
) -> str:
    """User/bot profile database — structured attributes like name, preferences, facts, personality traits.

    This is NOT the semantic memory system. Use this tool for structured profile data:
    - User profiles: name, location, occupation, preferences, interests, communication style
    - Bot profiles: personality traits, developed behaviors

    Use search_memories for free-text semantic memory search instead.

    Args:
        action: One of: "summary", "list", "get", "set", "delete".
            - summary: Get a human-readable profile overview.
            - list: List all profile attributes (optionally filtered by category).
            - get: Get a specific attribute by key.
            - set: Set/update an attribute (requires key and value).
            - delete: Delete an attribute by key.
        entity_type: "user" or "bot".
        entity_id: The user ID (for users) or bot slug (for bots).
            IMPORTANT: When entity_type="bot", pass YOUR bot slug as entity_id,
            not the user's name. e.g. entity_id="snark" for bot snark.
        category: Attribute category filter for list/set/delete.
            For users: "fact", "preference", "interest", "context", "communication".
            For bots: "personality", "preference", "interest".
        key: Attribute key (required for get/set/delete).
        value: Attribute value (required for set).

    Returns:
        Result as text.
    """
    from llm_bawt.profiles import EntityType, AttributeCategory

    # Validate entity_id matches entity_type
    if action in ("set", "delete", "list", "get"):
        pm = _get_profile_manager()
        if entity_type == "bot":
            from llm_bawt.bots import BotManager
            from llm_bawt.utils.config import Config
            bot = BotManager(Config()).get_bot(entity_id)
            if not bot:
                return f"Error: '{entity_id}' is not a valid bot. Check the bot slug."
        elif entity_type == "user":
            profile = pm.get_profile(EntityType.USER, entity_id)
            if not profile:
                # Auto-create user profiles, but check it's not a bot slug
                from llm_bawt.bots import BotManager
                from llm_bawt.utils.config import Config
                bot = BotManager(Config()).get_bot(entity_id)
                if bot:
                    return f"Error: '{entity_id}' is a bot, not a user. Use entity_type=\"bot\"."

    _ALL_CATEGORIES = [
        AttributeCategory.FACT, AttributeCategory.PREFERENCE,
        AttributeCategory.PERSONALITY, AttributeCategory.INTEREST,
        AttributeCategory.CONTEXT, AttributeCategory.COMMUNICATION,
    ]

    logger.debug("MCP tool invoked: profile action=%s entity=%s/%s", action, entity_type, entity_id)
    pm = _get_profile_manager()

    etype = EntityType.USER if entity_type == "user" else EntityType.BOT

    if action == "summary":
        if entity_type == "user":
            result = pm.get_user_profile_summary(entity_id)
        else:
            result = pm.get_bot_profile_summary(entity_id)
        return result or f"No profile data found for {entity_type} '{entity_id}'."

    elif action == "list":
        if category:
            attrs = pm.get_attributes_by_category(etype, entity_id, category)
        else:
            attrs = pm.get_all_attributes(etype, entity_id)
        if not attrs:
            return f"No attributes found for {entity_type} '{entity_id}'."
        lines = []
        for a in attrs:
            lines.append(f"[{a.category}] {a.key}: {a.value} (confidence={a.confidence})")
        return "\n".join(lines)

    elif action == "get":
        if not key:
            return "Error: 'key' is required for action='get'."
        if category:
            attr = pm.get_attribute(etype, entity_id, category, key)
        else:
            attr = None
            for cat in _ALL_CATEGORIES:
                attr = pm.get_attribute(etype, entity_id, cat, key)
                if attr:
                    break
        if not attr:
            return f"Attribute '{key}' not found."
        return f"[{attr.category}] {attr.key}: {attr.value} (confidence={attr.confidence})"

    elif action == "set":
        if not key or value is None:
            return "Error: 'key' and 'value' are required for action='set'."
        cat = category or AttributeCategory.FACT
        pm.set_attribute(etype, entity_id, cat, key, value)
        return f"Set [{cat}] {key} = {value}"

    elif action == "delete":
        if not key:
            return "Error: 'key' is required for action='delete'."
        if category:
            deleted = pm.delete_attribute(etype, entity_id, category, key)
        else:
            deleted = False
            for cat in _ALL_CATEGORIES:
                if pm.delete_attribute(etype, entity_id, cat, key):
                    deleted = True
                    break
        return f"Deleted: {deleted}"

    else:
        return f"Unknown action: {action}. Valid: summary, list, get, set, delete."


# ---------------------------------------------------------------------------
# Task system tools (registered via import side-effect)
# ---------------------------------------------------------------------------

from . import task_tools as _task_tools  # noqa: F401, E402


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
