#!/usr/bin/env python3
"""
Rebuild user profile from conversation history.

This script:
1. Reads recent message history from the database
2. Runs memory extraction on the messages to find facts about the user
3. Stores extracted facts as profile attributes
4. Optionally regenerates the profile summary

Usage:
    python scripts/rebuild_profile.py --user user --bot nova --days 30
    python scripts/rebuild_profile.py --user user --dry-run
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmbothub.utils.config import Config
from llmbothub.profiles import ProfileManager, EntityType
from llmbothub.memory_server.client import MemoryClient
from llmbothub.memory.extraction import MemoryExtractionService
from llmbothub.memory.profile_maintenance import ProfileMaintenanceService
from llmbothub.core.client import LLMBotHub

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_message_history(memory_client: MemoryClient, since_days: int = 30) -> list[dict]:
    """Get message history from the database."""
    since_minutes = since_days * 24 * 60
    messages = memory_client.get_messages(since_minutes=since_minutes)
    return messages


def derive_profile_from_memories(
    memories: list[dict],
    user_id: str,
    profile_manager: ProfileManager,
    llm_client,
    dry_run: bool = False,
) -> dict:
    """Derive profile attributes from existing memories using LLM."""
    
    if not memories:
        logger.info("No memories to process")
        return {"memories_processed": 0, "profile_attrs": 0}
    
    logger.info(f"Processing {len(memories)} memories...")
    
    # Build a prompt with all memories for the LLM to analyze
    memory_text = "\n".join([
        f"- [{m['importance']:.1f}] {m['content']} (tags: {m['tags']})"
        for m in memories
    ])
    
    prompt = f"""Analyze these stored memories about a user and extract structured profile attributes.

MEMORIES:
{memory_text}

For each distinct fact, preference, or interest, output a JSON object with:
- category: "fact" | "preference" | "interest" | "health"  
- key: short identifier (e.g., "name", "location", "occupation", "pet", "hobby")
- value: the actual information
- importance: 0.0-1.0

Group related memories into single attributes. For example:
- "User has a dog" and "User's dog is named Cabbie" → one "pet" attribute
- "User has back problems" and "User has ongoing back issue" → one "health_condition" attribute

Output JSON array only:
```json
[
  {{"category": "fact", "key": "location", "value": "Ohio", "importance": 0.9}},
  ...
]
```"""

    from llmbothub.models.message import Message
    
    try:
        response = llm_client.query(
            messages=[
                Message(role="system", content="You extract structured profile data from memories. Output only valid JSON."),
                Message(role="user", content=prompt),
            ],
            plaintext_output=True,
            stream=False,
        )
        
        # Parse JSON from response
        import json
        import re
        
        # Try to extract JSON array
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            logger.error(f"No JSON array found in response: {response[:200]}")
            return {"memories_processed": len(memories), "profile_attrs": 0, "error": "No JSON"}
        
        attrs = json.loads(json_match.group())
        
        if dry_run:
            logger.info(f"[DRY RUN] Would create {len(attrs)} profile attributes:")
            for a in attrs:
                logger.info(f"  {a['category']}.{a['key']}: {a['value']}")
            return {"memories_processed": len(memories), "profile_attrs": len(attrs), "dry_run": True}
        
        # Store each attribute
        count = 0
        for attr in attrs:
            category = attr.get("category", "fact")
            key = attr.get("key", "unknown")
            value = attr.get("value", "")
            importance = attr.get("importance", 0.7)
            
            if not value:
                continue
            
            profile_manager.set_attribute(
                EntityType.USER,
                user_id,
                category,
                key,
                value,
                confidence=importance,
                source="memory_derived",
            )
            count += 1
            logger.info(f"  Set {category}.{key}: {value[:50]}...")
        
        return {"memories_processed": len(memories), "profile_attrs": count}
        
    except Exception as e:
        logger.error(f"Failed to derive profile from memories: {e}")
        import traceback
        traceback.print_exc()
        return {"memories_processed": len(memories), "profile_attrs": 0, "error": str(e)}


def extract_profile_from_messages(
    messages: list[dict],
    user_id: str,
    bot_id: str,
    config: Config,
    llm_client,
    profile_manager: ProfileManager,
    memory_backend,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Extract profile information from messages using LLM."""
    
    # Group messages into conversation turns
    turns = []
    current_turn = []
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user":
            if current_turn:
                turns.append(current_turn)
            current_turn = [msg]
        elif role == "assistant" and current_turn:
            current_turn.append(msg)
    
    if current_turn:
        turns.append(current_turn)
    
    logger.info(f"Found {len(turns)} conversation turns to analyze")
    
    # Create extraction service with LLM
    extraction_service = MemoryExtractionService(llm_client=llm_client)
    
    total_facts = 0
    total_profile_attrs = 0
    
    for i, turn in enumerate(turns):
        if len(turn) < 2:
            continue
            
        user_msg = turn[0].get("content", "")
        assistant_msg = turn[-1].get("content", "") if len(turn) > 1 else ""
        
        if not user_msg or not assistant_msg:
            continue
        
        # Skip very short messages
        if len(user_msg) < 10 and len(assistant_msg) < 20:
            continue
        
        logger.info(f"Processing turn {i+1}/{len(turns)}: \"{user_msg[:50]}...\"")
        
        if dry_run:
            logger.info("  [DRY RUN] Would extract from this turn")
            continue
        
        try:
            # Format as conversation messages
            turn_messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            
            # Extract facts from this turn
            facts = extraction_service.extract_from_conversation(turn_messages, use_llm=True)
            
            if verbose:
                logger.info(f"  Raw facts returned: {len(facts) if facts else 0}")
                for f in (facts or []):
                    logger.info(f"    - {f.content[:60]}... (imp={f.importance}, profile={f.profile_attribute})")
            
            if not facts:
                continue
            
            # Filter by minimum importance
            min_importance = getattr(config, "MEMORY_EXTRACTION_MIN_IMPORTANCE", 0.5)
            facts = [f for f in facts if f.importance >= min_importance]
            
            if verbose:
                logger.info(f"  After importance filter (>={min_importance}): {len(facts)}")
            
            for fact in facts:
                # Store in memory
                memory_backend.add_memory(
                    content=fact.content,
                    importance=fact.importance,
                    tags=fact.tags,
                    source="extraction_rebuild",
                )
                total_facts += 1
                
                # Check for profile attribute
                if fact.profile_attribute:
                    category = fact.profile_attribute.get("category", "fact")
                    key = fact.profile_attribute.get("key", "unknown")
                    
                    profile_manager.set_attribute(
                        EntityType.USER,
                        user_id,
                        category,
                        key,
                        fact.content,
                        confidence=fact.importance,
                        source="extraction_rebuild",
                    )
                    total_profile_attrs += 1
                    logger.info(f"  Profile: {category}.{key} = {fact.content[:40]}...")
            
            if facts:
                logger.info(f"  Extracted: {len(facts)} facts, {total_profile_attrs} profile attrs")
                
        except Exception as e:
            logger.warning(f"  Error extracting from turn: {e}")
    
    return {
        "turns_processed": len(turns),
        "facts_extracted": total_facts,
        "profile_attrs_extracted": total_profile_attrs,
    }


def regenerate_profile_summary(
    user_id: str,
    profile_manager: ProfileManager,
    llm_client,
    dry_run: bool = False,
) -> dict:
    """Regenerate the profile summary from attributes."""
    
    service = ProfileMaintenanceService(profile_manager, llm_client)
    
    if dry_run:
        logger.info("[DRY RUN] Would regenerate profile summary")
        return {"dry_run": True}
    
    result = service.run(user_id, "user", dry_run=False)
    
    return {
        "attributes_before": result.attributes_before,
        "attributes_after": result.attributes_after,
        "categories_updated": result.categories_updated,
        "error": result.error,
    }


def main():
    parser = argparse.ArgumentParser(description="Rebuild user profile from message history")
    parser.add_argument("--user", "-u", default="user", help="User ID to rebuild profile for")
    parser.add_argument("--bot", "-b", default="nova", help="Bot ID")
    parser.add_argument("--days", "-d", type=int, default=30, help="Days of history to process")
    parser.add_argument("--limit", "-l", type=int, default=0, help="Limit number of turns to process (0=all)")
    parser.add_argument("--model", "-m", help="Model to use (default: from config)")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually save anything")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip extraction, only regenerate summary")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    config = Config()
    
    logger.info(f"=== Profile Rebuild for {args.user} ===")
    logger.info(f"Bot: {args.bot}")
    logger.info(f"Days of history: {args.days}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("")
    
    # Initialize components
    logger.info("Initializing components...")
    
    profile_manager = ProfileManager(config)
    
    # Memory client - try MCP server first, then fall back to direct PostgreSQL
    memory_url = getattr(config, "MEMORY_SERVER_URL", None)
    memory_client = None
    pg_backend = None  # Declare outside try block
    
    if memory_url:
        logger.info(f"Using MCP server: {memory_url}")
        memory_client = MemoryClient(server_url=memory_url, bot_id=args.bot)
        # For MCP mode, we don't have direct backend access
        logger.error("MCP mode not supported for rebuild - need direct PostgreSQL")
        return 1
    else:
        # Fall back to direct PostgreSQL access
        logger.info("Using direct PostgreSQL connection...")
        try:
            from llmbothub.memory.postgresql import PostgreSQLMemoryBackend
            from sqlalchemy import text
            
            pg_backend = PostgreSQLMemoryBackend(config, bot_id=args.bot)
            
            # Create a wrapper that matches MemoryClient interface
            class DirectMemoryClient:
                def __init__(self, backend, bot_id):
                    self.backend = backend
                    self.bot_id = bot_id
                    self._table_name = f"{backend.bot_id_sanitized}_messages"
                
                def get_messages(self, since_minutes=None):
                    """Get all messages, optionally filtered by time."""
                    import time as time_module
                    
                    with self.backend.engine.connect() as conn:
                        if since_minutes:
                            cutoff = time_module.time() - (since_minutes * 60)  # Convert to seconds
                            result = conn.execute(
                                text(f"SELECT id, role, content, timestamp FROM {self._table_name} WHERE timestamp > :cutoff ORDER BY timestamp ASC"),
                                {"cutoff": cutoff}
                            )
                        else:
                            result = conn.execute(
                                text(f"SELECT id, role, content, timestamp FROM {self._table_name} ORDER BY timestamp ASC")
                            )
                        
                        return [{"id": r[0], "role": r[1], "content": r[2], "timestamp": r[3]} for r in result]
                
                def store_memory(self, content, tags=None, importance=0.5, source="explicit"):
                    return self.backend.add_memory(
                        content=content,
                        tags=tags or [],
                        importance=importance,
                        source=source,
                    )
            
            memory_client = DirectMemoryClient(pg_backend, args.bot)
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # LLM client
    model = args.model or config.DEFAULT_MODEL_ALIAS or "gpt-5.2-2025-12-11"
    logger.info(f"Loading model: {model}")
    
    try:
        # Create LLMBotHub instance to get the client
        llmbothub = LLMBotHub(
            resolved_model_alias=model,
            config=config,
            bot_id=args.bot,
            user_id=args.user,
            local_mode=True,  # Don't need database for this
        )
        llm_client = llmbothub.client
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    logger.info(f"Model loaded: {model}")
    logger.info("")
    
    # Get memories from the database
    logger.info("Fetching memories...")
    from sqlalchemy import text
    with pg_backend.engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT content, importance, tags 
            FROM {args.bot}_memories 
            WHERE importance >= 0.5
            ORDER BY importance DESC
        """))
        memories = [{"content": r[0], "importance": r[1], "tags": r[2]} for r in result]
    logger.info(f"Found {len(memories)} memories (importance >= 0.5)")
    logger.info("")
    
    # Derive profile from memories
    if not args.skip_extraction and memories:
        logger.info("=== Deriving profile from memories ===")
        derive_result = derive_profile_from_memories(
            memories=memories,
            user_id=args.user,
            profile_manager=profile_manager,
            llm_client=llm_client,
            dry_run=args.dry_run,
        )
        logger.info("")
        logger.info(f"Derivation complete:")
        logger.info(f"  Memories processed: {derive_result.get('memories_processed', 0)}")
        logger.info(f"  Profile attrs created: {derive_result.get('profile_attrs', 0)}")
        logger.info("")
    
    # Regenerate summary
    logger.info("=== Regenerating profile summary ===")
    summary_result = regenerate_profile_summary(
        user_id=args.user,
        profile_manager=profile_manager,
        llm_client=llm_client,
        dry_run=args.dry_run,
    )
    
    if summary_result.get("error"):
        logger.error(f"Error regenerating summary: {summary_result['error']}")
    else:
        logger.info(f"Summary regeneration complete:")
        logger.info(f"  Categories updated: {summary_result.get('categories_updated', [])}")
    
    logger.info("")
    
    # Show current profile
    logger.info("=== Current Profile ===")
    current_summary = profile_manager.get_user_profile_summary(args.user)
    if current_summary:
        logger.info(current_summary)
    else:
        logger.info("(empty)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
