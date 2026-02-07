#!/usr/bin/env python3
"""
Clean up bloated profile attributes, keeping only core identity traits.

Usage:
    python scripts/cleanup_profile.py --user user --dry-run
    python scripts/cleanup_profile.py --user user --execute
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmbothub.utils.config import Config
from llmbothub.profiles import ProfileManager, EntityType
from llmbothub.memory_server.extraction import ALLOWED_PROFILE_KEYS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def cleanup_profile(user_id: str, dry_run: bool = True):
    """Remove non-core profile attributes."""
    config = Config()
    manager = ProfileManager(config)
    
    logger.info(f"=== Profile Cleanup for {user_id} ===")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    logger.info("")
    
    # Get all attributes
    attrs = manager.get_all_attributes(EntityType.USER, user_id)
    
    # Categorize
    to_keep = []
    to_remove = []
    
    for attr in attrs:
        key_lower = attr.key.lower()
        if key_lower in ALLOWED_PROFILE_KEYS:
            to_keep.append(attr)
        else:
            to_remove.append(attr)
    
    logger.info(f"Found {len(attrs)} profile attributes:")
    logger.info(f"  - Keeping: {len(to_keep)} (core identity)")
    logger.info(f"  - Removing: {len(to_remove)} (projects, tools, temporal)")
    logger.info("")
    
    if to_remove:
        logger.info("Attributes to REMOVE:")
        for attr in to_remove:
            val = str(attr.value)[:60] + "..." if len(str(attr.value)) > 60 else str(attr.value)
            logger.info(f"  - {attr.category}.{attr.key}: {val}")
        logger.info("")
    
    if to_keep:
        logger.info("Attributes to KEEP:")
        for attr in to_keep:
            val = str(attr.value)[:60] + "..." if len(str(attr.value)) > 60 else str(attr.value)
            logger.info(f"  ✓ {attr.category}.{attr.key}: {val}")
        logger.info("")
    
    if not dry_run and to_remove:
        logger.info("Executing removal...")
        from sqlalchemy import text
        
        # Get engine from profile manager
        engine = manager.engine
        
        with engine.connect() as conn:
            for attr in to_remove:
                sql = text("""
                    DELETE FROM user_profile_attributes 
                    WHERE entity_type = :entity_type 
                    AND entity_id = :entity_id 
                    AND category = :category 
                    AND key = :key
                """)
                conn.execute(sql, {
                    "entity_type": "user",
                    "entity_id": user_id,
                    "category": attr.category,
                    "key": attr.key,
                })
                logger.info(f"  Removed: {attr.category}.{attr.key}")
            conn.commit()
        
        logger.info("")
        logger.info("Cleanup complete!")
        
        # Regenerate summary
        logger.info("Regenerating profile summary...")
        # The summary will be regenerated on next use automatically
        
    elif dry_run:
        logger.info("DRY RUN - no changes made.")
        logger.info("Run with --execute to apply changes.")
    else:
        logger.info("No attributes to remove.")
    
    return len(to_keep), len(to_remove)


def main():
    parser = argparse.ArgumentParser(description="Clean up bloated profile attributes")
    parser.add_argument("--user", "-u", default="user", help="User ID to cleanup")
    parser.add_argument("--execute", action="store_true", help="Actually remove attributes (default is dry run)")
    
    args = parser.parse_args()
    
    dry_run = not args.execute
    kept, removed = cleanup_profile(args.user, dry_run)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
