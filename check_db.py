#!/usr/bin/env python3
"""Check what's in the database."""

from llm_bawt.utils.config import Config
from llm_bawt.profiles import ProfileManager, EntityType
from sqlmodel import Session, select
from llm_bawt.profiles import ProfileAttribute, EntityProfile

config = Config()
user_id = config.DEFAULT_USER  # Use configured user, not hardcoded "default"
manager = ProfileManager(config)

print("=== Entity Profiles ===")
with Session(manager.engine) as session:
    stmt = select(EntityProfile).where(EntityProfile.entity_type == EntityType.USER)
    user_profiles = list(session.exec(stmt).all())
    for p in user_profiles:
        print(f"  {p.entity_id}: {p.display_name or '(no name)'}")

print(f"\n=== Attributes for user '{user_id}' ===")
attrs = manager.get_all_attributes(EntityType.USER, user_id)
for a in attrs:
    print(f"  {a.category}.{a.key} = {a.value}")
    print(f"    confidence={a.confidence:.0%}, source={a.source}, created={a.created_at}")

if not attrs:
    print("  (no attributes found)")

# Now test SQL directly
print("\n=== Direct SQL Query ===")
with Session(manager.engine) as session:
    stmt = select(ProfileAttribute)
    all_attrs = list(session.exec(stmt).all())
    print(f"Total attributes in database: {len(all_attrs)}")
    for a in all_attrs[:10]:  # Show first 10
        print(f"  [{a.id}] {a.entity_type}/{a.entity_id}: {a.category}.{a.key} = {a.value}")
