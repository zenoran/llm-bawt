#!/usr/bin/env python3
"""Quick test of profile attribute saving."""

from llm_bawt.utils.config import Config
from llm_bawt.profiles import ProfileManager, EntityType, AttributeCategory

config = Config()
manager = ProfileManager(config)

# Test setting an attribute
print("Setting test attribute...")
attr = manager.set_attribute(
    entity_type=EntityType.USER,
    entity_id="default",
    category=AttributeCategory.FACT,
    key="test_age",
    value="25",
    confidence=0.9,
    source="test",
)

print(f"✓ Created attribute with ID: {attr.id}")
print(f"  Entity: {attr.entity_type}/{attr.entity_id}")
print(f"  Attribute: {attr.category}.{attr.key} = {attr.value}")

# Retrieve it
print("\nRetrieving attribute...")
retrieved = manager.get_attribute(EntityType.USER, "default", AttributeCategory.FACT, "test_age")
if retrieved:
    print(f"✓ Retrieved: {retrieved.category}.{retrieved.key} = {retrieved.value}")
else:
    print("✗ Could not retrieve attribute!")

# List all attributes
print("\nAll attributes for user 'default':")
attrs = manager.get_all_attributes(EntityType.USER, "default")
for a in attrs:
    print(f"  - {a.category}.{a.key} = {a.value} (conf={a.confidence:.0%}, source={a.source})")

print(f"\n✓ Found {len(attrs)} total attributes")
