#!/usr/bin/env python3
"""Test profile extraction from facts."""

import logging
logging.basicConfig(level=logging.DEBUG)

from llmbothub.utils.config import Config
from llmbothub.memory_server.extraction import extract_profile_attributes_from_fact, _extract_attribute_key

config = Config()
user_id = config.DEFAULT_USER  # Use configured user, not hardcoded "default"

# Test key extraction first
test_facts = [
    "User has 2 dogs named Nora and Cabbie",
    "User is 45 years old",
    "User works as a software engineer",
    "User likes Python programming",
    "User's favorite color is blue",
    "User lives in Seattle",
]

print("=== Testing key extraction ===")
for content in test_facts:
    key = _extract_attribute_key(content)
    print(f"  '{content[:50]}...' -> key={key}")

print("\n=== Testing full extraction ===")
# Test with a mock fact
test_fact = {
    "content": "User has 2 dogs named Nora and Cabbie",
    "tags": ["fact", "relationship"],
    "importance": 0.8,
}

result = extract_profile_attributes_from_fact(test_fact, user_id=user_id, config=config)
print(f"Result: {result}")

# Check what's in the database now
print(f"\n=== Database contents for user '{user_id}' ===")
from llmbothub.profiles import ProfileManager, EntityType
manager = ProfileManager(config)
attrs = manager.get_all_attributes(EntityType.USER, user_id)
for a in attrs:
    print(f"  {a.category}.{a.key} = {a.value[:60] if len(str(a.value)) > 60 else a.value}")
