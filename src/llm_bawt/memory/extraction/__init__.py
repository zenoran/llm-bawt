"""Memory extraction module for distilling important facts from conversations."""

from .prompts import (
    FACT_EXTRACTION_PROMPT,
    MEMORY_UPDATE_PROMPT,
    MEMORY_TAGS,
    get_fact_extraction_prompt,
    get_memory_update_prompt,
    estimate_importance,
)
from .service import MemoryExtractionService

__all__ = [
    "FACT_EXTRACTION_PROMPT",
    "MEMORY_UPDATE_PROMPT",
    "MEMORY_TAGS",
    "get_fact_extraction_prompt",
    "get_memory_update_prompt",
    "estimate_importance",
    "MemoryExtractionService",
]
