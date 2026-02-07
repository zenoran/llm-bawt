"""Search client abstraction for web search capabilities.

Provides a pluggable search client system supporting multiple providers:
- DuckDuckGo (free, no API key required)
- Tavily (production-ready, LLM-optimized)
- Brave (privacy-focused, API key required)

Search clients follow the same pattern as LLM clients - abstract base
with provider-specific implementations.
"""

from .base import SearchClient, SearchResult, SearchProvider
from .factory import get_search_client, get_search_unavailable_reason, is_search_available
from .brave_client import BraveSearchClient

__all__ = [
    "SearchClient",
    "SearchResult", 
    "SearchProvider",
    "get_search_client",
    "get_search_unavailable_reason",
    "is_search_available",
    "BraveSearchClient",
]
