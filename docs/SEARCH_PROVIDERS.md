# Search Providers

## Overview

llm-bawt supports web search through multiple providers via a pluggable search client architecture. Search results are used as context for LLM responses when the user's query benefits from current information.

## Available Providers

| Provider | API Key Required | Free Tier | Features |
|----------|-----------------|-----------|----------|
| DuckDuckGo | No | Unlimited | Web search, free fallback |
| Tavily | Yes | Limited | Web search, AI summaries |
| Brave | Yes | 2,000/month | Web, news, AI summaries, privacy-focused |

## Provider Selection

Provider selection is automatic with this priority:

1. Explicit `--search-provider` CLI argument
2. `LLM_BAWT_SEARCH_PROVIDER` config setting
3. Tavily (if API key configured)
4. Brave (if API key configured)
5. DuckDuckGo (free fallback, always available)

## Architecture

All providers implement the `SearchClient` base class (`search/base.py`):

```python
class SearchClient(ABC):
    PROVIDER: SearchProvider
    REQUIRES_API_KEY: bool

    def search(self, query: str, max_results: int | None = None) -> list[SearchResult]
    def is_available(self) -> bool
```

The factory function `get_search_client(config, provider, max_results)` in `search/factory.py` handles instantiation and fallback logic.

## Configuration

Add to `~/.config/llm-bawt/.env`:

```bash
# Provider selection (optional - auto-selects based on available keys)
LLM_BAWT_SEARCH_PROVIDER=brave

# Tavily
LLM_BAWT_TAVILY_API_KEY=your-key-here

# Brave Search
LLM_BAWT_BRAVE_API_KEY=your-key-here
LLM_BAWT_BRAVE_SAFESEARCH=moderate    # off, moderate, strict

# General search settings
LLM_BAWT_SEARCH_MAX_RESULTS=5
LLM_BAWT_SEARCH_TIMEOUT=10
```

### Getting API Keys

- **Tavily:** https://tavily.com/
- **Brave Search:** https://api.search.brave.com/ (free tier: 2,000 queries/month)
- **DuckDuckGo:** No key needed

## Brave Search Client

The Brave Search client (`search/brave_client.py`) provides:

- **Web search** with region filtering and relevance-based results
- **News search** with time range filtering (day, week, month, year)
- **AI summaries** via `search_with_summary()` (requires Brave Pro plan)
- Extra snippet merging for richer context
- Lazy HTTP client initialization with proper cleanup
- Specific error handling for auth (401) and rate limit (429) responses

Uses the Brave Search REST API directly via `httpx` (no SDK dependency).

## DuckDuckGo Client

The DuckDuckGo client (`search/ddgs_client.py`) uses the `ddgs` library. No API key required. Serves as the universal fallback provider.

## Tavily Client

The Tavily client (`search/tavily_client.py`) uses the `tavily-python` SDK. Supports search with optional AI-generated answer summaries.

## File Locations

```
src/llm_bawt/search/
├── __init__.py        # Exports: get_search_client, SearchClient, SearchResult, etc.
├── base.py            # SearchClient ABC, SearchResult, SearchProvider enum
├── factory.py         # get_search_client(), is_search_available()
├── ddgs_client.py     # DuckDuckGo client
├── tavily_client.py   # Tavily client
└── brave_client.py    # Brave Search client
```

## Adding a New Search Provider

1. Add a value to the `SearchProvider` enum in `search/base.py`
2. Create a client class in `search/` extending `SearchClient`
3. Implement `search()` and `is_available()` methods
4. Add API key config field to `Config` in `utils/config.py`
5. Register in `search/factory.py`:
   - Add to `get_search_client()` instantiation logic
   - Add to `is_search_available()` checks
   - Add to `get_search_unavailable_reason()` messages
6. Export from `search/__init__.py`

## CLI Usage

```bash
# Use auto-selected provider
llm "current weather in Seattle"

# Specify provider explicitly
llm --search-provider brave "latest AI news"

# Check search availability
llm --status
```
