# Web Fetch Tool Integration

## Overview

Add a `web_fetch` tool that allows bots to retrieve and read web page content. Uses [Crawl4AI](https://github.com/unclecode/crawl4ai) running as a local Docker service for robust extraction with JS rendering support.

## Architecture

```
Bot requests web_fetch tool
        │
        ▼
  ToolExecutor._execute_web_fetch()
        │
        ▼
  WebFetchClient.fetch(url)
        │
        ▼
  HTTP POST to Crawl4AI (localhost:11235)
        │
        ▼
  Returns markdown content to bot context
```

Crawl4AI runs as a sidecar Docker service (already added to `docker-compose.yml`). The client is a thin `httpx` wrapper — no heavy dependencies in the bot process.

## Infrastructure (Done)

Crawl4AI service in `docker-compose.yml`:

```yaml
crawl4ai:
  image: unclecode/crawl4ai:latest
  container_name: llm-bawt-crawl4ai
  ports:
    - "${CRAWL4AI_PORT:-11235}:11235"
  environment:
    - CRAWL4AI_API_TOKEN=${CRAWL4AI_API_TOKEN:-}
  restart: unless-stopped
```

## Implementation Steps

### 1. Client — `src/llm_bawt/integrations/web_fetch/client.py`

Follows the same pattern as `integrations/newsapi/client.py`: standalone class, lazy `httpx` client, returns formatted strings.

```python
class WebFetchClient:
    """Fetches web pages via Crawl4AI and returns markdown content."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 30,
        max_content_chars: int = 50_000,
    ):
        self._base_url = base_url or os.environ.get(
            "LLM_BAWT_CRAWL4AI_URL", "http://localhost:11235"
        )
        self._timeout = timeout
        self._max_content_chars = max_content_chars
        self._client: httpx.Client | None = None

    def fetch(self, url: str) -> str:
        """Fetch URL, return markdown content truncated to max_content_chars."""

    def is_available(self) -> bool:
        """Health check against Crawl4AI service."""

    def _get_client(self) -> httpx.Client:
        """Lazy-init HTTP client."""
```

**Key details:**
- Sync `httpx.Client` (matches newsapi pattern — executor is sync)
- `max_content_chars` truncation to avoid blowing up LLM context (default 50k chars)
- Crawl4AI API format: `POST /crawl` with `{"urls": [url], "browser_config": {...}, "crawler_config": {...}}`
- Response: `data["results"][0]["markdown"]["raw_markdown"]`
- Metadata (title, description) extracted from `data["results"][0]["metadata"]`
- Return format: metadata header + markdown content, as a plain string

**Return format example:**
```
Title: Large language model - Wikipedia
URL: https://en.wikipedia.org/wiki/Large_language_model

---

# Large language model

A large language model (LLM) is a type of...
[content truncated to 50000 chars]
```

### 2. Tool Definition — `src/llm_bawt/tools/definitions.py`

```python
WEB_FETCH_TOOL = Tool(
    name="web_fetch",
    description="Fetch a web page and read its content as text. Use this to read articles, documentation, blog posts, or any URL the user references.",
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="The full URL to fetch (must start with http:// or https://)",
        ),
    ],
)

WEB_FETCH_TOOLS = [WEB_FETCH_TOOL]
```

Single parameter — just the URL. No `prompt` or extraction query needed; the bot already has its own context for interpreting the content.

Add `include_web_fetch_tools` flag to `get_tools_list()` and `get_tools_prompt()`.

### 3. Executor Handler — `src/llm_bawt/tools/executor.py`

```python
# In __init__ handler dispatch table:
"web_fetch": self._execute_web_fetch,

# Handler:
def _execute_web_fetch(self, tool_call: ToolCall) -> str:
    url = tool_call.arguments.get("url", "")
    if not url:
        return format_tool_result(tool_call.name, None, error="Missing required parameter: url")

    self._ensure_web_fetch_client()
    if not self.web_fetch_client:
        return format_tool_result(tool_call.name, None, error="Web fetch service not available")

    try:
        result = self.web_fetch_client.fetch(url)
        return format_tool_result(tool_call.name, result)
    except Exception as e:
        return format_tool_result(tool_call.name, None, error=str(e))

def _ensure_web_fetch_client(self) -> None:
    if self.web_fetch_client:
        return
    try:
        from ..integrations.web_fetch.client import WebFetchClient
        self.web_fetch_client = WebFetchClient()
    except Exception as e:
        logger.warning(f"Failed to init web fetch client: {e}")
```

### 4. Wire Through Pipeline

These are the pass-through plumbing changes. Each file gets `web_fetch_client` added to its constructor/parameters:

| File | Change |
|------|--------|
| `tools/executor.py` | Add `web_fetch_client` param to `__init__` |
| `tools/loop.py` | Add `web_fetch_client` param, pass to `ToolExecutor` |
| `tools/loop.py` | Add to `query_with_tools()` function signature |
| `core/pipeline.py` | Add `web_fetch_client` param to `RequestPipeline.__init__`, pass to `query_with_tools()` |
| `tools/parser.py` | Add `"web_fetch"` to `KNOWN_TOOLS` set |

### 5. Enable Per-Bot

The tool should be opt-in via the same flags pattern used for news/search/home:

- `include_web_fetch_tools` in `get_tools_list()` / `get_tools_prompt()`
- Controlled by whether `web_fetch_client` is not None (same as news)
- In pipeline: `include_web_fetch_tools=self.web_fetch_client is not None`

### 6. Configuration

Env vars (in `.env` or `~/.config/llm-bawt/.env`):

```bash
# Crawl4AI service URL (default: http://localhost:11235)
LLM_BAWT_CRAWL4AI_URL=http://localhost:11235

# Optional API token for Crawl4AI
CRAWL4AI_API_TOKEN=

# Max content chars returned to bot (default: 50000)
LLM_BAWT_WEB_FETCH_MAX_CHARS=50000
```

No config changes needed in `utils/config.py` — the client reads from env directly (same as NewsAPI pattern).

## File Locations

```
src/llm_bawt/integrations/web_fetch/
├── __init__.py          # Exports: WebFetchClient
└── client.py            # WebFetchClient implementation
```

Changes to existing files:
```
src/llm_bawt/tools/definitions.py   # Add WEB_FETCH_TOOL, WEB_FETCH_TOOLS, flags
src/llm_bawt/tools/executor.py      # Add handler + web_fetch_client param
src/llm_bawt/tools/loop.py          # Pass-through web_fetch_client
src/llm_bawt/tools/parser.py        # Add "web_fetch" to KNOWN_TOOLS
src/llm_bawt/core/pipeline.py       # Pass-through web_fetch_client
```

## Docker Networking

When running inside Docker, the `app` container reaches Crawl4AI via Docker's internal DNS:

```
LLM_BAWT_CRAWL4AI_URL=http://crawl4ai:11235
```

For local dev (outside Docker), it's the default `http://localhost:11235`.

The dev compose already has `extra_hosts` for host.docker.internal, but since both services are in the same compose network, `crawl4ai` hostname works directly.
