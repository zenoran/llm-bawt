"""Web search MCP tool (registered via import side-effect from server.py).

Exposes ``web_search`` to every agent bot on the bawthub MCP server. This is the
local replacement for the Claude CLI's Anthropic *server-side* ``WebSearch``
tool, which only works against api.anthropic.com and hangs on the proxy path
(grok / openai bots). With no provider specified it fans the query out to every
configured provider in parallel (Brave + Reddit + Tavily as available) and
returns the merged, source-tagged results.

Page retrieval is already covered for agent bots by the crawl4ai MCP server, so
there is no separate ``web_fetch`` tool here.
"""

from __future__ import annotations

import logging

from .server import mcp

logger = logging.getLogger(__name__)


@mcp.tool(name="web_search")
async def web_search(
    query: str,
    max_results: int = 5,
    provider: str | None = None,
) -> dict:
    """Search the live web via local providers (Brave / Reddit / Tavily).

    With no ``provider``, fans the query out to every configured provider in
    parallel and merges the results, each tagged with its source. Use this for
    current events, docs, or anything outside the model's training data. For
    fetching a specific page's contents, use the crawl4ai tools instead.

    Args:
        query: The search query.
        max_results: Max results to request per provider (default 5).
        provider: Optional single provider to restrict to — one of
            ``brave``, ``reddit``, ``tavily``, ``duckduckgo``. Omit to fan out
            to all configured providers.

    Returns:
        Dict with ``query``, ``providers`` (list actually queried),
        ``count``, and ``results`` (list of ``{title, url, snippet, score,
        source}``). If no provider is configured, ``results`` is empty and
        ``error`` explains how to enable search.
    """
    from llm_bawt.search.multi import search_all
    from llm_bawt.utils.config import config

    q = (query or "").strip()
    if not q:
        return {"query": query, "providers": [], "count": 0, "results": [],
                "error": "Missing required parameter: query"}

    providers = [provider] if provider else None
    try:
        results, queried = await search_all(
            config, q, max_results=max_results, providers=providers
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("web_search failed for %r: %s", q, e)
        return {"query": q, "providers": [], "count": 0, "results": [],
                "error": f"Search failed: {e}"}

    if not queried:
        from llm_bawt.search import get_search_unavailable_reason
        return {"query": q, "providers": [], "count": 0, "results": [],
                "error": get_search_unavailable_reason(config)}

    logger.info("web_search %r → %d results across %s", q, len(results), queried)
    return {
        "query": q,
        "providers": queried,
        "count": len(results),
        "results": [r.to_dict() for r in results],
    }
