#!/usr/bin/env python3
"""POC: Crawl4AI web page extraction for bot tools."""

import httpx
import json
import sys

CRAWL4AI_URL = "http://localhost:11235"


async def fetch_webpage(url: str, timeout: float = 30.0) -> dict:
    """Fetch a webpage and return extracted markdown content."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{CRAWL4AI_URL}/crawl",
            json={
                "urls": [url],
                "browser_config": {
                    "type": "BrowserConfig",
                    "params": {"headless": True},
                },
                "crawler_config": {
                    "type": "CrawlerRunConfig",
                    "params": {
                        "cache_mode": "bypass",
                    },
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()

    result = data["results"][0]
    if not result["success"]:
        return {"success": False, "error": result.get("error_message", "Unknown error"), "url": url}

    md_data = result["markdown"]
    raw_markdown = md_data.get("raw_markdown", "")
    markdown_with_citations = md_data.get("markdown_with_citations", "")

    return {
        "success": True,
        "url": result.get("url") or url,
        "markdown": raw_markdown,
        "markdown_with_citations": markdown_with_citations,
        "metadata": result.get("metadata", {}),
        "char_count": len(raw_markdown),
    }


async def main():
    test_urls = [
        "https://en.wikipedia.org/wiki/Large_language_model",
        "https://httpbin.org/html",
        "https://news.ycombinator.com",
    ]

    url = test_urls[0] if len(sys.argv) < 2 else sys.argv[1]
    print(f"Fetching: {url}")
    print("-" * 60)

    result = await fetch_webpage(url)

    if not result["success"]:
        print(f"FAILED: {result['error']}")
        return

    print(f"URL: {result['url']}")
    print(f"Chars: {result['char_count']}")
    print(f"Metadata: {json.dumps(result.get('metadata', {}), indent=2)[:500]}")
    print()
    print("--- Markdown (first 3000 chars) ---")
    print(result["markdown"][:3000])


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
