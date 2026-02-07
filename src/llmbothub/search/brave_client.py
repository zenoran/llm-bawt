"""Brave Search client implementation."""

import logging

import httpx

from .base import SearchClient, SearchProvider, SearchResult

logger = logging.getLogger(__name__)

BRAVE_API_BASE = "https://api.search.brave.com/res/v1"


class BraveSearchClient(SearchClient):
    """Brave Search API client."""

    PROVIDER = SearchProvider.BRAVE
    REQUIRES_API_KEY = True

    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
        timeout: int = 10,
        include_summary: bool = False,
        safesearch: str = "moderate",
    ):
        """Initialize Brave Search client.

        Args:
            api_key: Brave Search API key
            max_results: Default number of results to return
            timeout: Request timeout in seconds
            include_summary: Include AI-generated summary (requires Pro plan)
            safesearch: Filter level - "off", "moderate", or "strict"
        """
        super().__init__(max_results=max_results)
        self._api_key = api_key
        self._timeout = timeout
        self._include_summary = include_summary
        self._safesearch = safesearch
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Lazy-initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=BRAVE_API_BASE,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self._api_key,
                },
                timeout=self._timeout,
            )
        return self._client

    def search(
        self,
        query: str,
        max_results: int | None = None,
        region: str | None = None,
    ) -> list[SearchResult]:
        """Perform a Brave web search."""
        if not self._api_key:
            logger.error("Brave API key not configured")
            return []

        max_results = max_results or self.max_results

        try:
            client = self._get_client()

            params = {
                "q": query,
                "count": max_results,
                "safesearch": self._safesearch,
            }

            if region:
                params["country"] = region

            if self._include_summary:
                params["summary"] = "1"

            response = client.get("/web/search", params=params)
            response.raise_for_status()
            data = response.json()

            results = []

            web_results = data.get("web", {}).get("results", [])
            for raw in web_results:
                snippet = raw.get("description", "")
                extra = raw.get("extra_snippets", [])
                if extra:
                    snippet = f"{snippet}\n{' '.join(extra[:2])}"

                results.append(SearchResult(
                    title=raw.get("title", ""),
                    url=raw.get("url", ""),
                    snippet=snippet,
                    score=self._normalize_score(raw),
                    source=self.PROVIDER,
                    raw=raw,
                ))

            return results[:max_results]

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("Brave API key is invalid")
            elif e.response.status_code == 429:
                logger.warning("Brave rate limit exceeded")
            else:
                logger.error(f"Brave search HTTP error: {e}")
            return []
        except httpx.RequestError as e:
            logger.error(f"Brave search request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Brave search failed: {e}")
            return []

    def search_news(
        self,
        query: str,
        max_results: int | None = None,
        time_range: str | None = None,
    ) -> list[SearchResult]:
        """Search Brave news."""
        if not self._api_key:
            logger.error("Brave API key not configured")
            return []

        max_results = max_results or self.max_results

        try:
            client = self._get_client()

            params = {
                "q": query,
                "count": max_results,
                "safesearch": self._safesearch,
            }

            if time_range:
                freshness_map = {
                    "d": "pd",
                    "w": "pw",
                    "m": "pm",
                    "y": "py",
                }
                if time_range in freshness_map:
                    params["freshness"] = freshness_map[time_range]

            response = client.get("/news/search", params=params)
            response.raise_for_status()
            data = response.json()

            results = []

            news_results = data.get("results", [])
            for raw in news_results:
                snippet = raw.get("description", "")
                age = raw.get("age", "")
                if age:
                    snippet = f"[{age}] {snippet}"

                results.append(SearchResult(
                    title=raw.get("title", ""),
                    url=raw.get("url", ""),
                    snippet=snippet,
                    score=None,
                    source=self.PROVIDER,
                    raw=raw,
                ))

            return results[:max_results]

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("Brave API key is invalid")
            elif e.response.status_code == 429:
                logger.warning("Brave rate limit exceeded")
            else:
                logger.error(f"Brave news search HTTP error: {e}")
            return []
        except httpx.RequestError as e:
            logger.error(f"Brave news search request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Brave news search failed: {e}")
            return []

    def search_with_summary(
        self,
        query: str,
        max_results: int | None = None,
    ) -> tuple[list[SearchResult], str | None]:
        """Search with AI-generated summary (requires Pro plan)."""
        if not self._api_key:
            return [], None

        max_results = max_results or self.max_results

        try:
            client = self._get_client()

            params = {
                "q": query,
                "count": max_results,
                "safesearch": self._safesearch,
                "summary": "1",
            }

            response = client.get("/web/search", params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            web_results = data.get("web", {}).get("results", [])
            for raw in web_results:
                results.append(SearchResult(
                    title=raw.get("title", ""),
                    url=raw.get("url", ""),
                    snippet=raw.get("description", ""),
                    score=self._normalize_score(raw),
                    source=self.PROVIDER,
                    raw=raw,
                ))

            summary = None
            summarizer = data.get("summarizer", {})
            if summarizer:
                summary = summarizer.get("summary")

            return results[:max_results], summary

        except Exception as e:
            logger.error(f"Brave search with summary failed: {e}")
            return [], None

    def _normalize_score(self, raw: dict) -> float | None:
        """Normalize Brave ranking to a 0.0-1.0 score."""
        return None

    def is_available(self) -> bool:
        """Check if Brave Search is properly configured."""
        return is_brave_available() and bool(self._api_key)

    def __del__(self):
        """Clean up HTTP client on deletion."""
        if self._client:
            self._client.close()


def is_brave_available() -> bool:
    """Check if httpx is installed (core dependency)."""
    try:
        import httpx  # noqa: F401
        return True
    except ImportError:
        return False
