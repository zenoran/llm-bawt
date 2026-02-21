"""Tests for search tool execution modes."""

from llm_bawt.search.base import SearchClient, SearchProvider, SearchResult
from llm_bawt.tools.executor import ToolExecutor
from llm_bawt.tools.parser import ToolCall


class _FakeSearchClient(SearchClient):
    PROVIDER = SearchProvider.DUCKDUCKGO
    REQUIRES_API_KEY = False

    def __init__(self) -> None:
        super().__init__(max_results=5)
        self.last_call: tuple[str, str, str | None] | None = None

    def search(
        self,
        query: str,
        max_results: int | None = None,
        region: str | None = None,
    ) -> list[SearchResult]:
        del max_results, region
        self.last_call = ("web", query, None)
        return [
            SearchResult(
                title="Web result",
                url="https://example.com",
                snippet="web snippet",
                source=self.PROVIDER,
            )
        ]

    def search_news(
        self,
        query: str,
        max_results: int | None = None,
        time_range: str | None = None,
    ) -> list[SearchResult]:
        del max_results
        self.last_call = ("news", query, time_range)
        return [
            SearchResult(
                title="News result",
                url="https://news.example.com",
                snippet="news snippet",
                source=self.PROVIDER,
            )
        ]

    def search_reddit(
        self,
        query: str,
        max_results: int | None = None,
        time_range: str | None = None,
    ) -> list[SearchResult]:
        del max_results
        self.last_call = ("reddit", query, time_range)
        return [
            SearchResult(
                title="Reddit result",
                url="https://www.reddit.com/r/python/",
                snippet="reddit snippet",
                source=self.PROVIDER,
            )
        ]


def test_search_tool_executes_reddit_type() -> None:
    search_client = _FakeSearchClient()
    executor = ToolExecutor(user_id="test-user", search_client=search_client)

    result = executor.execute(
        ToolCall(
            name="search",
            arguments={"type": "reddit", "query": "uv vs pip", "time_range": "w"},
            raw_text="",
        )
    )

    assert 'status="success"' in result
    assert "Reddit result" in result
    assert search_client.last_call == ("reddit", "uv vs pip", "w")


def test_legacy_reddit_search_alias_normalizes_to_search_tool() -> None:
    search_client = _FakeSearchClient()
    executor = ToolExecutor(user_id="test-user", search_client=search_client)

    result = executor.execute(
        ToolCall(
            name="reddit_search",
            arguments={"query": "local llm setup"},
            raw_text="",
        )
    )

    assert 'status="success"' in result
    assert "Reddit result" in result
    assert search_client.last_call == ("reddit", "local llm setup", "w")
