"""Tests for Brave Search client."""

from unittest.mock import Mock, patch

from llm_bawt.search.base import SearchProvider
from llm_bawt.search.brave_client import BraveSearchClient, is_brave_available


class TestBraveSearchClient:
    """Tests for BraveSearchClient."""

    def test_provider_constant(self):
        """Verify provider is set correctly."""
        assert BraveSearchClient.PROVIDER == SearchProvider.BRAVE
        assert BraveSearchClient.REQUIRES_API_KEY is True

    def test_init_with_api_key(self):
        """Test client initialization."""
        client = BraveSearchClient(api_key="test-key", max_results=10)
        assert client._api_key == "test-key"
        assert client.max_results == 10
        assert client.is_available() is True

    def test_init_without_api_key(self):
        """Test client without API key."""
        client = BraveSearchClient(api_key="", max_results=5)
        assert client.is_available() is False

    def test_search_no_api_key(self):
        """Search should return empty list without API key."""
        client = BraveSearchClient(api_key="")
        results = client.search("test query")
        assert results == []

    @patch("llm_bawt.search.brave_client.httpx.Client")
    def test_search_success(self, mock_client_class):
        """Test successful search."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Test Result",
                        "url": "https://example.com",
                        "description": "Test description",
                    }
                ]
            }
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = BraveSearchClient(api_key="test-key")
        results = client.search("test query")

        assert len(results) == 1
        assert results[0].title == "Test Result"
        assert results[0].url == "https://example.com"
        assert results[0].source == SearchProvider.BRAVE

    @patch("llm_bawt.search.brave_client.httpx.Client")
    def test_search_news(self, mock_client_class):
        """Test news search."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "News Article",
                    "url": "https://news.example.com",
                    "description": "Breaking news",
                    "age": "2 hours ago",
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = BraveSearchClient(api_key="test-key")
        results = client.search_news("test query", time_range="d")

        assert len(results) == 1
        assert "2 hours ago" in results[0].snippet

    @patch("llm_bawt.search.brave_client.httpx.Client")
    def test_search_reddit(self, mock_client_class):
        """Test reddit search uses web endpoint with site filter."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Reddit Thread",
                        "url": "https://www.reddit.com/r/test/comments/abc123/thread/",
                        "description": "Interesting discussion",
                    }
                ]
            }
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = BraveSearchClient(api_key="test-key")
        results = client.search_reddit("best mechanical keyboard", time_range="w")

        assert len(results) == 1
        assert results[0].title == "Reddit Thread"
        assert results[0].url.startswith("https://www.reddit.com/")

        mock_client.get.assert_called_once()
        called_path = mock_client.get.call_args.args[0]
        called_params = mock_client.get.call_args.kwargs["params"]
        assert called_path == "/web/search"
        assert called_params["q"].startswith("site:reddit.com")
        assert called_params["freshness"] == "pw"


class TestBraveAvailability:
    """Tests for availability checks."""

    def test_is_brave_available(self):
        """httpx should always be available."""
        assert is_brave_available() is True
