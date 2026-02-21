"""Tests for Reddit Search client."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

from llm_bawt.search.base import SearchProvider
from llm_bawt.search.reddit_client import RedditSearchClient, is_reddit_available


class TestRedditSearchClient:
    """Tests for RedditSearchClient."""

    def test_provider_constant(self):
        """Verify provider metadata."""
        assert RedditSearchClient.PROVIDER == SearchProvider.REDDIT
        assert RedditSearchClient.REQUIRES_API_KEY is True

    def test_init_without_credentials(self):
        """Client should be unavailable without both credentials."""
        client = RedditSearchClient(client_id="", client_secret="")
        assert client.is_available() is False

    @patch("llm_bawt.search.reddit_client.httpx.Client")
    def test_search_reddit_success_and_time_filter(self, mock_client_class):
        """Search should authenticate and return only results within time window."""
        now_ts = datetime.now(UTC).timestamp()
        old_ts = (datetime.now(UTC) - timedelta(days=14)).timestamp()

        auth_response = Mock()
        auth_response.raise_for_status = Mock()
        auth_response.json.return_value = {
            "access_token": "token-123",
            "expires_in": 3600,
        }

        search_response = Mock()
        search_response.raise_for_status = Mock()
        search_response.json.return_value = {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "Fresh post",
                            "permalink": "/r/test/comments/abc/fresh/",
                            "created_utc": now_ts,
                            "subreddit_name_prefixed": "r/test",
                            "score": 42,
                            "num_comments": 7,
                        }
                    },
                    {
                        "data": {
                            "title": "Old post",
                            "permalink": "/r/test/comments/def/old/",
                            "created_utc": old_ts,
                            "subreddit_name_prefixed": "r/test",
                            "score": 3,
                            "num_comments": 1,
                        }
                    },
                ]
            }
        }

        mock_client = Mock()
        mock_client.post.return_value = auth_response
        mock_client.get.return_value = search_response
        mock_client_class.return_value = mock_client

        client = RedditSearchClient(
            client_id="cid",
            client_secret="csecret",
            user_agent="llm-bawt-test/1.0",
        )
        results = client.search_reddit("python tooling", max_results=5, time_range="w")

        assert len(results) == 1
        assert results[0].title == "Fresh post"
        assert results[0].url.startswith("https://www.reddit.com/r/test/comments/abc/fresh/")

        mock_client.post.assert_called_once()
        mock_client.get.assert_called_once()

    @patch("llm_bawt.search.reddit_client.httpx.Client")
    def test_search_reuses_cached_token(self, mock_client_class):
        """Token request should be reused across searches until expiry."""
        auth_response = Mock()
        auth_response.raise_for_status = Mock()
        auth_response.json.return_value = {
            "access_token": "token-123",
            "expires_in": 3600,
        }

        search_response = Mock()
        search_response.raise_for_status = Mock()
        search_response.json.return_value = {"data": {"children": []}}

        mock_client = Mock()
        mock_client.post.return_value = auth_response
        mock_client.get.return_value = search_response
        mock_client_class.return_value = mock_client

        client = RedditSearchClient(client_id="cid", client_secret="csecret")
        _ = client.search("query one")
        _ = client.search("query two")

        assert mock_client.post.call_count == 1
        assert mock_client.get.call_count == 2


class TestRedditAvailability:
    """Tests for availability checks."""

    def test_is_reddit_available(self):
        """httpx should always be available in this project."""
        assert is_reddit_available() is True
