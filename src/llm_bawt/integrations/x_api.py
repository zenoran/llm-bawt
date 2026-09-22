"""Bounded, read-only X API access. No scraping, retries, or provider fallback."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

API_BASE = "https://api.x.com/2"


class XApiError(Exception):
    """Safe to expose to tools/UI; never includes tokens or upstream bodies."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def request_x(token: str, path: str, params: dict[str, Any] | None = None) -> dict:
    try:
        response = httpx.get(
            f"{API_BASE}/{path}",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=20.0,
            follow_redirects=False,
        )
    except httpx.HTTPError:
        raise XApiError("unreachable", "X API is unreachable or timed out; try again later.") from None
    if response.status_code == 401:
        raise XApiError("unauthorized", "X rejected the bearer token. Reconnect X in provider accounts.")
    if response.status_code == 402:
        raise XApiError("credits_required", "X API credits are exhausted or billing is not enabled. Check console.x.com.")
    if response.status_code == 403:
        raise XApiError("forbidden", "X denied this endpoint. Check app permissions, API access, and billing in console.x.com.")
    if response.status_code == 429:
        raise XApiError("rate_limited", "X API rate limit reached. Wait before trying again.")
    if response.status_code == 400:
        raise XApiError("invalid_query", "X rejected the query or time/pagination parameters. Check X search syntax.")
    if not 200 <= response.status_code < 300:
        raise XApiError("upstream_error", f"X API returned HTTP {response.status_code}.")
    try:
        data = response.json()
    except ValueError:
        raise XApiError("invalid_response", "X API returned an invalid response.") from None
    if not isinstance(data, dict):
        raise XApiError("invalid_response", "X API returned an invalid response.")
    return data


def _time(value: str, name: str, now: datetime) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        raise XApiError("invalid_request", f"{name} must be an ISO-8601 timestamp with a timezone.") from None
    if parsed.tzinfo is None:
        raise XApiError("invalid_request", f"{name} must include a timezone.")
    parsed = parsed.astimezone(timezone.utc)
    if parsed < now - timedelta(days=7) or parsed > now - timedelta(seconds=10):
        raise XApiError("invalid_request", f"{name} must be within the last seven days and at least 10 seconds ago.")
    return parsed


def recent_search(
    config,
    query: str,
    *,
    max_results: int = 10,
    start_time: str | None = None,
    end_time: str | None = None,
    next_token: str | None = None,
) -> dict:
    """One paid page, newest first; credential resolved fresh for disconnect/rotation."""
    from ..service.providers.api_key import resolve_api_key

    query = (query or "").strip()
    if not query or len(query) > 512:
        raise XApiError("invalid_request", "X search query must contain 1–512 characters.")
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 10 <= max_results <= 100:
        raise XApiError("invalid_request", "max_results must be an integer from 10 to 100 (X's minimum page size is 10).")
    now = datetime.now(timezone.utc)
    start = _time(start_time, "start_time", now) if start_time else None
    end = _time(end_time, "end_time", now) if end_time else None
    if start and end and start >= end:
        raise XApiError("invalid_request", "start_time must be earlier than end_time.")
    if next_token is not None and (not next_token.strip() or len(next_token) > 4096):
        raise XApiError("invalid_request", "Invalid X pagination token.")
    token = resolve_api_key(config, "x")
    if not token:
        raise XApiError("not_connected", "Connect X (Twitter) in BawtHub provider accounts using an app-only bearer token.")
    params: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "sort_order": "recency",
        "tweet.fields": "created_at,author_id",
    }
    for name, value in (("start_time", start), ("end_time", end)):
        if value:
            params[name] = value.isoformat().replace("+00:00", "Z")
    if next_token:
        params["next_token"] = next_token
    data = request_x(token, "tweets/search/recent", params)
    posts = data.get("data", [])
    meta = data.get("meta")
    if not isinstance(posts, list) or not isinstance(meta, dict) or "result_count" not in meta:
        raise XApiError("invalid_response", "X search returned an incomplete response.")
    results = []
    for post in posts:
        if not isinstance(post, dict) or not str(post.get("id", "")).isdigit() or not isinstance(post.get("text"), str):
            raise XApiError("invalid_response", "X search returned an invalid post.")
        results.append({
            "id": post["id"],
            "text": post["text"],
            "url": f"https://x.com/i/status/{post['id']}",
            "created_at": post.get("created_at"),
            "author_id": post.get("author_id"),
            "source": "x",
        })
    result = {
        "query": query,
        "provider": "x",
        "count": len(results),
        "results": results,
        "next_token": meta.get("next_token"),
    }
    if data.get("errors"):
        result["warning"] = "X returned partial results; some requested data could not be retrieved."
    return result
