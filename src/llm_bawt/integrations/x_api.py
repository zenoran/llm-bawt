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


SORT_ORDERS = ("recency", "relevancy")
GRANULARITIES = ("minute", "hour", "day")


def _prepare(config, query: str, start_time: str | None, end_time: str | None, next_token: str | None) -> tuple[str, dict[str, Any]]:
    """Shared validation for every paid X read; rejects before resolving or spending."""
    from ..service.providers.api_key import resolve_api_key

    query = (query or "").strip()
    if not query or len(query) > 512:
        raise XApiError("invalid_request", "X search query must contain 1–512 characters.")
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
    params: dict[str, Any] = {"query": query}
    for name, value in (("start_time", start), ("end_time", end)):
        if value:
            params[name] = value.isoformat().replace("+00:00", "Z")
    if next_token:
        params["next_token"] = next_token
    return token, params


def _choice(value: str, name: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise XApiError("invalid_request", f"{name} must be one of: {', '.join(allowed)}.")
    return value


def recent_search(
    config,
    query: str,
    *,
    max_results: int = 10,
    start_time: str | None = None,
    end_time: str | None = None,
    next_token: str | None = None,
    sort_order: str = "recency",
    include_authors: bool = False,
) -> dict:
    """One paid page; credential resolved fresh for disconnect/rotation.

    Engagement metrics are post fields (no extra resource). Author expansion is
    opt-in because X bills returned user objects separately.
    """
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 10 <= max_results <= 100:
        raise XApiError("invalid_request", "max_results must be an integer from 10 to 100 (X's minimum page size is 10).")
    _choice(sort_order, "sort_order", SORT_ORDERS)
    token, params = _prepare(config, query, start_time, end_time, next_token)
    params.update({
        "max_results": max_results,
        "sort_order": sort_order,
        "tweet.fields": "created_at,author_id,public_metrics",
    })
    if include_authors:
        params["expansions"] = "author_id"
        params["user.fields"] = "username,name,verified,public_metrics"
    data = request_x(token, "tweets/search/recent", params)
    posts = data.get("data", [])
    meta = data.get("meta")
    if not isinstance(posts, list) or not isinstance(meta, dict) or "result_count" not in meta:
        raise XApiError("invalid_response", "X search returned an incomplete response.")
    users = {}
    if include_authors:
        for user in (data.get("includes") or {}).get("users") or []:
            if isinstance(user, dict) and user.get("id"):
                users[str(user["id"])] = {
                    "username": user.get("username"),
                    "name": user.get("name"),
                    "verified": user.get("verified"),
                    "followers": (user.get("public_metrics") or {}).get("followers_count"),
                }
    results = []
    for post in posts:
        if not isinstance(post, dict) or not str(post.get("id", "")).isdigit() or not isinstance(post.get("text"), str):
            raise XApiError("invalid_response", "X search returned an invalid post.")
        metrics = post.get("public_metrics") or {}
        item = {
            "id": post["id"],
            "text": post["text"],
            "url": f"https://x.com/i/status/{post['id']}",
            "created_at": post.get("created_at"),
            "author_id": post.get("author_id"),
            "likes": metrics.get("like_count"),
            "reposts": metrics.get("retweet_count"),
            "replies": metrics.get("reply_count"),
            "quotes": metrics.get("quote_count"),
            "source": "x",
        }
        author = users.get(str(post.get("author_id")))
        if author:
            item["author"] = author
            if author.get("username"):
                item["url"] = f"https://x.com/{author['username']}/status/{post['id']}"
        results.append(item)
    result = {
        "query": query.strip(),
        "provider": "x",
        "sort_order": sort_order,
        "count": len(results),
        "results": results,
        "next_token": meta.get("next_token"),
    }
    if data.get("errors"):
        result["warning"] = "X returned partial results; some requested data could not be retrieved."
    return result


def recent_counts(
    config,
    query: str,
    *,
    granularity: str = "hour",
    start_time: str | None = None,
    end_time: str | None = None,
    next_token: str | None = None,
) -> dict:
    """Post volume per bucket (last 7 days) without returning posts: find spikes, then search them."""
    _choice(granularity, "granularity", GRANULARITIES)
    token, params = _prepare(config, query, start_time, end_time, next_token)
    params["granularity"] = granularity
    data = request_x(token, "tweets/counts/recent", params)
    buckets = data.get("data", [])
    meta = data.get("meta")
    if not isinstance(buckets, list) or not isinstance(meta, dict):
        raise XApiError("invalid_response", "X counts returned an incomplete response.")
    rows = []
    for bucket in buckets:
        if not isinstance(bucket, dict) or not isinstance(bucket.get("tweet_count"), int):
            raise XApiError("invalid_response", "X counts returned an invalid bucket.")
        rows.append({"start": bucket.get("start"), "end": bucket.get("end"), "count": bucket["tweet_count"]})
    peaks = sorted(rows, key=lambda r: r["count"], reverse=True)[:5]
    return {
        "query": query.strip(),
        "provider": "x",
        "granularity": granularity,
        "total": meta.get("total_tweet_count", sum(r["count"] for r in rows)),
        "peaks": peaks,
        "buckets": rows,
        "next_token": meta.get("next_token"),
    }
