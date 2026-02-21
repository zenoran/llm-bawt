"""Tests for reddit search alias parsing."""

from llm_bawt.tools.parser import parse_tool_calls


def test_parse_plain_reddit_search_alias() -> None:
    text = 'reddit_search {"query": "best homelab setup"}'
    calls, remaining = parse_tool_calls(text)

    assert len(calls) == 1
    assert calls[0].name == "reddit_search"
    assert calls[0].arguments["query"] == "best homelab setup"
    assert remaining == ""


def test_parse_raw_json_reddit_search_alias() -> None:
    text = '{"name":"reddit_search","arguments":{"query":"ollama tips"}}'
    calls, remaining = parse_tool_calls(text)

    assert len(calls) == 1
    assert calls[0].name == "reddit_search"
    assert calls[0].arguments["query"] == "ollama tips"
    assert remaining == ""
