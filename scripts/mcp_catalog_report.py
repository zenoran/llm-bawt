#!/usr/bin/env python3
"""Measure local or live MCP discovery metadata without calling any tools.

Run with the project's Python (tiktoken is already a dependency):
  .venv/bin/python scripts/mcp_catalog_report.py
  .venv/bin/python scripts/mcp_catalog_report.py --url http://app:8001/mcp
  .venv/bin/python scripts/mcp_catalog_report.py --baseline /tmp/catalog.json

--snapshot writes raw catalog metadata for a subsequent comparison. Do not put
request payloads, prompts or tool results in these snapshots.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

import tiktoken


def serialized(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def measure(tools: list[dict], encoding: str) -> dict:
    encoder = tiktoken.get_encoding(encoding)

    def count(value: object) -> int:
        return len(encoder.encode(value if isinstance(value, str) else serialized(value)))

    definitions = [
        {"name": f"mcp__bawthub__{tool['name']}",
         "description": tool.get("description") or "",
         "input_schema": tool["inputSchema"]}
        for tool in tools
    ]
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"tools": 0, "tokens": 0})
    largest = []
    for tool, definition in zip(tools, definitions, strict=True):
        size = count(definition)
        group = groups[tool["name"].split("_")[0]]
        group["tools"] += 1
        group["tokens"] += size
        largest.append({"name": tool["name"], "tokens": size})
    return {
        "encoding": encoding,
        "tool_count": len(tools),
        "full_definitions_tokens": count(definitions),
        "descriptions_tokens": sum(count(tool.get("description") or "") for tool in tools),
        "input_schemas_tokens": sum(count(tool["inputSchema"]) for tool in tools),
        "names_only_tokens": count("\n".join(d["name"] for d in definitions)),
        "groups": dict(sorted(groups.items(), key=lambda pair: -pair[1]["tokens"])),
        "largest": sorted(largest, key=lambda row: -row["tokens"])[:10],
        "scope": "Full name/description/input_schema JSON, not provider framing, results, instructions or billed usage. Deferred loading/cache costs are separate.",
    }


async def get_catalog(url: str | None) -> list[dict]:
    if url:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = []
                cursor = None
                while True:
                    page = await session.list_tools(cursor=cursor)
                    tools.extend(page.tools)
                    cursor = page.nextCursor
                    if not cursor:
                        break
    else:
        from llm_bawt.mcp_server.server import mcp

        tools = await mcp.list_tools()
    return [tool.model_dump(by_alias=True, exclude_none=True) for tool in tools]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Live MCP URL; omit for local source registration")
    parser.add_argument("--encoding", default="o200k_base")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    tools = await get_catalog(args.url)
    report = measure(tools, args.encoding)
    report["source"] = args.url or "local source (not proof of activation)"
    if args.snapshot:
        args.snapshot.write_text(json.dumps(tools, ensure_ascii=False, indent=2))
    if args.baseline:
        baseline = measure(json.loads(args.baseline.read_text()), args.encoding)
        old = baseline["full_definitions_tokens"]
        new = report["full_definitions_tokens"]
        report["comparison"] = {
            "before": old, "after": new, "saved": old - new,
            "reduction_percent": round(100 * (old - new) / old, 2) if old else None,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
