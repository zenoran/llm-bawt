"""Anthropic Messages body ↔ OpenAI Responses API body translation.

Anthropic shape we accept (subset of the public Messages API):

    {
      "model": "<provider>/<upstream_model>",
      "system": "...string..." | [{"type":"text","text":"..."}, ...],
      "messages": [
        {"role": "user", "content": "string" | [{"type":"text","text":...}, ...]},
        {"role": "assistant", "content": [
            {"type":"text","text":"..."},
            {"type":"tool_use","id":"tu_X","name":"...","input":{...}},
        ]},
        {"role": "user", "content": [
            {"type":"tool_result","tool_use_id":"tu_X","content":"..."}
        ]},
      ],
      "tools": [{"name":"...","description":"...","input_schema":{...}}],
      "max_tokens": 4096,
      "temperature": ...,
      "stream": true
    }

Responses API shape we emit:

    {
      "model": "<upstream_model>",
      "instructions": "...",   # merged system content
      "input": [
        {"role":"user","content":"..."} | with multimodal parts,
        {"role":"assistant","content":"..."},
        {"type":"function_call","name":"...","arguments":"<json>","call_id":"tu_X"},
        {"type":"function_call_output","call_id":"tu_X","output":"..."},
      ],
      "tools": [{"type":"function","name":"...","description":"...","parameters":{...}}],
      "max_output_tokens": 4096,
      "temperature": ...,
      "store": false,
      "stream": true
    }

Conversion follows ``llm_bawt/clients/responses_client.py`` for the
Responses-API side so the codebase stays consistent on that shape.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _flatten_system(system: Any) -> str | None:
    """Anthropic ``system`` can be a string or list of content blocks."""
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = block.get("text") or ""
                if text:
                    parts.append(text)
        return "\n\n".join(parts) if parts else None
    return None


def _user_content_to_responses(content: Any) -> tuple[list[dict], list[dict]]:
    """Split a user-role content payload into (regular content parts, tool_result items).

    Anthropic packs tool_result blocks inside a user message; Responses API
    breaks them out as separate ``function_call_output`` input items. So a
    single Anthropic user message can produce 0–1 user content items + N
    function_call_output items.
    """
    parts: list[dict] = []
    tool_results: list[dict] = []

    if isinstance(content, str):
        if content:
            parts.append({"type": "input_text", "text": content})
        return parts, tool_results

    if not isinstance(content, list):
        return parts, tool_results

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            txt = block.get("text") or ""
            if txt:
                parts.append({"type": "input_text", "text": txt})
        elif btype == "image":
            # Anthropic image block: {source: {type, media_type, data}} (base64)
            #                       or {source: {type:"url", url}} (newer API)
            src = block.get("source") or {}
            if src.get("type") == "base64":
                data = src.get("data") or ""
                media = src.get("media_type") or "image/png"
                if data:
                    parts.append({
                        "type": "input_image",
                        "image_url": f"data:{media};base64,{data}",
                        "detail": "auto",
                    })
            elif src.get("type") == "url":
                url = src.get("url") or ""
                if url:
                    parts.append({"type": "input_image", "image_url": url, "detail": "auto"})
        elif btype == "tool_result":
            call_id = block.get("tool_use_id") or ""
            # tool_result.content can be a string OR list of content blocks
            # (text/image). Responses API expects a string output, so we
            # flatten text content and drop images (rare in practice).
            raw = block.get("content")
            if isinstance(raw, str):
                output = raw
            elif isinstance(raw, list):
                chunks: list[str] = []
                for sub in raw:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        chunks.append(sub.get("text") or "")
                output = "\n".join(chunks)
            else:
                output = ""
            tool_results.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            })

    return parts, tool_results


def _assistant_content_to_responses(content: Any) -> list[dict]:
    """Assistant content → text item(s) + function_call items.

    Anthropic packs text and tool_use blocks together in one assistant
    message; Responses API wants them as separate input items.
    """
    items: list[dict] = []

    if isinstance(content, str):
        if content:
            items.append({"role": "assistant", "content": content})
        return items

    if not isinstance(content, list):
        return items

    text_buf: list[str] = []

    def flush_text() -> None:
        if text_buf:
            joined = "".join(text_buf)
            if joined:
                items.append({"role": "assistant", "content": joined})
            text_buf.clear()

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_buf.append(block.get("text") or "")
        elif btype == "tool_use":
            flush_text()
            tool_input = block.get("input") or {}
            items.append({
                "type": "function_call",
                "name": block.get("name") or "",
                "arguments": json.dumps(tool_input) if not isinstance(tool_input, str) else tool_input,
                "call_id": block.get("id") or "",
            })
    flush_text()
    return items


def _tools_to_responses(tools: list[dict] | None) -> list[dict] | None:
    """Anthropic tools schema → Responses API tools schema.

    Anthropic:  {name, description, input_schema}
    Responses:  {type:"function", name, description, parameters}
    """
    if not tools:
        return None
    converted: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        item: dict[str, Any] = {
            "type": "function",
            "name": tool.get("name") or "",
        }
        if "description" in tool:
            item["description"] = tool["description"]
        # ``input_schema`` is the Anthropic name; some callers already use
        # ``parameters`` so honor both.
        params = tool.get("input_schema") or tool.get("parameters")
        if params is not None:
            item["parameters"] = params
        converted.append(item)
    return converted or None


def anthropic_to_responses(body: dict, upstream_model: str) -> dict:
    """Translate an Anthropic Messages request body to a Responses API body.

    ``upstream_model`` is the post-prefix model name (e.g. ``gpt-5.4``),
    already split off from the Anthropic ``model`` field by the route.
    """
    instructions = _flatten_system(body.get("system"))
    input_items: list[dict] = []

    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or "user"
        content = msg.get("content")

        if role == "user":
            parts, tool_results = _user_content_to_responses(content)
            if parts:
                # All parts are input_text/input_image; user content items
                # take the list directly.
                input_items.append({"role": "user", "content": parts})
            input_items.extend(tool_results)
        elif role == "assistant":
            input_items.extend(_assistant_content_to_responses(content))
        else:
            # Anthropic Messages API technically only has user/assistant
            # message roles; anything else is the caller's mistake. Drop it
            # but log so we notice if a client starts sending something new.
            logger.debug("Dropping non-user/assistant message role=%r", role)

    payload: dict[str, Any] = {
        "model": upstream_model,
        "input": input_items,
        "store": False,
    }
    if instructions:
        payload["instructions"] = instructions

    if "max_tokens" in body:
        try:
            payload["max_output_tokens"] = int(body["max_tokens"])
        except (TypeError, ValueError):
            pass

    if "temperature" in body:
        try:
            payload["temperature"] = float(body["temperature"])
        except (TypeError, ValueError):
            pass

    if body.get("stream"):
        payload["stream"] = True

    tools = _tools_to_responses(body.get("tools"))
    if tools:
        payload["tools"] = tools
        tc = body.get("tool_choice")
        if tc:
            payload["tool_choice"] = tc

    return payload
