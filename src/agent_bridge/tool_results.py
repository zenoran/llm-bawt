"""Lossless tool-result serialization shared by agent bridges and the app.

The provider-facing bridges may transiently carry the complete accepted result on
short-lived per-run streams. Public unified events and history responses use the
bounded preview plus metadata produced here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

TOOL_RESULT_PREVIEW_CHARS = 2_000
TOOL_RESULT_PAYLOAD_VERSION = 1
STRUCTURED_TOOL_RESULT_TYPE = "application/x-tool-blocks+json"


def _without_inline_images(value: Any) -> Any:
    """Keep structured result shape without duplicating base64 image bytes.

    Screenshot bytes already have their own media-asset lifecycle. Retaining a
    small descriptor here keeps the result truthful and inspectable without
    storing the same multi-megabyte image twice.
    """
    if isinstance(value, list):
        return [_without_inline_images(item) for item in value]
    if isinstance(value, dict):
        block_type = str(value.get("type") or "").lower()
        source = value.get("source")
        if block_type in {"image", "image_url"} or (
            isinstance(source, dict) and source.get("type") == "base64"
        ):
            media_type = source.get("media_type") if isinstance(source, dict) else None
            return {"type": block_type or "image", "media_type": media_type, "omitted": True}
        return {str(key): _without_inline_images(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ToolResultPayload:
    version: int
    content: str
    content_type: str
    total_chars: int
    total_bytes: int
    sha256: str
    complete: bool

    @classmethod
    def from_value(cls, value: Any, *, complete: bool = True) -> "ToolResultPayload":
        if isinstance(value, str):
            content = value
            content_type = "text/plain"
        elif value is None:
            content = ""
            content_type = "text/plain"
        else:
            normalized = _without_inline_images(value)
            content = json.dumps(
                normalized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            content_type = STRUCTURED_TOOL_RESULT_TYPE
        raw = content.encode("utf-8")
        return cls(
            version=TOOL_RESULT_PAYLOAD_VERSION,
            content=content,
            content_type=content_type,
            total_chars=len(content),
            total_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            complete=bool(complete),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResultPayload":
        if int(data.get("version") or 0) != TOOL_RESULT_PAYLOAD_VERSION:
            raise ValueError("unsupported tool-result payload version")
        content = data.get("content")
        if not isinstance(content, str):
            raise ValueError("tool-result payload content must be text")
        payload = cls.from_value(content, complete=bool(data.get("complete", False)))
        content_type = str(data.get("content_type") or "text/plain")
        return cls(
            version=TOOL_RESULT_PAYLOAD_VERSION,
            content=content,
            content_type=content_type,
            total_chars=len(content),
            total_bytes=len(content.encode("utf-8")),
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            complete=bool(data.get("complete", False)),
        )

    @property
    def preview(self) -> str:
        return self.content[:TOOL_RESULT_PREVIEW_CHARS]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def result_meta(
        self,
        *,
        record_id: int | None = None,
        available: bool,
    ) -> dict[str, Any]:
        return {
            "record_id": record_id,
            "preview_chars": len(self.preview),
            "total_chars": self.total_chars,
            "total_bytes": self.total_bytes,
            "sha256": self.sha256,
            "content_type": self.content_type,
            "complete": self.complete,
            "available": bool(available),
            "legacy": False,
        }


def payload_from_event(payload: Any, result: Any) -> ToolResultPayload:
    """Prefer a versioned producer payload; tolerate rolling-deploy producers."""
    if isinstance(payload, dict):
        try:
            return ToolResultPayload.from_dict(payload)
        except (TypeError, ValueError):
            pass
    return ToolResultPayload.from_value(result, complete=True)
