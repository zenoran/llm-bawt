"""Focused tests for extracted streaming attachment preparation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.stream_request_attachments import (
    prepare_stream_request_attachments,
)


def _run(request, log):
    return asyncio.run(
        prepare_stream_request_attachments(request, user_id="nick", log=log)
    )


def test_plain_text_survives_unavailable_media_store(monkeypatch):
    monkeypatch.setattr(
        "llm_bawt.service.stream_request_attachments.get_media_store",
        Mock(side_effect=RuntimeError("offline")),
    )
    log = SimpleNamespace(error=Mock(), warning=Mock())
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hello")]
    )

    prompt, llm_images, durable_refs, store = _run(request, log)

    assert prompt == "hello"
    assert llm_images == []
    assert durable_refs == []
    assert store is None
    log.error.assert_called_once()


def test_legacy_inline_image_is_forwarded_and_uploaded(monkeypatch):
    asset = SimpleNamespace(id="ma_uploaded")
    store = SimpleNamespace(upload=Mock(return_value=asset))
    monkeypatch.setattr(
        "llm_bawt.service.stream_request_attachments.get_media_store",
        Mock(return_value=store),
    )
    log = SimpleNamespace(error=Mock(), warning=Mock())
    request = ChatCompletionRequest(
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "inspect"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                    },
                ],
            )
        ]
    )

    prompt, llm_images, durable_refs, returned_store = _run(request, log)

    assert prompt == "inspect"
    assert llm_images == [{"mimeType": "image/png", "content": "aGVsbG8="}]
    assert durable_refs == [{"asset_id": "ma_uploaded", "kind": "image"}]
    assert returned_store is store
    assert store.upload.call_args.kwargs["raw_bytes"] == b"hello"


def test_asset_id_uses_preview_and_keeps_durable_ref(monkeypatch):
    store = SimpleNamespace(
        read_preview_as_data_url=Mock(
            return_value="data:image/webp;base64,cHJldmlldw=="
        )
    )
    monkeypatch.setattr(
        "llm_bawt.service.stream_request_attachments.get_media_store",
        Mock(return_value=store),
    )
    log = SimpleNamespace(error=Mock(), warning=Mock())
    request = ChatCompletionRequest(
        messages=[
            ChatMessage(
                role="user",
                content="look",
                attachment_ids=["ma_existing"],
            )
        ]
    )

    prompt, llm_images, durable_refs, returned_store = _run(request, log)

    assert prompt == "look"
    assert llm_images == [{"mimeType": "image/webp", "content": "cHJldmlldw=="}]
    assert durable_refs == [{"asset_id": "ma_existing", "kind": "image"}]
    assert returned_store is store
