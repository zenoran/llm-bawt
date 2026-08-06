from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest

from llm_bawt.media.clients.base import GenerationResult, MediaClient
from llm_bawt.media.clients.openai_images import OpenAIImageClient
from llm_bawt.media.clients.registry import MediaProviderCapabilities, MediaProviderRegistry
from llm_bawt.media.generation_service import MediaGenerationService
from llm_bawt.service.routes.media import _row_to_response


class FakeImageClient(MediaClient):
    def __init__(self) -> None:
        self.closed = False
        self.calls: list[dict] = []

    async def generate(self, prompt, media_type, model, **kwargs):
        self.calls.append({"prompt": prompt, "media_type": media_type, "model": model, **kwargs})
        return GenerationResult(
            provider_job_id="job-1",
            status="completed",
            progress=100,
            media_url="data:image/png;base64,aW1hZ2U=",
        )

    async def poll_status(self, provider_job_id: str) -> GenerationResult:
        raise AssertionError("synchronous image provider should not be polled")

    async def download(self, media_url: str) -> bytes:
        return b"image"

    async def close(self) -> None:
        self.closed = True


def _fake_registry(client: FakeImageClient) -> MediaProviderRegistry:
    registry = MediaProviderRegistry()
    registry.register(
        MediaProviderCapabilities(
            provider="fake",
            label="Fake",
            media_types=("image",),
            default_models={"image": "fake-image"},
            models={"image": ("fake-image",)},
            image_input=True,
            aspect_ratios={"image": ("1:1",)},
            resolutions={"image": ("1k",)},
            default_aspect_ratios={"image": "1:1"},
            default_resolutions={"image": "1k"},
        ),
        lambda: client,
        aliases=("legacy-fake",),
    )
    return registry


def test_legacy_xai_rows_are_exposed_as_grok() -> None:
    response = _row_to_response({
        "id": "gen-old",
        "status": "failed",
        "media_type": "image",
        "prompt": "old",
        "provider": "xai",
        "model": "grok-imagine-image",
        "created_at": "2026-08-06T00:00:00Z",
    })

    assert response.provider == "grok"


def test_registry_alias_and_provider_defaults() -> None:
    client = FakeImageClient()
    service = MediaGenerationService(_fake_registry(client))

    assert service.resolve_request(
        provider="legacy-fake",
        media_type="image",
        model=None,
        aspect_ratio=None,
        resolution=None,
    ) == ("fake", "fake-image", "1:1", "1k")


def test_registry_rejects_unsupported_capabilities() -> None:
    service = MediaGenerationService(_fake_registry(FakeImageClient()))

    with pytest.raises(ValueError, match="aspect ratio"):
        service.resolve_request(
            provider="fake",
            media_type="image",
            model=None,
            aspect_ratio="16:9",
            resolution="1k",
        )


def test_generation_service_dispatches_and_closes_client() -> None:
    client = FakeImageClient()
    service = MediaGenerationService(_fake_registry(client))

    generated = asyncio.run(service.generate_image(prompt="draw a dog", provider="fake"))

    assert generated.data == b"image"
    assert generated.provider == "fake"
    assert generated.model == "fake-image"
    assert client.calls[0]["aspect_ratio"] == "1:1"
    assert client.closed is True


def test_openai_text_to_image_request_and_base64_response() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"data": [{"b64_json": base64.b64encode(b"png-bytes").decode()}]},
        )

    async def exercise() -> None:
        client = OpenAIImageClient(api_key="test", base_url="https://openai.test/v1")
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            base_url="https://openai.test/v1",
            transport=httpx.MockTransport(handler),
            headers={"Authorization": "Bearer test"},
        )
        try:
            result = await client.generate(
                "draw a dog",
                "image",
                "gpt-image-1",
                aspect_ratio="3:2",
                resolution="1k",
            )
            assert await client.download(result.media_url or "") == b"png-bytes"
        finally:
            await client.close()

    asyncio.run(exercise())
    payload = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/images/generations"
    assert payload == {
        "model": "gpt-image-1",
        "prompt": "draw a dog",
        "n": 1,
        "size": "1536x1024",
    }


def test_openai_reference_image_uses_edits_multipart() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"data": [{"b64_json": base64.b64encode(b"edited").decode()}]},
        )

    async def exercise() -> None:
        source = "data:image/png;base64," + base64.b64encode(b"source-png").decode()
        client = OpenAIImageClient(api_key="test", base_url="https://openai.test/v1")
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            base_url="https://openai.test/v1",
            transport=httpx.MockTransport(handler),
            headers={"Authorization": "Bearer test"},
        )
        try:
            await client.generate(
                "make it warmer",
                "image",
                "gpt-image-1",
                source_image=source,
                aspect_ratio="1:1",
                resolution="1k",
            )
        finally:
            await client.close()

    asyncio.run(exercise())
    request = requests[0]
    assert request.url.path == "/v1/images/edits"
    assert "multipart/form-data" in request.headers["content-type"]
    assert b"source-png" in request.content
    assert b"make it warmer" in request.content
