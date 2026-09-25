from __future__ import annotations

import asyncio
import base64

import httpx
import pytest
from fastapi import HTTPException

from llm_bawt.media.clients.local_video import LocalVideoClient
from llm_bawt.media.clients.registry import media_provider_registry
from llm_bawt.media.generation_service import MediaGenerationService
from local_model_bridge.video_models import VideoModelManager
from local_model_bridge.video_server import VideoJobs, VideoRequest
from local_model_bridge.video_worker import dimensions, load_source_image


def test_provider_is_visible_without_changing_default() -> None:
    capability = media_provider_registry.capabilities("local-video")
    assert capability.media_types == ("video",)
    assert capability.default_models["video"] == "wan2.2-ti2v-5b"
    assert MediaGenerationService().resolve_request(
        provider="local-video", media_type="video", model=None,
        aspect_ratio=None, resolution=None,
    ) == ("local-video", "wan2.2-ti2v-5b", "16:9", "480p")
    assert media_provider_registry.canonical_provider(None) == "grok"


def test_video_dimensions_and_image_input_are_bounded() -> None:
    assert dimensions("720p", "16:9") == (1280, 704)
    with pytest.raises(ValueError, match="Unsupported"):
        dimensions("4k", "16:9")
    with pytest.raises(ValueError, match="embedded image"):
        load_source_image("https://example.com/private")
    with pytest.raises(Exception):
        load_source_image("data:image/png;base64," + base64.b64encode(b"not an image").decode())


def test_video_jobs_reject_invalid_size_without_starting_process(tmp_path) -> None:
    jobs = VideoJobs(tmp_path)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(jobs.submit(VideoRequest(prompt="bird", resolution="4k")))
    assert exc.value.status_code == 400
    assert not list(tmp_path.iterdir())


def test_model_inventory_and_scoped_cleanup(tmp_path) -> None:
    manager = VideoModelManager(tmp_path)
    other = tmp_path / "models--other--keep"
    other.mkdir()
    (other / "weights.bin").write_bytes(b"keep")
    blobs = manager.repo_dir / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "partial.incomplete").write_bytes(b"partial")
    status = manager.status()
    assert status["size_bytes"] == 7
    assert not status["installed"]
    assert status["cache_path"] == str(manager.repo_dir)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(manager.remove(active=True))
    assert exc.value.status_code == 409
    assert manager.repo_dir.exists()
    asyncio.run(manager.remove())
    assert not manager.repo_dir.exists()
    assert (other / "weights.bin").read_bytes() == b"keep"


def test_model_cleanup_rejects_symlink(tmp_path) -> None:
    manager = VideoModelManager(tmp_path)
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    manager.repo_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(manager.remove())
    assert exc.value.status_code == 400
    assert outside.exists()
    manager.repo_dir.unlink()


def test_local_video_client_submit_poll_and_download() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"id": "job1", "status": "processing", "progress": 10})
        if request.method == "DELETE":
            return httpx.Response(200, json={"deleted": True})
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"mp4")
        return httpx.Response(200, json={"id": "job1", "status": "completed", "progress": 100, "width": 832, "height": 480})

    async def exercise() -> None:
        client = LocalVideoClient(base_url="http://local-model-bridge:8685")
        await client.close()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=client.base_url)
        try:
            generated = await client.generate("running dog", "video", "wan2.2-ti2v-5b")
            assert generated.provider_job_id == "job1"
            status = await client.poll_status("job1")
            assert status.width == 832
            assert await client.download(status.media_url or "") == b"mp4"
            await client.remove_job("job1")
            with pytest.raises(ValueError, match="Unexpected"):
                await client.download("https://example.com/steal")
        finally:
            await client.close()

    asyncio.run(exercise())
    assert [r.url.path for r in requests] == ["/videos", "/videos/job1", "/videos/job1/content", "/videos/job1"]
