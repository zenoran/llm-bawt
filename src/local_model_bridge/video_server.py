"""Private job API for local video inference, hosted in the existing GPU bridge.

The API is Docker-network-only. Generation runs in a separate process so CUDA
resources are released after each job and embeddings stay responsive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .video_models import VideoModelManager
from .video_worker import DIMENSIONS

logger = logging.getLogger(__name__)


class VideoRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    source_image: str | None = None
    aspect_ratio: str = "16:9"
    duration: float = Field(default=5, ge=1, le=15)
    resolution: str = "480p"


class VideoJobs:
    def __init__(self, directory: Path):
        self.directory = directory
        self.jobs: dict[str, dict] = {}
        self.lock = asyncio.Lock()
        self.process: asyncio.subprocess.Process | None = None

    async def submit(self, request: VideoRequest) -> dict:
        if request.aspect_ratio not in DIMENSIONS.get(request.resolution, {}):
            raise HTTPException(status_code=400, detail="Unsupported video size or aspect ratio")
        if request.source_image and (not request.source_image.startswith("data:image/") or len(request.source_image) > 30_000_000):
            raise HTTPException(status_code=400, detail="Source image must be an embedded image under 22MB")
        if not models.status()["worker_ready"]:
            raise HTTPException(status_code=503, detail="GPU video worker is not installed yet")
        if not models.installed():
            raise HTTPException(status_code=409, detail="Install Wan 2.2 weights in Studio before generating")
        async with self.lock:
            if self.process is not None:
                raise HTTPException(status_code=409, detail="Local video GPU is busy")
            job_id = uuid.uuid4().hex
            self.directory.mkdir(parents=True, exist_ok=True)
            input_path = self.directory / f"{job_id}.input.json"
            output_path = self.directory / f"{job_id}.mp4"
            input_path.write_text(request.model_dump_json(), encoding="utf-8")
            try:
                self.process = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "local_model_bridge.video_worker", str(input_path), str(output_path),
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
            except Exception:
                input_path.unlink(missing_ok=True)
                raise
            self.jobs[job_id] = {"status": "processing", "progress": 10}
            asyncio.create_task(self._finish(job_id, input_path, output_path, self.process))
            return {"id": job_id, **self.jobs[job_id]}

    async def _finish(self, job_id: str, input_path: Path, output_path: Path, process: asyncio.subprocess.Process) -> None:
        try:
            _, stderr = await process.communicate()
            if process.returncode != 0 or not output_path.is_file():
                self.jobs[job_id] = {"status": "failed", "progress": 0, "error": stderr.decode(errors="replace")[-1600:] or "Video worker produced no output"}
                output_path.unlink(missing_ok=True)
                output_path.with_suffix(".json").unlink(missing_ok=True)
                return
            metadata = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
            self.jobs[job_id] = {"status": "completed", "progress": 100, **metadata}
        except Exception as exc:
            logger.exception("Video job %s failed", job_id)
            self.jobs[job_id] = {"status": "failed", "progress": 0, "error": str(exc)}
            output_path.unlink(missing_ok=True)
            output_path.with_suffix(".json").unlink(missing_ok=True)
        finally:
            input_path.unlink(missing_ok=True)
            if self.process is process:
                self.process = None

    def status(self, job_id: str) -> dict:
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail="Unknown local video job")
        return {"id": job_id, **self.jobs[job_id]}

    def file(self, job_id: str) -> Path:
        if self.status(job_id)["status"] != "completed":
            raise HTTPException(status_code=404, detail="Video not ready")
        return self.directory / f"{job_id}.mp4"


jobs = VideoJobs(Path(tempfile.gettempdir()) / "llm-bawt-video-jobs")
models = VideoModelManager()
app = FastAPI(title="Local GPU video API", docs_url=None, redoc_url=None)


@app.get("/models/wan2.2-ti2v-5b")
def model_status() -> dict:
    return models.status(active=jobs.process is not None)


@app.post("/models/wan2.2-ti2v-5b/install")
async def install_model() -> dict:
    return await models.install()


@app.delete("/models/wan2.2-ti2v-5b")
async def remove_model() -> dict:
    return await models.remove(active=jobs.process is not None)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "worker": "wan2.2-ti2v-5b"}


@app.post("/videos")
async def submit(request: VideoRequest) -> dict:
    return await jobs.submit(request)


@app.get("/videos/{job_id}")
def status(job_id: str) -> dict:
    return jobs.status(job_id)


@app.get("/videos/{job_id}/content")
def content(job_id: str) -> FileResponse:
    return FileResponse(jobs.file(job_id), media_type="video/mp4")


@app.delete("/videos/{job_id}")
def remove(job_id: str) -> dict:
    if jobs.status(job_id)["status"] == "processing":
        raise HTTPException(status_code=409, detail="Generation in progress")
    jobs.jobs.pop(job_id, None)
    jobs.directory.joinpath(f"{job_id}.mp4").unlink(missing_ok=True)
    jobs.directory.joinpath(f"{job_id}.json").unlink(missing_ok=True)
    return {"deleted": True}


async def serve_video(port: int) -> None:
    import uvicorn

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning", access_log=False)
    await uvicorn.Server(config).serve()
