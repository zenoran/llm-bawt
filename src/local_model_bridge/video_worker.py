"""One-shot Wan 2.2 video inference subprocess for the local GPU bridge.

Each job has a fresh process so CUDA allocations and model weights are released
without affecting the bridge's CPU embedding server or Redis chat listener.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
FPS = 24
# Stay conservative on a 16 GB card; the model is offloaded to system RAM.
DIMENSIONS = {
    "480p": {"16:9": (832, 480), "9:16": (480, 832), "1:1": (512, 512)},
    "720p": {"16:9": (1280, 704), "9:16": (704, 1280), "1:1": (704, 704)},
}


def dimensions(resolution: str, aspect_ratio: str) -> tuple[int, int]:
    try:
        return DIMENSIONS[resolution][aspect_ratio]
    except KeyError as exc:
        raise ValueError(f"Unsupported local video dimensions: {resolution} {aspect_ratio}") from exc


def load_source_image(source_image: str):
    """Only accept embedded image bytes, never fetch user-supplied URLs from the GPU worker."""
    from PIL import Image

    if not source_image.startswith("data:image/") or ";base64," not in source_image:
        raise ValueError("Local video requires an embedded image (data:image/...;base64,...)")
    encoded = source_image.split(";base64,", 1)[1]
    if len(encoded) > 30_000_000:
        raise ValueError("Source image is too large")
    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
        image.load()
        return image.convert("RGB")


def run_job(job: dict, destination: Path) -> dict:
    # Explicit install is the only network path. Do not let a missing tokenizer
    # or model component trigger a surprise download during generation.
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline
    from diffusers.utils import export_to_video

    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available in local-model-bridge; check NVIDIA container runtime")
    width, height = dimensions(job["resolution"], job["aspect_ratio"])
    duration = float(job["duration"])
    if not 1 <= duration <= 15:
        raise ValueError("Duration must be between 1 and 15 seconds")
    # Wan temporal latents require 4n+1 frames. Trim the exported result to
    # the requested duration; rounding up would silently lengthen a 5s clip.
    requested_frames = max(1, round(duration * FPS))
    num_frames = 4 * ((requested_frames - 1 + 3) // 4) + 1
    source_image = job.get("source_image")
    # Installation is an explicit UI action. Generation must never silently
    # download 34 GB of weights or scatter copies across model directories.
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32, local_files_only=True,
    )
    # This TI2V repo declares WanPipeline in model_index.json; when given an
    # image, explicitly load the I2V pipeline to condition on its first frame.
    pipeline_class = WanImageToVideoPipeline if source_image else WanPipeline
    pipeline = pipeline_class.from_pretrained(
        MODEL_ID, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True,
    )
    if source_image and not pipeline.config.expand_timesteps:
        raise RuntimeError("Wan TI2V image conditioning requires expand_timesteps; check the checkpoint")
    pipeline.enable_sequential_cpu_offload()
    pipeline.vae.enable_tiling()
    kwargs = {
        "prompt": job["prompt"], "height": height, "width": width,
        "num_frames": num_frames, "num_inference_steps": 35, "guidance_scale": 5.0,
    }
    if source_image:
        kwargs["image"] = load_source_image(source_image)
    output = pipeline(**kwargs).frames[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(output[:requested_frames], str(destination), fps=FPS)
    return {"width": width, "height": height, "actual_duration": len(output[:requested_frames]) / FPS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one local Wan video generation")
    parser.add_argument("job", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    info = run_job(job, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(info), encoding="utf-8")


if __name__ == "__main__":
    main()
