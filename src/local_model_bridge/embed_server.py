"""HTTP embedding server for the local-model bridge (TASK-277).

Hosts the MiniLM (``all-MiniLM-L6-v2``) sentence-transformers model that used
to load in-process inside the main app, exposing it as a tiny FastAPI service
on its own port (default 8684). The app's ``memory/embeddings.py`` is now a
thin HTTP client that POSTs here, so the GPU/embedding model lives only in the
bridge — the app process holds no local model at all.

Endpoints:
    POST /embed   {"texts": [str, ...], "model"?: str}
                  -> {"embeddings": [[float, ...], ...], "model": str, "dim": int}
    GET  /health  -> {"ok": bool, "model": str, "dim": int}

Environment:
    LOCAL_MODEL_EMBED_MODEL   — sentence-transformers model (default all-MiniLM-L6-v2)
    LOCAL_MODEL_EMBED_DEVICE  — torch device for the model (default "cpu"; MiniLM
                                is tiny, and CPU keeps embeddings independent of
                                GPU/NVML state and off the contended card)
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("local_model_bridge.embed")

EMBED_MODEL_NAME = os.getenv("LOCAL_MODEL_EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DEVICE = os.getenv("LOCAL_MODEL_EMBED_DEVICE", "cpu")
# MiniLM is 384-d. Exposed in /health and used by the app's static dim fallback.
EMBED_DIM = 384

# Module-global cached model (loaded once at startup via load_model()).
_model: Any = None


def load_model() -> Any:
    """Load and cache the sentence-transformers model.

    Prefers the on-disk HF cache (``local_files_only=True``) so a warm container
    never round-trips to HuggingFace on startup; falls back to a network load
    only when the cache is cold. Returns the model, or ``None`` if the library
    or weights are unavailable (caller decides how to degrade).
    """
    global _model
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error(
            "sentence-transformers not installed in the bridge image; "
            "/embed cannot serve embeddings"
        )
        return None

    # Suppress the per-file HEAD-request spam HF emits during the ETag check.
    noisy = [
        "sentence_transformers",
        "transformers",
        "transformers.modeling_utils",
        "torch.distributed",
        "httpx",
        "httpcore",
        "huggingface_hub",
    ]
    old_levels = {n: logging.getLogger(n).level for n in noisy}
    for n in noisy:
        logging.getLogger(n).setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*layers were not sharded.*")
            warnings.filterwarnings("ignore", category=FutureWarning)
            try:
                model = SentenceTransformer(
                    EMBED_MODEL_NAME, device=EMBED_DEVICE, local_files_only=True
                )
                logger.info(
                    "Loaded embedding model %s from local cache (device=%s)",
                    EMBED_MODEL_NAME,
                    EMBED_DEVICE,
                )
            except Exception as cache_miss:
                logger.info(
                    "Embedding model %s not cached (%s); downloading from HuggingFace…",
                    EMBED_MODEL_NAME,
                    cache_miss,
                )
                model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)
    except Exception as e:
        logger.error("Failed to load embedding model %s: %s", EMBED_MODEL_NAME, e)
        return None
    finally:
        for n, lvl in old_levels.items():
            logging.getLogger(n).setLevel(lvl)

    try:
        dim = int(model.get_sentence_embedding_dimension())
        if dim != EMBED_DIM:
            logger.warning(
                "Embedding model %s reports dim=%d but EMBED_DIM=%d; app's static "
                "dimension fallback will be wrong — update EMBED_DIM/DDL",
                EMBED_MODEL_NAME,
                dim,
                EMBED_DIM,
            )
    except Exception:
        pass

    _model = model
    return _model


class EmbedRequest(BaseModel):
    texts: list[str]
    model: str | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="local-model-bridge embed", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> dict:
        return {"ok": _model is not None, "model": EMBED_MODEL_NAME, "dim": EMBED_DIM}

    @app.post("/embed")
    def embed(req: EmbedRequest) -> dict:
        if req.model and req.model != EMBED_MODEL_NAME:
            # All current callers use MiniLM; serve the configured model and
            # note the mismatch once rather than failing the hot path.
            logger.warning(
                "embed request asked for model=%s but server serves %s",
                req.model,
                EMBED_MODEL_NAME,
            )
        if not req.texts:
            return {"embeddings": [], "model": EMBED_MODEL_NAME, "dim": EMBED_DIM}

        model = load_model()
        if model is None:
            # 503 so the thin client treats it as a transient outage and falls
            # back to None (non-vector search) rather than caching a bad result.
            from fastapi import HTTPException

            raise HTTPException(status_code=503, detail="embedding model unavailable")

        vectors = model.encode(
            req.texts, convert_to_numpy=True, show_progress_bar=False
        ).tolist()
        return {"embeddings": vectors, "model": EMBED_MODEL_NAME, "dim": EMBED_DIM}

    return app


async def serve_embed(port: int) -> None:
    """Pre-warm the model, then serve the embed API on ``port`` until cancelled.

    Run as an asyncio task inside the bridge's existing event loop. The model
    load happens in a thread executor so the loop (and the Redis bridge/health
    tasks) are not blocked during the (cold) load.
    """
    import asyncio

    import uvicorn

    loop = asyncio.get_running_loop()
    logger.info("Pre-warming embedding model %s (device=%s)…", EMBED_MODEL_NAME, EMBED_DEVICE)
    await loop.run_in_executor(None, load_model)

    config = uvicorn.Config(
        create_app(),
        host="0.0.0.0",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    logger.info("Embed API listening on :%d (model=%s)", port, EMBED_MODEL_NAME)
    await server.serve()
