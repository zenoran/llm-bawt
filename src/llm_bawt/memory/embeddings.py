"""Memory embeddings — thin HTTP client to the local-model-bridge /embed API.

The MiniLM (``all-MiniLM-L6-v2``) model used to load in-process here; as of
TASK-277 it lives in the local-model-bridge and is reached over HTTP, so the
app process holds no local model. Public signatures are unchanged, so all
callers (memory/postgresql.py, maintenance.py, migrations.py,
service/animation_classifier.py, service/api.py warmup, mcp_server/storage.py)
are untouched.

This sits on the synchronous memory hot path (every store/recall), so:
- a single persistent httpx client is reused (no per-call connection setup),
- connect timeout is short and retries are bounded,
- persistent failure returns ``None`` — callers already guard ``if embedding:``
  and degrade to non-vector search rather than crashing,
- ``get_embedding_dimension`` returns a static constant and never does I/O, so
  schema/DDL creation never blocks on bridge availability.
"""

import logging
import os
import threading
import time

import httpx

logger = logging.getLogger(__name__)

# MiniLM dimension. Returned by get_embedding_dimension() with no network call
# so DDL/schema sizing never depends on the bridge being up. If the embedding
# model ever changes to a different dimension, update this AND the DB schema.
EMBED_DIM = 384

# Base URL of the bridge embed API. Default matches the compose service name.
LOCAL_MODEL_EMBED_URL = os.getenv(
    "LOCAL_MODEL_EMBED_URL", "http://local-model-bridge:8684"
).rstrip("/")

# Bounded retry for transient failures on the hot path.
_MAX_ATTEMPTS = 2
# Throttle outage logging to once per window so a bridge outage doesn't spam a
# line on every single memory operation.
_OUTAGE_LOG_WINDOW_S = 60.0

_client: httpx.Client | None = None
_client_lock = threading.Lock()
_last_outage_log = 0.0


def _get_client() -> httpx.Client:
    """Lazily build and cache a persistent httpx client (thread-safe)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(
                    base_url=LOCAL_MODEL_EMBED_URL,
                    timeout=httpx.Timeout(connect=2.0, read=30.0, write=10.0, pool=2.0),
                )
    return _client


def _log_outage(msg: str) -> None:
    """Log an embed outage at most once per window."""
    global _last_outage_log
    now = time.monotonic()
    if now - _last_outage_log >= _OUTAGE_LOG_WINDOW_S:
        _last_outage_log = now
        logger.warning(msg)


def _embed(texts: list[str], model_name: str) -> list[list[float]] | None:
    """POST texts to the bridge /embed endpoint with bounded retry.

    Returns a list of vectors (one per input text, same order) or ``None`` on
    persistent failure. ``texts`` must be non-empty.
    """
    payload = {"texts": texts, "model": model_name}
    last_err: Exception | str | None = None

    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = _get_client().post("/embed", json=payload)
        except Exception as e:  # connect/read/pool timeouts, conn refused, etc.
            last_err = e
            continue

        if resp.status_code == 200:
            data = resp.json()
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                last_err = (
                    f"malformed /embed response: got "
                    f"{len(embeddings) if isinstance(embeddings, list) else type(embeddings)} "
                    f"vectors for {len(texts)} texts"
                )
                break  # a well-formed 200 that's wrong won't fix itself on retry
            return embeddings

        # 4xx (except 408/429) is a client error that won't change on retry.
        if 400 <= resp.status_code < 500 and resp.status_code not in (408, 429):
            last_err = f"/embed HTTP {resp.status_code}: {resp.text[:200]}"
            break
        last_err = f"/embed HTTP {resp.status_code}"

    _log_outage(
        f"Embedding service unavailable at {LOCAL_MODEL_EMBED_URL} "
        f"({last_err}); memory ops will degrade to non-vector search until it recovers"
    )
    return None


def generate_embedding(
    text: str, model_name: str = "all-MiniLM-L6-v2", verbose: bool = False
) -> list[float] | None:
    """Generate an embedding vector for the given text.

    Args:
        text: Text to embed
        model_name: Sentence-transformers model name (served by the bridge)
        verbose: Accepted for signature compatibility; unused

    Returns:
        List of floats representing the embedding, or None on failure
    """
    if not text or text.isspace():
        return None

    result = _embed([text], model_name)
    if not result:
        return None
    return result[0]


def generate_embeddings_batch(
    texts: list[str], model_name: str = "all-MiniLM-L6-v2", verbose: bool = False
) -> list[list[float] | None]:
    """Generate embeddings for multiple texts efficiently.

    Args:
        texts: List of texts to embed
        model_name: Sentence-transformers model name (served by the bridge)
        verbose: Accepted for signature compatibility; unused

    Returns:
        List of embeddings (or None for failed/empty items)
    """
    if not texts:
        return []

    # Filter out empty texts, keeping track of original indices so the result
    # lines up 1:1 with the input (None for skipped/empty entries).
    valid_indices: list[int] = []
    valid_texts: list[str] = []
    for i, text in enumerate(texts):
        if text and not text.isspace():
            valid_indices.append(i)
            valid_texts.append(text)

    if not valid_texts:
        return [None] * len(texts)

    embeddings = _embed(valid_texts, model_name)
    if embeddings is None:
        return [None] * len(texts)

    result: list[list[float] | None] = [None] * len(texts)
    for i, emb in zip(valid_indices, embeddings):
        result[i] = emb
    return result


def get_embedding_dimension(
    model_name: str = "all-MiniLM-L6-v2", verbose: bool = False
) -> int:
    """Return the embedding dimension.

    Static constant by design: schema/DDL paths call this and must never block
    on the bridge being reachable.

    Args:
        model_name: Accepted for signature compatibility; unused
        verbose: Accepted for signature compatibility; unused

    Returns:
        Embedding dimension (384 for MiniLM)
    """
    return EMBED_DIM
