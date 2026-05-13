"""Post-hoc embedding-based animation classifier (TASK-215).

Replaces the old prompt-injection / tool-call mechanism. The main LLM call
runs unburdened; once the response text is final, we score it against the
animation catalog (sent by bawthub on the request) and pick the highest
cosine-similarity match above a threshold.

Why local embeddings instead of another LLM call:
  - `all-MiniLM-L6-v2` is already loaded (memory subsystem).
  - ~5–20 ms on CPU vs. ~200–400 ms for a network call.
  - Deterministic, debuggable similarity scores.

The catalog rarely changes during a session; embeddings are cached by a
SHA-256 hash of the (name, description) pairs so re-embedding only happens
when the catalog actually changes.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol

from ..memory.embeddings import generate_embedding, generate_embeddings_batch

logger = logging.getLogger(__name__)


# Anything with `.name` and `.description` works — duck-typed. In practice
# this is `ChatRequestAnimation` from `schemas.py`.
class _AnimationLike(Protocol):
    name: str
    description: str | None


# Empirically tuned against the 15 default animation descriptions: clear
# matches ("frustrated" → angry, "amazing news" → happy, "hmm let me think"
# → thoughtful) land in the 0.20–0.37 band, while plausible-but-wrong picks
# tend to be < 0.15. We err toward returning nothing rather than picking a
# random animation — the frontend then stays on its idle pose.
#
# Longer multi-sentence responses (which is what voice mode actually
# produces) score notably higher than the short smoke-test strings, so this
# threshold is conservative in practice.
DEFAULT_THRESHOLD = 0.20

# In-process cache keyed by catalog content hash → (names, normalized matrix).
# Cleared on process restart, which is fine — the cost of a single rebuild
# is a few tens of ms.
_CACHE: dict[str, tuple[list[str], list[list[float]]]] = {}


def _catalog_hash(animations: list[_AnimationLike]) -> str:
    fingerprint = "\n".join(f"{a.name}|{a.description or ''}" for a in animations)
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _embed_catalog(animations: list[_AnimationLike]) -> tuple[list[str], list[list[float]]] | None:
    """Embed every animation description once, cache by content hash.

    Returns (names, normalized_embeddings) or None on failure.
    """
    key = _catalog_hash(animations)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    # The description carries the semantic load; fall back to the name when
    # missing so we always have at least *something* to embed.
    texts = [
        f"{a.name}. {a.description}" if a.description else a.name
        for a in animations
    ]
    raw = generate_embeddings_batch(texts)
    if any(v is None for v in raw):
        logger.warning(
            "animation classifier: at least one embedding failed; skipping",
        )
        return None
    normalized = [_l2_normalize(v) for v in raw]  # type: ignore[arg-type]
    names = [a.name for a in animations]
    _CACHE[key] = (names, normalized)
    logger.debug("animation classifier: cached %d embeddings", len(names))
    return names, normalized


def classify_animation(
    response_text: str,
    animations: list[_AnimationLike],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> str | None:
    """Pick the best-matching animation for ``response_text``.

    Args:
        response_text: The full assistant response (text only, no tool calls).
        animations: Animation catalog passed in on the request. Caller is
            expected to filter to enabled rows.
        threshold: Minimum cosine similarity to return a pick. Below this,
            returns None — the frontend stays on idle.

    Returns:
        The animation `name`, or None when nothing crosses the threshold
        (or when embeddings aren't available — graceful degradation).
    """
    if not response_text or not response_text.strip():
        return None
    if not animations:
        return None

    catalog = _embed_catalog(animations)
    if catalog is None:
        return None
    names, matrix = catalog

    raw_q = generate_embedding(response_text)
    if raw_q is None:
        return None
    q = _l2_normalize(raw_q)

    # Linear scan — 15-ish entries, no point pulling in numpy just for this.
    best_idx = 0
    best_score = _dot(matrix[0], q)
    for i in range(1, len(matrix)):
        s = _dot(matrix[i], q)
        if s > best_score:
            best_score = s
            best_idx = i

    if best_score < threshold:
        logger.info(
            "🎭 classifier: best=%s score=%.3f below threshold=%.2f → none",
            names[best_idx], best_score, threshold,
        )
        return None
    logger.info(
        "🎭 classifier picked '%s' (score=%.3f)",
        names[best_idx], best_score,
    )
    return names[best_idx]
