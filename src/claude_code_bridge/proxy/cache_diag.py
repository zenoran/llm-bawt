"""Per-turn prefix-divergence diagnostic for the proxy prompt cache.

The ChatGPT codex backend caches on a byte-identical prefix. The proxy is
stateless (``store:false``, full history replayed every turn), so if the
replayed history isn't byte-stable the cache breakpoint lands right after the
tools and the whole transcript re-processes each turn. This module proves
*where* the prefix first diverges turn-over-turn, so we know what to
canonicalize.

OFF BY DEFAULT. Enable by setting the env var ``PROXY_CACHE_DIAG``:

- ``PROXY_CACHE_DIAG=1``              → log the breakpoint line each turn
- ``PROXY_CACHE_DIAG=/path/to/dir``   → also dump each turn's canonical
                                        stream to that dir for offline diffing

On each prepared request we:

1. Build a canonical token-stream of the cacheable prefix in cache order:
   ``[instructions, tool[0], tool[1], ..., input[0], input[1], ...]`` — each
   serialized deterministically (``sort_keys``, compact separators).
2. Group turns by a conversation key (hash of instructions + first user item,
   mirroring ``_prompt_cache_key``).
3. Compare to the previous turn's stream for that key.
4. Log the first diverging component:
   ``instructions`` | ``tools[N]`` | ``input[N]:<role/type>(…)``.

The shared-prefix length tells us how much *did* cache; the breakpoint label
tells us what busted it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FLAG = "PROXY_CACHE_DIAG"
_MAX_KEYS = 64  # bound the in-process history (oldest evicted FIFO)


def _enabled() -> tuple[bool, Path | None]:
    """Return (log_enabled, dump_dir_or_none)."""
    raw = os.getenv(_FLAG)
    if not raw:
        return False, None
    raw = raw.strip()
    if raw in ("1", "true", "yes", "on"):
        return True, None
    p = Path(raw)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Path unwritable — still log, just skip the dump.
        return True, None
    return True, p


def _canon(obj: Any) -> str:
    """Deterministic JSON serialization — byte-stable across runs."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _conv_key(responses_body: dict) -> str:
    """Stable per-conversation key (instructions + first user content item).

    Mirrors ``openai_chatgpt._prompt_cache_key`` so diag grouping matches the
    cache routing the backend actually sees. Falls back to the adapter-set
    ``prompt_cache_key`` when present.
    """
    existing = responses_body.get("prompt_cache_key")
    if isinstance(existing, str) and existing:
        return existing[:12]
    instructions = responses_body.get("instructions") or ""
    first_user = ""
    for item in responses_body.get("input") or []:
        if isinstance(item, dict) and item.get("role") == "user":
            first_user = _canon(item.get("content") or [])
            break
    seed = f"{instructions}\n\n{first_user}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()[:12]


def _input_label(item: dict, idx: int) -> str:
    """Human-readable label for an input item, including volatile id fields."""
    itype = item.get("type")
    if itype in ("function_call", "function_call_output"):
        return f"input[{idx}]:{itype}(call_id={item.get('call_id')})"
    role = item.get("role") or "?"
    return f"input[{idx}]:{role}"


def _build_stream(responses_body: dict) -> list[tuple[str, str]]:
    """Canonical (label, canonical_text) pairs in cache order."""
    parts: list[tuple[str, str]] = []
    instr = responses_body.get("instructions")
    if instr:
        parts.append(("instructions", str(instr)))
    for i, tool in enumerate(responses_body.get("tools") or []):
        if isinstance(tool, dict):
            parts.append((f"tools[{i}]:{tool.get('name') or '?'}", _canon(tool)))
    for i, item in enumerate(responses_body.get("input") or []):
        if isinstance(item, dict):
            parts.append((_input_label(item, i), _canon(item)))
    return parts


# In-process: conv_key -> (turn_index, [labels], [canonical_texts])
_state: dict[str, tuple[int, list[str], list[str]]] = {}
_lock = threading.Lock()


def record(responses_body: dict) -> None:
    """Compare this turn's prefix to the previous turn for the same convo.

    Call after the adapter's ``prepare_request`` so the body is exactly what
    goes upstream. No-op when the diag flag is unset.
    """
    log_enabled, dump_dir = _enabled()
    if not log_enabled:
        return

    key = _conv_key(responses_body)
    stream = _build_stream(responses_body)
    labels = [lbl for lbl, _ in stream]
    texts = [txt for _, txt in stream]

    with _lock:
        prev = _state.get(key)
        # Store before we compute, so we always advance the turn counter even
        # if comparison short-circuits below.
        turn = (prev[0] + 1) if prev else 1
        if len(_state) >= _MAX_KEYS and key not in _state:
            _state.pop(next(iter(_state)))  # evict oldest
        _state[key] = (turn, labels, texts)

    if dump_dir:
        try:
            (dump_dir / f"{key}_turn{turn:03d}.txt").write_text(
                "\n".join(f"{lbl}\t{txt}" for lbl, txt in stream),
                encoding="utf-8",
            )
        except OSError:
            pass

    if not prev:
        logger.info(
            "Cache diag key=%s turn=1 input_items=%d (baseline, no prior to compare)",
            key,
            len(stream),
        )
        return

    _, prev_labels, prev_texts = prev
    shared = 0
    breakpoint_label = None
    # A divergence is a VALUE mismatch at a shared index. Running off the end
    # of the previous turn is NOT a divergence — it's a newly appended tail
    # (the expected marginal turn, which is supposed to be uncached).
    for i in range(min(len(texts), len(prev_texts))):
        if texts[i] != prev_texts[i]:
            breakpoint_label = labels[i]
            break
        shared = i + 1

    total = len(stream)
    if breakpoint_label is None:
        logger.info(
            "Cache diag key=%s turn=%d shared_prefix=%d/%d → only new tail uncached (healthy)",
            key,
            turn,
            shared,
            total,
        )
    else:
        logger.info(
            "Cache diag key=%s turn=%d BREAKPOINT=%s shared_prefix=%d/%d "
            "(everything from here re-processes)",
            key,
            turn,
            breakpoint_label,
            shared,
            total,
        )
