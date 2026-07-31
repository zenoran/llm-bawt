"""Regression coverage for live turn_complete screenshot attachments."""

from __future__ import annotations

from types import SimpleNamespace

from llm_bawt.service.turn_stream_worker import TurnStreamWorker


def test_worker_captures_attachment_enricher_before_service_rebind() -> None:
    """The worker rebinds ``self`` to BackgroundService inside _stream_to_queue.

    Attachment enrichment must therefore be captured from the worker mixin before
    that rebind, matching the existing publish/persist helper pattern.
    """
    worker = object.__new__(TurnStreamWorker)
    worker.ctx = SimpleNamespace()

    calls: list[list[dict]] = []
    worker._enrich_attachment_refs = lambda refs: calls.append(refs) or [  # type: ignore[method-assign]
        {"asset_id": "ma_test", "kind": "image"}
    ]

    captured = worker._enrich_attachment_refs
    refs = [{"asset_id": "ma_test", "kind": "image"}]

    assert captured(refs) == refs
    assert calls == [refs]


def test_worker_source_uses_captured_enricher_after_self_rebind() -> None:
    import inspect

    source = inspect.getsource(TurnStreamWorker._stream_to_queue)
    assert "_enrich_attachment_refs = self._enrich_attachment_refs" in source
    assert "completed_attachments = _enrich_attachment_refs(" in source
    assert "completed_attachments = self._enrich_attachment_refs(" not in source
