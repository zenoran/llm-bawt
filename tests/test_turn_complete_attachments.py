"""Regression coverage for live turn_complete screenshot attachments."""

from __future__ import annotations

from types import SimpleNamespace

from llm_bawt.service.turn_stream_finalize import TurnStreamFinalizer
from llm_bawt.service.turn_stream_worker import TurnStreamWorker


def test_worker_passes_attachment_enricher_to_terminal_coordinator() -> None:
    """The worker captures its mixin helper before rebinding ``self`` to the service."""
    import inspect

    source = inspect.getsource(TurnStreamWorker._stream_to_queue)
    assert "_enrich_attachment_refs = self._enrich_attachment_refs" in source
    assert "TurnStreamFinalizer(" in source
    assert "enrich_attachment_refs=_enrich_attachment_refs" in source


def test_finalizer_uses_injected_attachment_enricher() -> None:
    calls: list[list[dict]] = []
    ctx = SimpleNamespace(agent_attachments_holder=[])
    finalizer = TurnStreamFinalizer(
        ctx,
        publish_event_direct=lambda _event: None,
        enrich_attachment_refs=lambda refs: calls.append(refs) or refs,
    )
    refs = [{"asset_id": "ma_test", "kind": "image"}]

    assert finalizer._enrich_attachment_refs(refs) == refs
    assert calls == [refs]
