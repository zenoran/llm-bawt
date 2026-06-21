"""Tests for extraction guards applied to generated summaries."""

from llm_bawt.memory.extraction.service import MemoryExtractionService


def test_all_placeholder_summary_is_not_extractable() -> None:
    summary = """[HISTORICAL SUMMARY]
Summary: N/A
Key Details: N/A
Intent: N/A
Tone: N/A
Open Loops: N/A
"""

    assert MemoryExtractionService.summary_has_extractable_content(summary) is False


def test_substantive_summary_is_extractable() -> None:
    summary = """[HISTORICAL SUMMARY]
Summary: User prefers PostgreSQL for durable storage.
Key Details: N/A
Intent: Choose a database.
Tone: Neutral
Open Loops: None
"""

    assert MemoryExtractionService.summary_has_extractable_content(summary) is True


def test_placeholder_summary_does_not_call_llm() -> None:
    class FailingClient:
        def query(self, **_kwargs):
            raise AssertionError("placeholder summary must not call the LLM")

    service = MemoryExtractionService(llm_client=FailingClient())
    facts = service.extract_from_summary(
        summary_text="Summary: N/A\nKey Details: None\nIntent: N/A\nTone: N/A\nOpen Loops: None",
        session_start=0,
        session_end=1,
        summary_id="summary-id",
        use_llm=True,
    )

    assert facts == []
