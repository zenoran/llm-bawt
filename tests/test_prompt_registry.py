"""Tests for prompt registry defaults and validation."""

from sqlmodel import SQLModel, create_engine

from llm_bawt.prompt_registry import (
    PromptResolver,
    PromptTemplate,
    PromptTemplateStore,
    PromptTemplateVersion,
    extract_placeholders,
)


def test_extract_placeholders_ignores_escaped_braces() -> None:
    body = "JSON example: {{\"ok\": true}} {messages}"
    assert extract_placeholders(body) == ["messages"]


def test_prompt_resolver_renders_default_summarization_prompt() -> None:
    resolver = PromptResolver()
    rendered = resolver.render(
        key="history.summarization.single",
        variables={"messages": "User: hi\nAssistant: hello"},
    )
    assert "User: hi" in rendered
    assert "Output format" in rendered


def test_prompt_validation_flags_missing_required_placeholders() -> None:
    resolver = PromptResolver()
    result = resolver.validate(
        key="history.summarization.single",
        body="Summarize this\nConversation:\n{wrong_name}",
    )
    assert result["valid"] is False
    assert "messages" in result["missing_required"]


def test_prompt_resolver_renders_memory_maintenance_prompt() -> None:
    resolver = PromptResolver()
    rendered = resolver.render(
        key="memory.maintenance.intent_content_only",
        variables={"fact": "User prefers compact summaries"},
    )
    assert "User prefers compact summaries" in rendered
    assert "intent phrase" in rendered


def test_prompt_store_seed_defaults_is_idempotent() -> None:
    store = PromptTemplateStore.__new__(PromptTemplateStore)
    store.config = None
    store.connection_url = "sqlite://test"
    store.engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(
        store.engine,
        tables=[PromptTemplate.__table__, PromptTemplateVersion.__table__],
    )

    first = store.seed_defaults()
    second = store.seed_defaults()

    assert first["created"] > 0
    assert second["created"] == 0
    assert second["skipped"] == first["total"]
