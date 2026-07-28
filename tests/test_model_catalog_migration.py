from llm_bawt.memory.model_catalog_migration import (
    _CREATE_SCHEMA_SQL,
    _derive_harness,
)


def test_access_path_schema_adds_harness_prompt_mapping_idempotently():
    assert "system_prompt_overrides JSONB NOT NULL DEFAULT '{}'::jsonb" in _CREATE_SCHEMA_SQL
    assert "ADD COLUMN IF NOT EXISTS system_prompt_overrides" in _CREATE_SCHEMA_SQL


def test_derive_harness_maps_backend_to_native_harness():
    assert _derive_harness("openclaw", "anything", "anything") == "openclaw"
    assert _derive_harness("codex", "openai", "responses") == "codex"


def test_derive_harness_claude_code_is_direct_only_for_native_anthropic():
    # Native Anthropic messages → the direct claude-code harness.
    assert (
        _derive_harness("claude-code", "anthropic", "anthropic-messages")
        == "claude-code"
    )
    # A non-Anthropic vendor (or non-messages protocol) must ride the proxy.
    assert _derive_harness("claude-code", "zai", "anthropic-messages") == "claude-proxy"
    assert _derive_harness("claude-code", "openai", "responses") == "claude-proxy"
    assert _derive_harness("claude-code", "xai", "chat-completions") == "claude-proxy"


def test_derive_harness_defaults_to_chat_for_direct_llm_backends():
    assert _derive_harness(None, "openai", "chat-completions") == "chat"
    assert _derive_harness("", "xai", "chat-completions") == "chat"
    assert _derive_harness("grok", "xai", "chat-completions") == "chat"
