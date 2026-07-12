from llm_bawt.memory.model_catalog_migration import (
    _derive_harness,
    map_legacy_definition,
)


def _row(**overrides):
    row = {
        "alias": "example",
        "type": "openai",
        "model_id": "example-upstream",
        "repo_id": None,
        "filename": None,
        "description": "Example",
        "extra": None,
        "created_at": None,
        "updated_at": None,
    }
    row.update(overrides)
    return row


def test_codex_and_claude_proxy_map_to_same_openai_oauth_endpoint_shape():
    codex = map_legacy_definition(
        _row(
            alias="gpt-5.6-sol",
            type="agent_backend",
            model_id="gpt-5.6-sol",
            extra={"backend": "codex"},
        )
    )
    proxy = map_legacy_definition(
        _row(
            alias="gpt-5.6-sol",
            type="claude-code",
            model_id="openai_chatgpt/gpt-5.6-sol",
            extra={"provider": "openai_chatgpt", "upstream_model": "gpt-5.6-sol"},
        )
    )

    assert (
        (codex.model_key, codex.access_path.key)
        == (
            proxy.model_key,
            proxy.access_path.key,
        )
        == ("gpt-5.6-sol", "openai-oauth")
    )
    assert codex.upstream_model_id == proxy.upstream_model_id == "gpt-5.6-sol"
    assert _derive_harness("codex", codex.access_path) == "codex"
    assert _derive_harness("claude-code", proxy.access_path) == "claude-proxy"


def test_native_claude_and_zai_passthrough_use_anthropic_message_harness():
    claude = map_legacy_definition(
        _row(alias="claude-opus-4-7", type="claude-code", model_id="claude-opus-4-7")
    )
    zai = map_legacy_definition(
        _row(
            alias="zai-glm-5.2",
            type="claude-code",
            model_id="zai/glm-5.2",
            extra={"provider": "zai", "upstream_model": "glm-5.2"},
        )
    )

    assert claude.access_path.key == "anthropic-oauth"
    assert zai.access_path.key == "zai-anthropic"
    assert _derive_harness("claude-code", claude.access_path) == "claude-code"
    assert _derive_harness("claude-code", zai.access_path) == "claude-code"


def test_local_gguf_serving_fields_belong_to_endpoint():
    mapped = map_legacy_definition(
        _row(
            alias="dolphin",
            type="gguf",
            model_id=None,
            repo_id="org/model-GGUF",
            filename="model-Q6_K.gguf",
            extra={
                "context_window": 32768,
                "n_gpu_layers": 41,
                "chat_format": "chatml",
            },
        )
    )

    assert mapped.access_path.key == "local-llamacpp"
    assert mapped.context_window_override == 32768
    assert mapped.serving_config["repo_id"] == "org/model-GGUF"
    assert mapped.serving_config["filename"] == "model-Q6_K.gguf"
    assert mapped.serving_config["n_gpu_layers"] == 41
    assert mapped.serving_config["chat_format"] == "chatml"


def test_xai_chat_and_responses_are_distinct_protocol_access_paths():
    direct = map_legacy_definition(
        _row(alias="grok-direct", type="grok", model_id="grok-4.20")
    )
    proxy = map_legacy_definition(
        _row(
            alias="grok-proxy",
            type="claude-code",
            model_id="xai/grok-4.20",
            extra={"provider": "xai", "upstream_model": "grok-4.20"},
        )
    )

    assert direct.access_path.key == "xai-chat"
    assert direct.access_path.protocol == "chat-completions"
    assert proxy.access_path.key == "xai-responses"
    assert proxy.access_path.protocol == "responses"
    assert _derive_harness("claude-code", proxy.access_path) == "claude-proxy"


def test_json_null_extra_is_preserved_for_compatibility_view():
    sql_null = map_legacy_definition(_row(extra=None, extra_is_sql_null=True))
    json_null = map_legacy_definition(_row(extra=None, extra_is_sql_null=False))

    assert "compat_extra" not in sql_null.serving_config
    assert json_null.serving_config["compat_extra"] is None
