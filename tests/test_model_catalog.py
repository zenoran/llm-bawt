import pytest

from llm_bawt.model_catalog import (
    AccessPath,
    AmbiguousModelError,
    IncompatibleModelError,
    ModelCatalog,
    ModelEndpoint,
    ModelIdentity,
    ProtocolCompatibility,
)


def _endpoint(
    endpoint_id: int,
    model_key: str,
    access_key: str,
    vendor: str,
    protocol: str,
    upstream: str,
    *,
    legacy_type: str = "openai",
    serving_config=None,
):
    return ModelEndpoint(
        id=endpoint_id,
        model=ModelIdentity(endpoint_id, model_key, vendor, model_key),
        access_path=AccessPath(
            endpoint_id,
            access_key,
            vendor,
            protocol,
            None,
            "test",
        ),
        upstream_model_id=upstream,
        serving_config=serving_config or {},
        legacy_type=legacy_type,
    )


def test_shared_openai_oauth_endpoint_resolves_differently_by_harness():
    endpoint = _endpoint(
        20,
        "gpt-5.6-sol",
        "openai-oauth",
        "openai",
        "responses",
        "gpt-5.6-sol",
        legacy_type="agent_backend",
    )
    catalog = ModelCatalog([endpoint])

    codex = catalog.resolve("gpt-5.6-sol", harness="codex")
    proxy = catalog.resolve("gpt-5.6-sol", harness="claude-proxy")

    assert codex["type"] == "agent_backend"
    assert codex["backend"] == "codex"
    assert codex["model_id"] == "gpt-5.6-sol"
    assert proxy["type"] == "claude-code"
    assert proxy["backend"] == "claude-code"
    assert proxy["model_id"] == "openai_chatgpt/gpt-5.6-sol"
    assert codex["endpoint_id"] == proxy["endpoint_id"] == 20


def test_bare_unique_proxy_ref_preserves_legacy_provider_prefix():
    endpoint = _endpoint(
        2,
        "grok-4.5",
        "xai-responses",
        "xai",
        "responses",
        "grok-4.5",
        legacy_type="claude-code",
    )

    resolved = ModelCatalog([endpoint]).resolve("grok-4.5")
    assert resolved["type"] == "claude-code"
    assert resolved["model_id"] == "xai/grok-4.5"


def test_protocol_compatibility_is_single_filter_rule():
    anthropic = _endpoint(
        1, "claude-opus", "anthropic-oauth", "anthropic", "anthropic-messages", "opus"
    )
    xai = _endpoint(2, "grok", "xai-responses", "xai", "responses", "grok-4")

    assert ProtocolCompatibility.is_compatible("claude-code", anthropic.access_path)
    assert not ProtocolCompatibility.is_compatible("codex", anthropic.access_path)
    assert ProtocolCompatibility.is_compatible("claude-proxy", xai.access_path)
    assert not ProtocolCompatibility.is_compatible(
        "claude-proxy", anthropic.access_path
    )


def test_bare_collision_requires_harness_or_endpoint_ref():
    chat = _endpoint(1, "same", "openai-api", "openai", "chat-completions", "same")
    responses = _endpoint(2, "same", "openai-oauth", "openai", "responses", "same")
    catalog = ModelCatalog([chat, responses])

    with pytest.raises(AmbiguousModelError):
        catalog.resolve("same")
    assert catalog.resolve("same", harness="chat")["endpoint_id"] == 1
    assert catalog.resolve("same", harness="codex")["endpoint_id"] == 2
    assert catalog.resolve("same@openai-oauth")["endpoint_id"] == 2


def test_incompatible_explicit_endpoint_is_rejected():
    endpoint = _endpoint(
        1, "claude-opus", "anthropic-oauth", "anthropic", "anthropic-messages", "opus"
    )
    catalog = ModelCatalog([endpoint])

    with pytest.raises(IncompatibleModelError):
        catalog.resolve(1, harness="chat")


def test_local_serving_config_is_flattened_for_existing_consumers():
    endpoint = _endpoint(
        3,
        "dolphin",
        "local-llamacpp",
        "local",
        "chat-completions",
        "dolphin",
        legacy_type="gguf",
        serving_config={
            "repo_id": "org/repo",
            "filename": "model.gguf",
            "n_gpu_layers": 33,
        },
    )
    catalog = ModelCatalog([endpoint])

    resolved = catalog.resolve("dolphin", harness="chat")
    assert resolved["type"] == "gguf"
    assert resolved["repo_id"] == "org/repo"
    assert resolved["filename"] == "model.gguf"
    assert resolved["n_gpu_layers"] == 33
