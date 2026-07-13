"""Provider attribution tests for the per-turn context usage popover."""

from types import SimpleNamespace

from llm_bawt.model_catalog import AccessPath, ModelCatalog, ModelEndpoint, ModelIdentity
from llm_bawt.service.routes import usage as usage_routes


def _catalog(*endpoints: ModelEndpoint) -> ModelCatalog:
    return ModelCatalog(endpoints)


def _service_and_bot(endpoint: ModelEndpoint, *, harness: str, backend: str):
    catalog = _catalog(endpoint)
    config = SimpleNamespace(
        ensure_model_catalog=lambda: catalog,
        resolve_model=lambda ref, harness=None, default=None: catalog.resolve(
            ref, harness=harness
        ),
    )
    bot = SimpleNamespace(
        slug="al",
        default_model=endpoint.model.key,
        endpoint_id=endpoint.id,
        harness=harness,
        agent_backend=backend,
    )
    return SimpleNamespace(config=config), bot


def _patch_bot(monkeypatch, bot) -> None:
    class FakeBotManager:
        def __init__(self, _config):
            pass

        def get_bot(self, _bot_id):
            return bot

    monkeypatch.setattr("llm_bawt.bots.BotManager", FakeBotManager)


def test_openai_oauth_claude_proxy_maps_to_chatgpt_usage(monkeypatch):
    endpoint = ModelEndpoint(
        id=20,
        model=ModelIdentity(20, "reasoning-primary", "openai", "Reasoning Primary"),
        access_path=AccessPath(
            11,
            "openai-oauth",
            "openai",
            "responses",
            "https://chatgpt.com/backend-api/codex",
            "oauth",
        ),
        upstream_model_id="opaque-upstream-name",
        legacy_type="agent_backend",
    )
    service, bot = _service_and_bot(
        endpoint, harness="claude-proxy", backend="claude-code"
    )
    _patch_bot(monkeypatch, bot)

    assert usage_routes._provider_for_bot(service, "al") == "openai_chatgpt"


def test_native_anthropic_endpoint_maps_to_claude(monkeypatch):
    endpoint = ModelEndpoint(
        id=17,
        model=ModelIdentity(17, "opus-current", "anthropic", "Opus"),
        access_path=AccessPath(
            6,
            "anthropic-oauth",
            "anthropic",
            "anthropic-messages",
            "https://api.anthropic.com",
            "oauth",
        ),
        upstream_model_id="claude-opus-current",
        legacy_type="claude-code",
    )
    service, bot = _service_and_bot(
        endpoint, harness="claude-code", backend="claude-code"
    )
    _patch_bot(monkeypatch, bot)

    assert usage_routes._provider_for_bot(service, "al") == "claude"


def test_legacy_gpt_alias_still_maps_to_chatgpt_usage(monkeypatch):
    config = SimpleNamespace(
        ensure_model_catalog=lambda: None,
        resolve_model=lambda ref, harness=None, default=None: {
            "type": "agent_backend",
            "model_id": "gpt-5.6-sol",
        },
    )
    bot = SimpleNamespace(
        slug="al",
        default_model="gpt-5.6-sol",
        endpoint_id=None,
        harness="claude-proxy",
        agent_backend="claude-code",
    )
    _patch_bot(monkeypatch, bot)

    assert usage_routes._provider_for_bot(SimpleNamespace(config=config), "al") == "openai_chatgpt"
