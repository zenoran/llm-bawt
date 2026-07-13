"""Tests for canonical bot model consolidation onto ``default_model``.

Covers:
  * ``agent_backend_for_model_def`` — the shared catalog-shape helper
  * universal SDK model injection in ``ServiceLLMBawt._init_bot``
    (claude-code now behaves like codex)
  * ``_resolve_request_model`` returning the REAL catalog alias for agent
    bots, with virtual-alias fallback and openclaw exemption
"""

from types import SimpleNamespace

from llm_bawt.bot_types import agent_backend_for_model_def
from llm_bawt.model_catalog import (
    AccessPath,
    ModelCatalog,
    ModelEndpoint,
    ModelIdentity,
    bot_model_ref,
)
from llm_bawt.service.core import ServiceLLMBawt
from llm_bawt.service.instance_manager import InstanceManagerMixin


# ---------------------------------------------------------------------------
# agent_backend_for_model_def
# ---------------------------------------------------------------------------

def test_helper_claude_code_type():
    assert agent_backend_for_model_def({"type": "claude-code"}) == "claude-code"


def test_helper_codex_legacy_type():
    assert agent_backend_for_model_def({"type": "codex"}) == "codex"


def test_helper_agent_backend_top_level():
    md = {"type": "agent_backend", "backend": "codex", "model_id": "gpt-5.5"}
    assert agent_backend_for_model_def(md) == "codex"


def test_helper_agent_backend_extra_nested():
    md = {"type": "agent_backend", "extra": {"backend": "codex"}}
    assert agent_backend_for_model_def(md) == "codex"


def test_helper_chat_models_and_edge_cases():
    assert agent_backend_for_model_def({"type": "openai", "model_id": "gpt-5"}) is None
    assert agent_backend_for_model_def({"type": "gguf"}) is None
    assert agent_backend_for_model_def({"type": "agent_backend"}) is None
    assert agent_backend_for_model_def({}) is None
    assert agent_backend_for_model_def(None) is None


# ---------------------------------------------------------------------------
# Canonical bot endpoint selection
# ---------------------------------------------------------------------------


def _dual_grok_catalog():
    model = ModelIdentity(
        id=1,
        key="grok-4.5",
        vendor="xai",
        display_name="Grok 4.5",
    )
    chat = AccessPath(
        id=1,
        key="xai-chat",
        vendor="xai",
        protocol="chat-completions",
        base_url="https://api.x.ai/v1",
        auth_mechanism="api-key",
    )
    responses = AccessPath(
        id=2,
        key="xai-responses",
        vendor="xai",
        protocol="responses",
        base_url="https://api.x.ai/v1",
        auth_mechanism="api-key",
    )
    return ModelCatalog([
        ModelEndpoint(17, model, chat, "grok-4.5", legacy_type="grok"),
        ModelEndpoint(18, model, responses, "grok-4.5", legacy_type="grok"),
    ])


def test_bot_model_ref_prefers_canonical_endpoint():
    catalog = _dual_grok_catalog()
    config = SimpleNamespace(ensure_model_catalog=lambda: catalog)
    bot = SimpleNamespace(
        default_model="grok-4.5",
        endpoint_id=18,
        harness="claude-proxy",
    )

    assert bot_model_ref(config, bot) == "grok-4.5@xai-responses"


def test_bot_model_ref_legacy_fallback_without_catalog():
    config = SimpleNamespace(ensure_model_catalog=lambda: None)
    bot = SimpleNamespace(
        default_model="grok-4.5",
        endpoint_id=18,
        harness="claude-proxy",
    )

    assert bot_model_ref(config, bot) == "grok-4.5"


# ---------------------------------------------------------------------------
# ServiceLLMBawt._init_bot injection
# ---------------------------------------------------------------------------

class _DummyAgentBackendClient:
    def __init__(self):
        self._bot_config = {}


def _run_init_bot(monkeypatch, bot, defined_models, config=None):
    config = config or SimpleNamespace(defined_models={"models": defined_models})

    def fake_init_bot(self, _config):
        self.bot = bot

    monkeypatch.setattr("llm_bawt.core.base.BaseLLMBawt._init_bot", fake_init_bot)
    monkeypatch.setattr(
        "llm_bawt.clients.agent_backend_client.AgentBackendClient",
        _DummyAgentBackendClient,
    )

    service = ServiceLLMBawt.__new__(ServiceLLMBawt)
    service.client = _DummyAgentBackendClient()
    service.bot_id = "loopy"
    service.user_id = "nick"
    service.config = config
    ServiceLLMBawt._init_bot(service, config)
    return service


def test_init_bot_injects_claude_code_model(monkeypatch):
    """claude-code bots get model_id injected from default_model — same as codex."""
    bot = SimpleNamespace(
        agent_backend="claude-code",
        agent_backend_config={"session_key": "abc-123"},
        default_model="opus-4-7",
    )
    service = _run_init_bot(
        monkeypatch,
        bot,
        {"opus-4-7": {"type": "claude-code", "model_id": "claude-opus-4-7"}},
    )
    assert service.client._bot_config["model"] == "claude-opus-4-7"
    assert service.client._bot_config["session_key"] == "abc-123"
    assert service.client._bot_config["bot_id"] == "loopy"


def test_init_bot_uses_canonical_endpoint_for_ambiguous_proxy_model(monkeypatch):
    catalog = _dual_grok_catalog()
    config = SimpleNamespace(
        defined_models={"models": catalog.compatibility_mapping()},
        ensure_model_catalog=lambda: catalog,
        resolve_model=lambda ref, harness=None, default=None: catalog.resolve(
            ref, harness=harness
        ),
    )
    bot = SimpleNamespace(
        agent_backend="claude-code",
        agent_backend_config={},
        default_model="grok-4.5",
        endpoint_id=18,
        harness="claude-proxy",
    )

    service = _run_init_bot(monkeypatch, bot, {}, config=config)

    assert service.client._bot_config["model"] == "xai/grok-4.5"


def test_init_bot_injection_overrides_legacy_config_model(monkeypatch):
    """Catalog model wins over stale agent_backend_config.model."""
    bot = SimpleNamespace(
        agent_backend="claude-code",
        agent_backend_config={"model": "claude-sonnet-4-20250514"},
        default_model="opus-4-7",
    )
    service = _run_init_bot(
        monkeypatch,
        bot,
        {"opus-4-7": {"type": "claude-code", "model_id": "claude-opus-4-7"}},
    )
    assert service.client._bot_config["model"] == "claude-opus-4-7"


def test_init_bot_legacy_fallback_when_default_model_unresolvable(monkeypatch):
    """No compatible catalog entry → legacy agent_backend_config.model survives."""
    bot = SimpleNamespace(
        agent_backend="claude-code",
        agent_backend_config={"model": "claude-sonnet-4-20250514"},
        default_model="grok-4-fast",
    )
    service = _run_init_bot(
        monkeypatch,
        bot,
        {"grok-4-fast": {"type": "grok", "model_id": "grok-4-fast"}},
    )
    assert service.client._bot_config["model"] == "claude-sonnet-4-20250514"


def test_init_bot_backend_mismatch_no_injection(monkeypatch):
    """default_model pointing at a DIFFERENT backend's entry is not injected."""
    bot = SimpleNamespace(
        agent_backend="claude-code",
        agent_backend_config={},
        default_model="codex-gpt-5-5",
    )
    service = _run_init_bot(
        monkeypatch,
        bot,
        {"codex-gpt-5-5": {"type": "agent_backend", "backend": "codex", "model_id": "gpt-5.5"}},
    )
    assert "model" not in service.client._bot_config


# ---------------------------------------------------------------------------
# _resolve_request_model — real alias for agent bots
# ---------------------------------------------------------------------------

def _make_manager(defined_models, available, agent_backend_models, default_model="nova-model"):
    mgr = InstanceManagerMixin.__new__(InstanceManagerMixin)
    mgr.config = SimpleNamespace(defined_models={"models": defined_models})
    mgr._available_models = available
    mgr._agent_backend_models = agent_backend_models
    mgr._default_model = default_model
    return mgr


def _patch_bot_manager(monkeypatch, bot):
    class FakeBotManager:
        def __init__(self, _config):
            pass

        def get_bot(self, _bot_id):
            return bot

        def get_default_bot(self, **_kw):
            return bot

    monkeypatch.setattr("llm_bawt.service.instance_manager.BotManager", FakeBotManager)


def test_resolve_agent_bot_returns_real_alias(monkeypatch):
    bot = SimpleNamespace(slug="loopy", agent_backend="claude-code", default_model="opus-4-7")
    _patch_bot_manager(monkeypatch, bot)
    mgr = _make_manager(
        defined_models={
            "opus-4-7": {"type": "claude-code", "model_id": "claude-opus-4-7"},
        },
        available=["opus-4-7", "claude-code"],
        agent_backend_models={"loopy": "claude-code"},
    )
    alias, warnings = mgr._resolve_request_model(None, "loopy", local_mode=False)
    assert alias == "opus-4-7"
    assert warnings == []


def test_resolve_agent_bot_uses_canonical_endpoint_ref(monkeypatch):
    catalog = _dual_grok_catalog()
    bot = SimpleNamespace(
        slug="nova",
        agent_backend="claude-code",
        default_model="grok-4.5",
        endpoint_id=18,
        harness="claude-proxy",
    )
    _patch_bot_manager(monkeypatch, bot)
    mgr = _make_manager(
        defined_models=catalog.compatibility_mapping(),
        available=["grok-4.5@xai-chat", "grok-4.5@xai-responses", "claude-code"],
        agent_backend_models={"nova": "claude-code"},
    )
    mgr.config.ensure_model_catalog = lambda: catalog
    mgr.config.resolve_model = lambda ref, harness=None, default=None: catalog.resolve(
        ref, harness=harness
    )

    alias, warnings = mgr._resolve_request_model(None, "nova", local_mode=False)

    assert alias == "grok-4.5@xai-responses"
    assert warnings == []


def test_resolve_agent_bot_accepts_single_endpoint_canonical_ref(monkeypatch):
    """A canonical endpoint ref need not appear in the compatibility key list."""
    model = ModelIdentity(20, "gpt-5.6-sol", "openai", "GPT 5.6 Sol")
    oauth = AccessPath(
        11,
        "openai-oauth",
        "openai",
        "responses",
        "https://chatgpt.com/backend-api/codex",
        "oauth",
    )
    catalog = ModelCatalog([
        ModelEndpoint(
            20,
            model,
            oauth,
            "gpt-5.6-sol",
            legacy_type="agent_backend",
            serving_config={"compat_extra": {"backend": "codex"}},
        )
    ])
    bot = SimpleNamespace(
        slug="al",
        agent_backend="claude-code",
        default_model="gpt-5.6-sol",
        endpoint_id=20,
        harness="claude-proxy",
    )
    _patch_bot_manager(monkeypatch, bot)
    # Single-endpoint compatibility mappings expose only the model key, not
    # ``model@access-path``. That must not invalidate bot_model_ref's result.
    mgr = _make_manager(
        defined_models=catalog.compatibility_mapping(),
        available=["gpt-5.6-sol", "claude-code"],
        agent_backend_models={"al": "claude-code"},
    )
    mgr.config.ensure_model_catalog = lambda: catalog
    mgr.config.resolve_model = lambda ref, harness=None, default=None: catalog.resolve(
        ref, harness=harness
    )

    alias, warnings = mgr._resolve_request_model(None, "al", local_mode=False)

    assert alias == "gpt-5.6-sol@openai-oauth"
    assert warnings == []


def test_resolve_agent_bot_ignores_requested_model(monkeypatch):
    bot = SimpleNamespace(slug="loopy", agent_backend="claude-code", default_model="opus-4-7")
    _patch_bot_manager(monkeypatch, bot)
    mgr = _make_manager(
        defined_models={"opus-4-7": {"type": "claude-code", "model_id": "claude-opus-4-7"}},
        available=["opus-4-7", "claude-code"],
        agent_backend_models={"loopy": "claude-code"},
    )
    alias, warnings = mgr._resolve_request_model("grok-4-fast", "loopy", local_mode=False)
    assert alias == "opus-4-7"
    assert any("ignoring requested model" in w for w in warnings)


def test_resolve_agent_bot_falls_back_to_virtual_alias(monkeypatch):
    """Missing/incompatible default_model → virtual backend alias."""
    bot = SimpleNamespace(slug="loopy", agent_backend="claude-code", default_model=None)
    _patch_bot_manager(monkeypatch, bot)
    mgr = _make_manager(
        defined_models={},
        available=["claude-code"],
        agent_backend_models={"loopy": "claude-code"},
    )
    alias, _ = mgr._resolve_request_model(None, "loopy", local_mode=False)
    assert alias == "claude-code"


def test_resolve_openclaw_bot_keeps_virtual_alias(monkeypatch):
    """Openclaw gateway owns its model — always virtual alias, even with default_model set."""
    bot = SimpleNamespace(slug="vex", agent_backend="openclaw", default_model="grok-4-fast")
    _patch_bot_manager(monkeypatch, bot)
    mgr = _make_manager(
        defined_models={"grok-4-fast": {"type": "grok", "model_id": "grok-4-fast"}},
        available=["grok-4-fast", "openclaw"],
        agent_backend_models={"vex": "openclaw"},
    )
    alias, _ = mgr._resolve_request_model(None, "vex", local_mode=False)
    assert alias == "openclaw"


# ---------------------------------------------------------------------------
# migration alias derivation
# ---------------------------------------------------------------------------

def test_derive_model_alias():
    from llm_bawt.memory.migrations import _derive_model_alias

    assert _derive_model_alias("claude-opus-4-20250514") == "opus-4-20250514"
    assert _derive_model_alias("claude-sonnet-4-5") == "sonnet-4-5"
    assert _derive_model_alias("gpt-5.5") == "gpt-5-5"
    assert _derive_model_alias("GPT-5.5 Codex") == "gpt-5-5-codex"
    assert _derive_model_alias("") == "model"


# ---------------------------------------------------------------------------
# settings-route config-time validation (Phase 3)
# ---------------------------------------------------------------------------

import pytest
from fastapi import HTTPException

from llm_bawt.service.routes import settings as settings_routes


class _FakeConn:
    def __init__(self, row):
        self._row = row

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def execute(self, *_a, **_k):
        row = self._row
        return SimpleNamespace(fetchone=lambda: row)


class _FakeEngine:
    def __init__(self, row):
        self._row = row

    def connect(self):
        return _FakeConn(self._row)


def _patch_catalog_row(monkeypatch, row):
    """Make _validate_agent_default_model see ``row`` as the catalog lookup."""
    monkeypatch.setattr(
        settings_routes, "get_service", lambda: SimpleNamespace(config=None)
    )
    monkeypatch.setattr(
        settings_routes,
        "get_bot_profile_store",
        lambda _cfg: SimpleNamespace(engine=_FakeEngine(row)),
    )


def test_normalize_mirrors_model_to_session_model():
    payload = {
        "slug": "loopy",
        "agent_backend_config": {"model": "claude-sonnet-4", "session_key": "k"},
    }
    settings_routes._normalize_agent_backend_config_model("claude-code", payload)
    config = payload["agent_backend_config"]
    # "model" is kept (not popped) so un-restarted bridges still see it for
    # session resume-vs-reset; new bridge code drops it on first persist.
    assert config["model"] == "claude-sonnet-4"
    assert config["session_model"] == "claude-sonnet-4"
    assert config["session_key"] == "k"


def test_normalize_drops_empty_model_key():
    payload = {"slug": "loopy", "agent_backend_config": {"model": ""}}
    settings_routes._normalize_agent_backend_config_model("claude-code", payload)
    assert payload["agent_backend_config"] == {}


def test_normalize_openclaw_untouched():
    payload = {"slug": "vex", "agent_backend_config": {"model": "anything"}}
    settings_routes._normalize_agent_backend_config_model("openclaw", payload)
    assert payload["agent_backend_config"] == {"model": "anything"}


def test_validate_missing_default_model_422():
    with pytest.raises(HTTPException) as exc:
        settings_routes._validate_agent_default_model(
            "claude-code", {"slug": "loopy", "default_model": None}
        )
    assert exc.value.status_code == 422
    assert "default_model" in exc.value.detail


def test_validate_unknown_alias_422(monkeypatch):
    _patch_catalog_row(monkeypatch, None)
    with pytest.raises(HTTPException) as exc:
        settings_routes._validate_agent_default_model(
            "claude-code", {"slug": "loopy", "default_model": "nope"}
        )
    assert exc.value.status_code == 422
    assert "not a known model" in exc.value.detail


def test_validate_backend_mismatch_422(monkeypatch):
    _patch_catalog_row(monkeypatch, ("grok", "grok-4-fast", ""))
    with pytest.raises(HTTPException) as exc:
        settings_routes._validate_agent_default_model(
            "claude-code", {"slug": "loopy", "default_model": "grok-4-fast"}
        )
    assert exc.value.status_code == 422
    assert "not compatible" in exc.value.detail


def test_validate_missing_model_id_422(monkeypatch):
    _patch_catalog_row(monkeypatch, ("claude-code", "", ""))
    with pytest.raises(HTTPException) as exc:
        settings_routes._validate_agent_default_model(
            "claude-code", {"slug": "loopy", "default_model": "opus-4-7"}
        )
    assert exc.value.status_code == 422
    assert "model_id" in exc.value.detail


def test_validate_compatible_entry_passes(monkeypatch):
    _patch_catalog_row(monkeypatch, ("claude-code", "claude-opus-4-7", ""))
    settings_routes._validate_agent_default_model(
        "claude-code", {"slug": "loopy", "default_model": "opus-4-7"}
    )


def test_validate_codex_agent_backend_shape_passes(monkeypatch):
    _patch_catalog_row(monkeypatch, ("agent_backend", "gpt-5.5", "codex"))
    settings_routes._validate_agent_default_model(
        "codex", {"slug": "byte", "default_model": "codex-gpt-5-5"}
    )


def test_validate_openclaw_exempt():
    # No DB patching needed: must return before any lookup.
    settings_routes._validate_agent_default_model(
        "openclaw", {"slug": "vex", "default_model": None}
    )


def test_validate_db_lookup_failure_does_not_gate(monkeypatch):
    class _BrokenEngine:
        def connect(self):
            raise RuntimeError("db down")

    monkeypatch.setattr(
        settings_routes, "get_service", lambda: SimpleNamespace(config=None)
    )
    monkeypatch.setattr(
        settings_routes,
        "get_bot_profile_store",
        lambda _cfg: SimpleNamespace(engine=_BrokenEngine()),
    )
    settings_routes._validate_agent_default_model(
        "claude-code", {"slug": "loopy", "default_model": "opus-4-7"}
    )


def test_resolve_chat_bot_unchanged(monkeypatch):
    bot = SimpleNamespace(slug="nova", agent_backend=None, default_model="grok-4-fast")
    _patch_bot_manager(monkeypatch, bot)
    mgr = _make_manager(
        defined_models={"grok-4-fast": {"type": "grok", "model_id": "grok-4-fast"}},
        available=["grok-4-fast"],
        agent_backend_models={},
    )
    alias, warnings = mgr._resolve_request_model(None, "nova", local_mode=False)
    assert alias == "grok-4-fast"
    assert warnings == []
