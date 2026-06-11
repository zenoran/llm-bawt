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
# ServiceLLMBawt._init_bot injection
# ---------------------------------------------------------------------------

class _DummyAgentBackendClient:
    def __init__(self):
        self._bot_config = {}


def _run_init_bot(monkeypatch, bot, defined_models):
    config = SimpleNamespace(defined_models={"models": defined_models})

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
