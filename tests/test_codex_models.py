from types import SimpleNamespace

from llm_bawt.service.core import ServiceLLMBawt
from llm_bawt.service.routes.models import _normalize_model_definition, _public_model_type


def test_public_model_type_exposes_codex_group():
    info = {"type": "agent_backend", "backend": "codex", "model_id": "gpt-5.5"}
    assert _public_model_type(info) == "codex"


def test_normalize_model_definition_maps_codex_to_agent_backend():
    normalized = _normalize_model_definition(
        {"type": "codex", "model_id": "gpt-5.4", "description": "test"}
    )
    assert normalized["type"] == "agent_backend"
    assert normalized["backend"] == "codex"
    assert normalized["tool_support"] == "none"


def test_service_bot_init_applies_codex_default_model(monkeypatch):
    bot = SimpleNamespace(
        agent_backend="codex",
        agent_backend_config={"session_key": "bot:user"},
        default_model="codex-gpt-5-5",
    )
    config = SimpleNamespace(
        defined_models={
            "models": {
                "codex-gpt-5-5": {
                    "type": "agent_backend",
                    "backend": "codex",
                    "model_id": "gpt-5.5",
                }
            }
        }
    )

    class DummyAgentBackendClient:
        def __init__(self):
            self._bot_config = {}

    def fake_init_bot(self, _config):
        self.bot = bot

    monkeypatch.setattr("llm_bawt.core.base.BaseLLMBawt._init_bot", fake_init_bot)
    monkeypatch.setattr(
        "llm_bawt.clients.agent_backend_client.AgentBackendClient",
        DummyAgentBackendClient,
    )

    service = ServiceLLMBawt.__new__(ServiceLLMBawt)
    service.client = DummyAgentBackendClient()
    service.bot_id = "builder"
    service.user_id = "nick"
    service.config = config

    ServiceLLMBawt._init_bot(service, config)

    assert service.client._bot_config["session_key"] == "bot:user"
    assert service.client._bot_config["bot_id"] == "builder"
    assert service.client._bot_config["user_id"] == "nick"
    assert service.client._bot_config["model"] == "gpt-5.5"
