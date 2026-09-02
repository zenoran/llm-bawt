"""HTTP contract tests for the first-run setup routes (TASK-355)."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from llm_bawt.service.routes import setup as setup_routes


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(setup_routes.router)
    return TestClient(app)


def _service() -> MagicMock:
    service = MagicMock()
    service.config = MagicMock(name="config")
    return service


def _adapter(state: str = "ok", label: str = "xAI") -> MagicMock:
    adapter = MagicMock()
    adapter.id = "xai"
    adapter.label = label
    adapter.health.return_value = {"state": state, "detail": None}
    return adapter


_FIRST_BOT = {
    "provider_id": "xai",
    "model_id": "grok-4.3",
    "context_window": 131072,
    "slug": "helper",
    "name": "Helper",
    "system_prompt": "You are helpful.",
}


# ---------------------------------------------------------------------------
# GET /v1/setup/providers
# ---------------------------------------------------------------------------


def test_setup_providers_reports_connection_state_and_defaults():
    adapters = {"xai": _adapter("ok"), "openai-api": _adapter("unconfigured", "OpenAI")}

    with (
        patch.object(setup_routes, "get_service", return_value=_service()),
        patch.object(setup_routes, "get_adapter", side_effect=lambda _c, pid: adapters.get(pid)),
    ):
        response = _client().get("/v1/setup/providers")

    assert response.status_code == 200
    payload = response.json()
    by_id = {row["id"]: row for row in payload["providers"]}
    assert by_id["xai"]["connected"] is True
    assert by_id["xai"]["access_path"] == "xai-chat"
    assert by_id["xai"]["discovery"] == "xai"
    assert by_id["openai-api"]["connected"] is False
    assert by_id["openai-api"]["state"] == "unconfigured"
    assert payload["default_system_prompt"] == setup_routes.DEFAULT_FIRST_BOT_PROMPT
    assert payload["default_context_window"] == setup_routes.DEFAULT_CONTEXT_WINDOW


def test_setup_providers_survives_one_failing_adapter():
    broken = _adapter()
    broken.health.side_effect = RuntimeError("boom")
    adapters = {"xai": broken, "openai-api": _adapter("ok", "OpenAI")}

    with (
        patch.object(setup_routes, "get_service", return_value=_service()),
        patch.object(setup_routes, "get_adapter", side_effect=lambda _c, pid: adapters.get(pid)),
    ):
        payload = _client().get("/v1/setup/providers").json()

    by_id = {row["id"]: row for row in payload["providers"]}
    assert by_id["xai"]["connected"] is False
    assert by_id["xai"]["state"] == "unknown"
    assert by_id["openai-api"]["connected"] is True


# ---------------------------------------------------------------------------
# POST /v1/setup/seed (infrastructure only)
# ---------------------------------------------------------------------------


def test_setup_seed_uses_service_config_and_returns_report():
    service = _service()
    report = {"catalog_migration": {}, "prompts": {"created": [], "existing": []}}

    with (
        patch.object(setup_routes, "get_service", return_value=service),
        patch.object(setup_routes, "seed_tenant", return_value=report) as seed,
    ):
        response = _client().post("/v1/setup/seed")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "report": report, "preferences": {}}
    seed.assert_called_once_with(service.config)


def test_setup_seed_persists_network_and_voice_preferences():
    store = MagicMock()
    store.engine = MagicMock()

    with (
        patch.object(setup_routes, "get_service", return_value=_service()),
        patch.object(setup_routes, "get_runtime_settings_store", return_value=store),
        patch.object(setup_routes, "seed_tenant", return_value={}),
    ):
        response = _client().post(
            "/v1/setup/seed",
            json={
                "exposure": "public",
                "domains": ["App.Example.com", "app.example.com"],
                "voice_enabled": False,
            },
        )

    assert response.status_code == 200
    assert response.json()["preferences"] == {
        "tenant.exposure": "public",
        "tenant.domains": ["app.example.com"],
        "tenant.voice_enabled": False,
    }
    assert store.set_value.call_args_list == [
        (("global", "*", "tenant.exposure", "public"),),
        (("global", "*", "tenant.domains", ["app.example.com"]),),
        (("global", "*", "tenant.voice_enabled", False),),
    ]


def test_setup_seed_rejects_invalid_domains():
    response = _client().post("/v1/setup/seed", json={"domains": ["not a domain"]})
    assert response.status_code == 422


def test_setup_seed_reports_database_unavailable_as_503():
    with (
        patch.object(setup_routes, "get_service", return_value=_service()),
        patch.object(
            setup_routes,
            "seed_tenant",
            side_effect=RuntimeError("Tenant seed requires database credentials"),
        ),
    ):
        response = _client().post("/v1/setup/seed")

    assert response.status_code == 503
    assert response.json()["detail"] == "Tenant seed requires database credentials"


def test_setup_seed_hides_unexpected_internal_error():
    with (
        patch.object(setup_routes, "get_service", return_value=_service()),
        patch.object(setup_routes, "seed_tenant", side_effect=KeyError("secret detail")),
    ):
        response = _client().post("/v1/setup/seed")

    assert response.status_code == 500
    assert response.json()["detail"] == "Tenant seed failed"


# ---------------------------------------------------------------------------
# POST /v1/setup/first-bot
# ---------------------------------------------------------------------------


@contextmanager
def _first_bot_env(adapter, *, seeder=None, persist=None):
    """Patch every collaborator of ``/v1/setup/first-bot``; yield the mocks."""
    seeder = seeder or MagicMock()
    if not isinstance(seeder.apply.side_effect, Exception) and seeder.apply.side_effect is None:
        seeder.apply.return_value = {"catalog_migration": {"ok": True}, "prompts": {"created": []}}
    if seeder.register_model.side_effect is None:
        seeder.register_model.return_value = {
            "models": {"created": ["grok-4.3"], "existing": []},
            "endpoints": {"created": ["grok-4.3@xai-chat"], "existing": []},
            "endpoint_id": 7,
        }
    persist = persist or AsyncMock(return_value={"slug": "helper", "name": "Helper"})
    with (
        patch.object(setup_routes, "get_service", return_value=_service()),
        patch.object(setup_routes, "get_adapter", return_value=adapter),
        patch.object(setup_routes, "TenantSeeder", return_value=seeder),
        patch.object(setup_routes, "reload_models_catalog") as reload,
        patch.object(setup_routes, "_persist_bot_profile", persist),
    ):
        yield seeder, persist, reload


def test_first_bot_happy_path_orders_infra_model_reload_then_bot():
    order: list[str] = []
    seeder = MagicMock()
    seeder.apply.side_effect = lambda: order.append("infra") or {"catalog_migration": {}, "prompts": {}}
    seeder.register_model.side_effect = lambda **kw: order.append("model") or {
        "models": {"created": ["grok-4.3"], "existing": []},
        "endpoints": {"created": ["grok-4.3@xai-chat"], "existing": []},
        "endpoint_id": 7,
    }

    async def _persist(payload, *, create_only):
        order.append("bot")
        return {"slug": payload["slug"], "endpoint_id": payload["endpoint_id"]}

    with _first_bot_env(_adapter("ok"), seeder=seeder, persist=AsyncMock(side_effect=_persist)) as (_, persist, reload):
        reload.side_effect = lambda: order.append("reload")
        response = _client().post("/v1/setup/first-bot", json=_FIRST_BOT)

    assert response.status_code == 200, response.text
    assert order == ["infra", "model", "reload", "bot"]
    seeder.register_model.assert_called_once_with(
        vendor="xai",
        model_id="grok-4.3",
        access_path_key="xai-chat",
        display_name=None,
        context_window=131072,
    )
    payload = persist.call_args.args[0]
    assert persist.call_args.kwargs == {"create_only": True}
    assert payload["slug"] == "helper"
    assert payload["harness"] == "chat"
    assert payload["bot_type"] == "chat"
    assert payload["endpoint_id"] == 7
    assert payload["uses_tools"] is True and payload["uses_search"] is True
    body = response.json()
    assert body["ok"] is True
    assert body["bot"] == {"slug": "helper", "endpoint_id": 7}
    assert body["report"]["models"]["created"] == ["grok-4.3"]


def test_first_bot_refuses_when_provider_not_connected():
    with _first_bot_env(_adapter("unconfigured")) as (seeder, persist, _):
        response = _client().post("/v1/setup/first-bot", json=_FIRST_BOT)

    assert response.status_code == 409
    assert response.json()["detail"] == "Connect xAI before creating the first bot"
    seeder.apply.assert_not_called()
    persist.assert_not_called()


def test_first_bot_warning_state_still_counts_as_connected():
    with _first_bot_env(_adapter("warning")) as (seeder, _, _):
        response = _client().post("/v1/setup/first-bot", json=_FIRST_BOT)

    assert response.status_code == 200
    seeder.apply.assert_called_once()


def test_first_bot_rejects_provider_that_cannot_back_a_chat_bot():
    with _first_bot_env(_adapter("ok")) as (seeder, _, _):
        response = _client().post("/v1/setup/first-bot", json={**_FIRST_BOT, "provider_id": "anthropic-api"})

    assert response.status_code == 400
    assert "cannot back the first bot" in response.json()["detail"]
    seeder.apply.assert_not_called()


def test_first_bot_validates_slug_and_prompt_shape():
    client = _client()
    assert client.post("/v1/setup/first-bot", json={**_FIRST_BOT, "slug": "Bad Slug"}).status_code == 422
    assert client.post("/v1/setup/first-bot", json={**_FIRST_BOT, "system_prompt": ""}).status_code == 422
    assert client.post("/v1/setup/first-bot", json={**_FIRST_BOT, "context_window": 0}).status_code == 422


def test_first_bot_passes_through_duplicate_slug_conflict():
    persist = AsyncMock(side_effect=HTTPException(status_code=409, detail="Bot 'helper' already exists"))
    with _first_bot_env(_adapter("ok"), persist=persist):
        response = _client().post("/v1/setup/first-bot", json=_FIRST_BOT)

    assert response.status_code == 409
    assert response.json()["detail"] == "Bot 'helper' already exists"


def test_first_bot_maps_engine_errors():
    seeder = MagicMock()
    seeder.apply.side_effect = RuntimeError("Tenant seed requires database credentials")
    with _first_bot_env(_adapter("ok"), seeder=seeder):
        response = _client().post("/v1/setup/first-bot", json=_FIRST_BOT)
    assert response.status_code == 503

    seeder.apply.side_effect = None
    seeder.apply.return_value = {}
    seeder.register_model.side_effect = ValueError("model_id is required")
    with _first_bot_env(_adapter("ok"), seeder=seeder):
        response = _client().post("/v1/setup/first-bot", json=_FIRST_BOT)
    assert response.status_code == 400
    assert response.json()["detail"] == "model_id is required"


def test_first_bot_persists_preferences_after_bot_create():
    store = MagicMock()
    store.engine = MagicMock()
    with (
        _first_bot_env(_adapter("ok")),
        patch.object(setup_routes, "get_runtime_settings_store", return_value=store),
    ):
        response = _client().post(
            "/v1/setup/first-bot",
            json={**_FIRST_BOT, "exposure": "private", "voice_enabled": True},
        )

    assert response.status_code == 200
    assert response.json()["preferences"] == {"tenant.exposure": "private", "tenant.voice_enabled": True}
