import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, select

from llm_bawt.integrations.google_home_probe import (
    DEVICE_ID, GoogleHomeProbe, ProbeRequest,
)
from llm_bawt.integrations.google_home_probe_store import ProbeCaptureStore, _captures
from llm_bawt.service.routes import google_home_probe as route


from llm_bawt.integrations.google_home_oauth import ProbeUnauthorized


class Captures:
    def __init__(self):
        self.rows = {}

    def record(self, key, payload):
        self.rows[key] = json.loads(payload)


def message(intent, payload=None):
    return {"requestId": "test-1", "inputs": [{"intent": "action.devices." + intent, "payload": payload or {}}]}


def test_sync_declares_only_probe():
    result = GoogleHomeProbe(Captures()).handle("owner", ProbeRequest.model_validate(message("SYNC")))
    devices = result["payload"]["devices"]
    assert [d["id"] for d in devices] == [DEVICE_ID]
    assert devices[0]["traits"] == ["action.devices.traits.AppSelector"]
    assert devices[0]["name"]["name"] == "BawtHub TV"
    apps = devices[0]["attributes"]["availableApplications"]
    assert apps[0]["key"] == "youtube"
    assert "YouTube" in apps[0]["names"][0]["name_synonym"]
    query = GoogleHomeProbe(Captures()).handle("owner", ProbeRequest.model_validate(
        message("QUERY", {"devices": [{"id": DEVICE_ID}]})))
    assert query["payload"]["devices"][DEVICE_ID]["currentApplication"] == "youtube"


def test_captures_exact_text_without_executing_or_custom_data():
    captures = Captures()
    text = "check why downloads are slow — don't execute this"
    req = ProbeRequest.model_validate(message("EXECUTE", {"commands": [{
        "devices": [{"id": DEVICE_ID, "customData": {"secret": "omit"}}],
        "execution": [{"command": "action.devices.commands.appSearch", "params": {"newApplicationName": text}}],
    }]}))
    probe = GoogleHomeProbe(captures)
    result = probe.handle("owner", req)
    assert result["payload"]["commands"][0]["status"] == "ERROR"
    event = next(iter(captures.rows.values()))
    group = event["inputs"][0]["payload"]["commands"][0]
    assert group["execution"][0]["params"]["newApplicationName"] == text
    assert "customData" not in group["devices"][0]
    assert req.inputs[0].payload.commands[0].devices[0].customData == {"secret": "omit"}


def test_unknown_device_is_not_success():
    result = GoogleHomeProbe(Captures()).handle("owner", ProbeRequest.model_validate(message("QUERY", {"devices": [{"id": "other"}]})))
    assert result["payload"]["devices"]["other"]["errorCode"] == "deviceNotFound"


def test_invalid_protocol_is_rejected():
    with pytest.raises(ValidationError):
        ProbeRequest.model_validate(message("UNKNOWN"))


def test_encrypted_storage_deduplicates_and_prunes(monkeypatch):
    from llm_bawt.service.providers import crypto
    monkeypatch.setattr(crypto, "encrypt", lambda s: "encrypted:" + s)
    monkeypatch.setattr(crypto, "decrypt", lambda s: s.removeprefix("encrypted:"))
    store = ProbeCaptureStore(create_engine("sqlite://"))
    store.record("same", '{"requestId":"one"}')
    store.record("same", '{"requestId":"two"}')
    assert store.recent()[0]["payload"]["requestId"] == "one"
    with store.engine.begin() as conn:
        assert conn.execute(select(_captures.c.payload_enc)).scalar().startswith("encrypted:")
        conn.execute(_captures.update().values(created_at=time.time() - 90000))
    assert store.recent() == []
    store.record("new", '{}')
    with store.engine.connect() as conn:
        assert len(conn.execute(select(_captures)).all()) == 1


@pytest.fixture
def client(monkeypatch):
    class Auth:
        async def authenticate(self, header, http=None):
            if header != "Bearer good":
                raise ProbeUnauthorized()
            return "owner"
    captures = Captures()
    monkeypatch.setattr(route, "get_probe", lambda: GoogleHomeProbe(captures))
    app = FastAPI()
    app.include_router(route.router)
    app.dependency_overrides[route.get_probe_auth] = lambda: Auth()
    return TestClient(app), captures


def test_rejection_diagnostics_never_log_values_or_unknown_keys(client, caplog):
    http, _ = client
    body = message("EXECUTE", {"commands": [{
        "devices": [{"id": DEVICE_ID}],
        "execution": [{"command": "action.devices.commands.OnOff", "params": {"on": True}}],
    }]})
    body["sensitive-extra-key"] = "sensitive-value"
    result = http.post("/integrations/google-home/fulfillment", json=body,
                       headers={"Authorization": "Bearer good"})
    assert result.status_code == 400
    assert "string_type" in caplog.text
    assert "extra_forbidden" in caplog.text
    assert '"on"' in caplog.text
    assert "sensitive" not in caplog.text
    assert "Bearer good" not in caplog.text
    assert "True" not in caplog.text


@pytest.mark.parametrize("intent", ["SYNC", "QUERY", "EXECUTE"])
def test_live_intent_context_accepted_but_not_captured(client, intent):
    http, captures = client
    payload = ({"commands": [{"devices": [{"id": DEVICE_ID}], "execution": [{
        "command": "action.devices.commands.appSelect", "params": {"newApplication": "youtube"},
    }]}]} if intent == "EXECUTE" else {})
    body = message(intent, payload)
    body["inputs"][0]["context"] = {"metadata": {"private": "do-not-retain"}}
    response = http.post("/integrations/google-home/fulfillment", json=body,
                         headers={"Authorization": "Bearer good"})
    assert response.status_code == 200
    event = next(iter(captures.rows.values()))
    assert "context" not in event["inputs"][0]
    assert "do-not-retain" not in json.dumps(event)
    if intent == "EXECUTE":
        assert response.json()["payload"]["commands"][0]["status"] == "ERROR"
        assert event["inputs"][0]["payload"]["commands"][0]["execution"][0]["params"] == {"newApplication": "youtube"}


def test_http_boundary(client):
    http, captures = client
    url = "/integrations/google-home/fulfillment"
    assert http.post(url, json=message("SYNC")).status_code == 401
    assert not captures.rows
    headers = {"Authorization": "Bearer good"}
    assert http.post(url, json=message("SYNC"), headers=headers).status_code == 200
    assert http.post(url, content=b"{}", headers=headers).status_code == 415
    headers["Content-Type"] = "application/json"
    assert http.post(url, content=b"broken", headers=headers).status_code == 400
    assert http.post(url, content=b"x" * 32769, headers=headers).status_code == 413
    assert http.get(url).status_code == 405
