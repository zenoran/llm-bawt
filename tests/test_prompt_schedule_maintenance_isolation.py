"""Generic maintenance APIs must not expose/trigger user-owned prompt jobs."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

from llm_bawt.service.prompt_schedule_store import PromptScheduleStore
from llm_bawt.service.prompt_timing import PromptTiming
from llm_bawt.service.routes import jobs
from llm_bawt.service.scheduler import create_scheduler_tables


@pytest.fixture
def api(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    create_scheduler_tables(engine)
    store = PromptScheduleStore(engine)
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    job = store.create(owner="private-owner", bot_id="test", name="Private", prompt="Private content",
                       timing=PromptTiming(kind="once", once_at=now + timedelta(minutes=1)), now=now)
    store.reserve_due(now + timedelta(minutes=1))
    monkeypatch.setattr(jobs, "_get_scheduler_engine", lambda: engine)
    app = FastAPI()
    app.include_router(jobs.router)
    with TestClient(app) as client:
        yield client, job
    engine.dispose()


def test_generic_lists_hide_prompt_jobs_and_runs(api):
    client, job = api
    assert client.get("/v1/jobs").json()["jobs"] == []
    assert client.get("/v1/jobs/runs").json()["runs"] == []
    assert client.get(f"/v1/jobs/runs?job_id={job}&include_result=true").json()["runs"] == []


def test_generic_trigger_cannot_corrupt_prompt_cadence(api):
    client, _ = api
    assert client.post("/v1/jobs/send_prompt/trigger").status_code == 400
    assert client.get("/v1/jobs?job_type=send_prompt").status_code == 400
    assert client.get("/v1/jobs/runs?job_type=send_prompt").status_code == 400
