from datetime import datetime, timedelta, timezone

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from llm_bawt.service.prompt_retention import prune_prompt_runs
from llm_bawt.service.prompt_schedule_models import PromptOccurrence
from llm_bawt.service.scheduler import JobRun, JobType, ScheduledJob, create_scheduler_tables


def test_retention_preserves_latest_and_unknown():
    engine = create_engine("sqlite://", poolclass=StaticPool)
    create_scheduler_tables(engine)
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        session.add(ScheduledJob(id="test", job_type=JobType.SEND_PROMPT, bot_id="test"))
        session.commit()
        for key, days, state in [("old", 110, "succeeded"), ("uncertain", 100, "unknown"), ("latest", 95, "succeeded")]:
            session.add(JobRun(id=key, job_id="test", bot_id="test"))
            session.flush()
            session.add(PromptOccurrence(run_id=key, job_id="test", occurrence_key=key,
                scheduled_for=now-timedelta(days=days), created_at=now-timedelta(days=days),
                updated_at=now-timedelta(days=days), schedule_revision=1, snapshot_json={}, state=state))
        session.commit()
    assert prune_prompt_runs(engine, now) == 1
    with Session(engine) as session:
        assert session.get(JobRun, "old") is None
        assert session.get(PromptOccurrence, "uncertain") is not None
        assert session.get(PromptOccurrence, "latest") is not None
    engine.dispose()
