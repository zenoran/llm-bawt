"""Tests for the background job scheduler."""

from datetime import datetime

from llm_bawt.service.scheduler import (
    JobScheduler,
    ScheduledJob,
    JobRun,
    JobType,
    JobStatus,
)


class TestScheduledJobModel:
    """Tests for ScheduledJob SQLModel."""
    
    def test_scheduled_job_creation_defaults(self):
        job = ScheduledJob(
            job_type=JobType.PROFILE_MAINTENANCE,
            bot_id="test",
        )
        assert job.enabled is True
        assert job.interval_minutes == 60
        assert job.job_type == JobType.PROFILE_MAINTENANCE
        assert job.bot_id == "test"
        assert job.last_run_at is None
        assert job.next_run_at is None
    
    def test_scheduled_job_custom_interval(self):
        job = ScheduledJob(
            job_type=JobType.MEMORY_CONSOLIDATION,
            bot_id="nova",
            interval_minutes=120,
        )
        assert job.interval_minutes == 120
        assert job.job_type == JobType.MEMORY_CONSOLIDATION
    
    def test_scheduled_job_can_be_disabled(self):
        job = ScheduledJob(
            job_type=JobType.PROFILE_MAINTENANCE,
            bot_id="test",
            enabled=False,
        )
        assert job.enabled is False


class TestJobRunModel:
    """Tests for JobRun SQLModel."""
    
    def test_job_run_creation_defaults(self):
        run = JobRun(
            job_id="test-job-id",
            bot_id="nova",
        )
        assert run.status == JobStatus.PENDING
        assert run.job_id == "test-job-id"
        assert run.bot_id == "nova"
        assert run.finished_at is None
        assert run.duration_ms is None
        assert run.result_json is None
        assert run.error_message is None
    
    def test_job_run_status_values(self):
        """Verify all status values work."""
        for status in [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.SKIPPED]:
            run = JobRun(job_id="test", bot_id="test", status=status)
            assert run.status == status


class TestJobTypes:
    """Tests for JobType enum."""
    
    def test_job_types_available(self):
        assert JobType.PROFILE_MAINTENANCE.value == "profile_maintenance"
        assert JobType.MEMORY_CONSOLIDATION.value == "memory_consolidation"
        assert JobType.MEMORY_DECAY.value == "memory_decay"
        assert JobType.HISTORY_SUMMARIZATION.value == "history_summarization"


class TestTaskFactories:
    """Tests for task factory functions."""
    
    def test_create_profile_maintenance_task(self):
        from llm_bawt.service.tasks import create_profile_maintenance_task, TaskType
        
        task = create_profile_maintenance_task(
            entity_id="user",
            entity_type="user",
            bot_id="nova",
        )
        assert task.task_type == TaskType.PROFILE_MAINTENANCE
        assert task.payload["entity_id"] == "user"
        assert task.payload["entity_type"] == "user"
        assert task.bot_id == "nova"
        assert task.user_id == "user"  # entity_id becomes user_id
    
    def test_create_maintenance_task_with_user_id(self):
        from llm_bawt.service.tasks import create_maintenance_task, TaskType
        
        task = create_maintenance_task(
            bot_id="nova",
            user_id="system",
            run_consolidation=True,
        )
        assert task.task_type == TaskType.MEMORY_MAINTENANCE
        assert task.bot_id == "nova"
        assert task.user_id == "system"
        assert task.payload["run_consolidation"] is True

    def test_create_history_summarization_task(self):
        from llm_bawt.service.tasks import create_history_summarization_task, TaskType

        task = create_history_summarization_task(
            bot_id="nova",
            user_id="system",
            use_heuristic_fallback=False,
            max_tokens_per_chunk=2048,
            model="grok-4-fast",
        )
        assert task.task_type == TaskType.HISTORY_SUMMARIZATION
        assert task.bot_id == "nova"
        assert task.user_id == "system"
        assert task.payload["use_heuristic_fallback"] is False
        assert task.payload["max_tokens_per_chunk"] == 2048
        assert task.payload["model"] == "grok-4-fast"


class TestJobActivityGates:
    """Tests for scheduler activity-gating logic."""

    def test_history_job_skips_when_no_new_messages(self):
        scheduler = JobScheduler(engine=None, task_processor=None)
        job = ScheduledJob(
            job_type=JobType.HISTORY_SUMMARIZATION,
            bot_id="nova",
            last_run_at=datetime.utcnow(),
        )

        scheduler._max_timestamp_message_activity = lambda _bot_id: None  # type: ignore[method-assign]
        should_run, reason = scheduler._has_new_activity_for_job(job)

        assert should_run is False
        assert reason == "No conversation messages found"

    def test_history_job_runs_when_messages_are_newer_than_last_run(self):
        scheduler = JobScheduler(engine=None, task_processor=None)
        last_run = datetime.utcnow()
        job = ScheduledJob(
            job_type=JobType.HISTORY_SUMMARIZATION,
            bot_id="nova",
            last_run_at=last_run,
        )

        scheduler._max_timestamp_message_activity = lambda _bot_id: last_run.timestamp() + 1  # type: ignore[method-assign]
        should_run, reason = scheduler._has_new_activity_for_job(job)

        assert should_run is True
        assert reason == "New conversation activity detected"

    def test_pending_probe_uses_trigger_budget_not_model_window(self, monkeypatch):
        """Regression: the pending-summary probe must size its eligibility budget
        from the summarization TRIGGER (default 12000), mirroring
        _process_history_summarization — NOT the model context window.

        Deriving the budget from the model window made this probe ~10x more
        lenient than execution, so any bot whose unsummarized backlog sat between
        the trigger and the window reported "0 pending" and was skipped every
        cycle (backlog starvation). Also pins the per-bot model resolution to the
        real bot_id rather than the previously hardcoded "nova".
        """
        import llm_bawt.bots as bots_mod
        import llm_bawt.memory.summarization as summ_mod
        from llm_bawt.service import scheduler as sched_mod

        captured: dict = {}

        class _Config:
            MAINTENANCE_MODEL = "grok-4-fast"
            SUMMARIZATION_MODEL = ""
            EXTRACTION_MODEL = ""
            SUMMARIZATION_TRIGGER_TOKENS = 12000

            def get_model_context_window(self, _alias=None):
                return 128000  # model window — must NOT be used as the budget

            def get_model_max_tokens(self, _alias=None):
                return 4096

        class _TaskProc:
            config = _Config()

            def _resolve_request_model(self, _preferred, bot_id=None, local_mode=False):
                return ("grok-4-fast", None)

        class _Bot:
            pass

        class _BotManager:
            def __init__(self, _config):
                pass

            def get_bot(self, _bot_id):
                return _Bot()

            def get_default_bot(self):
                return _Bot()

        class _Resolver:
            def __init__(self, **_kwargs):
                pass

            def resolve(self, _key, fallback):
                return fallback  # summarization_trigger_tokens -> 12000

        class _Summarizer:
            def __init__(self, _config, bot_id=None, settings_getter=None, max_context_tokens=0):
                captured["max_context_tokens"] = max_context_tokens
                captured["bot_id"] = bot_id

            def preview_summarizable_sessions(self):
                return ["session-a", "session-b"]

        monkeypatch.setattr(bots_mod, "BotManager", _BotManager)
        monkeypatch.setattr(summ_mod, "HistorySummarizer", _Summarizer)
        monkeypatch.setattr(sched_mod, "RuntimeSettingsResolver", _Resolver)

        scheduler = JobScheduler(engine=None, task_processor=_TaskProc())
        has_pending, count = scheduler._has_pending_history_summaries("loopy", {})

        # Budget is the trigger (12000), NOT the model window (128000 - 4096).
        assert captured["max_context_tokens"] == 12000
        # Model resolution uses the real bot, not the hardcoded "nova".
        assert captured["bot_id"] == "loopy"
        assert has_pending is True
        assert count == 2

    def test_profile_maintenance_skips_without_profile_changes(self):
        scheduler = JobScheduler(engine=None, task_processor=None)
        last_run = datetime.utcnow()
        job = ScheduledJob(
            job_type=JobType.PROFILE_MAINTENANCE,
            bot_id="*",
            last_run_at=last_run,
            config_json='{"entity_id":"user","entity_type":"user"}',
        )

        scheduler._profile_needs_rebuild = lambda _entity_id, _entity_type: (False, "No profile attribute changes since last summary build")  # type: ignore[method-assign]
        should_run, reason = scheduler._has_new_activity_for_job(job)

        assert should_run is False
        assert reason == "No profile attribute changes since last summary build"
