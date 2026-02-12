"""Tests for the background job scheduler."""

import pytest
from datetime import datetime, timedelta

from llm_bawt.service.scheduler import (
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
