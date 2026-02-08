"""Background job scheduler with database persistence."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel, Session, select
from sqlalchemy import Column, DateTime, Text

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a job run."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobType(str, Enum):
    """Types of scheduled jobs."""
    PROFILE_MAINTENANCE = "profile_maintenance"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    MEMORY_DECAY = "memory_decay"


class ScheduledJob(SQLModel, table=True):
    """Definition of a recurring job."""
    __tablename__ = "scheduled_jobs"
    
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    job_type: JobType = Field(index=True)
    bot_id: str = Field(index=True, description="Bot this job runs for, or '*' for all")
    enabled: bool = Field(default=True)
    interval_minutes: int = Field(default=60)
    last_run_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    next_run_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column=Column(DateTime(timezone=True)))
    config_json: Optional[str] = Field(default=None, sa_column=Column(Text), description="Job-specific config as JSON")


class JobRun(SQLModel, table=True):
    """History of job executions."""
    __tablename__ = "job_runs"
    
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    job_id: str = Field(index=True, foreign_key="scheduled_jobs.id")
    bot_id: str = Field(index=True)
    status: JobStatus = Field(default=JobStatus.PENDING)
    started_at: datetime = Field(default_factory=datetime.utcnow, sa_column=Column(DateTime(timezone=True)))
    finished_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    duration_ms: Optional[int] = Field(default=None)
    result_json: Optional[str] = Field(default=None, sa_column=Column(Text), description="Success result as JSON")
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text))


def create_scheduler_tables(engine) -> None:
    """Create scheduler tables if they don't exist."""
    SQLModel.metadata.create_all(engine, tables=[ScheduledJob.__table__, JobRun.__table__])


class JobScheduler:
    """
    Async scheduler that watches for due jobs and enqueues them.
    
    Usage:
        scheduler = JobScheduler(engine, task_processor)
        await scheduler.start()  # Runs until stopped
        await scheduler.stop()
    """
    
    def __init__(self, engine, task_processor, check_interval: int = 30):
        """
        Args:
            engine: SQLAlchemy engine for job tables
            task_processor: TaskProcessor instance to enqueue tasks
            check_interval: Seconds between checking for due jobs
        """
        self.engine = engine
        self.task_processor = task_processor
        self.check_interval = check_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.debug(f"Scheduler started (check interval: {self.check_interval}s)")
    
    async def stop(self) -> None:
        """Stop the scheduler loop gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.debug("Scheduler stopped")
    
    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self._check_and_run_due_jobs()
            except Exception as e:
                logger.exception(f"Scheduler loop error: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    async def _check_and_run_due_jobs(self) -> None:
        """Check for due jobs and enqueue them."""
        loop = asyncio.get_event_loop()
        
        def get_due_jobs():
            with Session(self.engine) as session:
                now = datetime.utcnow()
                statement = select(ScheduledJob).where(
                    ScheduledJob.enabled.is_(True),
                    (ScheduledJob.next_run_at.is_(None)) | (ScheduledJob.next_run_at <= now)
                )
                return session.exec(statement).all()
        
        due_jobs = await loop.run_in_executor(None, get_due_jobs)
        
        for job in due_jobs:
            await self._execute_job(job)
    
    async def _execute_job(self, job: ScheduledJob) -> None:
        """Execute a single job."""
        logger.debug(f"Running scheduled job: {job.job_type}")
        
        loop = asyncio.get_event_loop()
        
        # Create job run record
        def create_run():
            with Session(self.engine) as session:
                run = JobRun(
                    job_id=job.id,
                    bot_id=job.bot_id,
                    status=JobStatus.RUNNING,
                )
                session.add(run)
                session.commit()
                session.refresh(run)
                return run.id
        
        run_id = await loop.run_in_executor(None, create_run)
        start_time = datetime.utcnow()
        
        try:
            # Create and execute task based on job type
            task = self._create_task_for_job(job)
            if task:
                task_result = await self.task_processor.process_task(task)
                # TaskResult has .result (dict) and .status
                from llm_bawt.service.tasks import TaskStatus as TStatus
                if task_result.status == TStatus.COMPLETED:
                    result_data = task_result.result or {}
                    status = JobStatus.SUCCESS if result_data.get("error") is None else JobStatus.FAILED
                    result_json = json.dumps(result_data)
                    error_message = result_data.get("error")
                else:
                    status = JobStatus.FAILED
                    result_json = None
                    error_message = task_result.error
            else:
                status = JobStatus.SKIPPED
                result_json = None
                error_message = f"Unknown job type: {job.job_type}"
        except Exception as e:
            status = JobStatus.FAILED
            result_json = None
            error_message = str(e)
            logger.exception(f"Job {job.id} failed")
        
        # Update job run record
        finish_time = datetime.utcnow()
        duration_ms = int((finish_time - start_time).total_seconds() * 1000)
        
        def update_run():
            with Session(self.engine) as session:
                run = session.get(JobRun, run_id)
                if run:
                    run.status = status
                    run.finished_at = finish_time
                    run.duration_ms = duration_ms
                    run.result_json = result_json
                    run.error_message = error_message
                    session.add(run)
                
                # Update job's next_run_at
                db_job = session.get(ScheduledJob, job.id)
                if db_job:
                    db_job.last_run_at = start_time
                    db_job.next_run_at = start_time + timedelta(minutes=db_job.interval_minutes)
                    session.add(db_job)
                
                session.commit()
        
        await loop.run_in_executor(None, update_run)
        logger.debug(f"Job {job.job_type} completed with status {status}")
    
    def _create_task_for_job(self, job: ScheduledJob):
        """Create a Task from a ScheduledJob."""
        from llm_bawt.service.tasks import create_profile_maintenance_task, create_maintenance_task
        
        if job.job_type == JobType.PROFILE_MAINTENANCE:
            # For profile maintenance, entity_id comes from config or is the bot's user
            config = json.loads(job.config_json) if job.config_json else {}
            entity_id = config.get("entity_id", "user")  # Default user
            return create_profile_maintenance_task(
                entity_id=entity_id,
                entity_type=config.get("entity_type", "user"),
                bot_id=job.bot_id,
            )
        elif job.job_type == JobType.MEMORY_CONSOLIDATION:
            config = json.loads(job.config_json) if job.config_json else {}
            entity_id = config.get("entity_id", "system")
            return create_maintenance_task(
                bot_id=job.bot_id,
                user_id=entity_id,
                run_consolidation=True,
                run_recurrence_detection=False,
                run_decay_pruning=False,
            )
        elif job.job_type == JobType.MEMORY_DECAY:
            config = json.loads(job.config_json) if job.config_json else {}
            entity_id = config.get("entity_id", "system")
            return create_maintenance_task(
                bot_id=job.bot_id,
                user_id=entity_id,
                run_consolidation=False,
                run_recurrence_detection=False,
                run_decay_pruning=True,
            )
        
        return None


def init_default_jobs(engine, config) -> None:
    """Initialize default scheduled jobs if they don't exist."""
    with Session(engine) as session:
        # Check if profile maintenance job exists
        existing = session.exec(
            select(ScheduledJob).where(ScheduledJob.job_type == JobType.PROFILE_MAINTENANCE)
        ).first()
        
        if not existing:
            default_user = config.DEFAULT_USER or "user"
            job = ScheduledJob(
                job_type=JobType.PROFILE_MAINTENANCE,
                bot_id="*",  # All bots
                enabled=config.SCHEDULER_ENABLED,
                interval_minutes=config.PROFILE_MAINTENANCE_INTERVAL_MINUTES,
                config_json=json.dumps({"entity_id": default_user, "entity_type": "user"}),
            )
            session.add(job)
            session.commit()
            logger.debug("Created default profile maintenance job")
