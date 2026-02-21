"""Background job scheduler with database persistence."""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel, Session, select
from sqlalchemy import Column, DateTime, Text, text
from ..runtime_settings import RuntimeSettingsResolver

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
    HISTORY_SUMMARIZATION = "history_summarization"


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
    _migrate_postgres_jobtype_enum(engine)


def _migrate_postgres_jobtype_enum(engine) -> None:
    """Ensure Postgres jobtype enum includes all currently supported labels."""
    if engine.dialect.name != "postgresql":
        return

    with engine.begin() as conn:
        type_exists = conn.execute(
            text("SELECT 1 FROM pg_type WHERE typname = 'jobtype'")
        ).scalar()
        if not type_exists:
            return

        labels = {
            row[0]
            for row in conn.execute(
                text(
                    """
                    SELECT e.enumlabel
                    FROM pg_enum e
                    JOIN pg_type t ON t.oid = e.enumtypid
                    WHERE t.typname = 'jobtype'
                    """
                )
            )
        }

        needed_label = JobType.HISTORY_SUMMARIZATION.name
        if needed_label not in labels:
            safe_label = needed_label.replace("'", "''")
            conn.execute(text(f"ALTER TYPE jobtype ADD VALUE IF NOT EXISTS '{safe_label}'"))
            logger.info("Added missing enum value to jobtype: %s", needed_label)


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
        status = JobStatus.RUNNING
        result_json: str | None = None
        error_message: str | None = None
        
        try:
            should_run, skip_reason = await loop.run_in_executor(
                None,
                lambda: self._has_new_activity_for_job(job),
            )
            if not should_run:
                status = JobStatus.SKIPPED
                result_json = json.dumps(
                    {
                        "skipped": True,
                        "reason": skip_reason or "No new activity since last run",
                    }
                )
                logger.debug(
                    "Skipping scheduled job %s for bot %s: %s",
                    job.job_type,
                    job.bot_id,
                    skip_reason or "no new activity",
                )
            else:
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

    @staticmethod
    def _sanitize_identifier(identifier: str) -> str:
        """Sanitize table identifier to lowercase alnum/underscore."""
        return re.sub(r"[^a-z0-9_]", "", (identifier or "").lower())

    @staticmethod
    def _parse_config_json(config_json: str | None) -> dict:
        """Parse job config JSON safely."""
        if not config_json:
            return {}
        try:
            return json.loads(config_json)
        except Exception:
            return {}

    def _max_timestamp_message_activity(self, bot_id: str) -> float | None:
        """Return latest non-summary message timestamp for a bot."""
        table_bot = self._sanitize_identifier(bot_id)
        if not table_bot:
            return None
        table_name = f"{table_bot}_messages"
        sql = text(
            f"""
            SELECT MAX(timestamp) AS latest_ts
            FROM {table_name}
            WHERE role != 'summary'
              AND COALESCE(recalled_history, FALSE) = FALSE
            """
        )
        with self.engine.connect() as conn:
            row = conn.execute(sql).fetchone()
            if not row:
                return None
            latest_ts = row[0]
            if latest_ts is None:
                return None
            return float(latest_ts)

    def _max_memory_updated_at(self, bot_id: str) -> datetime | None:
        """Return latest memory update time for a bot."""
        table_bot = self._sanitize_identifier(bot_id)
        if not table_bot:
            return None
        table_name = f"{table_bot}_memories"
        sql = text(f"SELECT MAX(updated_at) AS latest_updated_at FROM {table_name}")
        with self.engine.connect() as conn:
            row = conn.execute(sql).fetchone()
            if not row:
                return None
            latest = row[0]
            if isinstance(latest, datetime):
                return latest
            return None

    def _max_profile_updated_at(self, entity_id: str, entity_type: str) -> datetime | None:
        """Return latest profile/attribute update time for an entity."""
        safe_entity_id = (entity_id or "").strip().lower()
        safe_entity_type = (entity_type or "user").strip().lower()
        if not safe_entity_id:
            return None

        latest: datetime | None = None
        with self.engine.connect() as conn:
            attr_row = conn.execute(
                text(
                    """
                    SELECT MAX(updated_at) AS latest_updated_at
                    FROM entity_profile_attributes
                    WHERE entity_id = :entity_id
                      AND entity_type = :entity_type
                    """
                ),
                {"entity_id": safe_entity_id, "entity_type": safe_entity_type},
            ).fetchone()
            profile_row = conn.execute(
                text(
                    """
                    SELECT MAX(updated_at) AS latest_updated_at
                    FROM entity_profiles
                    WHERE entity_id = :entity_id
                      AND entity_type = :entity_type
                    """
                ),
                {"entity_id": safe_entity_id, "entity_type": safe_entity_type},
            ).fetchone()

            candidates = []
            if attr_row and isinstance(attr_row[0], datetime):
                candidates.append(attr_row[0])
            if profile_row and isinstance(profile_row[0], datetime):
                candidates.append(profile_row[0])
            if candidates:
                latest = max(candidates)

        return latest

    def _has_pending_history_summaries(self, bot_id: str) -> tuple[bool, int]:
        """Return whether there are currently eligible unsummarized sessions.

        This guards against backlog starvation where no *new* activity happened
        since the previous check, but old sessions are still eligible and
        unsummarized.
        """
        try:
            from ..memory.summarization import HistorySummarizer

            summarizer = HistorySummarizer(self.task_processor.config, bot_id=bot_id)
            eligible_sessions = summarizer.preview_summarizable_sessions()
            count = len(eligible_sessions)
            return count > 0, count
        except Exception as e:
            logger.debug(
                "Pending-summary probe failed for bot %s: %s",
                bot_id,
                e,
            )
            return False, 0

    @staticmethod
    def _has_datetime_activity_since(
        latest_value: datetime | None, last_run_at: datetime | None
    ) -> bool:
        """Check if a datetime value indicates new activity since last run."""
        if latest_value is None:
            return False
        if last_run_at is None:
            return True
        return latest_value > last_run_at

    def _has_new_activity_for_job(self, job: ScheduledJob) -> tuple[bool, str | None]:
        """Return whether this job has relevant new activity since last run."""
        config = self._parse_config_json(job.config_json)
        last_run_at = job.last_run_at

        try:
            if job.job_type == JobType.HISTORY_SUMMARIZATION:
                # Primary gate: if there are eligible unsummarized sessions,
                # run even without fresh message activity since last_run_at.
                has_pending, pending_count = self._has_pending_history_summaries(job.bot_id)
                if has_pending:
                    return True, f"{pending_count} pending unsummarized session(s)"

                # Secondary gate: new raw activity may create soon-to-be-eligible
                # sessions even when none are currently eligible.
                latest_ts = self._max_timestamp_message_activity(job.bot_id)
                if latest_ts is None:
                    return False, "No conversation messages found"
                if last_run_at is None:
                    return False, "No eligible sessions yet"
                if latest_ts > last_run_at.timestamp():
                    return True, "New conversation activity detected"
                return False, "No new conversation activity since last run"

            if job.job_type == JobType.PROFILE_MAINTENANCE:
                entity_id = config.get("entity_id", "user")
                entity_type = config.get("entity_type", "user")
                latest_profile_update = self._max_profile_updated_at(entity_id, entity_type)
                if self._has_datetime_activity_since(latest_profile_update, last_run_at):
                    return True, None
                return False, "No profile attribute changes since last run"

            if job.job_type == JobType.MEMORY_CONSOLIDATION:
                latest_memory_update = self._max_memory_updated_at(job.bot_id)
                if self._has_datetime_activity_since(latest_memory_update, last_run_at):
                    return True, None
                return False, "No memory changes since last run"

            if job.job_type == JobType.MEMORY_DECAY:
                latest_memory_update = self._max_memory_updated_at(job.bot_id)
                if latest_memory_update is None:
                    return False, "No memories available for decay"
                return True, None
        except Exception as e:
            logger.debug(
                "Activity-gate check failed for job %s (%s); falling back to run: %s",
                job.id,
                job.job_type,
                e,
            )
            return True, None

        return True, None
    
    def _create_task_for_job(self, job: ScheduledJob):
        """Create a Task from a ScheduledJob."""
        from llm_bawt.service.tasks import (
            create_history_summarization_task,
            create_maintenance_task,
            create_profile_maintenance_task,
        )
        
        if job.job_type == JobType.PROFILE_MAINTENANCE:
            # For profile maintenance, entity_id comes from config or is the bot's user
            config = self._parse_config_json(job.config_json)
            entity_id = config.get("entity_id", "user")  # Default user
            return create_profile_maintenance_task(
                entity_id=entity_id,
                entity_type=config.get("entity_type", "user"),
                bot_id=job.bot_id,
            )
        elif job.job_type == JobType.MEMORY_CONSOLIDATION:
            config = self._parse_config_json(job.config_json)
            entity_id = config.get("entity_id", "system")
            return create_maintenance_task(
                bot_id=job.bot_id,
                user_id=entity_id,
                run_consolidation=True,
                run_recurrence_detection=False,
                run_decay_pruning=False,
            )
        elif job.job_type == JobType.MEMORY_DECAY:
            config = self._parse_config_json(job.config_json)
            entity_id = config.get("entity_id", "system")
            return create_maintenance_task(
                bot_id=job.bot_id,
                user_id=entity_id,
                run_consolidation=False,
                run_recurrence_detection=False,
                run_decay_pruning=True,
            )
        elif job.job_type == JobType.HISTORY_SUMMARIZATION:
            config = self._parse_config_json(job.config_json)
            return create_history_summarization_task(
                bot_id=job.bot_id,
                user_id=config.get("user_id", "system"),
                use_heuristic_fallback=config.get("use_heuristic_fallback", True),
                max_tokens_per_chunk=config.get("max_tokens_per_chunk", 4000),
                model=config.get("model"),
            )
        
        return None


def init_default_jobs(engine, config) -> None:
    """Initialize default scheduled jobs if they don't exist."""
    global_settings = RuntimeSettingsResolver(config=config)
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
                interval_minutes=int(
                    global_settings.resolve(
                        "profile_maintenance_interval_minutes",
                        config.PROFILE_MAINTENANCE_INTERVAL_MINUTES,
                    )
                ),
                config_json=json.dumps({"entity_id": default_user, "entity_type": "user"}),
            )
            session.add(job)
            session.commit()
            logger.debug("Created default profile maintenance job")

        # Ensure history summarization jobs exist for each configured bot.
        # Track A/Track F history behavior is bot-scoped; a single default-bot
        # job leaves other bots (e.g., mira) without summary rows.
        existing_history_jobs = session.exec(
            select(ScheduledJob).where(ScheduledJob.job_type == JobType.HISTORY_SUMMARIZATION)
        ).all()
        existing_history_bot_ids = {job.bot_id for job in existing_history_jobs}

        # Backward compatibility: if an all-bots wildcard exists, do nothing.
        if "*" in existing_history_bot_ids:
            return

        from llm_bawt.bots import BotManager

        bot_manager = BotManager(config)
        target_bots = [bot.slug for bot in bot_manager.list_bots()]
        if not target_bots:
            target_bots = [config.DEFAULT_BOT or "nova"]

        created = 0
        for bot_id in target_bots:
            if bot_id in existing_history_bot_ids:
                continue
            bot_settings = RuntimeSettingsResolver(config=config, bot_id=bot_id)

            history_job = ScheduledJob(
                job_type=JobType.HISTORY_SUMMARIZATION,
                bot_id=bot_id,
                enabled=config.SCHEDULER_ENABLED,
                interval_minutes=int(
                    bot_settings.resolve(
                        "history_summarization_interval_minutes",
                        getattr(config, "HISTORY_SUMMARIZATION_INTERVAL_MINUTES", 30),
                    )
                ),
                config_json=json.dumps(
                    {
                        "user_id": "system",
                        "use_heuristic_fallback": True,
                        "max_tokens_per_chunk": 4000,
                    }
                ),
            )
            session.add(history_job)
            created += 1

        if created:
            session.commit()
            logger.debug("Created %d history summarization job(s)", created)
