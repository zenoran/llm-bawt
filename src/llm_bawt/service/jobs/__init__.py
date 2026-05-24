"""Scheduler-invoked job implementations (TASK-231 onwards).

This package houses the *body* of each scheduled job — code that the
:class:`llm_bawt.service.scheduler.JobScheduler` dispatches to once a
:class:`llm_bawt.service.tasks.Task` has been built and routed.

Today only :mod:`media_gc` lives here; older jobs (memory consolidation,
profile maintenance, history summarization) still live inside
:mod:`background_tasks` and pre-date this package. New scheduled jobs
should land here so the service layer stays thin.
"""
