"""DB-configured operations catalog + job ledger (TASK-639).

Public surface:

- :mod:`.models` — ``OpsOperation`` (catalog) + ``OpsJob`` (ledger) SQLModel tables
- :mod:`.store` — ``OpsStore``: CRUD, atomic state transitions, per-slug seeding
- :mod:`.service` — ``OpsService``: catalog validation + job dispatch orchestration
- :mod:`.executor` — ``Executor`` ABC + ``SystemdSshExecutor`` (host-owned jobs)
- :mod:`.seeds` — canonical operation rows (per-slug insert-if-missing)

Why this exists (TASK-639 invariants):

1. Operation names / scripts / arg schemas / hosts / limits live in Postgres,
   editable in BawtHub. Adding an operation is a DB row, not a code change.
2. The agent never supplies arbitrary shell. ``ops_run`` accepts only a slug
   + JSON-Schema-validated args. Args become ``OPS_ARG_<NAME>`` env vars for
   the operator-authored script.
3. Jobs execute as host-owned systemd units (``systemd-run --user``) so an
   ``app`` / ``claude-code-bridge`` / ``redis`` restart cannot kill the child.
4. Job records snapshot the operation version + script hash at dispatch —
   later edits to the catalog never mutate an in-flight job.
"""

from .executor import Executor, ExecutorError, SystemdSshExecutor
from .models import (
    JOB_CANCELLED,
    JOB_DISPATCHING,
    JOB_FAILED,
    JOB_LOST,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    JOB_TERMINAL_STATES,
    JOB_TIMED_OUT,
    OpsJob,
    OpsOperation,
)
from .service import OpsDispatchError, OpsService
from .store import OpsStore
from .validation import ArgValidationError, validate_args

__all__ = [
    "OpsOperation",
    "OpsJob",
    "OpsStore",
    "OpsService",
    "OpsDispatchError",
    "Executor",
    "ExecutorError",
    "SystemdSshExecutor",
    "validate_args",
    "ArgValidationError",
    "JOB_QUEUED",
    "JOB_DISPATCHING",
    "JOB_RUNNING",
    "JOB_SUCCEEDED",
    "JOB_FAILED",
    "JOB_TIMED_OUT",
    "JOB_CANCELLED",
    "JOB_LOST",
    "JOB_TERMINAL_STATES",
]
