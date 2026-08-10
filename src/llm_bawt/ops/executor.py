"""Executor contract + NohupSshExecutor (TASK-639).

An executor turns a validated :class:`OpsJob` into a detached host-side
child process and later reconciles its exit state back into the DB. The
executor never touches the DB directly — it hands intermediate state back
to :class:`OpsService`, which owns the row transitions.

The transport is just SSH + ``nohup``: no host-side runner script, no
transient systemd unit, no ``systemd --user`` linger setup. Every job's
working dir on the host is::

    ~/.local/share/llm-bawt/ops-jobs/<job_id>/
        pid          # writer's PID (echo $! right after backgrounding)
        exit         # exit code of the wrapped command (present iff done)
        output.log   # combined stdout+stderr, size-capped by tail on read

Dispatch = one SSH call that mkdirs the job dir, ``nohup``s a bash wrapper
that (a) optionally sleeps ``start_delay_seconds``, (b) runs the operator's
script under ``timeout``, (c) writes ``$?`` to ``exit`` — then prints the
child PID so we can record it. The wrapper is fully self-contained; the
container can exit or restart and the child keeps going.

Reconcile = one SSH call that either cats ``exit`` (job done) or ``kill -0``s
the PID (still alive vs disappeared with no exit code). Terminal → a second
SSH call tails ``output.log``.

Deployment prerequisite: the ``app`` container needs ``openssh-client`` and a
mounted SSH identity that authenticates to ``target_host``. Until that lands,
:meth:`NohupSshExecutor.available` returns False and :meth:`OpsService.dispatch`
fails cleanly instead of pretending the job started.
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# Host-side working root for all ops jobs. Each job gets a subdir named by
# its job_id. The reconciler reads exit/output.log from these paths.
DEFAULT_HOST_JOBS_ROOT = "/home/nick/.local/share/llm-bawt/ops-jobs"

# ``timeout`` exits 124 when it kills the child for exceeding its limit.
# We map that back to :data:`OpsJob.state` = timed_out.
TIMEOUT_EXIT_CODE = 124


class ExecutorError(RuntimeError):
    """Raised when an executor can't dispatch / reconcile a job."""


@dataclass
class DispatchResult:
    """What the executor tells the store after a successful dispatch.

    ``host_unit_name`` reuses the existing DB column for observability —
    format is ``pid:<pid>`` so operators can spot orphans in ``ps`` and the
    reconciler has a durable liveness handle without a second read.
    """

    host_unit_name: str
    status_file_path: str  # path to the ``exit`` file on the host
    log_file_path: str


@dataclass
class ReconcileResult:
    """What the executor tells the store after polling one job.

    ``state`` uses the executor-facing vocabulary
    (``running`` | ``succeeded`` | ``failed`` | ``timed_out``) which the
    service maps to :data:`OpsJob.state`. ``None`` = no signal yet (the
    child hasn't materialized any files), so keep the current DB state.
    """

    state: str | None
    exit_code: int | None
    output_tail: str | None
    error: str | None
    started_at: str | None
    finished_at: str | None


class Executor(ABC):
    """Abstract executor contract. All methods are synchronous — call from
    a threadpool if you need concurrency; the operations are I/O-bound but
    short-lived so a simple executor pool is fine.
    """

    @abstractmethod
    def kind(self) -> str: ...

    @abstractmethod
    def available(self) -> bool:
        """Return True if this executor can dispatch right now.

        Called before every dispatch so the service can fail cleanly with a
        clear message ("openssh-client not installed") instead of the SSH
        binary going missing mid-dispatch.
        """

    @abstractmethod
    def dispatch(
        self,
        *,
        job_id: str,
        operation_slug: str,
        target_host: str,
        run_as_user: str | None,
        working_directory: str | None,
        command_script: str,
        env_args: dict[str, Any],
        timeout_seconds: int,
        start_delay_seconds: int,
        max_output_bytes: int,
    ) -> DispatchResult: ...

    @abstractmethod
    def reconcile(
        self,
        *,
        job_id: str,
        target_host: str,
        status_file_path: str,
        log_file_path: str,
        output_tail_bytes: int = 4096,
    ) -> ReconcileResult: ...


# ---------------------------------------------------------------------------
# NohupSshExecutor
# ---------------------------------------------------------------------------

class SshTransport:
    """Thin subprocess-based SSH transport. Isolated so tests can swap it."""

    def __init__(self, *, ssh_bin: str = "ssh") -> None:
        self.ssh_bin = ssh_bin

    def available(self) -> bool:
        return shutil.which(self.ssh_bin) is not None

    def run(
        self,
        target: str,
        command: str,
        *,
        timeout: int = 30,
    ) -> tuple[int, str, str]:
        """Run ``command`` remotely. Returns (exit_code, stdout, stderr)."""
        args = [
            self.ssh_bin,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
            target,
            command,
        ]
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr


class NohupSshExecutor(Executor):
    """Dispatch jobs as detached ``nohup`` children on an SSH-reachable host.

    No host-side runner script, no systemd unit. The child owns its own
    working dir; ``exit`` file presence is the source of truth for "done."
    """

    def __init__(
        self,
        *,
        transport: SshTransport | None = None,
        host_jobs_root: str = DEFAULT_HOST_JOBS_ROOT,
    ) -> None:
        self.transport = transport or SshTransport()
        self.host_jobs_root = host_jobs_root

    def kind(self) -> str:
        return "nohup_ssh"

    def available(self) -> bool:
        return self.transport.available()

    # ---- dispatch --------------------------------------------------------

    def dispatch(
        self,
        *,
        job_id: str,
        operation_slug: str,
        target_host: str,
        run_as_user: str | None,
        working_directory: str | None,
        command_script: str,
        env_args: dict[str, Any],
        timeout_seconds: int,
        start_delay_seconds: int,
        max_output_bytes: int,
    ) -> DispatchResult:
        if not self.transport.available():
            raise ExecutorError(
                "NohupSshExecutor unavailable: ssh not found in the container "
                "(openssh-client required, TASK-639 Slice C)"
            )
        if not target_host:
            raise ExecutorError("operation missing target_host")

        job_dir = f"{self.host_jobs_root.rstrip('/')}/{job_id}"
        exit_path = f"{job_dir}/exit"
        log_path = f"{job_dir}/output.log"
        pid_path = f"{job_dir}/pid"

        # Args → OPS_ARG_<NAME> env exports. Values are shlex-quoted so no
        # agent-supplied value can break out of its var into shell syntax.
        env_exports = ""
        for k, v in (env_args or {}).items():
            key = f"OPS_ARG_{str(k).upper()}"
            val = "" if v is None else str(v)
            env_exports += f"export {key}={shlex.quote(val)}; "

        # Working directory prefix. Empty string when not configured so the
        # child inherits the SSH login shell's cwd.
        cd_prefix = ""
        if working_directory:
            cd_prefix = f"cd {shlex.quote(working_directory)} && "

        # Optional pre-start sleep. Non-zero for self-affecting ops so the
        # caller's response has time to stream before the restart lands.
        delay_prefix = ""
        if start_delay_seconds and start_delay_seconds > 0:
            delay_prefix = f"sleep {int(start_delay_seconds)}; "

        # The bash body that will run inside `nohup`. `timeout` bounds the
        # runtime; its exit code (124 on kill) surfaces as the child exit.
        # The exit file is written AFTER the command completes so its
        # presence is the "done" signal for the reconciler.
        inner = (
            f"{delay_prefix}"
            f"{env_exports}"
            f"{cd_prefix}"
            f"timeout {int(timeout_seconds or 300)}s bash -c {shlex.quote(command_script)}; "
            f"echo $? > {shlex.quote(exit_path)}"
        )

        # Outer: create the job dir, background the inner with nohup, capture
        # PID, then print it so we can record host_unit_name.
        outer = (
            f"install -d -m 700 {shlex.quote(job_dir)} && "
            f"( nohup bash -c {shlex.quote(inner)} > {shlex.quote(log_path)} 2>&1 & "
            f"echo $! > {shlex.quote(pid_path)} ) && "
            f"cat {shlex.quote(pid_path)}"
        )

        rc, stdout, err = self.transport.run(target_host, outer, timeout=30)
        if rc != 0:
            raise ExecutorError(
                f"ssh dispatch to {target_host} failed: {err.strip() or f'rc={rc}'}"
            )
        pid = (stdout or "").strip().splitlines()[-1].strip() if stdout else ""
        if not pid or not pid.isdigit():
            raise ExecutorError(
                f"ssh dispatch to {target_host} returned no PID (stdout={stdout!r})"
            )

        return DispatchResult(
            host_unit_name=f"pid:{pid}",
            status_file_path=exit_path,
            log_file_path=log_path,
        )

    # ---- reconcile -------------------------------------------------------

    def reconcile(
        self,
        *,
        job_id: str,
        target_host: str,
        status_file_path: str,
        log_file_path: str,
        output_tail_bytes: int = 4096,
    ) -> ReconcileResult:
        # Sibling paths — the dispatcher always writes pid + exit + output.log
        # into the same job dir, so we can derive the pid path from the exit
        # path without threading it through the store.
        job_dir = status_file_path.rsplit("/", 1)[0]
        pid_path = f"{job_dir}/pid"

        # Single round-trip: exit file wins, else PID liveness. Sentinels
        # are wrapped in double-underscores so they don't collide with any
        # exit code integer.
        probe = (
            f"if [ -f {shlex.quote(status_file_path)} ]; then "
            f"cat {shlex.quote(status_file_path)}; "
            f"else "
            f"pid=$(cat {shlex.quote(pid_path)} 2>/dev/null); "
            f'if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo __running__; '
            f"else echo __missing__; fi; "
            f"fi"
        )
        rc, stdout, err = self.transport.run(target_host, probe, timeout=20)
        if rc != 0:
            raise ExecutorError(
                f"reconcile ssh to {target_host} failed: {err.strip() or f'rc={rc}'}"
            )
        line = (stdout or "").strip()

        if line == "__running__":
            return ReconcileResult(
                state="running", exit_code=None, output_tail=None, error=None,
                started_at=None, finished_at=None,
            )
        if line == "__missing__" or not line:
            # Either the child hasn't backgrounded yet (dispatch just fired)
            # OR the pid file was cleaned up externally. Keep polling.
            return ReconcileResult(
                state=None, exit_code=None, output_tail=None, error=None,
                started_at=None, finished_at=None,
            )

        # Exit file present — its content is a single integer.
        try:
            exit_code = int(line)
        except ValueError:
            return ReconcileResult(
                state="failed", exit_code=None, output_tail=None,
                error=f"malformed exit file: {line!r}",
                started_at=None, finished_at=None,
            )

        if exit_code == 0:
            state = "succeeded"
        elif exit_code == TIMEOUT_EXIT_CODE:
            state = "timed_out"
        else:
            state = "failed"

        # Terminal → fetch a bounded tail of the log for the DB row.
        tail: str | None = None
        if log_file_path:
            tail_cmd = (
                f"if [ -f {shlex.quote(log_file_path)} ]; then "
                f"tail -c {int(output_tail_bytes)} {shlex.quote(log_file_path)}; "
                f"fi"
            )
            trc, tstdout, _terr = self.transport.run(target_host, tail_cmd, timeout=20)
            if trc == 0 and tstdout:
                tail = tstdout

        return ReconcileResult(
            state=state, exit_code=exit_code, output_tail=tail, error=None,
            started_at=None, finished_at=None,
        )


__all__ = [
    "Executor",
    "ExecutorError",
    "DispatchResult",
    "ReconcileResult",
    "SshTransport",
    "NohupSshExecutor",
    "DEFAULT_HOST_JOBS_ROOT",
    "TIMEOUT_EXIT_CODE",
]
