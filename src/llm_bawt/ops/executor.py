"""Executor contract + SystemdSshExecutor (TASK-639).

An executor turns a validated :class:`OpsJob` into a host-owned systemd
unit and later reconciles the host's status file back into the DB. The
executor never touches the DB directly — it hands intermediate state back
to :class:`OpsService`, which owns the row transitions.

Deployment prerequisite: the ``app`` container must have ``openssh-client``
installed and the read-only bridge SSH identity mounted (Slice C's
Dockerfile change). Until that lands, ``SystemdSshExecutor.available()``
returns False and ``OpsService.dispatch`` fails cleanly instead of pretending
the job started.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Host-side working root for all ops jobs. Bind-mounted on the executor's
# target_host so the reconciler can read status/output without another SSH.
DEFAULT_HOST_JOBS_ROOT = "/home/nick/.local/share/llm-bawt/ops-jobs"

# Host-side installed location of the runner script. Deploy path is:
#   scripts/ops-job-runner.py  →  ~/.local/share/llm-bawt/ops-job-runner.py
DEFAULT_HOST_RUNNER_PATH = "/home/nick/.local/share/llm-bawt/ops-job-runner.py"


class ExecutorError(RuntimeError):
    """Raised when an executor can't dispatch / reconcile a job."""


@dataclass
class DispatchResult:
    """What the executor tells the store after a successful dispatch."""

    host_unit_name: str
    status_file_path: str
    log_file_path: str


@dataclass
class ReconcileResult:
    """What the executor tells the store after polling one job.

    ``state`` uses the host-runner status file vocabulary
    (``running`` | ``succeeded`` | ``failed`` | ``timed_out``) which the
    service maps to :data:`OpsJob.state`.
    """

    state: str | None  # None = status file not yet present
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
# SystemdSshExecutor
# ---------------------------------------------------------------------------

class SshTransport:
    """Thin subprocess-based SSH transport. Isolated so tests can swap it."""

    def __init__(self, *, ssh_bin: str = "ssh", scp_bin: str = "scp") -> None:
        self.ssh_bin = ssh_bin
        self.scp_bin = scp_bin

    def available(self) -> bool:
        return shutil.which(self.ssh_bin) is not None and shutil.which(self.scp_bin) is not None

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

    def upload(
        self,
        target: str,
        local_path: str,
        remote_path: str,
        *,
        timeout: int = 30,
    ) -> tuple[int, str, str]:
        args = [
            self.scp_bin,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
            local_path,
            f"{target}:{remote_path}",
        ]
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr


class SystemdSshExecutor(Executor):
    """Dispatch jobs as ``systemd-run --user`` transient units on an
    SSH-reachable host. Host-owned so the app container's PID namespace
    can be torn down without killing the child.
    """

    def __init__(
        self,
        *,
        transport: SshTransport | None = None,
        host_jobs_root: str = DEFAULT_HOST_JOBS_ROOT,
        host_runner_path: str = DEFAULT_HOST_RUNNER_PATH,
    ) -> None:
        self.transport = transport or SshTransport()
        self.host_jobs_root = host_jobs_root
        self.host_runner_path = host_runner_path

    def kind(self) -> str:
        return "systemd_ssh"

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
                "SystemdSshExecutor unavailable: ssh/scp not found in the "
                "container (openssh-client required, TASK-639 Slice C)"
            )
        if not target_host:
            raise ExecutorError("operation missing target_host")

        job_dir = f"{self.host_jobs_root.rstrip('/')}/{job_id}"
        status_path = f"{job_dir}/status.json"
        log_path = f"{job_dir}/output.log"
        request_path = f"{job_dir}/request.json"
        unit_name = f"llm-bawt-ops-{job_id[:16]}"

        # Create the job dir on the host.
        mkdir_cmd = (
            f"install -d -m 700 {shlex.quote(job_dir)} && "
            f"install -d -m 755 {shlex.quote(os.path.dirname(self.host_runner_path))}"
        )
        rc, _out, err = self.transport.run(target_host, mkdir_cmd, timeout=20)
        if rc != 0:
            raise ExecutorError(f"mkdir on {target_host} failed: {err.strip() or f'rc={rc}'}")

        # Materialize the immutable request snapshot the runner reads.
        # OPS_ARG_<NAME> keys become env vars in the child's environment.
        env: dict[str, str] = {}
        for k, v in (env_args or {}).items():
            env[f"OPS_ARG_{str(k).upper()}"] = "" if v is None else str(v)

        request_body = json.dumps(
            {
                "job_id": job_id,
                "operation_slug": operation_slug,
                "command_script": command_script,
                "env": env,
                "working_directory": working_directory,
                "timeout_seconds": int(timeout_seconds or 300),
                "max_output_bytes": int(max_output_bytes or 65536),
            },
            ensure_ascii=False,
        )
        self._upload_text(target_host, request_body, request_path)

        # Start the transient unit — non-blocking; runner writes status.json.
        # RuntimeMaxSec adds a hard timeout on top of the runner's own limit
        # so a wedged runner is still cleaned up by systemd.
        unit_args = [
            "systemd-run", "--user", "--no-block",
            f"--unit={unit_name}",
            f"--description=llm-bawt ops {operation_slug} ({job_id})",
            f"--property=RuntimeMaxSec={int(timeout_seconds or 300) + 30}",
        ]
        if start_delay_seconds and start_delay_seconds > 0:
            unit_args.append(f"--on-active={int(start_delay_seconds)}s")
        if run_as_user:
            # `--uid=` needs `--system`; for `--user` mode this is just a hint
            # — we skip it, since the SSH login already selected the user.
            pass
        unit_args.extend([
            "/usr/bin/env", "python3", shlex.quote(self.host_runner_path),
            "--job-dir", shlex.quote(job_dir),
        ])
        systemd_cmd = " ".join(shlex.quote(a) if " " in a else a for a in unit_args)
        rc, _out, err = self.transport.run(target_host, systemd_cmd, timeout=30)
        if rc != 0:
            raise ExecutorError(
                f"systemd-run on {target_host} failed: {err.strip() or f'rc={rc}'}"
            )

        return DispatchResult(
            host_unit_name=unit_name,
            status_file_path=status_path,
            log_file_path=log_path,
        )

    def _upload_text(self, target: str, body: str, remote_path: str) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            tf.write(body)
            tmp_path = tf.name
        try:
            rc, _out, err = self.transport.upload(target, tmp_path, remote_path, timeout=30)
            if rc != 0:
                raise ExecutorError(f"scp to {target}:{remote_path} failed: {err.strip() or f'rc={rc}'}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

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
        # Try to read status.json first; if it doesn't exist yet, the runner
        # hasn't started (or the systemd unit hasn't been activated for
        # start_delay_seconds). Return state=None so the caller keeps polling.
        cat_status = (
            f"if [ -f {shlex.quote(status_file_path)} ]; then "
            f"cat {shlex.quote(status_file_path)}; "
            f"else echo __missing__; fi"
        )
        rc, stdout, err = self.transport.run(target_host, cat_status, timeout=20)
        if rc != 0:
            raise ExecutorError(
                f"reconcile ssh to {target_host} failed: {err.strip() or f'rc={rc}'}"
            )
        stdout = stdout.strip()
        if stdout == "__missing__" or not stdout:
            return ReconcileResult(
                state=None, exit_code=None, output_tail=None, error=None,
                started_at=None, finished_at=None,
            )
        try:
            status = json.loads(stdout)
        except json.JSONDecodeError as exc:
            return ReconcileResult(
                state="failed", exit_code=None, output_tail=None,
                error=f"malformed status.json: {exc}",
                started_at=None, finished_at=None,
            )

        # Read the last N bytes of the log for the tail.
        tail: str | None = None
        if status.get("state") in ("succeeded", "failed", "timed_out"):
            tail_cmd = (
                f"if [ -f {shlex.quote(log_file_path)} ]; then "
                f"tail -c {int(output_tail_bytes)} {shlex.quote(log_file_path)}; "
                f"fi"
            )
            trc, tstdout, _terr = self.transport.run(target_host, tail_cmd, timeout=20)
            if trc == 0 and tstdout:
                tail = tstdout

        return ReconcileResult(
            state=status.get("state"),
            exit_code=status.get("exit_code"),
            output_tail=tail,
            error=status.get("error"),
            started_at=status.get("started_at"),
            finished_at=status.get("finished_at"),
        )


__all__ = [
    "Executor",
    "ExecutorError",
    "DispatchResult",
    "ReconcileResult",
    "SshTransport",
    "SystemdSshExecutor",
    "DEFAULT_HOST_JOBS_ROOT",
    "DEFAULT_HOST_RUNNER_PATH",
]
