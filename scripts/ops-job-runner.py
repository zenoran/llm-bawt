#!/usr/bin/env python3
"""Host-owned runner for a single ops job (TASK-639).

Deployed to the executor's ``target_host`` (typically echo). Invoked from a
transient systemd unit created by ``SystemdSshExecutor`` so the process is
owned by the host, not the app container's PID namespace — an ``app`` /
``claude-code-bridge`` / ``redis`` restart cannot kill an in-flight job.

Contract (do not break — the reconciler reads these files):

* Working directory for job artifacts: ``$JOB_DIR``
* Reads: ``$JOB_DIR/request.json`` (immutable snapshot written by executor)
* Writes atomically: ``$JOB_DIR/status.json`` (``.tmp`` + rename per update)
* Writes streaming: ``$JOB_DIR/output.log`` (combined stdout+stderr, capped)

status.json fields:
  { "state": "running" | "succeeded" | "failed" | "timed_out",
    "started_at": "<ISO-8601>", "finished_at": "<ISO-8601>|null",
    "exit_code": <int|null>, "error": "<str|null>",
    "output_bytes": <int>, "truncated": <bool> }

Standalone-Python-3 only. No llm-bawt imports, no third-party deps — must
run on any modern host without a venv setup.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, data: str) -> None:
    """Write ``data`` to ``path`` atomically via a tmp file + rename in the
    same directory so a reader never sees a torn write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_status(job_dir: Path, **fields) -> None:
    _atomic_write(job_dir / "status.json", json.dumps(fields, ensure_ascii=False))


def _read_request(job_dir: Path) -> dict:
    with (job_dir / "request.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _run_capped(
    script_path: Path,
    *,
    cwd: str | None,
    env: dict[str, str],
    timeout: int,
    max_output_bytes: int,
    log_path: Path,
) -> tuple[int | None, int, bool, str | None]:
    """Execute the script, capping combined output. Returns
    ``(exit_code, output_bytes, truncated, error_msg)``.

    - ``exit_code`` is None on timeout.
    - ``truncated`` is True if output was cut at the cap.
    - ``error_msg`` non-None when we synthesize one (e.g. timeout, spawn fail).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    truncated = False
    error_msg: str | None = None

    try:
        proc = subprocess.Popen(
            ["/bin/bash", str(script_path)],
            cwd=cwd or None,
            env={**os.environ, **env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group so we can kill children on timeout
        )
    except Exception as exc:  # noqa: BLE001
        return None, 0, False, f"spawn failed: {exc}"

    try:
        with log_path.open("wb") as log_fh:
            assert proc.stdout is not None
            deadline_hit = False
            try:
                # Poll in small chunks so we can enforce the byte cap AND the
                # timeout without blocking on read().
                while True:
                    chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") \
                        else proc.stdout.read(4096)
                    if not chunk:
                        # Process may have exited; drain will confirm.
                        if proc.poll() is not None:
                            break
                        continue
                    if written < max_output_bytes:
                        remaining = max_output_bytes - written
                        if len(chunk) > remaining:
                            log_fh.write(chunk[:remaining])
                            written += remaining
                            truncated = True
                        else:
                            log_fh.write(chunk)
                            written += len(chunk)
                    else:
                        truncated = True
                    log_fh.flush()
                    # Timeout check.
                    try:
                        proc.wait(timeout=0.01)
                    except subprocess.TimeoutExpired:
                        pass
            except subprocess.TimeoutExpired:
                deadline_hit = True

            # Second timeout gate wraps the total runtime — even a chatty
            # child that keeps writing must not run forever.
            try:
                exit_code = proc.wait(timeout=max(0, timeout - 0.01))
            except subprocess.TimeoutExpired:
                deadline_hit = True
                exit_code = None
    finally:
        # Clean up the process tree if it's still alive (timeout, or an
        # exception broke us out of the loop early).
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
            except (ProcessLookupError, OSError):
                pass

    if deadline_hit and exit_code is None:
        error_msg = f"timed out after {timeout}s"

    return exit_code, written, truncated, error_msg


def main() -> int:
    ap = argparse.ArgumentParser(description="Host-owned ops job runner")
    ap.add_argument(
        "--job-dir",
        required=True,
        help="Directory containing request.json; status/output land here.",
    )
    args = ap.parse_args()

    job_dir = Path(args.job_dir).resolve()
    if not job_dir.is_dir():
        print(f"job dir not found: {job_dir}", file=sys.stderr)
        return 2

    try:
        req = _read_request(job_dir)
    except Exception as exc:  # noqa: BLE001
        _write_status(
            job_dir,
            state="failed",
            started_at=_utcnow_iso(),
            finished_at=_utcnow_iso(),
            exit_code=None,
            error=f"failed to read request.json: {exc}",
            output_bytes=0,
            truncated=False,
        )
        print(f"failed to read request.json: {exc}", file=sys.stderr)
        return 2

    script_body: str = req.get("command_script", "") or ""
    if not script_body.strip():
        _write_status(
            job_dir,
            state="failed",
            started_at=_utcnow_iso(),
            finished_at=_utcnow_iso(),
            exit_code=None,
            error="empty command_script",
            output_bytes=0,
            truncated=False,
        )
        return 2

    script_path = job_dir / "script.sh"
    script_path.write_text(script_body, encoding="utf-8")
    script_path.chmod(0o750)

    env: dict[str, str] = {}
    for k, v in (req.get("env") or {}).items():
        # Args land as OPS_ARG_<NAME>=<value>. Coerce to string; the runner
        # never trusts non-string values from an agent-supplied dict.
        env[str(k)] = str(v) if not isinstance(v, str) else v
    env["OPS_JOB_ID"] = str(req.get("job_id") or job_dir.name)
    if req.get("working_directory"):
        env["OPS_WORKING_DIR"] = str(req["working_directory"])

    timeout = int(req.get("timeout_seconds") or 300)
    max_output = int(req.get("max_output_bytes") or 65536)
    cwd = req.get("working_directory") or None

    started_at = _utcnow_iso()
    _write_status(
        job_dir,
        state="running",
        started_at=started_at,
        finished_at=None,
        exit_code=None,
        error=None,
        output_bytes=0,
        truncated=False,
    )

    exit_code, output_bytes, truncated, error_msg = _run_capped(
        script_path,
        cwd=cwd, env=env,
        timeout=timeout, max_output_bytes=max_output,
        log_path=job_dir / "output.log",
    )

    finished_at = _utcnow_iso()
    if error_msg and exit_code is None:
        terminal = "timed_out" if "timed out" in error_msg else "failed"
    elif exit_code == 0:
        terminal = "succeeded"
    else:
        terminal = "failed"

    _write_status(
        job_dir,
        state=terminal,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        error=error_msg,
        output_bytes=output_bytes,
        truncated=truncated,
    )
    return 0 if terminal == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
