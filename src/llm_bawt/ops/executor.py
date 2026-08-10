"""Executor contract + DockerExecutor (TASK-639).

An executor turns a validated :class:`OpsJob` into an actual side effect.
For llm-bawt today that means "restart / stop / start / pull a container
in this Docker daemon" — the daemon is reached through the mounted socket
at ``/var/run/docker.sock``, no SSH and no host-side runner.

Operations describe themselves via a JSON spec stored in
:attr:`OpsOperation.command_script`. The DockerExecutor parses that spec
and drives the Docker SDK. Example specs::

    {"action": "restart", "container_name": "llm-bawt-app"}

    {"action": "restart", "compose_project": "llm-bawt", "compose_service": "app"}

    # Service selected from a validated arg:
    {"action": "restart", "compose_project": "llm-bawt",
     "compose_service_from_arg": "service"}

Every dispatch is synchronous unless ``start_delay_seconds`` is non-zero
OR the target container is us. Delayed dispatch spawns a background
thread that fires after the delay, so the caller's response can drain
before we (or any container we're restarting) get killed. The
:class:`DispatchResult` returned in the delayed case carries
``terminal_state="succeeded"`` immediately — we can't observe the
outcome from the corpse of the app process, so the "success" refers to
"the schedule was placed", not "the restart completed."

Deployment prerequisite: ``/var/run/docker.sock`` must be bind-mounted
into the container (see docker-compose.yml). Until that lands,
:meth:`DockerExecutor.available` returns False and
:meth:`OpsService.dispatch` fails cleanly with a clear reason instead of
pretending the job started.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class ExecutorError(RuntimeError):
    """Raised when an executor can't dispatch / reconcile a job."""


@dataclass
class DispatchResult:
    """What the executor tells the store after dispatching.

    Docker-side ops complete synchronously in most cases — dispatch
    returns the final outcome directly via ``terminal_state`` /
    ``exit_code`` / ``output``. The service marks the job terminal in
    one shot instead of waiting on a reconciler.

    ``host_unit_name`` is a short human-readable identifier for logs and
    the BawtHub UI (e.g. ``"docker:restart:llm-bawt-app"``). It uses the
    existing DB column of that name; the label there has no semantic role
    beyond display.

    When ``terminal_state`` is ``None`` the store still marks the job
    ``DISPATCHING`` and relies on ``reconcile()`` — that path is unused
    by :class:`DockerExecutor` today but preserved for future executors.
    """

    host_unit_name: str
    status_file_path: str = ""
    log_file_path: str = ""
    terminal_state: str | None = None  # "succeeded" | "failed" | "timed_out"
    exit_code: int | None = None
    output: str | None = None


@dataclass
class ReconcileResult:
    """What the executor tells the store after polling one job.

    Docker jobs terminal-mark at dispatch time, so this only matters if
    a future executor needs async polling. Kept for contract stability.
    """

    state: str | None  # None = no signal, keep current DB state
    exit_code: int | None = None
    output_tail: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class Executor(ABC):
    """Abstract executor contract. All methods are synchronous — call from
    a threadpool if you need concurrency; ops are I/O-bound but short-lived
    so a simple executor pool is fine.
    """

    @abstractmethod
    def kind(self) -> str: ...

    @abstractmethod
    def available(self) -> bool:
        """Return True if this executor can dispatch right now.

        Called before every dispatch so the service can fail cleanly with
        a clear message instead of surprising the caller mid-dispatch.
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
# DockerExecutor
# ---------------------------------------------------------------------------

# Supported actions. Extending this = one new elif branch in
# :meth:`DockerExecutor._fire_action`, and mirrored allowlisting here so the
# validation error surfaces before we start reaching for containers.
SUPPORTED_ACTIONS = frozenset({"restart", "start", "stop", "pull"})


class DockerExecutor(Executor):
    """Drive Docker via the mounted socket + Python SDK.

    Injectable ``client_factory`` so tests can pass a fake without needing
    the real docker package or a real socket.
    """

    def __init__(
        self,
        *,
        client_factory=None,
        self_container_names: tuple[str, ...] = ("llm-bawt-app", "app"),
    ) -> None:
        self._client_factory = client_factory
        self._client = None
        # Container names/service labels that identify "this process's
        # container" — used to force delayed dispatch on self-restart so
        # the response drains before Docker kills us.
        self.self_container_names = tuple(self_container_names)

    def kind(self) -> str:
        return "docker"

    # ---- lazy client ------------------------------------------------------

    def _default_factory(self):
        try:
            import docker  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - install-time guard
            raise ExecutorError(
                "docker executor unavailable: python `docker` package not installed"
            ) from exc
        return docker.from_env()

    @property
    def client(self):
        if self._client is None:
            factory = self._client_factory or self._default_factory
            self._client = factory()
        return self._client

    def available(self) -> bool:
        try:
            self.client.ping()
        except Exception as exc:  # noqa: BLE001
            logger.debug("docker executor unavailable: %s", exc)
            return False
        return True

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
        if not self.available():
            raise ExecutorError(
                "docker executor unavailable: cannot reach /var/run/docker.sock "
                "(is the socket bind-mounted into this container?)"
            )

        spec = self._parse_spec(command_script)
        action = str(spec.get("action") or "").strip().lower()
        if action not in SUPPORTED_ACTIONS:
            raise ExecutorError(
                f"unsupported docker action {action!r} "
                f"(supported: {sorted(SUPPORTED_ACTIONS)})"
            )

        selector = self._resolve_selector(spec, env_args)
        target_label = self._selector_label(selector)

        # Force delayed dispatch when we'd be killing ourselves — the caller
        # response has to drain before Docker cuts us off.
        is_self = self._is_self_target(selector)
        delay = int(start_delay_seconds or 0)
        if is_self and delay <= 0:
            delay = 5  # sensible default — enough for SSE to flush

        unit_name = f"docker:{action}:{target_label}"
        if delay > 0:
            unit_name += f":delay={delay}s"

        if delay > 0:
            self._fire_delayed(action, selector, delay, timeout_seconds)
            return DispatchResult(
                host_unit_name=unit_name,
                terminal_state="succeeded",
                exit_code=0,
                output=(
                    f"scheduled: {action} {target_label} in {delay}s "
                    f"(self-restart)" if is_self else
                    f"scheduled: {action} {target_label} in {delay}s"
                ),
            )

        # Immediate synchronous dispatch.
        try:
            output = self._fire_action(action, selector, timeout_seconds)
        except ExecutorError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExecutorError(f"docker {action} failed: {exc}") from exc

        return DispatchResult(
            host_unit_name=unit_name,
            terminal_state="succeeded",
            exit_code=0,
            output=output,
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
        # DockerExecutor terminal-marks at dispatch time, so reconcile
        # is a no-op — service.get_job_status won't call it for terminal
        # jobs, and there are no non-terminal docker jobs to poll.
        return ReconcileResult(state=None)

    # ---- internals -------------------------------------------------------

    def _parse_spec(self, command_script: str) -> dict[str, Any]:
        raw = (command_script or "").strip()
        if not raw:
            raise ExecutorError("operation command_script is empty (need JSON spec)")
        try:
            spec = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ExecutorError(
                f"operation command_script is not valid JSON: {exc}"
            ) from exc
        if not isinstance(spec, dict):
            raise ExecutorError(
                f"operation command_script must be a JSON object, got {type(spec).__name__}"
            )
        return spec

    def _resolve_selector(
        self, spec: dict[str, Any], env_args: dict[str, Any]
    ) -> dict[str, str]:
        """Return a container-locator dict with keys in
        {``name``, ``project``, ``service``}. Merges spec-literal fields
        with ``*_from_arg`` fields that pull from the validated args.
        """
        sel: dict[str, str] = {}
        if "container_name" in spec:
            sel["name"] = str(spec["container_name"])
        if "container_name_from_arg" in spec:
            key = str(spec["container_name_from_arg"])
            val = env_args.get(key)
            if val is None:
                raise ExecutorError(
                    f"selector references arg {key!r} but it wasn't supplied"
                )
            sel["name"] = str(val)
        if "compose_project" in spec:
            sel["project"] = str(spec["compose_project"])
        if "compose_service" in spec:
            sel["service"] = str(spec["compose_service"])
        if "compose_service_from_arg" in spec:
            key = str(spec["compose_service_from_arg"])
            val = env_args.get(key)
            if val is None:
                raise ExecutorError(
                    f"selector references arg {key!r} but it wasn't supplied"
                )
            sel["service"] = str(val)
        if not sel:
            raise ExecutorError(
                "operation spec has no container selector "
                "(need container_name, compose_project+compose_service, or *_from_arg)"
            )
        if "project" in sel and "service" not in sel:
            raise ExecutorError(
                "compose_project without compose_service (or compose_service_from_arg)"
            )
        return sel

    def _selector_label(self, sel: dict[str, str]) -> str:
        if "name" in sel:
            return sel["name"]
        return f"{sel.get('project','?')}/{sel.get('service','?')}"

    def _find_container(self, sel: dict[str, str]):
        """Resolve selector → docker container object. Raises ExecutorError
        on any lookup miss so the caller gets a specific reason."""
        try:
            import docker.errors as derrs  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover
            derrs = None  # type: ignore[assignment]

        if "name" in sel:
            try:
                return self.client.containers.get(sel["name"])
            except Exception as exc:  # docker.errors.NotFound + friends
                if derrs and isinstance(exc, derrs.NotFound):
                    raise ExecutorError(f"container not found: {sel['name']!r}")
                raise ExecutorError(f"container lookup failed: {exc}") from exc

        # Compose project + service — filter by the standard compose labels.
        filters = {
            "label": [
                f"com.docker.compose.project={sel['project']}",
                f"com.docker.compose.service={sel['service']}",
            ]
        }
        try:
            matches = self.client.containers.list(all=True, filters=filters)
        except Exception as exc:  # noqa: BLE001
            raise ExecutorError(f"container list failed: {exc}") from exc
        if not matches:
            raise ExecutorError(
                f"no container for compose project={sel['project']!r} "
                f"service={sel['service']!r}"
            )
        if len(matches) > 1:
            names = ", ".join(c.name for c in matches)
            raise ExecutorError(
                f"selector matched multiple containers ({names}); "
                "narrow it with container_name"
            )
        return matches[0]

    def _is_self_target(self, sel: dict[str, str]) -> bool:
        # If the selector names the app container by name OR by compose service.
        if sel.get("name") in self.self_container_names:
            return True
        if sel.get("service") in self.self_container_names:
            return True
        return False

    def _fire_action(
        self, action: str, sel: dict[str, str], timeout_seconds: int,
    ) -> str:
        """Execute one docker action synchronously. Returns a short summary
        line for the job's ``output_tail``."""
        container = self._find_container(sel)
        name = container.name
        if action == "restart":
            container.restart(timeout=max(int(timeout_seconds or 30), 10))
            return f"restarted {name}"
        if action == "start":
            container.start()
            return f"started {name}"
        if action == "stop":
            container.stop(timeout=max(int(timeout_seconds or 30), 10))
            return f"stopped {name}"
        if action == "pull":
            image = None
            try:
                image = container.image.tags[0] if container.image.tags else None
            except Exception:  # noqa: BLE001
                image = None
            if not image:
                raise ExecutorError(
                    f"container {name!r} has no tagged image to pull"
                )
            self.client.images.pull(image)
            return f"pulled {image} (container: {name})"
        raise ExecutorError(f"unsupported docker action: {action!r}")

    def _fire_delayed(
        self,
        action: str,
        sel: dict[str, str],
        delay_seconds: int,
        timeout_seconds: int,
    ) -> None:
        """Fire ``action`` after ``delay_seconds`` in a background daemon
        thread. Errors are logged (never re-raised) since the caller has
        already returned by the time this runs — the job row is stamped
        succeeded at dispatch time on the "schedule placed" contract.
        """

        label = self._selector_label(sel)

        def _worker() -> None:
            try:
                time.sleep(max(int(delay_seconds), 0))
                self._fire_action(action, sel, timeout_seconds)
                logger.info("ops delayed dispatch: %s %s done", action, label)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ops delayed dispatch failed: %s %s: %s", action, label, exc
                )

        t = threading.Thread(
            target=_worker,
            name=f"ops-delayed-{action}-{label}",
            daemon=True,
        )
        t.start()


__all__ = [
    "Executor",
    "ExecutorError",
    "DispatchResult",
    "ReconcileResult",
    "DockerExecutor",
    "SUPPORTED_ACTIONS",
]
