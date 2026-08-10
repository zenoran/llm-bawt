"""Tests for :class:`llm_bawt.ops.executor.DockerExecutor` (TASK-639).

Exercised with an injected fake docker client — no real socket, no real
containers, no threading race conditions in the immediate path.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from llm_bawt.ops.executor import (
    DockerExecutor,
    DispatchResult,
    ExecutorError,
    SUPPORTED_ACTIONS,
)


# ---------------------------------------------------------------------------
# Fake docker client — mimics just enough of docker.DockerClient for the
# executor to drive it end-to-end.
# ---------------------------------------------------------------------------

class FakeContainer:
    def __init__(self, name: str, tags: list[str] | None = None):
        self.name = name
        self.image = type("Img", (), {"tags": tags or []})
        self.restart_calls: list[int] = []
        self.stop_calls: list[int] = []
        self.start_calls: int = 0

    def restart(self, timeout: int = 10):
        self.restart_calls.append(timeout)

    def start(self):
        self.start_calls += 1

    def stop(self, timeout: int = 10):
        self.stop_calls.append(timeout)


class FakeContainers:
    def __init__(self, by_name: dict[str, FakeContainer],
                 by_label: dict[tuple[str, str], list[FakeContainer]]):
        self._by_name = by_name
        self._by_label = by_label

    def get(self, name: str) -> FakeContainer:
        if name not in self._by_name:
            raise LookupError(name)
        return self._by_name[name]

    def list(self, *, all: bool = False, filters: dict | None = None) -> list[FakeContainer]:
        if not filters or "label" not in filters:
            return list(self._by_name.values())
        labels = filters["label"]
        project = service = None
        for lbl in labels:
            k, _, v = lbl.partition("=")
            if k == "com.docker.compose.project":
                project = v
            elif k == "com.docker.compose.service":
                service = v
        key = (project, service)
        return list(self._by_label.get(key, []))


class FakeImages:
    def __init__(self):
        self.pulled: list[str] = []

    def pull(self, image: str):
        self.pulled.append(image)


class FakeDockerClient:
    def __init__(self, containers: FakeContainers, ping_ok: bool = True):
        self.containers = containers
        self.images = FakeImages()
        self._ping_ok = ping_ok

    def ping(self):
        if not self._ping_ok:
            raise RuntimeError("no socket")
        return True


def _executor(client: FakeDockerClient, **kw) -> DockerExecutor:
    return DockerExecutor(client_factory=lambda: client, **kw)


def _dispatch(ex: DockerExecutor, *, spec: dict, args: dict | None = None,
              delay: int = 0, timeout: int = 30) -> DispatchResult:
    return ex.dispatch(
        job_id="j1",
        operation_slug="test",
        target_host="",
        run_as_user=None,
        working_directory=None,
        command_script=json.dumps(spec),
        env_args=args or {},
        timeout_seconds=timeout,
        start_delay_seconds=delay,
        max_output_bytes=65536,
    )


# ---------------------------------------------------------------------------
# available()
# ---------------------------------------------------------------------------

def test_available_true_when_ping_ok():
    c = FakeDockerClient(FakeContainers({}, {}))
    assert _executor(c).available() is True


def test_available_false_when_ping_fails():
    c = FakeDockerClient(FakeContainers({}, {}), ping_ok=False)
    assert _executor(c).available() is False


# ---------------------------------------------------------------------------
# spec parsing + selector resolution
# ---------------------------------------------------------------------------

def test_empty_command_script_raises():
    ex = _executor(FakeDockerClient(FakeContainers({}, {})))
    with pytest.raises(ExecutorError, match="empty"):
        ex.dispatch(job_id="j", operation_slug="o", target_host="",
                    run_as_user=None, working_directory=None,
                    command_script="", env_args={}, timeout_seconds=30,
                    start_delay_seconds=0, max_output_bytes=1024)


def test_non_json_command_script_raises():
    ex = _executor(FakeDockerClient(FakeContainers({}, {})))
    with pytest.raises(ExecutorError, match="not valid JSON"):
        ex.dispatch(job_id="j", operation_slug="o", target_host="",
                    run_as_user=None, working_directory=None,
                    command_script="not-json-at-all",
                    env_args={}, timeout_seconds=30,
                    start_delay_seconds=0, max_output_bytes=1024)


def test_command_script_must_be_object():
    ex = _executor(FakeDockerClient(FakeContainers({}, {})))
    with pytest.raises(ExecutorError, match="JSON object"):
        ex.dispatch(job_id="j", operation_slug="o", target_host="",
                    run_as_user=None, working_directory=None,
                    command_script='["not", "an", "object"]',
                    env_args={}, timeout_seconds=30,
                    start_delay_seconds=0, max_output_bytes=1024)


def test_unsupported_action_raises():
    ex = _executor(FakeDockerClient(FakeContainers({}, {})))
    with pytest.raises(ExecutorError, match="unsupported docker action"):
        _dispatch(ex, spec={"action": "nuke", "container_name": "x"})


def test_no_selector_raises():
    ex = _executor(FakeDockerClient(FakeContainers({}, {})))
    with pytest.raises(ExecutorError, match="no container selector"):
        _dispatch(ex, spec={"action": "restart"})


def test_project_without_service_raises():
    ex = _executor(FakeDockerClient(FakeContainers({}, {})))
    with pytest.raises(ExecutorError, match="compose_project without"):
        _dispatch(ex, spec={"action": "restart", "compose_project": "p"})


def test_selector_from_arg_missing_raises():
    ex = _executor(FakeDockerClient(FakeContainers({}, {})))
    with pytest.raises(ExecutorError, match="wasn't supplied"):
        _dispatch(
            ex,
            spec={"action": "restart", "compose_project": "p",
                  "compose_service_from_arg": "svc"},
            args={},
        )


# ---------------------------------------------------------------------------
# restart by name / by compose labels
# ---------------------------------------------------------------------------

def test_restart_by_container_name_sync():
    c1 = FakeContainer("bawthub-frontend-1")
    client = FakeDockerClient(FakeContainers({"bawthub-frontend-1": c1}, {}))
    ex = _executor(client)
    result = _dispatch(ex, spec={"action": "restart",
                                  "container_name": "bawthub-frontend-1"})
    assert result.terminal_state == "succeeded"
    assert result.exit_code == 0
    assert "restarted bawthub-frontend-1" in (result.output or "")
    assert c1.restart_calls == [30]  # timeout_seconds=30 passed through


def test_restart_by_compose_labels_sync():
    c = FakeContainer("llm-bawt-crawl4ai")
    client = FakeDockerClient(FakeContainers(
        {"llm-bawt-crawl4ai": c},
        {("llm-bawt", "crawl4ai"): [c]},
    ))
    ex = _executor(client)
    result = _dispatch(
        ex,
        spec={"action": "restart",
              "compose_project": "llm-bawt",
              "compose_service": "crawl4ai"},
    )
    assert result.terminal_state == "succeeded"
    assert c.restart_calls == [30]


def test_selector_service_from_arg_wires_through():
    c = FakeContainer("llm-bawt-crawl4ai")
    client = FakeDockerClient(FakeContainers(
        {"llm-bawt-crawl4ai": c},
        {("llm-bawt", "crawl4ai"): [c]},
    ))
    ex = _executor(client)
    result = _dispatch(
        ex,
        spec={"action": "restart", "compose_project": "llm-bawt",
              "compose_service_from_arg": "svc"},
        args={"svc": "crawl4ai"},
    )
    assert result.terminal_state == "succeeded"
    assert c.restart_calls  # actually invoked


def test_container_not_found_raises():
    client = FakeDockerClient(FakeContainers({}, {}))
    ex = _executor(client)
    with pytest.raises(ExecutorError, match="container lookup failed|not found"):
        _dispatch(ex, spec={"action": "restart", "container_name": "nope"})


def test_multiple_matches_raises():
    a = FakeContainer("a")
    b = FakeContainer("b")
    client = FakeDockerClient(FakeContainers(
        {"a": a, "b": b},
        {("p", "s"): [a, b]},
    ))
    ex = _executor(client)
    with pytest.raises(ExecutorError, match="multiple containers"):
        _dispatch(ex, spec={"action": "restart",
                             "compose_project": "p", "compose_service": "s"})


# ---------------------------------------------------------------------------
# start / stop / pull
# ---------------------------------------------------------------------------

def test_start_action():
    c = FakeContainer("x")
    ex = _executor(FakeDockerClient(FakeContainers({"x": c}, {})))
    result = _dispatch(ex, spec={"action": "start", "container_name": "x"})
    assert c.start_calls == 1
    assert result.terminal_state == "succeeded"


def test_stop_action_passes_timeout():
    c = FakeContainer("x")
    ex = _executor(FakeDockerClient(FakeContainers({"x": c}, {})))
    _dispatch(ex, spec={"action": "stop", "container_name": "x"}, timeout=60)
    assert c.stop_calls == [60]


def test_pull_action_pulls_container_image_tag():
    c = FakeContainer("x", tags=["ghcr.io/foo/bar:latest"])
    client = FakeDockerClient(FakeContainers({"x": c}, {}))
    ex = _executor(client)
    _dispatch(ex, spec={"action": "pull", "container_name": "x"})
    assert client.images.pulled == ["ghcr.io/foo/bar:latest"]


def test_pull_without_tag_raises():
    c = FakeContainer("x", tags=[])
    ex = _executor(FakeDockerClient(FakeContainers({"x": c}, {})))
    with pytest.raises(ExecutorError, match="no tagged image"):
        _dispatch(ex, spec={"action": "pull", "container_name": "x"})


# ---------------------------------------------------------------------------
# self-restart forces a default delay
# ---------------------------------------------------------------------------

def test_self_restart_by_name_forces_delay():
    c = FakeContainer("llm-bawt-app")
    client = FakeDockerClient(FakeContainers({"llm-bawt-app": c}, {}))
    ex = _executor(client)
    result = _dispatch(
        ex,
        spec={"action": "restart", "container_name": "llm-bawt-app"},
        delay=0,
    )
    # No sync restart yet — delayed to daemon thread.
    assert c.restart_calls == []
    assert result.terminal_state == "succeeded"
    assert "delay=5s" in result.host_unit_name
    assert "self-restart" in (result.output or "")


def test_self_restart_by_compose_service_forces_delay():
    c = FakeContainer("llm-bawt-app")
    client = FakeDockerClient(FakeContainers(
        {"llm-bawt-app": c},
        {("llm-bawt", "app"): [c]},
    ))
    ex = _executor(client)
    result = _dispatch(
        ex,
        spec={"action": "restart", "compose_project": "llm-bawt",
              "compose_service": "app"},
        delay=0,
    )
    assert c.restart_calls == []
    assert result.terminal_state == "succeeded"


def test_explicit_delay_used_when_larger_than_default():
    c = FakeContainer("llm-bawt-app")
    client = FakeDockerClient(FakeContainers({"llm-bawt-app": c}, {}))
    ex = _executor(client)
    result = _dispatch(
        ex, spec={"action": "restart", "container_name": "llm-bawt-app"},
        delay=30,
    )
    assert "delay=30s" in result.host_unit_name


def test_non_self_restart_with_delay_still_delays():
    c = FakeContainer("llm-bawt-redis")
    client = FakeDockerClient(FakeContainers({"llm-bawt-redis": c}, {}))
    ex = _executor(client)
    result = _dispatch(
        ex, spec={"action": "restart", "container_name": "llm-bawt-redis"},
        delay=2,
    )
    assert c.restart_calls == []  # deferred to bg thread
    assert result.terminal_state == "succeeded"
    assert "delay=2s" in result.host_unit_name


def test_delayed_dispatch_actually_fires():
    """End-to-end delayed path: bg thread does eventually call restart."""
    c = FakeContainer("llm-bawt-redis")
    client = FakeDockerClient(FakeContainers({"llm-bawt-redis": c}, {}))
    ex = _executor(client)
    _dispatch(
        ex, spec={"action": "restart", "container_name": "llm-bawt-redis"},
        delay=0,  # keep sync-adjacent: not self-target, no forced delay
    )
    # Sanity: the sync path DID fire immediately since it's not self.
    assert c.restart_calls == [30]

    # Now force a delayed dispatch via an explicit delay of 0.1s worth via
    # the "self target" trigger — reuse the app container.
    c2 = FakeContainer("llm-bawt-app")
    client2 = FakeDockerClient(FakeContainers({"llm-bawt-app": c2}, {}))
    ex2 = _executor(client2)
    ex2._fire_delayed("restart", {"name": "llm-bawt-app"}, delay_seconds=0, timeout_seconds=30)
    # Give the daemon thread a moment.
    for _ in range(50):
        if c2.restart_calls:
            break
        time.sleep(0.02)
    assert c2.restart_calls, "delayed restart never fired"


# ---------------------------------------------------------------------------
# unavailable executor
# ---------------------------------------------------------------------------

def test_dispatch_when_daemon_unreachable_raises_available():
    client = FakeDockerClient(FakeContainers({}, {}), ping_ok=False)
    ex = _executor(client)
    with pytest.raises(ExecutorError, match="cannot reach"):
        _dispatch(ex, spec={"action": "restart", "container_name": "x"})


# ---------------------------------------------------------------------------
# reconcile is a no-op (docker terminal-marks at dispatch time)
# ---------------------------------------------------------------------------

def test_reconcile_returns_none_state():
    ex = _executor(FakeDockerClient(FakeContainers({}, {})))
    r = ex.reconcile(job_id="j", target_host="", status_file_path="",
                     log_file_path="")
    assert r.state is None


def test_supported_actions_enumeration():
    assert SUPPORTED_ACTIONS == frozenset({"restart", "start", "stop", "pull"})
