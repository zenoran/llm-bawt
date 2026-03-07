"""Lightweight InfluxDB v2 metrics emitter using the HTTP write API.

No extra dependency — uses the stdlib urllib to POST line-protocol points.
Falls back to no-op if INFLUXDB_URL is not configured.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Flush interval in seconds
_FLUSH_INTERVAL = 10.0


@dataclass
class _MetricsConfig:
    url: str = ""          # e.g. http://10.0.2.30:8086
    token: str = ""
    org: str = "homeassistant"
    bucket: str = "llm-bawt"

    @classmethod
    def from_env(cls) -> _MetricsConfig:
        return cls(
            url=os.environ.get("INFLUXDB_URL", ""),
            token=os.environ.get("INFLUXDB_TOKEN", ""),
            org=os.environ.get("INFLUXDB_ORG", "homeassistant"),
            bucket=os.environ.get("INFLUXDB_BUCKET", "llm-bawt"),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.token)


class MetricsCollector:
    """Thread-safe metrics collector that batches and flushes to InfluxDB."""

    def __init__(self, config: _MetricsConfig | None = None) -> None:
        self._config = config or _MetricsConfig.from_env()
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._tags: dict[str, str] = {}
        self._timer: threading.Timer | None = None

        if self._config.enabled:
            self._start_flush_timer()
            logger.info("Metrics enabled -> %s (bucket=%s)", self._config.url, self._config.bucket)
        else:
            logger.info("Metrics disabled (INFLUXDB_URL or INFLUXDB_TOKEN not set)")

    def set_tags(self, **tags: str) -> None:
        """Set global tags applied to all points (e.g. host, service)."""
        with self._lock:
            self._tags.update(tags)

    def incr(self, name: str, value: int = 1, **tags: str) -> None:
        """Increment a counter."""
        key = self._make_key(name, tags)
        with self._lock:
            self._counters[key] += value

    def gauge(self, name: str, value: float, **tags: str) -> None:
        """Set a gauge value."""
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] = value

    def flush(self) -> None:
        """Flush all buffered metrics to InfluxDB."""
        if not self._config.enabled:
            return

        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            self._counters.clear()
            # gauges persist until overwritten

        if not counters and not gauges:
            return

        ts_ns = int(time.time() * 1e9)
        lines: list[str] = []

        for key, value in counters.items():
            lines.append(f"{key} count={value}i {ts_ns}")

        for key, value in gauges.items():
            lines.append(f"{key} value={value} {ts_ns}")

        body = "\n".join(lines)
        self._write(body)

    def close(self) -> None:
        if self._timer:
            self._timer.cancel()
        self.flush()

    # -- internal --

    def _make_key(self, measurement: str, extra_tags: dict[str, str]) -> str:
        """Build InfluxDB line-protocol measurement + tag set."""
        all_tags = {**self._tags, **extra_tags}
        if all_tags:
            tag_str = ",".join(f"{k}={_escape_tag(v)}" for k, v in sorted(all_tags.items()))
            return f"{measurement},{tag_str}"
        return measurement

    def _write(self, body: str) -> None:
        url = (
            f"{self._config.url}/api/v2/write"
            f"?org={self._config.org}&bucket={self._config.bucket}&precision=ns"
        )
        req = urllib.request.Request(
            url,
            data=body.encode(),
            headers={
                "Authorization": f"Token {self._config.token}",
                "Content-Type": "text/plain; charset=utf-8",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        except (urllib.error.URLError, OSError) as e:
            logger.warning("Failed to write metrics to InfluxDB: %s", e)

    def _start_flush_timer(self) -> None:
        def _tick() -> None:
            try:
                self.flush()
            except Exception:
                logger.exception("Metrics flush error")
            finally:
                self._start_flush_timer()

        self._timer = threading.Timer(_FLUSH_INTERVAL, _tick)
        self._timer.daemon = True
        self._timer.start()


def _escape_tag(v: str) -> str:
    """Escape special characters in InfluxDB tag values."""
    return v.replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")


# Module-level singleton
_collector: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
