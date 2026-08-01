"""Low-overhead in-process concurrency and error counters for proxy logs."""

from __future__ import annotations

from collections import Counter
from threading import Lock


class ProxyObservability:
    """Tracks active streams and cumulative transient upstream failures.

    The proxy deliberately has no semaphore: current evidence shows real
    parallelism and a responsive event loop. These counters make future
    saturation visible without serializing unrelated bots.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._active_provider: Counter[str] = Counter()
        self._active_account: Counter[tuple[str, str]] = Counter()
        self._errors_429: Counter[str] = Counter()
        self._errors_5xx: Counter[str] = Counter()

    def begin(self, provider: str, account_hash: str) -> tuple[int, int]:
        with self._lock:
            self._active_provider[provider] += 1
            self._active_account[(provider, account_hash)] += 1
            return (
                self._active_provider[provider],
                self._active_account[(provider, account_hash)],
            )

    def end(self, provider: str, account_hash: str) -> None:
        with self._lock:
            self._active_provider[provider] -= 1
            self._active_account[(provider, account_hash)] -= 1

    def record_error(self, provider: str, status: int | None) -> None:
        with self._lock:
            if status == 429:
                self._errors_429[provider] += 1
            elif status is not None and 500 <= status <= 599:
                self._errors_5xx[provider] += 1

    def error_totals(self, provider: str) -> tuple[int, int]:
        with self._lock:
            return self._errors_429[provider], self._errors_5xx[provider]


OBSERVABILITY = ProxyObservability()
