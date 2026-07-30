"""Concurrency-safe runtime schema bootstrap helpers.

Runtime DDL is still a compatibility bridge for deployments that predate a
given table or column.  This module keeps that bridge off steady-state request
paths by making each logical schema unit single-flight per SQLAlchemy engine.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Hashable
from weakref import WeakKeyDictionary, WeakSet

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

logger = logging.getLogger(__name__)

_ADVISORY_LOCK_NAME = "llm-bawt-runtime-schema"

# One process-local order for every runtime schema unit.  The PostgreSQL
# advisory lock below extends the same ordering across app processes.
_bootstrap_lock = threading.RLock()
_guards: WeakSet[SchemaBootstrapGuard] = WeakSet()


class SchemaBootstrapGuard:
    """Run a transactional schema callback once per engine and logical scope."""

    def __init__(self) -> None:
        self._completed: WeakKeyDictionary[Engine, set[Hashable]] = WeakKeyDictionary()
        _guards.add(self)

    def run(
        self,
        engine: Engine,
        logical_scope: Hashable,
        bootstrap: Callable[[Connection], None],
    ) -> bool:
        """Run ``bootstrap`` once, returning whether this call performed it.

        Completion is recorded only after ``engine.begin()`` exits cleanly and
        commits.  Exceptions therefore leave the scope retryable.
        """

        with _bootstrap_lock:
            completed = self._completed.setdefault(engine, set())
            if logical_scope in completed:
                return False

            try:
                with engine.begin() as conn:
                    if conn.dialect.name == "postgresql":
                        conn.execute(
                            text(
                                "SELECT pg_advisory_xact_lock("
                                "hashtext(:schema_lock_name))"
                            ),
                            {"schema_lock_name": _ADVISORY_LOCK_NAME},
                        )
                    bootstrap(conn)
            except Exception:
                logger.exception(
                    "Runtime schema bootstrap failed for scope %r", logical_scope
                )
                raise

            # The transaction has committed successfully at this point.
            completed.add(logical_scope)
            logger.info("Runtime schema ready: %s", logical_scope)
            return True

    def reset_for_tests(self, engine: Engine | None = None) -> None:
        """Forget completion state. Intended only for isolated tests."""

        with _bootstrap_lock:
            if engine is None:
                self._completed.clear()
            else:
                self._completed.pop(engine, None)


def reset_schema_guards_for_tests(engine: Engine | None = None) -> None:
    """Reset all registered guards without touching or disposing engines."""

    with _bootstrap_lock:
        for guard in list(_guards):
            guard.reset_for_tests(engine)
