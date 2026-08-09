"""Test-suite-wide fixtures and the deselection visibility guard (TASK-780).

The default marker filter in ``pyproject.toml`` (`-m 'not service and not
llm_call and not integration'`) silently drops whole test files — the biggest
being ``test_inter_bot_delivery.py`` (~25 tests, module-scoped
``pytestmark = pytest.mark.integration``). Under a bare ``pytest`` / ``make
test`` those files vanish from the report with no visible signal, so an agent
verifying an inter-bot / dispatcher / capability change could see green and
have executed none of the relevant tests. TASK-780.

The ``pytest_deselected`` hook below records every item pytest drops (marker
filter, ``-k``, ``--lf``, whatever) and the ``pytest_terminal_summary`` hook
prints a bright end-of-run warning naming any FILE that was fully deselected
(zero surviving items) plus the exact invocation to run them.

The default stays hermetic — this hook does not change what runs, only what
gets reported. A dev/CI run without a live DB still passes; the warning just
tells whoever reads the output "there are ~25 more tests you didn't run,
here's how."
"""

from __future__ import annotations

from collections import defaultdict


def pytest_configure(config):
    # Per-file counters of deselected vs surviving items. Populated across the
    # collection lifecycle (pytest_deselected + pytest_collection_finish) and
    # read in pytest_terminal_summary.
    config._task780_deselected_by_file = defaultdict(int)
    config._task780_surviving_by_file = defaultdict(int)


def pytest_deselected(items):
    # pytest calls this with every item removed by the -m/-k expression.
    if not items:
        return
    counters = getattr(items[0].config, "_task780_deselected_by_file", None)
    if counters is None:
        return
    for item in items:
        counters[str(item.path)] += 1


def pytest_collection_finish(session):
    # ``session.items`` here is the POST-deselection surviving list — the
    # tests that will actually run. Anything in _task780_deselected_by_file
    # whose file has 0 survivors is fully dropped by the filter.
    # (pytest_collection_modifyitems runs BEFORE deselection under `-m`,
    # so its items list would count the deselected tests as "selected"
    # and make the diff useless. Use this hook instead.)
    for item in session.items:
        session.config._task780_surviving_by_file[str(item.path)] += 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    deselected = getattr(config, "_task780_deselected_by_file", {}) or {}
    surviving = getattr(config, "_task780_surviving_by_file", {}) or {}
    fully_dropped = sorted(
        path for path, count in deselected.items()
        if count > 0 and surviving.get(path, 0) == 0
    )
    if not fully_dropped:
        return
    tr = terminalreporter
    tr.write_sep("=", "MARKER-DESELECTED FILES (TASK-780)", yellow=True, bold=True)
    tr.write_line(
        "The default filter (-m 'not service and not llm_call and not integration')",
        yellow=True,
    )
    tr.write_line(
        "silently dropped these whole files. Green above does NOT mean they passed:",
        yellow=True,
    )
    for path in fully_dropped:
        count = deselected.get(path, 0)
        tr.write_line(f"  - {path}  ({count} test{'s' if count != 1 else ''})", yellow=True)
    tr.write_line("", yellow=True)
    tr.write_line("To run them:", yellow=True, bold=True)
    tr.write_line("  make test-integration   # only -m integration", yellow=True)
    tr.write_line(
        "  make test-all           # everything (hermetic + integration + service + llm_call)",
        yellow=True,
    )
