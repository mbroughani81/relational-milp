"""Legacy (pre-unified) result formatting for the old-schema runner shims.

The ``nnequiv run`` CLI uses :mod:`nnequiv.report.schema` / :mod:`.progress`.
These older ``expected=unsat:yes`` / phase-timing helpers are what the moved
``benchmarks.run_*`` shims still emit, so their CSV schema and progress lines
stay byte-for-byte identical for the fleet. ``benchmarks.common`` re-exports them.
"""

from __future__ import annotations

import sys
from typing import TextIO

from nnequiv.core import InstanceResult, SolveStats


def format_expected(result: InstanceResult) -> str:
    if result.expected_status is None:
        return ""
    matched = "yes" if result.matched_expected else "no"
    return f"{result.expected_status}:{matched}"


def format_solve_stats(stats: list[SolveStats]) -> str:
    parts = []
    for solve_stats in stats:
        timing_text = ",".join(
            f"{phase}={runtime_sec:.3f}" for phase, runtime_sec in solve_stats.timings
        )
        parts.append(f"{solve_stats.name}[{timing_text}]")
    measured_total_sec = sum(solve_stats.measured_total_sec for solve_stats in stats)
    parts.append(f"total={measured_total_sec:.3f}")
    return " ".join(parts)


def print_progress(
    index: int,
    total: int,
    result: InstanceResult,
    *,
    extra_fields: dict[str, str] | None = None,
    stream: TextIO | None = None,
) -> None:
    middle = "".join(f"{key}={value} " for key, value in (extra_fields or {}).items())
    phase_text = f" phases: {format_solve_stats(result.stats)}" if result.stats else ""
    print(
        f"[{index}/{total}] {result.instance_id}: "
        f"status={result.status} {middle}"
        f"expected={format_expected(result) or '-'} "
        f"runtime_sec={result.runtime_sec:.3f} epsilon={result.epsilon:.17g}"
        f"{phase_text}",
        file=stream if stream is not None else sys.stdout,
        flush=True,
    )
