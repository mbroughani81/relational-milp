"""Backward-compat facade over the pure domain model, plus harness helpers.

The domain model (Instance, polytopes, properties, results) now lives in
:mod:`nnequiv.core`; this module re-exports it so existing ``from
benchmarks.common import ...`` sites keep working during the migration. The
suite-option parser and the progress/format helpers below are report/CLI
concerns that will move to ``nnequiv.report`` / the suite loader in a later
phase; import the domain types from :mod:`nnequiv.core` in new code.
"""

from __future__ import annotations

import json
import sys
from typing import TextIO

from nnequiv.core import (
    EQUIVALENCE_PROPERTIES,
    AbstractPolytope,
    Bounds,
    EquivalenceProperty,
    HalfSpace,
    Hyperrectangle,
    Instance,
    InstanceResult,
    InstanceStatus,
    InstanceSuite,
    NeuralNetwork,
    SolveStats,
    constraints_list,
    contains,
    dim,
    validate_instance,
)

SuiteOptions = dict[str, str]


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


def parse_suite_options(raw_options: list[str] | None) -> SuiteOptions:
    options: SuiteOptions = {}
    for raw_option in raw_options or []:
        raw_option = raw_option.strip()
        if not raw_option:
            continue
        if raw_option.startswith("{"):
            parsed = json.loads(raw_option)
            if not isinstance(parsed, dict):
                raise ValueError("--suite-options JSON value must be an object")
            options.update({str(key): str(value) for key, value in parsed.items()})
            continue

        for part in raw_option.split(";"):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(
                    "--suite-options entries must be KEY=VALUE pairs"
                )
            key, value = part.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError("--suite-options keys must be non-empty")
            options[key] = value.strip()

    return options


__all__ = [
    # re-exported domain model (canonical home: nnequiv.core)
    "AbstractPolytope",
    "HalfSpace",
    "Hyperrectangle",
    "constraints_list",
    "contains",
    "dim",
    "EquivalenceProperty",
    "EQUIVALENCE_PROPERTIES",
    "InstanceStatus",
    "SolveStats",
    "InstanceResult",
    "Instance",
    "InstanceSuite",
    "validate_instance",
    "Bounds",
    "NeuralNetwork",
    # harness helpers still living here for now
    "SuiteOptions",
    "format_expected",
    "format_solve_stats",
    "print_progress",
    "parse_suite_options",
]
