"""Backward-compat facade over the pure domain model, plus harness helpers.

The domain model (Instance, polytopes, properties, results) now lives in
:mod:`nnequiv.core`; this module re-exports it so existing ``from
benchmarks.common import ...`` sites keep working during the migration. The
suite-option parser and the progress/format helpers below are report/CLI
concerns that will move to ``nnequiv.report`` / the suite loader in a later
phase; import the domain types from :mod:`nnequiv.core` in new code.
"""

from __future__ import annotations

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
from nnequiv.suites import SuiteOptions, parse_suite_options
from nnequiv.report.legacy import (
    format_expected,
    format_solve_stats,
    print_progress,
)


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
