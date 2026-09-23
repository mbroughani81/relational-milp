"""Backward-compat shim: moved to :mod:`nnequiv.verifiers.milp.cplex_log`."""

from nnequiv.verifiers.milp.cplex_log import (
    CplexPresolveLogStats,
    parse_cplex_presolve_log,
)

__all__ = ["CplexPresolveLogStats", "parse_cplex_presolve_log"]
