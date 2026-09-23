"""Sweep experiment registry.

Loaded lazily so importing this package (e.g. for the ``nnequiv sweep`` choices)
needs no RUNTIME_DIR and does not import the benchmark fixtures until a specific
experiment is requested.
"""

from __future__ import annotations

from nnequiv.sweep.engine import Experiment

EXPERIMENT_NAMES = ("prune", "distillation")


def get_experiment(name: str) -> Experiment:
    if name == "prune":
        from nnequiv.sweep.experiments.prune import EXPERIMENT

        return EXPERIMENT
    if name == "distillation":
        from nnequiv.sweep.experiments.distillation import EXPERIMENT

        return EXPERIMENT
    raise ValueError(f"unknown experiment {name!r}; choose from {list(EXPERIMENT_NAMES)}")


__all__ = ["EXPERIMENT_NAMES", "get_experiment"]
