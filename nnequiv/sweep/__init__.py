"""Experiment sweeps: one shared engine + per-experiment grids.

The ``*-experiment/recreate.py`` scripts and ``scripts/worker.py`` still drive
these (their ``--dry-run`` JSON is unchanged); ``nnequiv sweep <name>`` is the
equivalent entry point on the unified CLI.
"""

from nnequiv.sweep.engine import Experiment, build_plan, run, run_cli
from nnequiv.sweep.experiments import EXPERIMENT_NAMES, get_experiment

__all__ = [
    "Experiment",
    "build_plan",
    "run",
    "run_cli",
    "EXPERIMENT_NAMES",
    "get_experiment",
]
