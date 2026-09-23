"""Distillation experiment grid: the GPE paper's MNIST 8x8 teacher/student pairs.

milp_abcrown only (every pair differs in architecture, so ReluDiff/NeuroDiff
cannot run it). ``linf`` sweeps an epsilon ladder at each pair's published
radius; ``top1`` sweeps an input-radius ladder at a fixed strictness margin. The
shared plumbing lives in :mod:`nnequiv.sweep.engine`.
"""

from __future__ import annotations

from pathlib import Path

from nnequiv.sweep.engine import (
    Experiment,
    build_plan as _build_plan,
    run_cli,
    venv_python,
)
from nn_equivalence.nnequiv_benchmarks import PAIRS

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = REPO_ROOT / "distillation-experiment"
RESULTS_DIR = BASE_DIR / "results"
SKIP_FILE = BASE_DIR / "skip.conf"

METHODS = ("milp_abcrown",)
PAIR_IDS = tuple(PAIRS)

# epsilon ladder for linf; 15.0 is the paper's MNIST_large value (comparable with
# their Table I), the rest bracket it by decade.
EPSILONS = (1.0, 5.0, 15.0, 50.0)

# Radius ladder for top1 (which has no epsilon); 0.2 and 0.7 are the paper's own.
RADII = (0.1, 0.2, 0.3, 0.7)

# Strictness margin for top1 (NOT a tolerance on the outputs).
TOP1_MARGIN = "1e-4"

# Ten published cluster centers per cell.
LIMIT = 10

# Per-instance budget in seconds. linf splits into two independently timed
# directions, top1 is a single symmetric solve, so the per-solve value differs
# by property to keep the wall-clock budget equal.
INSTANCE_BUDGET = 1200
SOLVES_PER_INSTANCE = {"linf": 2, "top1": 1}

CROWN_PROFILE = "relu-kfsb"


def _python() -> str:
    return venv_python(REPO_ROOT)


def tag_for(method: str, pair_id: str, property_kind: str, knob: float) -> str:
    """Cell tag; the knob is the epsilon for linf and the radius for top1."""
    prefix = "e" if property_kind == "linf" else "r"
    return f"{method}__{pair_id}__{property_kind}__{prefix}{knob:g}"


def iter_cells():
    """Yield (method, pair_id, property_kind, knob, tag) for every grid cell."""
    for method in METHODS:
        for pair_id in PAIR_IDS:
            for epsilon in EPSILONS:
                yield method, pair_id, "linf", epsilon, tag_for(
                    method, pair_id, "linf", epsilon
                )
            for radius in RADII:
                yield method, pair_id, "top1", radius, tag_for(
                    method, pair_id, "top1", radius
                )


def timeout_for(property_kind: str) -> int:
    return INSTANCE_BUDGET // SOLVES_PER_INSTANCE[property_kind]


def _common_opts(pair_id: str, property_kind: str, knob: float) -> list[str]:
    if property_kind == "linf":
        knob_opts = [f"epsilon={knob:g}"]
    else:
        knob_opts = [f"radius={knob:g}", f"epsilon={TOP1_MARGIN}"]
    return [
        "--suite", "distillation_mnist8",
        "--suite-options", f"pairs={pair_id}",
        "--suite-options", f"property={property_kind}",
        *[arg for opt in knob_opts for arg in ("--suite-options", opt)],
        "--suite-options", f"limit={LIMIT}",
        "--suite-options", f"timeout={timeout_for(property_kind)}",
    ]


def build_command(
    method: str,
    pair_id: str,
    property_kind: str,
    knob: float,
    out: Path,
) -> list[str]:
    """Return the argv that runs one cell, writing its clean CSV to ``out``."""
    py = _python()
    opts = _common_opts(pair_id, property_kind, knob)
    if method == "milp_abcrown":
        return [py, "-m", "benchmarks.run_pyomo", *opts,
                "--solver", "cplex", "--bound-tightening", "abcrown", "--csv", str(out)]
    if method == "abcrown":
        raise ValueError(
            "abcrown cannot express the linf or top1 properties; this sweep is "
            "milp_abcrown only (see the module docstring)"
        )
    raise ValueError(f"unknown method: {method}")


def build_plan(skip_patterns: list[str] | None = None) -> list[dict]:
    return _build_plan(list(iter_cells()), build_command, RESULTS_DIR, SKIP_FILE, skip_patterns)


EXPERIMENT = Experiment(
    name="distillation",
    base_dir=BASE_DIR,
    repo_root=REPO_ROOT,
    iter_cells=iter_cells,
    build_command=build_command,
)


def main(argv: list[str] | None = None) -> int:
    return run_cli(EXPERIMENT, argv)


__all__ = [
    "METHODS",
    "PAIR_IDS",
    "EPSILONS",
    "RADII",
    "TOP1_MARGIN",
    "LIMIT",
    "INSTANCE_BUDGET",
    "SOLVES_PER_INSTANCE",
    "CROWN_PROFILE",
    "tag_for",
    "iter_cells",
    "timeout_for",
    "build_command",
    "build_plan",
    "EXPERIMENT",
    "main",
]
