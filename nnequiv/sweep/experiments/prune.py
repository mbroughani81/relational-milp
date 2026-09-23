"""Pruning experiment grid: ReluDiff MNIST nets vs a magnitude-pruned copy.

Verifies ``|N(x)[c] - P_p(N)(x)[c]| <= epsilon`` across four verifiers, sweeping
the pruning rate. three_pixel is kept in the grid but note it opens different
pixels for MILP vs ReluDiff/NeuroDiff, so cross-family comparisons in that mode
are invalid (see the repo notes). The shared plumbing lives in
:mod:`nnequiv.sweep.engine`.
"""

from __future__ import annotations

from pathlib import Path

from nnequiv.sweep.engine import (
    Experiment,
    build_plan as _build_plan,
    run_cli,
    venv_python,
)
from nn_equivalence.paths import runtime_path

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = REPO_ROOT / "prune-experiment"
RESULTS_DIR = BASE_DIR / "results"
SKIP_FILE = BASE_DIR / "skip.conf"

METHODS = ("milp_abcrown", "reludiff", "neurodiff")
ARCHS = ("mnist_relu_3_100", "mnist_relu_2_512", "mnist_relu_4_1024")
MODES = ("global", "three_pixel")
RATES = (5, 20, 50)
LIMIT = 100

# Per-instance budget in seconds, held equal across methods. milp_abcrown splits
# each instance into two independently timed CPLEX solves, so its per-solve value
# is half the budget; the diff tools get the whole budget as one subprocess.
INSTANCE_BUDGET = 600
SOLVES_PER_INSTANCE = {"milp_abcrown": 2}

EPSILON = "1.0"
RADIUS = "3"  # +/- radius/255 per pixel, global-mode input region
CROWN_PROFILE = "relu-kfsb"


def _python() -> str:
    return venv_python(REPO_ROOT)


def tag_for(method: str, arch: str, mode: str, rate: int) -> str:
    return f"{method}__{arch}__{mode}__p{rate}"


def iter_cells():
    """Yield (method, arch, mode, rate, tag) for every grid cell."""
    for method in METHODS:
        for arch in ARCHS:
            for mode in MODES:
                for rate in RATES:
                    yield method, arch, mode, rate, tag_for(method, arch, mode, rate)


def timeout_for(method: str) -> int:
    return INSTANCE_BUDGET // SOLVES_PER_INSTANCE.get(method, 1)


def _common_opts(method: str, arch: str, mode: str, rate: int) -> list[str]:
    sparsity = f"{rate / 100:.4f}"
    return [
        "--suite", "pruning_mnist",
        "--suite-options", f"networks={arch}",
        "--suite-options", f"modes={mode}",
        "--suite-options", f"sparsity={sparsity}",
        "--suite-options", f"radius={RADIUS}",
        "--suite-options", f"epsilon={EPSILON}",
        "--suite-options", f"limit={LIMIT}",
        "--suite-options", f"timeout={timeout_for(method)}",
    ]


def build_command(method: str, arch: str, mode: str, rate: int, out: Path) -> list[str]:
    """Return the argv that runs one cell, writing its clean CSV to ``out``."""
    py = _python()
    opts = _common_opts(method, arch, mode, rate)
    if method == "milp_abcrown":
        return [py, "-m", "benchmarks.run_pyomo", *opts,
                "--solver", "cplex", "--bound-tightening", "abcrown", "--csv", str(out)]
    if method == "abcrown":
        return [py, "-m", "benchmarks.run_crown", *opts,
                "--profile", CROWN_PROFILE, "--csv", str(out)]
    if method in ("reludiff", "neurodiff"):
        binary = runtime_path("third_party/NeuroDiff-ASE2020-Artifact/DiffNN-Code", method)
        return [py, "-m", "benchmarks.run_diffverifier", "--tool", method,
                "--binary", str(binary), *opts, "--csv", str(out)]
    raise ValueError(f"unknown method: {method}")


def build_plan(skip_patterns: list[str] | None = None) -> list[dict]:
    return _build_plan(list(iter_cells()), build_command, RESULTS_DIR, SKIP_FILE, skip_patterns)


EXPERIMENT = Experiment(
    name="prune",
    base_dir=BASE_DIR,
    repo_root=REPO_ROOT,
    iter_cells=iter_cells,
    build_command=build_command,
)


def main(argv: list[str] | None = None) -> int:
    return run_cli(EXPERIMENT, argv)


__all__ = [
    "METHODS",
    "ARCHS",
    "MODES",
    "RATES",
    "LIMIT",
    "INSTANCE_BUDGET",
    "SOLVES_PER_INSTANCE",
    "EPSILON",
    "RADIUS",
    "CROWN_PROFILE",
    "tag_for",
    "iter_cells",
    "timeout_for",
    "build_command",
    "build_plan",
    "EXPERIMENT",
    "main",
]
