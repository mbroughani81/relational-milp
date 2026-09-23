#!/usr/bin/env python3
"""Distillation experiment sweep runner (mirrors prune-experiment/recreate.py).

Runs our relational MILP over the knowledge-distillation benchmarks of Teuber
et al. 2021 ("Geometric Path Enumeration for Equivalence Verification of Neural
Networks") -- their own MNIST 8x8 teacher/student network pairs, input regions
and properties -- so the solve times sit next to their published NNEquiv /
MilpEquiv numbers (Table I, Fig. 3-4).

Two properties, each with its own difficulty knob:

  * ``linf``  max_i |z1(x)_i - z2(x)_i| <= epsilon.  Swept over an epsilon
              ladder, the paper's Fig. 4 axis, at each pair's published radius.
  * ``top1``  argmax z1(x) == argmax z2(x).  Has no epsilon, so the knob is the
              input radius instead; epsilon is only a strictness margin (see
              ``encoder_pyomo.TOP1_MIN_MARGIN``).

Every pair differs in architecture, so ReluDiff/NeuroDiff cannot run these and
the sweep is ``milp_abcrown`` only; the comparison baseline is the paper.

Each cell verifies the ten published cluster centers. Instances whose two
networks already disagree at the center point are kept, not filtered: the
results CSV records ``center_top1_agrees`` / ``center_linf_gap`` so the
analysis can separate a genuine proof from a cheap counterexample.

Skip logic matches the prune sweep -- a cell is skipped when its tag matches a
glob in skip.conf, or (run mode only) its result CSV already exists.

Modes:
  * default        run every not-skipped cell whose result CSV is missing.
  * --dry-run      print the plan as a JSON array of {"result", "command"}
                   objects; scripts/worker.py consumes this to claim + run
                   cells across a fleet.

RUNTIME_DIR must be set (it locates the downloaded benchmark fixtures).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nn_equivalence.nnequiv_benchmarks import PAIRS  # noqa: E402

RESULTS_DIR = SCRIPT_DIR / "results"
LOGS_DIR = SCRIPT_DIR / "logs"
SKIP_FILE = SCRIPT_DIR / "skip.conf"

# --------------------------------------------------------------------------- #
# Fixed sweep grid. Edit here to change the experiment; there are no env/CLI
# overrides by design, matching prune-experiment/recreate.py.
# --------------------------------------------------------------------------- #
METHODS = ("milp_abcrown",)
PAIR_IDS = tuple(PAIRS)

# epsilon ladder for linf. 15.0 is the paper's own MNIST_large-epsilon value, so
# that cell is directly comparable with Table I; the rest bracket it by decade.
EPSILONS = (1.0, 5.0, 15.0, 50.0)

# Radius ladder for top1, which has no epsilon. All of these are published for
# every cluster center; 0.2 and 0.7 are the paper's own choices.
RADII = (0.1, 0.2, 0.3, 0.7)

# Strictness margin for top1 (NOT a tolerance on the outputs).
TOP1_MARGIN = "1e-4"

# Ten published cluster centers per cell.
LIMIT = 10

# Per-instance verification budget in seconds. milp_abcrown splits a linf
# instance into two independently timed directions (run_pyomo.create_solver
# hands each its own CPLEX timelimit), while top1 is a single symmetric solve,
# so the per-solve value differs by property to keep the budget equal.
INSTANCE_BUDGET = 1200
SOLVES_PER_INSTANCE = {"linf": 2, "top1": 1}

CROWN_PROFILE = "relu-kfsb"


def _python() -> str:
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else (sys.executable or "python3")


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
    """The ``--suite-options timeout``, in seconds, for ``property_kind``.

    Divides INSTANCE_BUDGET by how many separately-timed solves the property
    needs per instance, so every cell gets the same wall-clock budget.
    """
    return INSTANCE_BUDGET // SOLVES_PER_INSTANCE[property_kind]


def _common_opts(pair_id: str, property_kind: str, knob: float) -> list[str]:
    # linf sweeps epsilon at the pair's published radius (radius left empty, so
    # the suite uses it); top1 sweeps radius at a fixed strictness margin.
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
        # run_crown only encodes the single-output margin, so it cannot express
        # linf or top1; kept here only so an explicit request fails loudly.
        raise ValueError(
            "abcrown cannot express the linf or top1 properties; this sweep is "
            "milp_abcrown only (see the module docstring)"
        )
    raise ValueError(f"unknown method: {method}")


def load_skip_patterns(path: Path = SKIP_FILE) -> list[str]:
    """Glob patterns from skip.conf (one per line, # comments and blanks ignored)."""
    if not path.exists():
        return []
    patterns = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            patterns.append(line)
    return patterns


def tag_skipped(tag: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(tag, pat) for pat in patterns)


def iter_jobs(skip_patterns: list[str], results_dir: Path = RESULTS_DIR):
    """Yield (out_path, argv) for every cell not excluded by skip.conf."""
    for method, pair_id, property_kind, knob, tag in iter_cells():
        if tag_skipped(tag, skip_patterns):
            continue
        out = results_dir / f"{tag}.csv"
        yield out, build_command(method, pair_id, property_kind, knob, out)


def build_plan(skip_patterns: list[str] | None = None,
               results_dir: Path = RESULTS_DIR) -> list[dict]:
    """The plan as a list of {"result", "command"} objects (skip.conf applied)."""
    if skip_patterns is None:
        skip_patterns = load_skip_patterns()
    return [
        {"result": str(out), "command": shlex.join(cmd)}
        for out, cmd in iter_jobs(skip_patterns, results_dir)
    ]


def run() -> int:
    """Run every not-skipped cell whose result CSV is missing, sequentially."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for out, cmd in iter_jobs(load_skip_patterns()):
        if out.exists():
            print(f"--- skip (exists) {out.name}")
            continue
        log = LOGS_DIR / f"{out.stem}.log"
        print(f">>> {out.stem}")
        with open(log, "w", encoding="utf-8") as log_file:
            rc = subprocess.run(
                cmd, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT
            ).returncode
        if rc == 0:
            print(f"    done -> {out.name}")
        else:
            failures += 1
            print(f"    FAILED (rc={rc}); see {log.relative_to(REPO_ROOT)}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan as a JSON array of {result, command} objects and exit",
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(build_plan(), indent=2))
        return 0
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
