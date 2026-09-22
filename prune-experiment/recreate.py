#!/usr/bin/env python3
"""Pruning experiment sweep runner (simplified rewrite of recreate.sh).

Verifies ``|N(x)[c] - P_p(N)(x)[c]| <= epsilon`` for the ReluDiff MNIST networks
N and their globally magnitude-pruned counterparts P_p(N), across four verifiers,
sweeping the pruning rate. It builds one command per (method, arch, mode, rate)
cell and either runs the undone ones or prints the plan.

Skip logic is deliberately minimal — a cell is skipped when:
  * its tag matches a glob pattern in skip.conf, OR
  * (run mode only) its result CSV already exists.

Modes:
  * default        run every not-skipped cell whose result CSV is missing.
  * --dry-run      print the plan as a JSON array of {"result", "command"}
                   objects (every cell not excluded by skip.conf). A worker
                   consumes this to claim + run cells across a fleet.

RUNTIME_DIR must be set (it locates the ReluDiff/NeuroDiff binaries).
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

from nn_equivalence.paths import runtime_path  # noqa: E402

RESULTS_DIR = SCRIPT_DIR / "results"
LOGS_DIR = SCRIPT_DIR / "logs"
SKIP_FILE = SCRIPT_DIR / "skip.conf"

# --------------------------------------------------------------------------- #
# Fixed sweep grid. Edit here to change the experiment; there are no env/CLI
# overrides by design (the old EXP_* / --light machinery is gone).
# three_pixel is intentionally omitted: it opens different pixels for MILP vs
# ReluDiff/NeuroDiff, so cross-family comparisons are invalid. Add it back to
# MODES only for within-family runs.
# --------------------------------------------------------------------------- #
METHODS = ("milp_abcrown", "reludiff", "neurodiff")
ARCHS = ("mnist_relu_3_100", "mnist_relu_2_512", "mnist_relu_4_1024")
MODES = ("global", "three_pixel")
RATES = (5, 20, 50)
LIMIT = 100

# Per-instance verification budget, in seconds, held equal across methods so the
# cross-family comparison is fair. It does NOT map 1:1 onto each runner's
# --suite-options timeout, because the runners spend it differently:
#   * milp_abcrown splits an instance into two independent directions
#     (nn1_minus_nn2, nn2_minus_nn1) and hands each its own CPLEX timelimit
#     (run_pyomo.create_solver), so its per-solve value is half the budget.
#   * reludiff/neurodiff get one wall-clock subprocess timeout for the whole
#     instance (run_diffverifier), so they take the full budget.
INSTANCE_BUDGET = 600
SOLVES_PER_INSTANCE = {"milp_abcrown": 2}

# Fixed verifier parameters, held constant across the sweep.
EPSILON = "1.0"
RADIUS = "3"  # +/- radius/255 per pixel, global-mode input region
CROWN_PROFILE = "relu-kfsb"


def _python() -> str:
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else (sys.executable or "python3")


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
    """The ``--suite-options timeout`` for ``method``, in seconds.

    Divides INSTANCE_BUDGET by how many separately-timed solves the method runs
    per instance, so every method gets the same wall-clock budget per instance.
    """
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
    for method, arch, mode, rate, tag in iter_cells():
        if tag_skipped(tag, skip_patterns):
            continue
        out = results_dir / f"{tag}.csv"
        yield out, build_command(method, arch, mode, rate, out)


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
