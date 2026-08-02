from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
from pathlib import Path
import re
import time
from typing import Any
from typing import Literal

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition as TC

from benchmarks.abcrown_bounds import (
    ABCrownBoundCache,
    ABCrownBoundOptions,
    compute_network_bounds,
)
from benchmarks.common import (
    Hyperrectangle,
    Instance,
    InstanceResult,
    InstanceStatus,
    InstanceSuite,
    SolveStats,
    SuiteOptions,
    parse_suite_options,
    validate_instance,
)
import nn_equivalence.encoder_pyomo as encoder
from nn_equivalence.nn_types import Bounds, NeuralNetwork

BoundTighteningMode = Literal["interval", "abcrown"]
SolverName = Literal["highs", "gurobi", "cplex"]


@dataclass(frozen=True)
class BoundOverrideResult:
    overrides: encoder.BoundOverrides | None
    nn1_runtime_sec: float
    nn2_runtime_sec: float


@dataclass(frozen=True)
class DirectionResult:
    status: InstanceStatus
    stats: SolveStats


def load_suite(name: str, suite_options: SuiteOptions) -> InstanceSuite:
    module = importlib.import_module(f"benchmarks.{name}")
    return module.load_suite(suite_options)


def format_expected(result: InstanceResult) -> str:
    if result.expected_status is None:
        return ""
    matched = "yes" if result.matched_expected else "no"
    return f"{result.expected_status}:{matched}"


def results_csv(results: list[InstanceResult]) -> str:
    lines = ["instance_id,status,expected,runtime_sec,epsilon"]
    for result in results:
        lines.append(
            f"{result.instance_id},"
            f"{result.status},"
            f"{format_expected(result)},"
            f"{result.runtime_sec:.6f},"
            f"{result.epsilon:.17g}"
        )
    return "\n".join(lines) + "\n"


def print_results(results: list[InstanceResult]) -> None:
    print(results_csv(results), end="")


def write_results_csv(path: Path, results: list[InstanceResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results_csv(results), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an NN equivalence instance suite with Pyomo."
    )
    parser.add_argument("--suite", default="sample")
    parser.add_argument(
        "--solver",
        default="highs",
        choices=("highs", "gurobi", "cplex"),
    )
    parser.add_argument(
        "--suite-options",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Suite-specific option. Repeat for multiple options; values may "
            "contain commas, e.g. --suite-options modes=global,three_pixel."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print backend solver logs to stdout.",
    )
    parser.add_argument(
        "--solver-log-dir",
        type=Path,
        default=None,
        help=(
            "Directory for backend solver logs. Writes one log file per "
            "instance direction."
        ),
    )
    parser.add_argument(
        "--bound-tightening",
        default="interval",
        choices=("interval", "abcrown"),
        help=(
            "Bound source for Pyomo Big-M constants. 'interval' uses interval "
            "arithmetic; 'abcrown' tightens interval bounds with certified "
            "alpha-beta-CROWN compute_bounds results when available."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Also write the final CSV results to this file.",
    )
    return parser.parse_args()


def pyomo_solver_name(solver_name: SolverName) -> str:
    if solver_name == "cplex":
        return "cplex_direct"
    return solver_name


def set_solver_timeout(solver: Any, solver_name: SolverName, timeout_sec: float) -> None:
    if hasattr(solver, "options"):
        if solver_name == "gurobi":
            solver.options["TimeLimit"] = timeout_sec
        elif solver_name == "cplex":
            solver.options["timelimit"] = timeout_sec
        else:
            solver.options["time_limit"] = timeout_sec
    elif hasattr(solver, "config") and hasattr(solver.config, "time_limit"):
        solver.config.time_limit = timeout_sec


def create_solver(
    solver_name: SolverName,
    timeout_sec: float,
) -> Any:
    backend_name = pyomo_solver_name(solver_name)
    solver = pyo.SolverFactory(backend_name)
    if not solver.available(False):
        raise RuntimeError(
            f"Pyomo solver '{solver_name}' is not available through backend "
            f"'{backend_name}'. Install the solver backend and make it "
            "available to Pyomo."
        )

    set_solver_timeout(solver, solver_name, timeout_sec)
    return solver


def set_solver_log_file(solver: Any, solver_name: SolverName, logfile: Path) -> None:
    if not hasattr(solver, "options"):
        raise RuntimeError(
            f"Pyomo solver '{solver_name}' does not support solver log files."
        )
    if solver_name == "gurobi":
        solver.options["LogFile"] = str(logfile)
    elif solver_name == "cplex":
        raise RuntimeError(
            "--solver-log-dir is not supported for --solver cplex because this "
            "runner uses Pyomo's cplex_direct backend. Use --verbose for "
            "interactive CPLEX logs."
        )
    elif solver_name == "highs":
        solver.options["log_file"] = str(logfile)
        solver.options["output_flag"] = True
    else:
        raise RuntimeError(f"Unsupported solver for log files: {solver_name}")


def safe_log_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def status_from_pyomo(termination_condition: TC) -> InstanceStatus:
    if termination_condition in {TC.optimal, TC.feasible, TC.globallyOptimal}:
        return "sat"
    if termination_condition == TC.infeasible:
        return "unsat"
    if termination_condition == TC.maxTimeLimit:
        return "timeout"
    return "unknown"


def solve_instance_direction(
    instance: Instance,
    solver_name: str,
    first_network_name: str,
    second_network_name: str,
    first_network: NeuralNetwork,
    second_network: NeuralNetwork,
    verbose: bool,
    solver_log_dir: Path | None,
    bound_overrides: encoder.BoundOverrides | None,
) -> DirectionResult:
    encode_start = time.perf_counter()
    model, input_vars = encoder.encode_instance_direction(
        instance,
        first_network_name,
        second_network_name,
        first_network,
        second_network,
        bound_overrides,
    )
    encode_runtime_sec = time.perf_counter() - encode_start

    solver_setup_start = time.perf_counter()
    solver = create_solver(
        solver_name,
        instance.timeout_sec,
    )
    direction_name = f"{first_network_name}_minus_{second_network_name}"
    logfile = None
    if solver_log_dir is not None:
        solver_log_dir.mkdir(parents=True, exist_ok=True)
        logfile = solver_log_dir / (
            f"{safe_log_name(instance.instance_id)}_{direction_name}.log"
        )
        set_solver_log_file(solver, solver_name, logfile)
    solver_setup_runtime_sec = time.perf_counter() - solver_setup_start

    start_time = time.perf_counter()
    result = solver.solve(
        model,
        tee=verbose,
        load_solutions=False,
    )
    runtime_sec = time.perf_counter() - start_time
    status = status_from_pyomo(result.solver.termination_condition)

    if status == "sat":
        model.solutions.load_from(result)
        encoder.validate_directional_witness(
            instance,
            input_vars,
            first_network_name,
            second_network_name,
            first_network,
            second_network,
        )

    return DirectionResult(
        status=status,
        stats=SolveStats(
            name=direction_name,
            timings=[
                ("encode", encode_runtime_sec),
                ("solver_setup", solver_setup_runtime_sec),
                ("solve", runtime_sec),
            ],
        ),
    )


def combine_directional_statuses(statuses: list[InstanceStatus]) -> InstanceStatus:
    if "sat" in statuses:
        return "sat"
    if all(status == "unsat" for status in statuses):
        return "unsat"
    if "timeout" in statuses:
        return "timeout"
    return "unknown"


def run_instance(
    instance: Instance,
    solver_name: str,
    bound_tightening: BoundTighteningMode,
    abcrown_bound_cache: ABCrownBoundCache | None,
    verbose: bool = False,
    solver_log_dir: Path | None = None,
) -> InstanceResult:
    validate_instance(instance)
    bound_result = compute_bound_overrides(
        instance,
        bound_tightening,
        abcrown_bound_cache,
    )
    bound_overrides = bound_result.overrides

    first_result = solve_instance_direction(
        instance,
        solver_name,
        "nn1",
        "nn2",
        instance.nn1,
        instance.nn2,
        verbose,
        solver_log_dir,
        bound_overrides,
    )
    second_result = solve_instance_direction(
        instance,
        solver_name,
        "nn2",
        "nn1",
        instance.nn2,
        instance.nn1,
        verbose,
        solver_log_dir,
        bound_overrides,
    )
    status = combine_directional_statuses([first_result.status, second_result.status])
    stats = [
        SolveStats(
            name="bound_tightening",
            timings=[
                ("nn1", bound_result.nn1_runtime_sec),
                ("nn2", bound_result.nn2_runtime_sec),
            ],
        ),
        first_result.stats,
        second_result.stats,
    ]

    return InstanceResult(
        instance_id=instance.instance_id,
        suite_name=instance.suite_name,
        status=status,
        runtime_sec=(
            bound_result.nn1_runtime_sec
            + bound_result.nn2_runtime_sec
            + timing_total(stats, "solve")
        ),
        epsilon=instance.epsilon,
        expected_status=instance.expected_status,
        stats=stats,
    )


def compute_bound_overrides(
    instance: Instance,
    bound_tightening: BoundTighteningMode,
    abcrown_bound_cache: ABCrownBoundCache | None,
) -> BoundOverrideResult:
    if bound_tightening == "interval":
        return BoundOverrideResult(
            overrides=None,
            nn1_runtime_sec=0.0,
            nn2_runtime_sec=0.0,
        )
    if bound_tightening != "abcrown":
        raise ValueError(f"unsupported bound tightening mode: {bound_tightening}")

    input_box = Hyperrectangle.overapproximate(instance.input_region)
    input_bounds: Bounds = input_box.bounds()
    options = ABCrownBoundOptions(timeout_sec=instance.timeout_sec)
    overrides: encoder.BoundOverrides = {}

    nn1_start = time.perf_counter()
    nn1_bounds = compute_network_bounds(
        instance.nn1,
        input_bounds,
        options,
        abcrown_bound_cache,
    )
    nn1_runtime_sec = time.perf_counter() - nn1_start
    if nn1_bounds is not None:
        overrides["nn1"] = nn1_bounds

    nn2_start = time.perf_counter()
    nn2_bounds = compute_network_bounds(
        instance.nn2,
        input_bounds,
        options,
        abcrown_bound_cache,
    )
    nn2_runtime_sec = time.perf_counter() - nn2_start
    if nn2_bounds is not None:
        overrides["nn2"] = nn2_bounds

    return BoundOverrideResult(
        overrides=overrides or None,
        nn1_runtime_sec=nn1_runtime_sec,
        nn2_runtime_sec=nn2_runtime_sec,
    )


def timing_total(stats: list[SolveStats], phase_name: str) -> float:
    return sum(
        runtime_sec
        for solve_stats in stats
        for phase, runtime_sec in solve_stats.timings
        if phase == phase_name
    )


def format_solve_stats(stats: list[SolveStats]) -> str:
    parts = []
    for solve_stats in stats:
        timing_text = ",".join(
            f"{phase}={runtime_sec:.3f}"
            for phase, runtime_sec in solve_stats.timings
        )
        parts.append(f"{solve_stats.name}[{timing_text}]")
    measured_total_sec = sum(solve_stats.measured_total_sec for solve_stats in stats)
    parts.append(f"total={measured_total_sec:.3f}")
    return " ".join(parts)


def print_progress(index: int, total: int, result: InstanceResult) -> None:
    phase_text = ""
    if result.stats:
        phase_text = f" phases: {format_solve_stats(result.stats)}"
    print(
        f"[{index}/{total}] {result.instance_id}: "
        f"status={result.status} expected={format_expected(result) or '-'} "
        f"runtime_sec={result.runtime_sec:.3f} epsilon={result.epsilon:.17g}"
        f"{phase_text}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    try:
        suite = load_suite(args.suite, parse_suite_options(args.suite_options))
        abcrown_bound_cache = (
            ABCrownBoundCache() if args.bound_tightening == "abcrown" else None
        )
        results: list[InstanceResult] = []
        total_instances = len(suite.instances)
        for index, instance in enumerate(suite.instances, start=1):
            result = run_instance(
                instance,
                args.solver,
                args.bound_tightening,
                abcrown_bound_cache,
                args.verbose,
                args.solver_log_dir,
            )
            results.append(result)
            print_progress(index, total_instances, result)
    except RuntimeError as error:
        print(error)
        raise SystemExit(2) from error

    print_results(results)
    if args.csv is not None:
        write_results_csv(args.csv, results)


if __name__ == "__main__":
    main()
