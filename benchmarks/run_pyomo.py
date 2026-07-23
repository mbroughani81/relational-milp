from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import re
import sys
import time
from typing import Any
from typing import Literal

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition as TC

from benchmarks.common import (
    Instance,
    InstanceResult,
    InstanceStatus,
    InstanceSuite,
    SuiteOptions,
    parse_suite_options,
    validate_instance,
)
from benchmarks.abcrown_bounds import (
    ABCrownBoundCache,
    ABCrownBoundOptions,
    compute_network_bounds,
)
import nn_equivalence.encoder_pyomo as encoder
from nn_equivalence.nn_types import Bounds, NeuralNetwork

BoundTighteningMode = Literal["interval", "abcrown"]
SolverName = Literal["highs", "gurobi", "cplex"]


def load_suite(name: str, suite_options: SuiteOptions) -> InstanceSuite:
    module = importlib.import_module(f"benchmarks.{name}")
    return module.load_suite(suite_options)


def format_expected(result: InstanceResult) -> str:
    if result.expected_status is None:
        return ""
    matched = "yes" if result.matched_expected else "no"
    return f"{result.expected_status}:{matched}"


def print_results(results: list[InstanceResult]) -> None:
    print("instance_id,status,expected,runtime_sec,epsilon")
    for result in results:
        print(
            f"{result.instance_id},"
            f"{result.status},"
            f"{format_expected(result)},"
            f"{result.runtime_sec:.6f},"
            f"{result.epsilon:.17g}"
        )


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
        "--solver-tee",
        action="store_true",
        help=(
            "Print backend solver logs interactively. This may mix solver logs "
            "with CSV output on stdout; use --solver-log-dir to keep logs separate."
        ),
    )
    parser.add_argument(
        "--solver-log-dir",
        type=Path,
        default=None,
        help=(
            "Directory for backend solver logs. Writes one log file per "
            "instance direction and keeps CSV results on stdout."
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
        "--solver-threads",
        type=int,
        default=None,
        help="Thread count passed to solvers that support it.",
    )
    parser.add_argument(
        "--highs-parallel",
        choices=("choose", "on", "off"),
        default=None,
        help="HiGHS parallel option. Use 'on' to force parallel mode.",
    )
    parser.add_argument(
        "--highs-mip-heuristic-effort",
        type=float,
        default=None,
        help="HiGHS MIP heuristic effort, e.g. 0.2.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help=(
            "Print one status line to stderr after each instance finishes. "
            "Stdout remains the final CSV."
        ),
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


def set_solver_threads(
    solver: Any,
    solver_name: SolverName,
    solver_threads: int | None,
) -> None:
    if solver_threads is None:
        return
    if solver_threads < 1:
        raise ValueError("--solver-threads must be positive")
    if solver_name == "gurobi":
        solver.options["Threads"] = solver_threads
    elif solver_name == "highs":
        solver.options["threads"] = solver_threads
    elif solver_name == "cplex":
        solver.options["threads"] = solver_threads


def set_highs_options(
    solver: Any,
    solver_name: SolverName,
    highs_parallel: str | None,
    highs_mip_heuristic_effort: float | None,
) -> None:
    if solver_name != "highs":
        if highs_parallel is not None or highs_mip_heuristic_effort is not None:
            raise ValueError("HiGHS-specific options require --solver highs")
        return
    if highs_parallel is not None:
        solver.options["parallel"] = highs_parallel
    if highs_mip_heuristic_effort is not None:
        if highs_mip_heuristic_effort < 0:
            raise ValueError("--highs-mip-heuristic-effort must be non-negative")
        solver.options["mip_heuristic_effort"] = highs_mip_heuristic_effort


def create_solver(
    solver_name: SolverName,
    timeout_sec: float,
    solver_threads: int | None,
    highs_parallel: str | None,
    highs_mip_heuristic_effort: float | None,
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
    set_solver_threads(solver, solver_name, solver_threads)
    set_highs_options(
        solver,
        solver_name,
        highs_parallel,
        highs_mip_heuristic_effort,
    )
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
            "runner uses Pyomo's cplex_direct backend. Use --solver-tee for "
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
    solver_tee: bool,
    solver_log_dir: Path | None,
    bound_overrides: encoder.BoundOverrides | None,
    solver_threads: int | None,
    highs_parallel: str | None,
    highs_mip_heuristic_effort: float | None,
) -> tuple[InstanceStatus, float]:
    model, input_vars = encoder.encode_instance_direction(
        instance,
        first_network_name,
        second_network_name,
        first_network,
        second_network,
        bound_overrides,
    )
    solver = create_solver(
        solver_name,
        instance.timeout_sec,
        solver_threads,
        highs_parallel,
        highs_mip_heuristic_effort,
    )
    direction_name = f"{first_network_name}_minus_{second_network_name}"
    logfile = None
    if solver_log_dir is not None:
        solver_log_dir.mkdir(parents=True, exist_ok=True)
        logfile = solver_log_dir / (
            f"{safe_log_name(instance.instance_id)}_{direction_name}.log"
        )
        set_solver_log_file(solver, solver_name, logfile)

    start_time = time.perf_counter()
    result = solver.solve(
        model,
        tee=solver_tee,
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

    return status, runtime_sec


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
    solver_tee: bool = False,
    solver_log_dir: Path | None = None,
    solver_threads: int | None = None,
    highs_parallel: str | None = None,
    highs_mip_heuristic_effort: float | None = None,
) -> InstanceResult:
    validate_instance(instance)
    bounds_start = time.perf_counter()
    bound_overrides = compute_bound_overrides(
        instance,
        bound_tightening,
        abcrown_bound_cache,
    )
    bounds_runtime = time.perf_counter() - bounds_start

    first_status, first_runtime = solve_instance_direction(
        instance,
        solver_name,
        "nn1",
        "nn2",
        instance.nn1,
        instance.nn2,
        solver_tee,
        solver_log_dir,
        bound_overrides,
        solver_threads,
        highs_parallel,
        highs_mip_heuristic_effort,
    )
    second_status, second_runtime = solve_instance_direction(
        instance,
        solver_name,
        "nn2",
        "nn1",
        instance.nn2,
        instance.nn1,
        solver_tee,
        solver_log_dir,
        bound_overrides,
        solver_threads,
        highs_parallel,
        highs_mip_heuristic_effort,
    )
    status = combine_directional_statuses([first_status, second_status])

    return InstanceResult(
        instance_id=instance.instance_id,
        suite_name=instance.suite_name,
        status=status,
        runtime_sec=bounds_runtime + first_runtime + second_runtime,
        epsilon=instance.epsilon,
        expected_status=instance.expected_status,
    )


def compute_bound_overrides(
    instance: Instance,
    bound_tightening: BoundTighteningMode,
    abcrown_bound_cache: ABCrownBoundCache | None,
) -> encoder.BoundOverrides | None:
    if bound_tightening == "interval":
        return None
    if bound_tightening != "abcrown":
        raise ValueError(f"unsupported bound tightening mode: {bound_tightening}")

    input_bounds: Bounds = instance.input_region.bounds()
    options = ABCrownBoundOptions(timeout_sec=instance.timeout_sec)
    overrides: encoder.BoundOverrides = {}

    nn1_bounds = compute_network_bounds(
        instance.nn1,
        input_bounds,
        options,
        abcrown_bound_cache,
    )
    if nn1_bounds is not None:
        overrides["nn1"] = nn1_bounds

    nn2_bounds = compute_network_bounds(
        instance.nn2,
        input_bounds,
        options,
        abcrown_bound_cache,
    )
    if nn2_bounds is not None:
        overrides["nn2"] = nn2_bounds

    return overrides or None


def print_progress(index: int, total: int, result: InstanceResult) -> None:
    print(
        f"[{index}/{total}] {result.instance_id}: "
        f"status={result.status} expected={format_expected(result) or '-'} "
        f"runtime_sec={result.runtime_sec:.3f} epsilon={result.epsilon:.17g}",
        file=sys.stderr,
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
                args.solver_tee,
                args.solver_log_dir,
                args.solver_threads,
                args.highs_parallel,
                args.highs_mip_heuristic_effort,
            )
            results.append(result)
            if args.progress:
                print_progress(index, total_instances, result)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from error

    print_results(results)


if __name__ == "__main__":
    main()
