from __future__ import annotations

import argparse
import importlib
import sys
import time
from typing import Any

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
import nn_equivalence.encoder_pyomo as encoder
from nn_equivalence.nn_types import NeuralNetwork


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
    parser.add_argument("--solver", default="highs", choices=("highs", "gurobi"))
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
    return parser.parse_args()


def set_solver_timeout(solver: Any, solver_name: str, timeout_sec: float) -> None:
    if hasattr(solver, "options"):
        if solver_name == "gurobi":
            solver.options["TimeLimit"] = timeout_sec
        else:
            solver.options["time_limit"] = timeout_sec
    elif hasattr(solver, "config") and hasattr(solver.config, "time_limit"):
        solver.config.time_limit = timeout_sec


def create_solver(solver_name: str, timeout_sec: float) -> Any:
    solver = pyo.SolverFactory(solver_name)
    if not solver.available(False):
        raise RuntimeError(
            f"Pyomo solver '{solver_name}' is not available. "
            "Install the solver backend and make it available to Pyomo."
        )

    set_solver_timeout(solver, solver_name, timeout_sec)
    return solver


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
) -> tuple[InstanceStatus, float]:
    model, input_vars = encoder.encode_instance_direction(
        instance,
        first_network_name,
        second_network_name,
        first_network,
        second_network,
    )
    solver = create_solver(solver_name, instance.timeout_sec)
    start_time = time.perf_counter()
    result = solver.solve(model, tee=False, load_solutions=False)
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


def run_instance(instance: Instance, solver_name: str) -> InstanceResult:
    validate_instance(instance)

    first_status, first_runtime = solve_instance_direction(
        instance,
        solver_name,
        "nn1",
        "nn2",
        instance.nn1,
        instance.nn2,
    )
    second_status, second_runtime = solve_instance_direction(
        instance,
        solver_name,
        "nn2",
        "nn1",
        instance.nn2,
        instance.nn1,
    )
    status = combine_directional_statuses([first_status, second_status])

    return InstanceResult(
        instance_id=instance.instance_id,
        suite_name=instance.suite_name,
        status=status,
        runtime_sec=first_runtime + second_runtime,
        epsilon=instance.epsilon,
        expected_status=instance.expected_status,
    )


def main() -> None:
    args = parse_args()
    try:
        suite = load_suite(args.suite, parse_suite_options(args.suite_options))
        results = [
            run_instance(instance, args.solver)
            for instance in suite.instances
        ]
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from error

    print_results(results)


if __name__ == "__main__":
    main()
