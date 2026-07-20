from __future__ import annotations

import argparse
import importlib
import gurobipy as gp
from gurobipy import GRB

from benchmarks.common import (
    Instance,
    InstanceResult,
    InstanceStatus,
    InstanceSuite,
    InputRegion,
    SuiteOptions,
    parse_suite_options,
    validate_instance,
)
import nn_equivalence.encoder as encoder
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
    parser = argparse.ArgumentParser(description="Run an NN equivalence instance suite.")
    parser.add_argument("--suite", default="sample")
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

def _status_from_gurobi(status: int) -> InstanceStatus:
    if status == GRB.OPTIMAL:
        return "sat"
    if status == GRB.INFEASIBLE:
        return "unsat"
    if status == GRB.TIME_LIMIT:
        return "timeout"
    return "unknown"


def _add_directional_output_distance_constraint(
    model: gp.Model,
    first_output_vars: list[gp.Var],
    second_output_vars: list[gp.Var],
    epsilon: float,
    name_prefix: str,
) -> None:
    if len(first_output_vars) != len(second_output_vars):
        raise ValueError("output variable lists must have the same length")

    selectors: list[gp.Var] = []
    for output_index, (first_var, second_var) in enumerate(
        zip(first_output_vars, second_output_vars)
    ):
        selector = model.addVar(
            vtype=GRB.BINARY,
            name=f"{name_prefix}_{output_index}",
        )
        selectors.append(selector)
        model.addGenConstrIndicator(
            selector,
            True,
            first_var - second_var,
            GRB.GREATER_EQUAL,
            epsilon,
            name=f"{name_prefix}_{output_index}_indicator",
        )

    model.addConstr(
        gp.quicksum(selectors) >= 1,
        name=f"{name_prefix}_at_least_one_coordinate",
    )


def solve_differential_model(
    instance: Instance,
    first_network_name: str,
    second_network_name: str,
    first_network: NeuralNetwork,
    second_network: NeuralNetwork,
) -> tuple[InstanceStatus, float]:
    model = gp.Model(
        f"{instance.instance_id}_{first_network_name}_minus_{second_network_name}"
    )
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = instance.timeout_sec
    model.Params.FeasibilityTol = 1e-9
    model.Params.IntFeasTol = 1e-9

    input_bounds = instance.input_region.bounds()
    x = [
        model.addVar(lb=lower, ub=upper, name=f"x_{i}")
        for i, (lower, upper) in enumerate(input_bounds)
    ]

    first_output_vars = encoder.add_hidden_variables(
        model,
        x,
        first_network,
        first_network_name,
        input_bounds=input_bounds,
    )
    second_output_vars = encoder.add_hidden_variables(
        model,
        x,
        second_network,
        second_network_name,
        input_bounds=input_bounds,
    )
    encoder.add_output_distance_constraint(
        model,
        first_output_vars,
        second_output_vars,
        instance.epsilon,
        name_prefix=f"{first_network_name}_minus_{second_network_name}",
    )
    model.setObjective(0.0, GRB.MINIMIZE)
    model.optimize()

    return _status_from_gurobi(model.Status), model.Runtime


def _combine_directional_statuses(statuses: list[InstanceStatus]) -> InstanceStatus:
    if "sat" in statuses:
        return "sat"
    if all(status == "unsat" for status in statuses):
        return "unsat"
    if "timeout" in statuses:
        return "timeout"
    return "unknown"


def run_instance(instance: Instance) -> InstanceResult:
    validate_instance(instance)

    first_status, first_runtime = solve_differential_model(
        instance,
        "nn1",
        "nn2",
        instance.nn1,
        instance.nn2,
    )
    second_status, second_runtime = solve_differential_model(
        instance,
        "nn2",
        "nn1",
        instance.nn2,
        instance.nn1,
    )
    status = _combine_directional_statuses([first_status, second_status])

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
    suite = load_suite(args.suite, parse_suite_options(args.suite_options))
    results = [run_instance(instance) for instance in suite.instances]
    print_results(results)


if __name__ == "__main__":
    main()
