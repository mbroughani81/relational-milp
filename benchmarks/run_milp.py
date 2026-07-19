from __future__ import annotations

import argparse
import importlib
import gurobipy as gp
from gurobipy import GRB

from benchmarks.common import Instance, InstanceSuite, InstanceResult, InstanceStatus, InputRegion, validate_instance
import nn_equivalence.encoder as encoder

def load_suite(name: str) -> InstanceSuite:
    module = importlib.import_module(f"benchmarks.{name}")
    return module.load_suite()


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
    return parser.parse_args()

def _status_from_gurobi(status: int) -> InstanceStatus:
    if status == GRB.OPTIMAL:
        return "sat"
    if status == GRB.INFEASIBLE:
        return "unsat"
    if status == GRB.TIME_LIMIT:
        return "timeout"
    return "unknown"


def run_instance(instance: Instance) -> InstanceResult:
    validate_instance(instance)

    model = gp.Model(instance.instance_id)
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = instance.timeout_sec
    model.Params.FeasibilityTol = 1e-9
    model.Params.IntFeasTol = 1e-9

    input_bounds = instance.input_region.bounds()
    x = [
        model.addVar(lb=lower, ub=upper, name=f"x_{i}")
        for i, (lower, upper) in enumerate(input_bounds)
    ]

    nn1_output_vars = encoder.add_hidden_variables(
        model,
        x,
        instance.nn1,
        "nn1",
        input_bounds=input_bounds,
    )
    nn2_output_vars = encoder.add_hidden_variables(
        model,
        x,
        instance.nn2,
        "nn2",
        input_bounds=input_bounds,
    )
    encoder.add_output_distance_constraint(
        model,
        nn1_output_vars,
        nn2_output_vars,
        instance.epsilon,
    )
    model.setObjective(0.0, GRB.MINIMIZE)
    model.optimize()

    status = _status_from_gurobi(model.Status)

    return InstanceResult(
        instance_id=instance.instance_id,
        suite_name=instance.suite_name,
        status=status,
        runtime_sec=model.Runtime,
        epsilon=instance.epsilon,
        expected_status=instance.expected_status,
    )



def main() -> None:
    args = parse_args()
    suite = load_suite(args.suite)
    results = [run_instance(instance) for instance in suite.instances]
    print_results(results)


if __name__ == "__main__":
    main()
