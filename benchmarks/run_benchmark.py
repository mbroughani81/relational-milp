from __future__ import annotations

import argparse
import importlib
import gurobipy as gp
from gurobipy import GRB

from benchmarks.common import Benchmark, BenchmarkSuite, BenchmarkResult, BenchmarkStatus, InputRegion, ReluStats, validate_benchmark
import nn_equivalence.encoder as encoder
from nn_equivalence.nn_types import Bounds, NeuralNetwork

def load_suite(name: str) -> BenchmarkSuite:
    module = importlib.import_module(f"benchmarks.{name}")
    return module.load_suite()

def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def format_expected(result: BenchmarkResult) -> str:
    if result.expected_status is None:
        return ""
    matched = "yes" if result.matched_expected else "no"
    return f"{result.expected_status}:{matched}"


def print_results(results: list[BenchmarkResult]) -> None:
    print(
        "benchmark_id,status,expected,runtime_sec,vars,binaries,"
        "constraints,num_relu,active_relu,inactive_relu,unstable_relu,max_output_diff"
    )
    for result in results:
        print(
            f"{result.benchmark_id},"
            f"{result.status},"
            f"{format_expected(result)},"
            f"{result.runtime_sec:.6f},"
            f"{result.num_vars},"
            f"{result.num_binary_vars},"
            f"{result.num_constraints},"
            f"{result.num_relu},"
            f"{result.num_active_relu},"
            f"{result.num_inactive_relu},"
            f"{result.num_unstable_relu},"
            f"{format_float(result.max_output_diff)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an NN equivalence benchmark suite.")
    parser.add_argument("--suite", default="sample")
    return parser.parse_args()

def _status_from_gurobi(status: int) -> BenchmarkStatus:
    if status == GRB.OPTIMAL:
        return "sat"
    if status == GRB.INFEASIBLE:
        return "unsat"
    if status == GRB.TIME_LIMIT:
        return "timeout"
    return "unknown"


def _add_input_variables(model: gp.Model, input_region: InputRegion) -> list[gp.Var]:
    return [
        model.addVar(lb=lower, ub=upper, name=f"x_{i}")
        for i, (lower, upper) in enumerate(input_region.bounds())
    ]


def _relu_stats_from_bounds(bounds: Bounds) -> ReluStats:
    active = 0
    inactive = 0
    unstable = 0

    for lower, upper in bounds:
        if lower >= 0:
            active += 1
        elif upper <= 0:
            inactive += 1
        else:
            unstable += 1

    return ReluStats(active=active, inactive=inactive, unstable=unstable)


def _relu_stats_for_network(nn: NeuralNetwork, input_bounds: Bounds) -> ReluStats:
    current_bounds = input_bounds
    stats = ReluStats()

    for layer_index, (weights, bias) in enumerate(nn, start=1):
        z_bounds = encoder.affine_bounds(weights, bias, current_bounds)
        if layer_index == len(nn):
            break

        stats += _relu_stats_from_bounds(z_bounds)
        current_bounds = encoder.relu_bounds(z_bounds)

    return stats


def run_benchmark(benchmark: Benchmark) -> BenchmarkResult:
    validate_benchmark(benchmark)

    model = gp.Model(benchmark.benchmark_id)
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = benchmark.timeout_sec
    model.Params.FeasibilityTol = 1e-9
    model.Params.IntFeasTol = 1e-9

    input_bounds = benchmark.input_region.bounds()
    relu_stats = _relu_stats_for_network(
        benchmark.nn1, input_bounds
    ) + _relu_stats_for_network(benchmark.nn2, input_bounds)
    x = _add_input_variables(model, benchmark.input_region)
    _, _, nn1_output_vars, nn1_deltas = encoder.add_hidden_variables(
        model,
        x,
        benchmark.nn1,
        "nn1",
        input_bounds=input_bounds,
    )
    _, _, nn2_output_vars, nn2_deltas = encoder.add_hidden_variables(
        model,
        x,
        benchmark.nn2,
        "nn2",
        input_bounds=input_bounds,
    )
    encoder.add_output_distance_constraint(
        model,
        nn1_output_vars,
        nn2_output_vars,
        benchmark.epsilon,
    )
    model.setObjective(0.0, GRB.MINIMIZE)
    model.optimize()

    status = _status_from_gurobi(model.Status)
    counterexample = None
    max_output_diff = None
    if status == "sat":
        counterexample = [var.X for var in x]
        nn1_output_values = [var.X for var in nn1_output_vars]
        nn2_output_values = [var.X for var in nn2_output_vars]
        max_output_diff = max(
            abs(first - second)
            for first, second in zip(nn1_output_values, nn2_output_values)
        )

    return BenchmarkResult(
        benchmark_id=benchmark.benchmark_id,
        suite_name=benchmark.suite_name,
        status=status,
        runtime_sec=model.Runtime,
        epsilon=benchmark.epsilon,
        input_dim=len(x),
        output_dim=len(nn1_output_vars),
        num_layers=len(benchmark.nn1),
        num_relu=relu_stats.total,
        num_active_relu=relu_stats.active,
        num_inactive_relu=relu_stats.inactive,
        num_unstable_relu=relu_stats.unstable,
        num_vars=model.NumVars,
        num_binary_vars=model.NumBinVars,
        num_constraints=model.NumConstrs,
        max_output_diff=max_output_diff,
        counterexample=counterexample,
        expected_status=benchmark.expected_status,
    )



def main() -> None:
    args = parse_args()
    suite = load_suite(args.suite)
    results = [run_benchmark(benchmark) for benchmark in suite.benchmarks]
    print_results(results)


if __name__ == "__main__":
    main()
