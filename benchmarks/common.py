from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import gurobipy as gp
from gurobipy import GRB

import nn_equivalence.encoder as encoder
from nn_equivalence.nn_types import Bounds, NeuralNetwork

BenchmarkStatus = Literal["sat", "unsat", "timeout", "unknown"]


@dataclass(frozen=True)
class InputRegion:
    lower_bounds: list[float]
    upper_bounds: list[float]

    def bounds(self) -> Bounds:
        if len(self.lower_bounds) != len(self.upper_bounds):
            raise ValueError("lower_bounds and upper_bounds must have the same length")
        return list(zip(self.lower_bounds, self.upper_bounds))


@dataclass(frozen=True)
class Benchmark:
    benchmark_id: str
    suite_name: str
    nn1: NeuralNetwork
    nn2: NeuralNetwork
    input_region: InputRegion
    epsilon: float
    expected_status: BenchmarkStatus | None = None
    timeout_sec: float = 30.0
    metadata: dict[str, str | int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    benchmarks: list[Benchmark]


@dataclass(frozen=True)
class ReluStats:
    active: int = 0
    inactive: int = 0
    unstable: int = 0

    @property
    def total(self) -> int:
        return self.active + self.inactive + self.unstable

    def __add__(self, other: "ReluStats") -> "ReluStats":
        return ReluStats(
            active=self.active + other.active,
            inactive=self.inactive + other.inactive,
            unstable=self.unstable + other.unstable,
        )


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark_id: str
    suite_name: str
    status: BenchmarkStatus
    runtime_sec: float
    epsilon: float
    input_dim: int
    output_dim: int
    num_layers: int
    num_relu: int
    num_active_relu: int
    num_inactive_relu: int
    num_unstable_relu: int
    num_vars: int
    num_binary_vars: int
    num_constraints: int
    max_output_diff: float | None
    counterexample: list[float] | None
    expected_status: BenchmarkStatus | None

    @property
    def matched_expected(self) -> bool | None:
        if self.expected_status is None:
            return None
        return self.status == self.expected_status


def _validate_benchmark(benchmark: Benchmark) -> None:
    if benchmark.epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if len(benchmark.nn1) != len(benchmark.nn2):
        raise ValueError("nn1 and nn2 must have the same number of layers")
    if len(benchmark.nn1[0][0][0]) != len(benchmark.input_region.lower_bounds):
        raise ValueError("input region dimension does not match network input size")

    for layer_index, ((weights1, bias1), (weights2, bias2)) in enumerate(
        zip(benchmark.nn1, benchmark.nn2), start=1
    ):
        if len(weights1) != len(weights2) or len(bias1) != len(bias2):
            raise ValueError(f"layer {layer_index} output sizes differ")
        if len(weights1[0]) != len(weights2[0]):
            raise ValueError(f"layer {layer_index} input sizes differ")


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
    _validate_benchmark(benchmark)

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
