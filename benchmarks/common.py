from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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


def validate_benchmark(benchmark: Benchmark) -> None:
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


