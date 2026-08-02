from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Literal

from nn_equivalence.nn_types import Bounds, NeuralNetwork

InstanceStatus = Literal["sat", "unsat", "timeout", "unknown"]
SuiteOptions = dict[str, str]


class AbstractPolytope:
    pass


@dataclass(frozen=True)
class HalfSpace(AbstractPolytope):
    a: list[float]
    b: float

    def validate_dimension(self, dimension: int) -> None:
        if len(self.a) != dimension:
            raise ValueError("halfspace dimension does not match input region")


@dataclass(frozen=True)
class Hyperrectangle(AbstractPolytope):
    low: list[float]
    high: list[float]

    def bounds(self) -> Bounds:
        if len(self.low) != len(self.high):
            raise ValueError("lower_bounds and upper_bounds must have the same length")
        region_bounds = list(zip(self.low, self.high))
        for lower, upper in region_bounds:
            if lower > upper:
                raise ValueError("input lower bound exceeds upper bound")
        return region_bounds

    @staticmethod
    def overapproximate(set_: AbstractPolytope) -> Hyperrectangle:
        if isinstance(set_, Hyperrectangle):
            return set_
        if not isinstance(set_, HPolytope):
            raise TypeError(f"unsupported polytope type: {type(set_).__name__}")

        lower_bounds = [-float("inf")] * dim(set_)
        upper_bounds = [float("inf")] * dim(set_)
        for constraint in constraints_list(set_):
            nonzero_indices = [
                index
                for index, coefficient in enumerate(constraint.a)
                if coefficient != 0.0
            ]
            if len(nonzero_indices) != 1:
                continue
            index = nonzero_indices[0]
            coefficient = constraint.a[index]
            bound = constraint.b / coefficient
            if coefficient > 0:
                upper_bounds[index] = min(upper_bounds[index], bound)
            else:
                lower_bounds[index] = max(lower_bounds[index], bound)

        for lower, upper in zip(lower_bounds, upper_bounds):
            if lower > upper:
                raise ValueError("input lower bound exceeds upper bound")
            if lower == -float("inf") or upper == float("inf"):
                raise ValueError(
                    "HPolytope must include finite axis-aligned bounds to "
                    "overapproximate it as a Hyperrectangle"
                )
        return Hyperrectangle(low=lower_bounds, high=upper_bounds)


class HPolytope(AbstractPolytope):
    def __init__(self, constraints: list[HalfSpace]) -> None:
        if not all(isinstance(constraint, HalfSpace) for constraint in constraints):
            raise ValueError("HPolytope constraints must be HalfSpace instances")
        self.constraints = tuple(constraints)
        dimensions = {len(constraint.a) for constraint in self.constraints}
        if len(dimensions) != 1:
            raise ValueError("HPolytope halfspaces must all have the same dimension")
        self._dimension = dimensions.pop()


def constraints_list(set_: AbstractPolytope | HalfSpace) -> tuple[HalfSpace, ...]:
    if isinstance(set_, HalfSpace):
        return (set_,)
    if isinstance(set_, HPolytope):
        for constraint in set_.constraints:
            constraint.validate_dimension(set_._dimension)
        return set_.constraints
    if isinstance(set_, Hyperrectangle):
        return ()
    raise TypeError(f"unsupported polytope type: {type(set_).__name__}")


def dim(set_: AbstractPolytope) -> int:
    if isinstance(set_, HalfSpace):
        return len(set_.a)
    if isinstance(set_, Hyperrectangle):
        return len(set_.low)
    if isinstance(set_, HPolytope):
        return set_._dimension
    raise TypeError(f"unsupported polytope type: {type(set_).__name__}")


def contains(
    set_: AbstractPolytope,
    values: list[float],
    tolerance: float = 0.0,
) -> bool:
    if isinstance(set_, HalfSpace):
        value = sum(
            coefficient * input_value
            for coefficient, input_value in zip(set_.a, values)
        )
        return value <= set_.b + tolerance

    if len(values) != dim(set_):
        return False
    if isinstance(set_, Hyperrectangle):
        region_bounds = set_.bounds()
        bounds_satisfied = all(
            lower - tolerance <= value <= upper + tolerance
            for value, (lower, upper) in zip(values, region_bounds)
        )
        if not bounds_satisfied:
            return False
    return all(
        contains(constraint, values, tolerance)
        for constraint in constraints_list(set_)
    )


@dataclass(frozen=True)
class Instance:
    instance_id: str
    suite_name: str
    nn1: NeuralNetwork
    nn2: NeuralNetwork
    input_region: AbstractPolytope
    epsilon: float
    expected_status: InstanceStatus | None = None
    timeout_sec: float = 30.0
    metadata: dict[str, str | int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class InstanceSuite:
    name: str
    instances: list[Instance]


@dataclass(frozen=True)
class SolveStats:
    name: str
    timings: list[tuple[str, float]] = field(default_factory=list)

    @property
    def measured_total_sec(self) -> float:
        return sum(runtime_sec for _, runtime_sec in self.timings)


@dataclass(frozen=True)
class InstanceResult:
    instance_id: str
    suite_name: str
    status: InstanceStatus
    runtime_sec: float
    epsilon: float
    expected_status: InstanceStatus | None
    stats: list[SolveStats] = field(default_factory=list)

    @property
    def matched_expected(self) -> bool | None:
        if self.expected_status is None:
            return None
        return self.status == self.expected_status


def parse_suite_options(raw_options: list[str] | None) -> SuiteOptions:
    options: SuiteOptions = {}
    for raw_option in raw_options or []:
        raw_option = raw_option.strip()
        if not raw_option:
            continue
        if raw_option.startswith("{"):
            parsed = json.loads(raw_option)
            if not isinstance(parsed, dict):
                raise ValueError("--suite-options JSON value must be an object")
            options.update({str(key): str(value) for key, value in parsed.items()})
            continue

        for part in raw_option.split(";"):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(
                    "--suite-options entries must be KEY=VALUE pairs"
                )
            key, value = part.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError("--suite-options keys must be non-empty")
            options[key] = value.strip()

    return options


def validate_instance(instance: Instance) -> None:
    if instance.epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if len(instance.nn1) != len(instance.nn2):
        raise ValueError("nn1 and nn2 must have the same number of layers")
    input_dimension = dim(instance.input_region)
    constraints_list(instance.input_region)
    Hyperrectangle.overapproximate(instance.input_region)
    if len(instance.nn1[0][0][0]) != input_dimension:
        raise ValueError("input region dimension does not match network input size")

    for layer_index, ((weights1, bias1), (weights2, bias2)) in enumerate(
        zip(instance.nn1, instance.nn2), start=1
    ):
        if len(weights1) != len(weights2) or len(bias1) != len(bias2):
            raise ValueError(f"layer {layer_index} output sizes differ")
        if len(weights1[0]) != len(weights2[0]):
            raise ValueError(f"layer {layer_index} input sizes differ")
