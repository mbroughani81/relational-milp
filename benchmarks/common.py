from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from nn_equivalence.nn_types import Bounds, NeuralNetwork

InstanceStatus = Literal["sat", "unsat", "timeout", "unknown"]


@dataclass(frozen=True)
class InputRegion:
    lower_bounds: list[float]
    upper_bounds: list[float]

    def bounds(self) -> Bounds:
        if len(self.lower_bounds) != len(self.upper_bounds):
            raise ValueError("lower_bounds and upper_bounds must have the same length")
        return list(zip(self.lower_bounds, self.upper_bounds))


@dataclass(frozen=True)
class Instance:
    instance_id: str
    suite_name: str
    nn1: NeuralNetwork
    nn2: NeuralNetwork
    input_region: InputRegion
    epsilon: float
    expected_status: InstanceStatus | None = None
    timeout_sec: float = 30.0
    metadata: dict[str, str | int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class InstanceSuite:
    name: str
    instances: list[Instance]


@dataclass(frozen=True)
class InstanceStats:
    pass


@dataclass(frozen=True)
class InstanceResult:
    instance_id: str
    suite_name: str
    status: InstanceStatus
    runtime_sec: float
    epsilon: float
    expected_status: InstanceStatus | None
    stats: InstanceStats = field(default_factory=InstanceStats)

    @property
    def matched_expected(self) -> bool | None:
        if self.expected_status is None:
            return None
        return self.status == self.expected_status


def validate_instance(instance: Instance) -> None:
    if instance.epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if len(instance.nn1) != len(instance.nn2):
        raise ValueError("nn1 and nn2 must have the same number of layers")
    if len(instance.nn1[0][0][0]) != len(instance.input_region.lower_bounds):
        raise ValueError("input region dimension does not match network input size")

    for layer_index, ((weights1, bias1), (weights2, bias2)) in enumerate(
        zip(instance.nn1, instance.nn2), start=1
    ):
        if len(weights1) != len(weights2) or len(bias1) != len(bias2):
            raise ValueError(f"layer {layer_index} output sizes differ")
        if len(weights1[0]) != len(weights2[0]):
            raise ValueError(f"layer {layer_index} input sizes differ")

