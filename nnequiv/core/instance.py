"""A single verification instance and its structural validation."""

from __future__ import annotations

from dataclasses import dataclass, field

from nnequiv.core.property import EQUIVALENCE_PROPERTIES, EquivalenceProperty
from nnequiv.core.region import (
    AbstractPolytope,
    Hyperrectangle,
    constraints_list,
    dim,
)
from nnequiv.core.result import InstanceStatus
from nnequiv.core.types import NeuralNetwork


@dataclass(frozen=True)
class Instance:
    instance_id: str
    suite_name: str
    nn1: NeuralNetwork
    nn2: NeuralNetwork
    input_region: AbstractPolytope
    epsilon: float
    output_index: int = 0
    expected_status: InstanceStatus | None = None
    timeout_sec: float = 30.0
    metadata: dict[str, str | int | float] = field(default_factory=dict)
    property_kind: EquivalenceProperty = "logit_class"

    @property
    def output_indices(self) -> tuple[int, ...] | None:
        """Outputs the property compares, or ``None`` when it is not a distance.

        ``logit_class`` compares one output, ``linf`` compares them all, and
        ``top1`` compares argmaxes rather than distances.
        """
        if self.property_kind == "top1":
            return None
        if self.property_kind == "linf":
            return tuple(range(len(self.nn1[-1][1])))
        return (self.output_index,)


@dataclass(frozen=True)
class InstanceSuite:
    name: str
    instances: list[Instance]


def validate_instance(instance: Instance) -> None:
    if instance.epsilon < 0:
        # For top1 this is the tie margin, which must also be non-negative.
        raise ValueError("epsilon must be non-negative")
    if instance.property_kind not in EQUIVALENCE_PROPERTIES:
        raise ValueError(
            f"unknown property_kind {instance.property_kind!r}; "
            f"expected one of {list(EQUIVALENCE_PROPERTIES)}"
        )
    if not instance.nn1 or not instance.nn2:
        raise ValueError("nn1 and nn2 must each have at least one layer")

    input_dimension = dim(instance.input_region)
    constraints_list(instance.input_region)
    Hyperrectangle.overapproximate(instance.input_region)

    nn1_input_size = len(instance.nn1[0][0][0])
    nn2_input_size = len(instance.nn2[0][0][0])
    if nn1_input_size != input_dimension:
        raise ValueError("input region dimension does not match nn1 input size")
    if nn2_input_size != input_dimension:
        raise ValueError("input region dimension does not match nn2 input size")

    # Teacher/student pairs (e.g. knowledge distillation) may have different
    # hidden widths and depths. Only the shared input/output interface is required.
    nn1_output_size = len(instance.nn1[-1][1])
    nn2_output_size = len(instance.nn2[-1][1])
    if nn1_output_size != nn2_output_size:
        raise ValueError(
            "nn1 and nn2 must have the same output size: "
            f"nn1={nn1_output_size}, nn2={nn2_output_size}"
        )
    # output_index only selects an output for logit_class; linf compares every
    # output and top1 compares argmaxes, so neither reads it.
    if instance.property_kind == "logit_class" and (
        instance.output_index < 0 or instance.output_index >= nn1_output_size
    ):
        raise ValueError(
            "output_index is outside the network output range: "
            f"index={instance.output_index}, output_size={nn1_output_size}"
        )
