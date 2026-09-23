"""Pure domain model for the NN-equivalence harness.

Nothing here imports a solver, torch, or subprocess — every verifier and the
runner depend on these types, never the other way around.
"""

from nnequiv.core.instance import Instance, InstanceSuite, validate_instance
from nnequiv.core.property import EQUIVALENCE_PROPERTIES, EquivalenceProperty
from nnequiv.core.region import (
    AbstractPolytope,
    HalfSpace,
    Hyperrectangle,
    constraints_list,
    contains,
    dim,
)
from nnequiv.core.result import InstanceResult, InstanceStatus, SolveStats
from nnequiv.core.types import (
    Bounds,
    JsonObject,
    JsonValue,
    LinearLayer,
    Matrix,
    NeuralNetwork,
    Vector,
)

__all__ = [
    "AbstractPolytope",
    "HalfSpace",
    "Hyperrectangle",
    "constraints_list",
    "contains",
    "dim",
    "EquivalenceProperty",
    "EQUIVALENCE_PROPERTIES",
    "InstanceStatus",
    "SolveStats",
    "InstanceResult",
    "Instance",
    "InstanceSuite",
    "validate_instance",
    "Vector",
    "Matrix",
    "Bounds",
    "LinearLayer",
    "NeuralNetwork",
    "JsonValue",
    "JsonObject",
]
