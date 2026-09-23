"""Backward-compat shim: the core type aliases now live in ``nnequiv.core.types``.

Kept so existing ``from nn_equivalence.nn_types import ...`` sites keep working
during the migration; import from :mod:`nnequiv.core.types` in new code.
"""

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
    "Vector",
    "Matrix",
    "Bounds",
    "LinearLayer",
    "NeuralNetwork",
    "JsonValue",
    "JsonObject",
]
