"""One entry point for per-network pre-activation bounds.

``network_bounds(net, box, mode)`` returns the per-layer bounds a MILP/CROWN
encoding needs. ``interval`` is pure; ``abcrown`` tightens the interval bounds
with certified alpha-beta-CROWN results, importing torch only then.
"""

from __future__ import annotations

from typing import Literal

from nnequiv.bounds.interval import compute_interval_bounds
from nnequiv.core.types import Bounds, NeuralNetwork

BoundMode = Literal["interval", "abcrown"]


def network_bounds(
    network: NeuralNetwork,
    input_bounds: Bounds,
    mode: BoundMode = "interval",
) -> list[Bounds]:
    if mode == "interval":
        return compute_interval_bounds(network, input_bounds)
    if mode == "abcrown":
        # Imported lazily so this module (and the encoder that reaches it via
        # interval.py) stays torch-free until abcrown bounds are requested.
        from nnequiv.bounds.crown import compute_network_bounds

        crown_bounds = compute_network_bounds(network, input_bounds)
        return compute_interval_bounds(network, input_bounds, crown_bounds)
    raise ValueError(f"unsupported bound mode: {mode!r}")
