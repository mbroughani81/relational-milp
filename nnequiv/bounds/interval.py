"""Pure interval-arithmetic pre-activation bound propagation.

No torch / auto_LiRPA here, so this module is safe to import from the encoder
and anywhere else in the pure core. Certified alpha-beta-CROWN tightening lives
in :mod:`nnequiv.bounds.crown`.
"""

from __future__ import annotations

from nnequiv.core.types import Bounds, NeuralNetwork


def affine_bounds(
    weights: list[list[float]],
    bias: list[float],
    input_bounds: Bounds,
) -> Bounds:
    output_bounds: Bounds = []

    for row, bias_value in zip(weights, bias):
        lower = bias_value
        upper = bias_value
        for weight, (input_lower, input_upper) in zip(row, input_bounds):
            if weight >= 0:
                lower += weight * input_lower
                upper += weight * input_upper
            else:
                lower += weight * input_upper
                upper += weight * input_lower
        output_bounds.append((lower, upper))

    return output_bounds


def relu_bounds(z_bounds: Bounds) -> Bounds:
    return [(max(0.0, lower), max(0.0, upper)) for lower, upper in z_bounds]


def tighten_bounds(interval_bounds: Bounds, bound: Bounds | None) -> Bounds:
    if bound is None:
        return interval_bounds
    if len(interval_bounds) != len(bound):
        raise ValueError("bound length does not match interval bounds")

    tightened: Bounds = []
    for (interval_lower, interval_upper), (bound_lower, bound_upper) in zip(
        interval_bounds,
        bound,
    ):
        lower = max(interval_lower, bound_lower)
        upper = min(interval_upper, bound_upper)
        if lower > upper:
            if lower - upper <= 1e-8:
                midpoint = 0.5 * (lower + upper)
                lower = midpoint
                upper = midpoint
            else:
                raise ValueError(
                    "bound is inconsistent with interval bounds: "
                    f"interval=({interval_lower}, {interval_upper}), "
                    f"bound=({bound_lower}, {bound_upper})"
                )
        tightened.append((lower, upper))
    return tightened


def compute_interval_bounds(
    network: NeuralNetwork,
    input_bounds: Bounds,
    bounds: list[Bounds] | None = None,
) -> list[Bounds]:
    if not network:
        raise ValueError("neural network must have at least one layer")
    if bounds is not None and len(bounds) != len(network):
        raise ValueError("bound layer count does not match network")

    network_bounds: list[Bounds] = []
    current_bounds = input_bounds
    for layer_index, (weights, bias) in enumerate(network):
        interval_z_bounds = affine_bounds(weights, bias, current_bounds)
        bound = None if bounds is None else bounds[layer_index]
        z_bounds = tighten_bounds(interval_z_bounds, bound)
        network_bounds.append(z_bounds)
        if layer_index != len(network) - 1:
            current_bounds = relu_bounds(z_bounds)

    return network_bounds
