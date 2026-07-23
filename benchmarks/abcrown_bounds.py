from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import hashlib
import io
import json
import sys

import torch
from torch import nn

from nn_equivalence.nn_types import Bounds, NeuralNetwork


@dataclass
class ABCrownBoundOptions:
    timeout_sec: float
    method: str = "CROWN-Optimized"


@dataclass
class ABCrownBoundCache:
    values: dict[str, list[Bounds]] = field(default_factory=dict)


class PrefixPreActivationNetwork(nn.Module):
    def __init__(self, network: NeuralNetwork, target_layer_index: int) -> None:
        super().__init__()
        if target_layer_index < 0 or target_layer_index >= len(network):
            raise ValueError("target_layer_index is out of range")

        modules: list[nn.Module] = []
        for layer_index, (weights, bias) in enumerate(network):
            linear = nn.Linear(len(weights[0]), len(bias))
            with torch.no_grad():
                linear.weight.copy_(torch.tensor(weights, dtype=torch.float32))
                linear.bias.copy_(torch.tensor(bias, dtype=torch.float32))
            modules.append(linear)

            if layer_index == target_layer_index:
                break
            modules.append(nn.ReLU())

        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def network_bounds_key(
    network: NeuralNetwork,
    input_bounds: Bounds,
    options: ABCrownBoundOptions,
) -> str:
    payload = json.dumps(
        {
            "network": network,
            "input_bounds": input_bounds,
            "timeout_sec": options.timeout_sec,
            "method": options.method,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def compute_layer_bounds(
    network: NeuralNetwork,
    input_bounds: Bounds,
    target_layer_index: int,
    options: ABCrownBoundOptions,
) -> Bounds:
    if target_layer_index == 0:
        weights, bias = network[0]
        return affine_bounds(weights, bias, input_bounds)

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

        input_dim = len(input_bounds)
        dummy_input = torch.zeros(1, input_dim, dtype=torch.float32)
        model = PrefixPreActivationNetwork(network, target_layer_index).eval()
        bounded_model = BoundedModule(model, dummy_input, device="cpu")
        lower = torch.tensor(
            [[bound[0] for bound in input_bounds]],
            dtype=torch.float32,
        )
        upper = torch.tensor(
            [[bound[1] for bound in input_bounds]],
            dtype=torch.float32,
        )
        perturbation = PerturbationLpNorm(
            norm=float("inf"),
            x_L=lower,
            x_U=upper,
        )
        bounded_input = BoundedTensor(dummy_input, perturbation)
        lower_tensor, upper_tensor = bounded_model.compute_bounds(
            x=(bounded_input,),
            method=options.method,
        )

    if not bool(torch.all(torch.isfinite(lower_tensor))):
        raise RuntimeError(
            f"alpha-beta-CROWN lower bounds are not finite for layer "
            f"{target_layer_index + 1}"
        )
    if not bool(torch.all(torch.isfinite(upper_tensor))):
        raise RuntimeError(
            f"alpha-beta-CROWN upper bounds are not finite for layer "
            f"{target_layer_index + 1}"
        )

    lower_bounds = lower_tensor.detach().cpu().reshape(-1).tolist()
    upper_bounds = upper_tensor.detach().cpu().reshape(-1).tolist()
    return [
        (float(lower_value), float(upper_value))
        for lower_value, upper_value in zip(lower_bounds, upper_bounds)
    ]


def compute_network_bounds(
    network: NeuralNetwork,
    input_bounds: Bounds,
    options: ABCrownBoundOptions,
    cache: ABCrownBoundCache | None = None,
) -> list[Bounds] | None:
    key = network_bounds_key(network, input_bounds, options)
    if cache is not None and key in cache.values:
        return cache.values[key]

    bounds: list[Bounds] = []
    current_bounds = input_bounds
    for layer_index, (weights, bias) in enumerate(network):
        interval_bounds = affine_bounds(weights, bias, current_bounds)
        if layer_index == 0:
            layer_bounds = interval_bounds
        else:
            try:
                layer_bounds = compute_layer_bounds(
                    network,
                    input_bounds,
                    layer_index,
                    options,
                )
            except Exception as error:
                print(
                    "alpha-beta-CROWN bound tightening failed for "
                    f"layer {layer_index + 1}; using interval bounds for "
                    f"that layer. reason={error}",
                    file=sys.stderr,
                )
                layer_bounds = interval_bounds
        bounds.append(layer_bounds)
        if layer_index != len(network) - 1:
            current_bounds = relu_bounds(layer_bounds)

    if cache is not None:
        cache.values[key] = bounds
    return bounds
