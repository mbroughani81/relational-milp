from __future__ import annotations

import sys

import pyomo.environ as pyo

from benchmarks.common import Instance
from nn_equivalence.nn_types import Bounds, NeuralNetwork

WITNESS_TOLERANCE = 1e-6
BoundOverrides = dict[str, list[Bounds]]


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


def tighten_bounds(interval_bounds: Bounds, override_bounds: Bounds | None) -> Bounds:
    if override_bounds is None:
        return interval_bounds
    if len(interval_bounds) != len(override_bounds):
        raise ValueError("bound override length does not match interval bounds")

    tightened: Bounds = []
    for (interval_lower, interval_upper), (override_lower, override_upper) in zip(
        interval_bounds,
        override_bounds,
    ):
        lower = max(interval_lower, override_lower)
        upper = min(interval_upper, override_upper)
        if lower > upper:
            if lower - upper <= 1e-8:
                midpoint = 0.5 * (lower + upper)
                lower = midpoint
                upper = midpoint
            else:
                raise ValueError(
                    "bound override is inconsistent with interval bounds: "
                    f"interval=({interval_lower}, {interval_upper}), "
                    f"override=({override_lower}, {override_upper})"
                )
        tightened.append((lower, upper))
    return tightened


def affine_values(
    weights: list[list[float]],
    bias: list[float],
    inputs: list[float],
) -> list[float]:
    return [
        sum(weight * input_value for weight, input_value in zip(row, inputs))
        + bias_value
        for row, bias_value in zip(weights, bias)
    ]


def forward_values(nn: NeuralNetwork, inputs: list[float]) -> list[float]:
    values = inputs
    for weights, bias in nn[:-1]:
        values = [max(0.0, value) for value in affine_values(weights, bias, values)]

    output_weights, output_bias = nn[-1]
    return affine_values(output_weights, output_bias, values)


def add_vars(
    model: pyo.ConcreteModel,
    name: str,
    bounds: Bounds,
    domain: pyo.Set = pyo.Reals,
) -> list[pyo.Var]:
    component = pyo.Var(
        range(len(bounds)),
        domain=domain,
        bounds=lambda _, index: bounds[index],
    )
    model.add_component(name, component)
    return [component[index] for index in range(len(bounds))]


def add_affine_constraints(
    constraints: pyo.ConstraintList,
    output_vars: list[pyo.Var],
    weights: list[list[float]],
    input_vars: list[pyo.Var],
    bias: list[float],
) -> None:
    for output_index, output_var in enumerate(output_vars):
        constraints.add(
            output_var
            == sum(
                weights[output_index][input_index] * input_vars[input_index]
                for input_index in range(len(input_vars))
            )
            + bias[output_index]
        )


def add_relu_big_m_constraints(
    model: pyo.ConcreteModel,
    constraints: pyo.ConstraintList,
    z_vars: list[pyo.Var],
    a_vars: list[pyo.Var],
    z_bounds: Bounds,
    layer_name: str,
) -> None:
    delta_vars = add_vars(
        model,
        f"{layer_name}_delta",
        [(0.0, 1.0)] * len(z_vars),
        domain=pyo.Binary,
    )

    for index, (z_var, a_var) in enumerate(zip(z_vars, a_vars)):
        lower, upper = z_bounds[index]

        constraints.add(a_var >= z_var)
        constraints.add(a_var >= 0)
        constraints.add(a_var <= z_var - lower * (1 - delta_vars[index]))
        constraints.add(a_var <= upper * delta_vars[index])


def add_network_variables(
    model: pyo.ConcreteModel,
    constraints: pyo.ConstraintList,
    input_vars: list[pyo.Var],
    nn: NeuralNetwork,
    name_prefix: str,
    input_bounds: Bounds,
    bound_overrides: list[Bounds] | None = None,
) -> tuple[list[pyo.Var], Bounds]:
    if bound_overrides is not None and len(bound_overrides) != len(nn):
        raise ValueError(
            f"{name_prefix} bound override layer count does not match network"
        )

    current_bounds = input_bounds
    previous_vars = input_vars

    for layer_index, (weights, bias) in enumerate(nn, start=1):
        interval_z_bounds = affine_bounds(weights, bias, current_bounds)
        override_z_bounds = (
            None if bound_overrides is None else bound_overrides[layer_index - 1]
        )
        z_bounds = tighten_bounds(interval_z_bounds, override_z_bounds)
        current_vars = add_vars(
            model,
            f"{name_prefix}_z{layer_index}",
            z_bounds,
        )
        add_affine_constraints(
            constraints,
            current_vars,
            weights,
            previous_vars,
            bias,
        )

        is_output_layer = layer_index == len(nn)
        if is_output_layer:
            return current_vars, z_bounds

        current_activation_bounds = relu_bounds(z_bounds)
        current_activation_vars = add_vars(
            model,
            f"{name_prefix}_a{layer_index}",
            current_activation_bounds,
        )
        add_relu_big_m_constraints(
            model,
            constraints,
            current_vars,
            current_activation_vars,
            z_bounds,
            layer_name=f"{name_prefix}_layer_{layer_index}",
        )
        previous_vars = current_activation_vars
        current_bounds = current_activation_bounds

    raise ValueError("neural network must have at least one layer")


def add_output_distance_constraint(
    model: pyo.ConcreteModel,
    constraints: pyo.ConstraintList,
    first_output_vars: list[pyo.Var],
    second_output_vars: list[pyo.Var],
    first_output_bounds: Bounds,
    second_output_bounds: Bounds,
    epsilon: float,
    name_prefix: str,
) -> None:
    if len(first_output_vars) != len(second_output_vars):
        raise ValueError("output variable lists must have the same length")
    if not first_output_vars:
        raise ValueError("output variable lists must be non-empty")

    selectors = add_vars(
        model,
        f"{name_prefix}_selector",
        [(0.0, 1.0)] * len(first_output_vars),
        domain=pyo.Binary,
    )
    for index, (first_var, second_var) in enumerate(
        zip(first_output_vars, second_output_vars)
    ):
        first_lower, _ = first_output_bounds[index]
        _, second_upper = second_output_bounds[index]
        min_difference = first_lower - second_upper
        big_m = max(0.0, epsilon - min_difference)
        constraints.add(
            first_var - second_var >= epsilon - big_m * (1 - selectors[index])
        )

    constraints.add(sum(selectors) >= 1)


def encode_instance_direction(
    instance: Instance,
    first_network_name: str,
    second_network_name: str,
    first_network: NeuralNetwork,
    second_network: NeuralNetwork,
    bound_overrides: BoundOverrides | None = None,
) -> tuple[pyo.ConcreteModel, list[pyo.Var]]:
    model = pyo.ConcreteModel(
        name=f"{instance.instance_id}_{first_network_name}_minus_{second_network_name}"
    )
    model.constraints = pyo.ConstraintList()

    input_bounds = instance.input_region.bounds()
    input_vars = add_vars(model, "x", input_bounds)
    first_output_vars, first_output_bounds = add_network_variables(
        model,
        model.constraints,
        input_vars,
        first_network,
        first_network_name,
        input_bounds,
        None if bound_overrides is None else bound_overrides.get(first_network_name),
    )
    second_output_vars, second_output_bounds = add_network_variables(
        model,
        model.constraints,
        input_vars,
        second_network,
        second_network_name,
        input_bounds,
        None if bound_overrides is None else bound_overrides.get(second_network_name),
    )
    add_output_distance_constraint(
        model,
        model.constraints,
        first_output_vars,
        second_output_vars,
        first_output_bounds,
        second_output_bounds,
        instance.epsilon,
        name_prefix=f"{first_network_name}_minus_{second_network_name}",
    )
    model.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)

    return model, input_vars


def validate_directional_witness(
    instance: Instance,
    input_vars: list[pyo.Var],
    first_network_name: str,
    second_network_name: str,
    first_network: NeuralNetwork,
    second_network: NeuralNetwork,
) -> None:
    input_values = [float(pyo.value(var)) for var in input_vars]
    input_bounds = instance.input_region.bounds()
    input_verified = all(
        lower - WITNESS_TOLERANCE <= value <= upper + WITNESS_TOLERANCE
        for value, (lower, upper) in zip(input_values, input_bounds)
    )
    first_outputs = forward_values(first_network, input_values)
    second_outputs = forward_values(second_network, input_values)
    witness_margin = max(
        first_output - second_output
        for first_output, second_output in zip(first_outputs, second_outputs)
    )
    target_verified = witness_margin >= instance.epsilon - WITNESS_TOLERANCE
    witness_verified = input_verified and target_verified
    if not witness_verified:
        print(
            "Solver returned a feasible point, but the numeric witness did not "
            f"verify. direction={first_network_name}-{second_network_name}, "
            f"witness_margin={witness_margin}, required_margin={instance.epsilon}, "
            f"target_verified={target_verified}, input_verified={input_verified}",
            file=sys.stderr,
        )
