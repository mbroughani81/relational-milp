from __future__ import annotations

import sys
from typing import Any

import pyomo.environ as pyo
from pyomo.core.base.constraint import IndexedConstraint

from nn_equivalence.nn_types import Bounds, NeuralNetwork
from benchmarks.common import Hyperrectangle
from benchmarks.common import Instance
from benchmarks.common import constraints_list
from benchmarks.common import contains

WITNESS_TOLERANCE = 1e-6
BoundOverrides = dict[str, list[Bounds]]
PyomoVar = Any


def add_constraint(constraints: IndexedConstraint, expr: Any) -> None:
    constraints.add(len(constraints), expr)


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
) -> list[PyomoVar]:
    component = pyo.Var(
        range(len(bounds)),
        domain=domain,
        bounds=lambda _, index: bounds[index],
    )
    model.add_component(name, component)
    return [component[index] for index in range(len(bounds))]


def add_input_region_constraints(
    constraints: IndexedConstraint,
    input_vars: list[PyomoVar],
    instance: Instance,
) -> None:
    for region_constraint in constraints_list(instance.input_region):
        expression = sum(
            coefficient * input_vars[index]
            for index, coefficient in enumerate(region_constraint.a)
        )
        add_constraint(constraints, expression <= region_constraint.b)


def add_affine_constraints(
    constraints: IndexedConstraint,
    output_vars: list[PyomoVar],
    weights: list[list[float]],
    input_vars: list[PyomoVar],
    bias: list[float],
) -> None:
    for output_index, output_var in enumerate(output_vars):
        add_constraint(
            constraints,
            output_var
            == sum(
                weights[output_index][input_index] * input_vars[input_index]
                for input_index in range(len(input_vars))
            )
            + bias[output_index]
        )


def add_relu_big_m_constraints(
    model: pyo.ConcreteModel,
    constraints: IndexedConstraint,
    z_vars: list[PyomoVar],
    a_vars: list[PyomoVar],
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

        add_constraint(constraints, a_var >= z_var)
        add_constraint(constraints, a_var >= 0)
        add_constraint(constraints, a_var <= z_var - lower * (1 - delta_vars[index]))
        add_constraint(constraints, a_var <= upper * delta_vars[index])


def add_network_variables(
    model: pyo.ConcreteModel,
    constraints: IndexedConstraint,
    input_vars: list[PyomoVar],
    nn: NeuralNetwork,
    name_prefix: str,
    input_bounds: Bounds,
    bound_overrides: list[Bounds] | None = None,
) -> tuple[list[PyomoVar], Bounds]:
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
    constraints: IndexedConstraint,
    first_output_vars: list[PyomoVar],
    second_output_vars: list[PyomoVar],
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
        add_constraint(
            constraints,
            first_var - second_var >= epsilon - big_m * (1 - selectors[index])
        )

    add_constraint(constraints, sum(selectors) >= 1)


def encode_instance_direction(
    instance: Instance,
    first_network_name: str,
    second_network_name: str,
    first_network: NeuralNetwork,
    second_network: NeuralNetwork,
    bound_overrides: BoundOverrides | None = None,
) -> tuple[pyo.ConcreteModel, list[PyomoVar]]:
    model = pyo.ConcreteModel(
        name=f"{instance.instance_id}_{first_network_name}_minus_{second_network_name}"
    )
    constraints: IndexedConstraint = IndexedConstraint(pyo.Any)
    model.constraints = constraints

    input_box = Hyperrectangle.overapproximate(instance.input_region)
    input_bounds = input_box.bounds()
    input_vars = add_vars(model, "x", input_bounds)
    add_input_region_constraints(constraints, input_vars, instance)
    first_output_vars, first_output_bounds = add_network_variables(
        model,
        constraints,
        input_vars,
        first_network,
        first_network_name,
        input_bounds,
        None if bound_overrides is None else bound_overrides.get(first_network_name),
    )
    second_output_vars, second_output_bounds = add_network_variables(
        model,
        constraints,
        input_vars,
        second_network,
        second_network_name,
        input_bounds,
        None if bound_overrides is None else bound_overrides.get(second_network_name),
    )
    add_output_distance_constraint(
        model,
        constraints,
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
    input_vars: list[PyomoVar],
    first_network_name: str,
    second_network_name: str,
    first_network: NeuralNetwork,
    second_network: NeuralNetwork,
) -> None:
    input_values: list[float] = []
    for var in input_vars:
        value = pyo.value(var)
        if value is None:
            raise ValueError("solver returned a witness with an uninitialized input")
        input_values.append(float(value))
    input_verified = contains(instance.input_region, input_values, WITNESS_TOLERANCE)
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
