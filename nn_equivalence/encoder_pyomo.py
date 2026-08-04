from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any
from typing import Literal

import pyomo.environ as pyo
from pyomo.core.base.constraint import IndexedConstraint

from nn_equivalence.nn_types import Bounds, NeuralNetwork
from benchmarks.common import Hyperrectangle
from benchmarks.common import Instance
from benchmarks.common import constraints_list
from benchmarks.common import contains

WITNESS_TOLERANCE = 1e-6
NetworkBounds = dict[str, list[Bounds]]
PyomoVar = Any
ReLUBinaryPhase = Literal["stable_active", "stable_inactive", "unstable"]


@dataclass(frozen=True)
class ReLUBinaryVariable:
    network_name: str
    phase: ReLUBinaryPhase
    var: PyomoVar


@dataclass(frozen=True)
class ReLUBinaryStats:
    network_name: str
    all_binary_variables: int
    relu_binary_variables: int
    stable_active_relu_binary_variables: int
    stable_inactive_relu_binary_variables: int
    unstable_relu_binary_variables: int
    unfixed_binary_variables: int


@dataclass(frozen=True)
class EncodingDebugStats:
    direction_name: str
    network_stats: list[ReLUBinaryStats]
    relu_binary_variables: list[ReLUBinaryVariable]
    output_selector_binary_variables: int
    all_binary_variables: int
    unfixed_binary_variables: int


@dataclass(frozen=True)
class EncodedDirection:
    model: pyo.ConcreteModel
    input_vars: list[PyomoVar]
    debug_stats: EncodingDebugStats


def add_constraint(constraints: IndexedConstraint, expr: Any) -> None:
    constraints.add(len(constraints), expr)


def relu_bounds(z_bounds: Bounds) -> Bounds:
    return [(max(0.0, lower), max(0.0, upper)) for lower, upper in z_bounds]


def relu_binary_stats(
    network_name: str,
    layer_bounds: list[Bounds],
    fix_stable_relu_binaries: bool,
) -> ReLUBinaryStats:
    stable_active = 0
    stable_inactive = 0
    unstable = 0

    for z_bounds in layer_bounds[:-1]:
        for lower, upper in z_bounds:
            if lower >= 0.0:
                stable_active += 1
            elif upper <= 0.0:
                stable_inactive += 1
            else:
                unstable += 1

    relu_binaries = stable_active + stable_inactive + unstable
    return ReLUBinaryStats(
        network_name=network_name,
        all_binary_variables=relu_binaries,
        relu_binary_variables=relu_binaries,
        stable_active_relu_binary_variables=stable_active,
        stable_inactive_relu_binary_variables=stable_inactive,
        unstable_relu_binary_variables=unstable,
        unfixed_binary_variables=(
            unstable if fix_stable_relu_binaries else relu_binaries
        ),
    )


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


def add_relu_bound_constraints(
    model: pyo.ConcreteModel,
    constraints: IndexedConstraint,
    z_vars: list[PyomoVar],
    a_vars: list[PyomoVar],
    z_bounds: Bounds,
    network_name: str,
    layer_name: str,
    fix_stable_relu_binaries: bool,
) -> list[ReLUBinaryVariable]:
    delta_vars = add_vars(
        model,
        f"{layer_name}_delta",
        [(0.0, 1.0)] * len(z_vars),
        domain=pyo.Binary,
    )
    debug_variables: list[ReLUBinaryVariable] = []

    for index, (z_var, a_var) in enumerate(zip(z_vars, a_vars)):
        lower, upper = z_bounds[index]
        delta_var = delta_vars[index]
        if lower >= 0.0:
            phase: ReLUBinaryPhase = "stable_active"
            if fix_stable_relu_binaries:
                delta_var.fix(1.0)
        elif upper <= 0.0:
            phase = "stable_inactive"
            if fix_stable_relu_binaries:
                delta_var.fix(0.0)
        else:
            phase = "unstable"
        debug_variables.append(
            ReLUBinaryVariable(
                network_name=network_name,
                phase=phase,
                var=delta_var,
            )
        )

        add_constraint(constraints, a_var >= z_var)
        add_constraint(constraints, a_var >= 0)
        add_constraint(constraints, a_var <= z_var - lower * (1 - delta_var))
        add_constraint(constraints, a_var <= upper * delta_var)

    return debug_variables


def add_network_variables(
    model: pyo.ConcreteModel,
    constraints: IndexedConstraint,
    input_vars: list[PyomoVar],
    nn: NeuralNetwork,
    name_prefix: str,
    bound: list[Bounds],
    fix_stable_relu_binaries: bool,
) -> tuple[list[PyomoVar], Bounds, ReLUBinaryStats, list[ReLUBinaryVariable]]:
    if len(bound) != len(nn):
        raise ValueError(f"{name_prefix} bound layer count does not match network")

    previous_vars = input_vars
    debug_stats = relu_binary_stats(
        name_prefix,
        bound,
        fix_stable_relu_binaries,
    )
    debug_variables: list[ReLUBinaryVariable] = []

    for layer_index, (weights, bias) in enumerate(nn, start=1):
        z_bounds = bound[layer_index - 1]
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
            return current_vars, z_bounds, debug_stats, debug_variables

        current_activation_bounds = relu_bounds(z_bounds)
        current_activation_vars = add_vars(
            model,
            f"{name_prefix}_a{layer_index}",
            current_activation_bounds,
        )
        debug_variables.extend(
            add_relu_bound_constraints(
                model,
                constraints,
                current_vars,
                current_activation_vars,
                z_bounds,
                network_name=name_prefix,
                layer_name=f"{name_prefix}_layer_{layer_index}",
                fix_stable_relu_binaries=fix_stable_relu_binaries,
            )
        )
        previous_vars = current_activation_vars

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
) -> int:
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
    return len(selectors)


def encode_instance_direction(
    instance: Instance,
    first_network_name: str,
    second_network_name: str,
    first_network: NeuralNetwork,
    second_network: NeuralNetwork,
    bounds: NetworkBounds,
    fix_stable_relu_binaries: bool = True,
) -> EncodedDirection:
    model = pyo.ConcreteModel(
        name=f"{instance.instance_id}_{first_network_name}_minus_{second_network_name}"
    )
    constraints: IndexedConstraint = IndexedConstraint(pyo.Any)
    model.constraints = constraints

    input_box = Hyperrectangle.overapproximate(instance.input_region)
    input_bounds = input_box.bounds()
    input_vars = add_vars(model, "x", input_bounds)
    add_input_region_constraints(constraints, input_vars, instance)
    (
        first_output_vars,
        first_output_bounds,
        first_debug_stats,
        first_debug_variables,
    ) = add_network_variables(
        model,
        constraints,
        input_vars,
        first_network,
        first_network_name,
        bounds[first_network_name],
        fix_stable_relu_binaries,
    )
    (
        second_output_vars,
        second_output_bounds,
        second_debug_stats,
        second_debug_variables,
    ) = add_network_variables(
        model,
        constraints,
        input_vars,
        second_network,
        second_network_name,
        bounds[second_network_name],
        fix_stable_relu_binaries,
    )
    selector_binary_count = add_output_distance_constraint(
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

    all_binary_variables = (
        first_debug_stats.all_binary_variables
        + second_debug_stats.all_binary_variables
        + selector_binary_count
    )
    unfixed_binary_variables = (
        first_debug_stats.unfixed_binary_variables
        + second_debug_stats.unfixed_binary_variables
        + selector_binary_count
    )
    direction_name = f"{first_network_name}_minus_{second_network_name}"
    return EncodedDirection(
        model=model,
        input_vars=input_vars,
        debug_stats=EncodingDebugStats(
            direction_name=direction_name,
            network_stats=[first_debug_stats, second_debug_stats],
            relu_binary_variables=first_debug_variables + second_debug_variables,
            output_selector_binary_variables=selector_binary_count,
            all_binary_variables=all_binary_variables,
            unfixed_binary_variables=unfixed_binary_variables,
        ),
    )


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
