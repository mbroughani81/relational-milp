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
# Smallest top-1 strictness margin the encoding will accept. MILP solvers carry
# a primal feasibility tolerance around 1e-6, so a margin at or below that is
# met "within tolerance" by points that do not actually violate top-1
# equivalence. Two orders of magnitude above it is still negligible next to
# logit scales while staying safely enforceable.
TOP1_MIN_MARGIN = 1e-4
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
) -> tuple[list[PyomoVar], ReLUBinaryStats, list[ReLUBinaryVariable]]:
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
            return current_vars, debug_stats, debug_variables

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


def output_difference_bounds(
    first_output_bounds: Bounds,
    second_output_bounds: Bounds,
    output_index: int,
) -> tuple[float, float]:
    """Interval bounds on ``first[i] - second[i]`` from each network's own bounds."""
    first_lower, first_upper = first_output_bounds[output_index]
    second_lower, second_upper = second_output_bounds[output_index]
    return first_lower - second_upper, first_upper - second_lower


def add_output_distance_constraint(
    model: pyo.ConcreteModel,
    constraints: IndexedConstraint,
    first_output_vars: list[PyomoVar],
    second_output_vars: list[PyomoVar],
    epsilon: float,
    output_indices: list[int],
    first_output_bounds: Bounds | None = None,
    second_output_bounds: Bounds | None = None,
) -> int:
    """Assert that some output in ``output_indices`` violates epsilon-equivalence.

    A single index is the tight constraint ``first[i] - second[i] >= epsilon``
    and needs no binaries. Several indices need a disjunction -- the violation
    may occur at *any* of them -- encoded with one selector binary per index:

        sum_i y_i == 1
        first[i] - second[i] >= epsilon - M_i * (1 - y_i)

    where ``M_i = epsilon - min(first[i] - second[i])`` makes the constraint
    vacuous whenever ``y_i == 0``. The big-M comes from the caller's already
    computed per-network output bounds (interval or abcrown), so it is as tight
    as the bound tightening in force.

    Returns the number of selector binaries added.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if len(first_output_vars) != len(second_output_vars):
        raise ValueError("output variable lists must have the same length")
    if not first_output_vars:
        raise ValueError("output variable lists must be non-empty")
    if not output_indices:
        raise ValueError("output_indices must be non-empty")
    for output_index in output_indices:
        if output_index < 0 or output_index >= len(first_output_vars):
            raise ValueError("output_index is outside the output variable range")

    if len(output_indices) == 1:
        output_index = output_indices[0]
        add_constraint(
            constraints,
            first_output_vars[output_index] - second_output_vars[output_index] >= epsilon,
        )
        return 0

    if first_output_bounds is None or second_output_bounds is None:
        raise ValueError(
            "output bounds are required to encode a multi-output distance "
            "constraint (they supply the disjunction's big-M)"
        )

    selectors = pyo.Var(range(len(output_indices)), domain=pyo.Binary)
    model.add_component("output_selector", selectors)
    add_constraint(
        constraints,
        sum(selectors[position] for position in range(len(output_indices))) == 1,
    )
    for position, output_index in enumerate(output_indices):
        difference_lower, _ = output_difference_bounds(
            first_output_bounds,
            second_output_bounds,
            output_index,
        )
        # Never negative: epsilon >= 0 and difference_lower <= the achievable min.
        big_m = max(0.0, epsilon - difference_lower)
        add_constraint(
            constraints,
            first_output_vars[output_index] - second_output_vars[output_index]
            >= epsilon - big_m * (1 - selectors[position]),
        )
    return len(output_indices)


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
    output_indices = instance.output_indices
    if output_indices is None:
        raise ValueError(
            f"property_kind {instance.property_kind!r} is not a distance property; "
            "encode it with encode_instance_top1 instead"
        )
    selector_binary_count = add_output_distance_constraint(
        model,
        constraints,
        first_output_vars,
        second_output_vars,
        instance.epsilon,
        list(output_indices),
        first_output_bounds=bounds[first_network_name][-1],
        second_output_bounds=bounds[second_network_name][-1],
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


def add_argmax_binaries(
    model: pyo.ConcreteModel,
    constraints: IndexedConstraint,
    output_vars: list[PyomoVar],
    output_bounds: Bounds,
    name: str,
) -> list[PyomoVar]:
    """Add binaries marking which output of ``output_vars`` is the maximum.

    Exactly one binary is set, and setting ``selector[i]`` forces output ``i``
    to be at least every other output. Ties are permitted: at an exact tie any
    maximizer may be selected.
    """
    selectors = pyo.Var(range(len(output_vars)), domain=pyo.Binary)
    model.add_component(name, selectors)
    add_constraint(
        constraints,
        sum(selectors[index] for index in range(len(output_vars))) == 1,
    )
    for index in range(len(output_vars)):
        index_lower, _ = output_bounds[index]
        for other in range(len(output_vars)):
            if other == index:
                continue
            _, other_upper = output_bounds[other]
            big_m = max(0.0, other_upper - index_lower)
            add_constraint(
                constraints,
                output_vars[index]
                >= output_vars[other] - big_m * (1 - selectors[index]),
            )
    return [selectors[index] for index in range(len(output_vars))]


def encode_instance_top1(
    instance: Instance,
    first_network_name: str,
    second_network_name: str,
    bounds: NetworkBounds,
    fix_stable_relu_binaries: bool = True,
) -> EncodedDirection:
    """Encode "the two networks disagree on argmax somewhere in the region".

    Unlike the epsilon properties this is a single model rather than two
    directions. Feasible means a top-1 counterexample exists.

    The condition follows Teuber et al. Section IV-A: for each output ``j``
    they restrict to the (closed) region where the first network's output
    ``j`` is maximal, then require the second network's ``j`` to be maximal
    there too. So a counterexample is an ``x`` and an output ``j`` with

        z1[j] maximal in z1   and   z2[m] > z2[j] for some m

    -- an index the first network is willing to predict but the second
    network strictly rules out. Note the *same* ``j`` appears on both sides.
    Asking instead for "the two argmax selections differ" would be wrong: at
    any point where either network ties, two different maximizers can be
    selected, so every region containing a tie would report a counterexample.

    Both networks get argmax selector binaries (ties may pick any maximizer),
    linked by::

        z2[m] - z2[j] >= margin - M * (2 - first_selector[j] - second_selector[m])

    plus an integer constraint keeping the two selectors off the same output.

    ``instance.epsilon`` is the strictness margin. It must be positive -- a
    top-1 counterexample is a *strict* violation and a MILP cannot express a
    strict inequality -- and comfortably above the solver's feasibility
    tolerance (~1e-6), or the solver will accept near-violations; see
    :data:`TOP1_MIN_MARGIN`. ``unsat`` then means "no point where the second
    network beats the first network's argmax by more than the margin".
    """
    if instance.property_kind != "top1":
        raise ValueError(
            f"encode_instance_top1 requires property_kind 'top1', "
            f"got {instance.property_kind!r}"
        )
    margin = instance.epsilon
    if margin < TOP1_MIN_MARGIN:
        raise ValueError(
            f"top1 needs a strictness margin (Instance.epsilon) of at least "
            f"{TOP1_MIN_MARGIN:g}, got {margin:g}. A margin of 0 makes every tie "
            "point a counterexample, and one near the MILP feasibility tolerance "
            "lets the solver accept near-violations as real ones."
        )
    model = pyo.ConcreteModel(name=f"{instance.instance_id}_top1_disagreement")
    constraints: IndexedConstraint = IndexedConstraint(pyo.Any)
    model.constraints = constraints

    input_box = Hyperrectangle.overapproximate(instance.input_region)
    input_vars = add_vars(model, "x", input_box.bounds())
    add_input_region_constraints(constraints, input_vars, instance)

    networks = {
        first_network_name: instance.nn1,
        second_network_name: instance.nn2,
    }
    output_vars: dict[str, list[PyomoVar]] = {}
    network_stats: list[ReLUBinaryStats] = []
    relu_variables: list[ReLUBinaryVariable] = []
    for name, network in networks.items():
        vars_, stats, variables = add_network_variables(
            model,
            constraints,
            input_vars,
            network,
            name,
            bounds[name],
            fix_stable_relu_binaries,
        )
        output_vars[name] = vars_
        network_stats.append(stats)
        relu_variables.extend(variables)

    first_vars = output_vars[first_network_name]
    second_vars = output_vars[second_network_name]
    first_bounds = bounds[first_network_name][-1]
    second_bounds = bounds[second_network_name][-1]

    first_selectors = add_argmax_binaries(
        model, constraints, first_vars, first_bounds, "first_argmax_selector"
    )
    second_selectors = add_argmax_binaries(
        model, constraints, second_vars, second_bounds, "second_argmax_selector"
    )
    # The two selectors must not land on the same output. Stated as an integer
    # constraint rather than left to the big-M block below: the margin is tiny
    # and a big-M constraint that tight is satisfied "within tolerance" by the
    # solver's feasibility tolerance, which would admit non-counterexamples.
    for index in range(len(first_vars)):
        add_constraint(constraints, first_selectors[index] + second_selectors[index] <= 1)

    # The second network must beat the first network's argmax by the margin.
    for first_index in range(len(first_vars)):
        _, first_index_upper = second_bounds[first_index]
        for second_index in range(len(second_vars)):
            if second_index == first_index:
                continue
            second_index_lower, _ = second_bounds[second_index]
            big_m = max(0.0, margin - (second_index_lower - first_index_upper))
            add_constraint(
                constraints,
                second_vars[second_index] - second_vars[first_index]
                >= margin
                - big_m
                * (2 - first_selectors[first_index] - second_selectors[second_index]),
            )

    model.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)

    selector_binary_count = len(first_selectors) + len(second_selectors)
    return EncodedDirection(
        model=model,
        input_vars=input_vars,
        debug_stats=EncodingDebugStats(
            direction_name="top1_disagreement",
            network_stats=network_stats,
            relu_binary_variables=relu_variables,
            output_selector_binary_variables=selector_binary_count,
            all_binary_variables=sum(
                stats.all_binary_variables for stats in network_stats
            )
            + selector_binary_count,
            unfixed_binary_variables=sum(
                stats.unfixed_binary_variables for stats in network_stats
            )
            + selector_binary_count,
        ),
    )


def validate_top1_witness(
    instance: Instance,
    input_vars: list[PyomoVar],
) -> None:
    """Warn when a claimed top-1 counterexample does not reproduce numerically."""
    input_values: list[float] = []
    for var in input_vars:
        value = pyo.value(var)
        if value is None:
            raise ValueError("solver returned a witness with an uninitialized input")
        input_values.append(float(value))

    input_verified = contains(instance.input_region, input_values, WITNESS_TOLERANCE)
    first_outputs = forward_values(instance.nn1, input_values)
    second_outputs = forward_values(instance.nn2, input_values)

    # Mirror the encoding: some output the first network calls maximal must be
    # one the second network beats by more than the margin.
    first_max = max(first_outputs)
    second_max = max(second_outputs)
    witness_margin = max(
        (
            second_max - second_outputs[index]
            for index, value in enumerate(first_outputs)
            if value >= first_max - WITNESS_TOLERANCE
        ),
        default=0.0,
    )
    target_verified = witness_margin >= instance.epsilon - WITNESS_TOLERANCE
    if not (input_verified and target_verified):
        print(
            "Solver returned a feasible point, but the numeric witness did not "
            f"verify. property=top1, nn1_argmax={first_outputs.index(first_max)}, "
            f"nn2_argmax={second_outputs.index(second_max)}, "
            f"witness_margin={witness_margin}, required_margin={instance.epsilon}, "
            f"target_verified={target_verified}, input_verified={input_verified}",
            file=sys.stderr,
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
    output_indices = instance.output_indices
    if output_indices is None:
        raise ValueError(
            f"property_kind {instance.property_kind!r} has no directional witness"
        )
    # The disjunction is satisfied as soon as one output violates epsilon, so
    # the witness margin is the best margin over the compared outputs.
    witness_index = max(
        output_indices,
        key=lambda index: first_outputs[index] - second_outputs[index],
    )
    witness_margin = first_outputs[witness_index] - second_outputs[witness_index]
    target_verified = witness_margin >= instance.epsilon - WITNESS_TOLERANCE
    witness_verified = input_verified and target_verified
    if not witness_verified:
        print(
            "Solver returned a feasible point, but the numeric witness did not "
            f"verify. direction={first_network_name}-{second_network_name}, "
            f"witness_margin={witness_margin}, required_margin={instance.epsilon}, "
            f"output_index={witness_index}, "
            f"target_verified={target_verified}, input_verified={input_verified}",
            file=sys.stderr,
        )
