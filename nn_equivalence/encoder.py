import gurobipy as gp

from nn_equivalence.nn_types import Bounds, NeuralNetwork


def add_input_variables(
    model: gp.Model,
    input_size: int,
    lower_bound: float = -1,
    upper_bound: float = 1,
) -> list[gp.Var]:
    return [
        model.addVar(lb=lower_bound, ub=upper_bound, name=f"x_{i}")
        for i in range(input_size)
    ]


def add_affine_constraints(
    model: gp.Model,
    output_vars: list[gp.Var],
    weights: list[list[float]],
    input_vars: list[gp.Var],
    bias: list[float],
    layer_name: str,
) -> None:
    for i, output_var in enumerate(output_vars):
        model.addConstr(
            output_var
            == gp.quicksum(
                weights[i][j] * input_vars[j] for j in range(len(input_vars))
            )
            + bias[i],
            name=f"{layer_name}_{i}",
        )


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


def add_relu_big_m_constraints(
    model: gp.Model,
    z_vars: list[gp.Var],
    a_vars: list[gp.Var],
    z_bounds: Bounds,
    layer_name: str,
) -> list[gp.Var]:
    deltas: list[gp.Var] = []

    for i, (z, a) in enumerate(zip(z_vars, a_vars)):
        lower, upper = z_bounds[i]

        delta = model.addVar(vtype=gp.GRB.BINARY, name=f"{layer_name}_delta_{i}")
        deltas.append(delta)

        model.addConstr(a >= z, name=f"{layer_name}_relu_{i}_a_ge_z")
        model.addConstr(a >= 0, name=f"{layer_name}_relu_{i}_a_ge_0")
        model.addConstr(
            a <= z - lower * (1 - delta),
            name=f"{layer_name}_relu_{i}_a_le_z_minus_L_inactive",
        )
        model.addConstr(a <= upper * delta, name=f"{layer_name}_relu_{i}_a_le_U_active")

    return deltas


def _big_m_for_lower_bound(
    lower_bound: float,
    threshold: float,
    fallback_big_m: float,
) -> float:
    if lower_bound <= -gp.GRB.INFINITY / 2:
        return fallback_big_m
    return max(0.0, threshold - lower_bound)


def add_output_distance_constraint(
    model: gp.Model,
    first_output_vars: list[gp.Var],
    second_output_vars: list[gp.Var],
    epsilon: float,
    name_prefix: str = "output_distance",
    fallback_big_m: float = 1_000_000.0,
) -> list[gp.Var]:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if len(first_output_vars) != len(second_output_vars):
        raise ValueError("output variable lists must have the same length")

    model.update()

    selectors: list[gp.Var] = []
    for i, (first_var, second_var) in enumerate(
        zip(first_output_vars, second_output_vars)
    ):
        positive_selector = model.addVar(
            vtype=gp.GRB.BINARY, name=f"{name_prefix}_{i}_positive"
        )
        negative_selector = model.addVar(
            vtype=gp.GRB.BINARY, name=f"{name_prefix}_{i}_negative"
        )
        selectors.extend([positive_selector, negative_selector])

        diff_lower = first_var.LB - second_var.UB
        diff_upper = first_var.UB - second_var.LB
        positive_big_m = _big_m_for_lower_bound(
            diff_lower, epsilon, fallback_big_m
        )
        negative_big_m = _big_m_for_lower_bound(
            -diff_upper, epsilon, fallback_big_m
        )

        model.addConstr(
            first_var - second_var
            >= epsilon - positive_big_m * (1 - positive_selector),
            name=f"{name_prefix}_{i}_first_minus_second_ge_epsilon",
        )
        model.addConstr(
            second_var - first_var
            >= epsilon - negative_big_m * (1 - negative_selector),
            name=f"{name_prefix}_{i}_second_minus_first_ge_epsilon",
        )

    model.addConstr(
        gp.quicksum(selectors) >= 1,
        name=f"{name_prefix}_at_least_one_coordinate",
    )

    return selectors


def add_hidden_variables(
    model: gp.Model,
    input_vars: list[gp.Var],
    nn: NeuralNetwork,
    name_prefix: str,
    input_lower_bound: float = -1,
    input_upper_bound: float = 1,
) -> tuple[list[list[gp.Var]], list[list[gp.Var]], list[gp.Var], list[gp.Var]]:
    input_bounds: Bounds = [
        (input_lower_bound, input_upper_bound) for _ in input_vars
    ]
    current_bounds = input_bounds
    pre_activation_vars: list[list[gp.Var]] = []
    activation_vars: list[list[gp.Var]] = []
    all_deltas: list[gp.Var] = []
    previous_vars = input_vars

    for layer_index, (weights, bias) in enumerate(nn, start=1):
        z_bounds = affine_bounds(weights, bias, current_bounds)
        current_vars = [
            model.addVar(lb=lower, ub=upper, name=f"{name_prefix}_z{layer_index}_{i}")
            for i, (lower, upper) in enumerate(z_bounds)
        ]
        add_affine_constraints(
            model=model,
            output_vars=current_vars,
            weights=weights,
            input_vars=previous_vars,
            bias=bias,
            layer_name=f"{name_prefix}_layer_{layer_index}",
        )
        pre_activation_vars.append(current_vars)

        is_output_layer = layer_index == len(nn)
        if is_output_layer:
            previous_vars = current_vars
            current_bounds = z_bounds
            continue

        current_activation_vars = [
            model.addVar(lb=0.0, ub=max(0.0, upper), name=f"{name_prefix}_a{layer_index}_{i}")
            for i, (_, upper) in enumerate(z_bounds)
        ]
        all_deltas.extend(
            add_relu_big_m_constraints(
                model=model,
                z_vars=current_vars,
                a_vars=current_activation_vars,
                z_bounds=z_bounds,
                layer_name=f"{name_prefix}_layer_{layer_index}",
            )
        )
        activation_vars.append(current_activation_vars)
        previous_vars = current_activation_vars
        current_bounds = relu_bounds(z_bounds)

    return pre_activation_vars, activation_vars, previous_vars, all_deltas
