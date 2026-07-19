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


def add_output_distance_constraint(
    model: gp.Model,
    first_output_vars: list[gp.Var],
    second_output_vars: list[gp.Var],
    epsilon: float,
    name_prefix: str = "output_distance",
) -> None:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if len(first_output_vars) != len(second_output_vars):
        raise ValueError("output variable lists must have the same length")



def add_hidden_variables(
    model: gp.Model,
    input_vars: list[gp.Var],
    nn: NeuralNetwork,
    name_prefix: str,
    input_bounds: Bounds,
) -> list[gp.Var]:
    current_bounds = input_bounds
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

        is_output_layer = layer_index == len(nn)
        if is_output_layer:
            previous_vars = current_vars
            current_bounds = z_bounds
            continue

        current_activation_vars = [
            model.addVar(lb=0.0, ub=max(0.0, upper), name=f"{name_prefix}_a{layer_index}_{i}")
            for i, (_, upper) in enumerate(z_bounds)
        ]
        add_relu_big_m_constraints(
            model=model,
            z_vars=current_vars,
            a_vars=current_activation_vars,
            z_bounds=z_bounds,
            layer_name=f"{name_prefix}_layer_{layer_index}",
        )
        previous_vars = current_activation_vars
        current_bounds = relu_bounds(z_bounds)

    return previous_vars
