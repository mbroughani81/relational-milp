from typing import Sequence

import gurobipy as gp
from gurobipy import GRB

Vector = list[float]
Matrix = list[list[float]]


def relu(value: float) -> float:
    return max(0.0, value)


def affine_layer(weights: Matrix, bias: Vector, inputs: Sequence[float]) -> Vector:
    return [
        sum(weight * input_value for weight, input_value in zip(row, inputs))
        + bias_value
        for row, bias_value in zip(weights, bias)
    ]


def relu_layer(values: Sequence[float]) -> Vector:
    return [relu(value) for value in values]


def predict(scores: Sequence[float]) -> int:
    return max(range(len(scores)), key=lambda i: scores[i])


def print_vector(name: str, values: Sequence[float]) -> None:
    formatted = ", ".join(f"{value:.6f}" for value in values)
    print(f"{name}: [{formatted}]")


def print_gurobi_solution(name: str, variables: Sequence[gp.Var]) -> None:
    values = [var.X for var in variables]
    print_vector(name, values)


def print_constraints(model: gp.Model) -> None:
    print()
    print("Gurobi linear constraints")
    print("=" * 36)

    for constr in model.getConstrs():
        row = model.getRow(constr)

        terms = []
        for i in range(row.size()):
            coeff = row.getCoeff(i)
            var = row.getVar(i)
            terms.append(f"{coeff:+g} {var.VarName}")

        lhs = " ".join(terms)
        sense = constr.Sense
        rhs = constr.RHS

        print(f"{constr.ConstrName}: {lhs} {sense} {rhs:g}")


def add_affine_constraints(
    model: gp.Model,
    output_vars: list[gp.Var],
    weights: Matrix,
    input_vars: list[gp.Var],
    bias: Vector,
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


def build_affine_gurobi_model(
    x_values: Vector,
    w1: Matrix,
    b1: Vector,
    a1_values: Vector,
    w2: Matrix,
    b2: Vector,
) -> tuple[gp.Model, list[gp.Var], list[gp.Var], list[gp.Var], list[gp.Var]]:
    model = gp.Model("toy_dnn")
    model.Params.OutputFlag = 0

    x = [
        model.addVar(lb=value, ub=value, name=f"x_{i}")
        for i, value in enumerate(x_values)
    ]

    z1 = [model.addVar(lb=-GRB.INFINITY, name=f"z1_{i}") for i in range(len(b1))]
    a1 = [
        model.addVar(lb=value, ub=value, name=f"a1_{i}")
        for i, value in enumerate(a1_values)
    ]
    z2 = [model.addVar(lb=-GRB.INFINITY, name=f"z2_{i}") for i in range(len(b2))]
    add_affine_constraints(
        model=model,
        output_vars=z1,
        weights=w1,
        input_vars=x,
        bias=b1,
        layer_name="hidden_affine",
    )
    add_affine_constraints(
        model=model,
        output_vars=z2,
        weights=w2,
        input_vars=a1,
        bias=b2,
        layer_name="output_affine",
    )
    model.setObjective(0.0, GRB.MINIMIZE)
    return model, x, z1, a1, z2


def main() -> None:
    x_values: Vector = [0.5, -0.2]
    w1: Matrix = [[1.0, -1.0], [-0.5, 1], [0.75, 0.25]]
    b1: Vector = [0.0, 0.1, -0.2]
    w2: Matrix = [[1.2, -0.7, 0.5], [-0.4, 1.0, -0.8]]
    b2: Vector = [0.05, -0.1]
    z1_values = affine_layer(w1, b1, x_values)
    a1_values = relu_layer(z1_values)
    z2_values = affine_layer(w2, b2, a1_values)

    # Step 1: Sample inference on NN
    predicted_class = predict(z2_values)
    print("Tiny hand-written ReLU network")
    print("=" * 36)

    print()
    print("Input")
    print_vector("x", x_values)

    print()
    print("Layer 1: z1 = W1 x + b1")
    print_vector("z1", z1_values)

    print()
    print("Layer 1 activation: a1 = ReLU(z1)")
    print_vector("a1", a1_values)

    print()
    print("Output layer: z2 = W2 a1 + b2")
    print_vector("z2 / class scores", z2_values)

    print()
    print(f"predicted class: {predicted_class}")
    # Step 2: Encoding
    model, x, z1, a1, z2 = build_affine_gurobi_model(
        x_values=x_values, w1=w1, b1=b1, a1_values=a1_values, w2=w2, b2=b2
    )
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        print()
        print(f"Gurobi status: {model.Status}")
        return
    print()
    print("Gurobi affine-only solution")
    print("-" * 36)
    print_gurobi_solution("x", x)
    print_gurobi_solution("z1", z1)
    print_gurobi_solution("a1 fixed for now", a1)
    print_gurobi_solution("z2", z2)

    print_constraints(model)


if __name__ == "__main__":
    main()
