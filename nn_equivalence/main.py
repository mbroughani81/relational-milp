from pathlib import Path
import gurobipy as gp

from nn_equivalence.generator import generate_dnn_pair
from nn_equivalence.nn_loader import load_linear_layers, load_nn_pair_1, load_nn_pair_2
from nn_equivalence.nn_types import LinearLayer
import nn_equivalence.encoder as encoder

def main() -> None:
    pair = generate_dnn_pair(
        hidden_sizes=[64, 64],
        name="smoky"
    )
    nn1: list[LinearLayer] = load_linear_layers(pair.first)
    nn2: list[LinearLayer] = load_linear_layers(pair.second)

    # nn1, nn2 = load_nn_pair_1()
    # nn1, nn2 = load_nn_pair_2()

    model = gp.Model("nn_eq")
    model.Params.OutputFlag = 0

    epsilon = 0.06
    input_size = len(nn1[0][0][0])
    print(input_size)

    # add nn constraints and vars
    x = encoder.add_input_variables(
        model,
        input_size,
        0.0,
        1.0
    )
    _, _, nn1_output_vars, _ = encoder.add_hidden_variables(
        model,
        x,
        nn1,
        "nn1"
    )
    _, _, nn2_output_vars, _ = encoder.add_hidden_variables(
        model,
        x,
        nn2,
        "nn2"
    )

    # add equivalence contraint
    encoder.add_output_distance_constraint(
        model,
        nn1_output_vars,
        nn2_output_vars,
        epsilon
    )

    # solving
    model.optimize()

    # report
    print()
    print("Equivalence check result")
    print("=" * 36)
    print(f"status: {model.Status}")
    print(f"epsilon: {epsilon}")

    if model.Status == gp.GRB.OPTIMAL:
        x_values = [var.X for var in x]
        nn1_output_values = [var.X for var in nn1_output_vars]
        nn2_output_values = [var.X for var in nn2_output_vars]
        differences = [
            abs(first - second)
            for first, second in zip(nn1_output_values, nn2_output_values)
        ]

        print(f"x: {x_values}")
        print(f"nn1 output: {nn1_output_values}")
        print(f"nn2 output: {nn2_output_values}")
        print(f"absolute differences: {differences}")
        print(f"L-inf difference: {max(differences)}")
        print("equivalent: no")
    elif model.Status == gp.GRB.INFEASIBLE:
        print("equivalent: yes")
    else:
        print("equivalent: unknown")


if __name__ == "__main__":
    main()
