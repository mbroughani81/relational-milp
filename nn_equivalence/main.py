from pathlib import Path
import gurobipy as gp

from nn_equivalence.generator import generate_dnn_pair
from nn_equivalence.nn_loader import load_linear_layers, load_nn_pair_1
from nn_equivalence.nn_types import LinearLayer
import nn_equivalence.encoder as encoder

def main() -> None:
    # pair = generate_dnn_pair(
    #     hidden_sizes=[32, 32],
    #     name="smoky"
    # )

    # print("Generated DNN pair")
    # print("=" * 36)
    # print(f"first: {pair.first}")
    # print(f"second: {pair.second}")

    # nn1: list[LinearLayer] = load_linear_layers(pair.first)
    # nn2: list[LinearLayer] = load_linear_layers(pair.second)
    nn1, nn2 = load_nn_pair_1()

    model = gp.Model("nn_eq")
    input_size = len(nn1[0][0][0])
    print(input_size)
    # add nn constraints and vars
    x = encoder.add_input_variables(
        model,
        input_size
    )
    encoder.add_hidden_variables(
        model,
        x,
        nn1,
        "nn1"
    )
    encoder.add_hidden_variables(
        model,
        x,
        nn2,
        "nn2"
    )

    # add equivalence contraint


if __name__ == "__main__":
    main()
