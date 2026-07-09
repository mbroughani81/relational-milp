from pathlib import Path

from nn_equivalence.generator import generate_dnn_pair
from nn_equivalence.nn_loader import load_linear_layers, load_nn_pair_1
from nn_equivalence.nn_types import LinearLayer

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


if __name__ == "__main__":
    main()
