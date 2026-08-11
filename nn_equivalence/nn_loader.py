from nn_equivalence.nn_types import LinearLayer, NeuralNetwork


def load_toy_nn_1() -> NeuralNetwork:
    return [
        (
            [[1.0, -0.5], [0.25, 0.75]],
            [0.10, -0.20],
        ),
        (
            [[0.60, 0.40], [-0.30, 1.10]],
            [0.00, 0.15],
        ),
        (
            [[1.20, -0.70], [0.50, 0.90]],
            [-0.05, 0.20],
        ),
    ]


def load_nn_pair_1() -> tuple[NeuralNetwork, NeuralNetwork]:
    nn1 = load_toy_nn_1()
    nn2: list[LinearLayer] = [
        (
            [[1.02, -0.48], [0.24, 0.77]],
            [0.11, -0.19],
        ),
        (
            [[0.58, 0.43], [-0.28, 1.08]],
            [0.01, 0.14],
        ),
        (
            [[1.18, -0.72], [0.52, 0.88]],
            [-0.04, 0.18],
        ),
    ]

    return nn1, nn2


def load_nn_pair_2() -> tuple[NeuralNetwork, NeuralNetwork]:
    nn = load_toy_nn_1()
    return nn, nn
