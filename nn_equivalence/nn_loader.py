from pathlib import Path
import json
import torch
from nn_equivalence.nn_types import JsonObject, LinearLayer, NeuralNetwork


def load_weights(path: Path) -> JsonObject:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_linear_layers(path: Path) -> list[LinearLayer]:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    linear_indices = sorted(
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("layers.") and key.endswith(".weight")
    )

    return [
        (
            state_dict[f"layers.{index}.weight"].detach().cpu().tolist(),
            state_dict[f"layers.{index}.bias"].detach().cpu().tolist(),
        )
        for index in linear_indices
    ]


def load_nn_pair_1() -> tuple[NeuralNetwork, NeuralNetwork]:
    nn1: list[LinearLayer] = [
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
