from pathlib import Path
import json
import torch
from nn_equivalence.nn_types import JsonObject, LinearLayer


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

