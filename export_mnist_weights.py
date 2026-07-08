from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

LayerWeights = dict[str, list[list[float]] | list[float]]


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run train_mnist_dnn.py first, or pass --checkpoint."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Expected {checkpoint_path} to contain 'model_state_dict'. "
            "This exporter expects checkpoints produced by train_mnist_dnn.py."
        )

    return checkpoint


def tensor_to_list(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()


def extract_weights(checkpoint: dict[str, Any]) -> dict[str, Any]:
    state_dict = checkpoint["model_state_dict"]
    required_keys = [
        "layers.0.weight",
        "layers.0.bias",
        "layers.2.weight",
        "layers.2.bias",
    ]

    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        raise KeyError(
            "Checkpoint does not match the expected SmallMnistReluNet architecture. "
            f"Missing keys: {missing_keys}"
        )

    w1 = state_dict["layers.0.weight"].detach().cpu()
    b1 = state_dict["layers.0.bias"].detach().cpu()
    w2 = state_dict["layers.2.weight"].detach().cpu()
    b2 = state_dict["layers.2.bias"].detach().cpu()

    input_shape = tuple(checkpoint.get("input_shape", (1, 28, 28)))
    hidden_size = int(checkpoint.get("hidden_size", b1.numel()))
    num_classes = int(checkpoint.get("num_classes", b2.numel()))

    return {
        "architecture": "SmallMnistReluNet",
        "description": "Flatten(28x28) -> Linear -> ReLU -> Linear -> 10 scores",
        "input_shape": list(input_shape),
        "input_size": int(w1.shape[1]),
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "test_accuracy": checkpoint.get("test_accuracy"),
        "layers": [
            {
                "name": "linear1",
                "type": "linear",
                "weight_name": "W1",
                "bias_name": "b1",
                "weight_shape": list(w1.shape),
                "bias_shape": list(b1.shape),
                "activation_after": "relu",
            },
            {
                "name": "linear2",
                "type": "linear",
                "weight_name": "W2",
                "bias_name": "b2",
                "weight_shape": list(w2.shape),
                "bias_shape": list(b2.shape),
                "activation_after": None,
            },
        ],
        "W1": tensor_to_list(w1),
        "b1": tensor_to_list(b1),
        "W2": tensor_to_list(w2),
        "b2": tensor_to_list(b2),
    }


def save_json(path: Path, weights: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(weights, file)


def save_npz(path: Path, weights: dict[str, Any]) -> None:
    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit(
            "Saving .npz requires NumPy. Either install numpy or use the default .json output."
        ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        W1=np.asarray(weights["W1"], dtype=np.float32),
        b1=np.asarray(weights["b1"], dtype=np.float32),
        W2=np.asarray(weights["W2"], dtype=np.float32),
        b2=np.asarray(weights["b2"], dtype=np.float32),
        input_shape=np.asarray(weights["input_shape"], dtype=np.int64),
        hidden_size=np.asarray(weights["hidden_size"], dtype=np.int64),
        num_classes=np.asarray(weights["num_classes"], dtype=np.int64),
    )


def save_weights(path: Path, weights: dict[str, Any]) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        save_json(path, weights)
    elif suffix == ".npz":
        save_npz(path, weights)
    else:
        raise ValueError("Output path must end with .json or .npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SmallMnistReluNet checkpoint weights for MILP verification."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/mnist_relu.pt"),
        help="Checkpoint produced by train_mnist_dnn.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/mnist_relu_weights.json"),
        help="Export path. Supported suffixes: .json, .npz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    weights = extract_weights(checkpoint)
    save_weights(args.output, weights)

    print("Exported MNIST ReLU weights")
    print("=" * 36)
    print(f"checkpoint: {args.checkpoint}")
    print(f"output: {args.output}")
    print(f"architecture: {weights['architecture']}")
    print(f"input size: {weights['input_size']}")
    print(f"hidden size: {weights['hidden_size']}")
    print(f"num classes: {weights['num_classes']}")
    print(f"W1 shape: {weights['layers'][0]['weight_shape']}")
    print(f"b1 shape: {weights['layers'][0]['bias_shape']}")
    print(f"W2 shape: {weights['layers'][1]['weight_shape']}")
    print(f"b2 shape: {weights['layers'][1]['bias_shape']}")


if __name__ == "__main__":
    main()
