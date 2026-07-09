from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GeneratedDnnPair:
    first: Path
    second: Path


def architecture_name(hidden_sizes: list[int]) -> str:
    return "_".join(str(size) for size in hidden_sizes)


def train_dnn(
    checkpoint: Path,
    hidden_sizes: list[int],
    seed: int,
    epochs: int,
    batch_size: int,
    data_dir: Path,
    force_train: bool,
) -> None:
    command = [
        sys.executable,
        "train/train_mnist_dnn.py",
        "--checkpoint",
        str(checkpoint),
        "--hidden-sizes",
        ",".join(str(size) for size in hidden_sizes),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--data-dir",
        str(data_dir),
    ]

    if force_train:
        command.append("--force-train")

    subprocess.run(command, check=True)


def generate_dnn_pair(
    hidden_sizes: list[int],
    name: str,
    output_dir: Path = Path("models/nn_equivalence"),
    first_seed: int = 0,
    second_seed: int = 1,
    epochs: int = 1,
    batch_size: int = 256,
    data_dir: Path = Path("data"),
    force_train: bool = False,
) -> GeneratedDnnPair:
    output_dir.mkdir(parents=True, exist_ok=True)

    first_path = output_dir / f"{name}_{architecture_name(hidden_sizes)}_seed{first_seed}.pt"
    second_path = output_dir / f"{name}_{architecture_name(hidden_sizes)}_seed{second_seed}.pt"

    train_dnn(
        checkpoint=first_path,
        hidden_sizes=hidden_sizes,
        seed=first_seed,
        epochs=epochs,
        batch_size=batch_size,
        data_dir=data_dir,
        force_train=force_train,
    )
    train_dnn(
        checkpoint=second_path,
        hidden_sizes=hidden_sizes,
        seed=second_seed,
        epochs=epochs,
        batch_size=batch_size,
        data_dir=data_dir,
        force_train=force_train,
    )

    return GeneratedDnnPair(first=first_path, second=second_path)
