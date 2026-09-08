"""Train MNIST teacher/student pairs with Hinton-style knowledge distillation.

Pipeline:
  1. Train teacher with cross-entropy
  2. Freeze teacher
  3. Distill student with ``α L_KD + (1-α) L_CE``
  4. Export ``teacher.nnet`` / ``student.nnet`` plus ``metadata.json``
  5. Characterize behavioral similarity on the ReluDiff 100-image set
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from nn_equivalence.nn_types import NeuralNetwork
from nn_equivalence.reludiff_nnet import (
    load_reludiff_mnist_tests,
    network_architecture,
    write_nnet_from_scratch,
)

PAIR_PRESETS: dict[str, dict[str, Any]] = {
    "kd_1": {
        "teacher_hidden": [64, 32],
        "student_hidden": [32, 16],
        "temperature": 2.0,
        "alpha": 0.5,
        "description": "baseline capacity gap",
    },
    "kd_2": {
        "teacher_hidden": [64, 32],
        "student_hidden": [32, 16],
        "temperature": 4.0,
        "alpha": 0.5,
        "description": "same architectures as kd_1, softer targets (T=4)",
    },
    "kd_3": {
        "teacher_hidden": [128, 64],
        "student_hidden": [32, 16],
        "temperature": 2.0,
        "alpha": 0.5,
        "description": "larger teacher/student capacity gap",
    },
}


class MnistReluMLP(nn.Module):
    """Fully-connected ReLU MLP: Flatten → (Linear+ReLU)* → Linear(10)."""

    def __init__(self, hidden_sizes: list[int], num_classes: int = 10) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must be non-empty")
        if any(size < 1 for size in hidden_sizes):
            raise ValueError("hidden layer sizes must be positive")

        modules: list[nn.Module] = [nn.Flatten()]
        input_size = 28 * 28
        for size in hidden_sizes:
            modules.append(nn.Linear(input_size, size))
            modules.append(nn.ReLU())
            input_size = size
        modules.append(nn.Linear(input_size, num_classes))
        self.layers = nn.Sequential(*modules)
        self.hidden_sizes = list(hidden_sizes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@dataclass(frozen=True)
class BehavioralStats:
    teacher_accuracy: float
    student_accuracy: float
    disagreement_rate: float
    logit_gap_min: float
    logit_gap_median: float
    logit_gap_mean: float
    logit_gap_p95: float
    logit_gap_max: float
    teacher_param_count: int
    student_param_count: int


def parse_hidden_sizes(value: str) -> list[int]:
    sizes = [int(part) for part in value.split(",") if part.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("must include at least one hidden size")
    if any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("hidden sizes must be positive")
    return sizes


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def torch_mlp_to_network(model: MnistReluMLP) -> NeuralNetwork:
    layers: NeuralNetwork = []
    for module in model.layers:
        if isinstance(module, nn.Linear):
            weights = module.weight.detach().cpu().tolist()
            bias = module.bias.detach().cpu().tolist()
            layers.append((weights, bias))
    if not layers:
        raise ValueError("model contains no Linear layers")
    return layers


def freeze_model(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def get_mnist_loaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    train_subset_size: int | None,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    if train_subset_size is not None:
        if train_subset_size < 1 or train_subset_size > len(train_set):
            raise ValueError(
                f"train_subset_size must be in [1, {len(train_set)}], "
                f"got {train_subset_size}"
            )
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_set), generator=generator)[
            :train_subset_size
        ].tolist()
        train_set = Subset(train_set, indices)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1000,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


def hinton_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (
        temperature * temperature
    )
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * kd_loss + (1.0 - alpha) * ce_loss


def train_teacher(
    model: MnistReluMLP,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        print(
            f"  teacher epoch {epoch:02d}: "
            f"loss={total_loss / total:.4f}, acc={correct / total:.4%}"
        )


def train_student_kd(
    student: MnistReluMLP,
    teacher: MnistReluMLP,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    temperature: float,
    alpha: float,
) -> None:
    freeze_model(teacher)
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)
            loss = hinton_kd_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=temperature,
                alpha=alpha,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (student_logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        print(
            f"  student epoch {epoch:02d}: "
            f"loss={total_loss / total:.4f}, acc={correct / total:.4%}"
        )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile of empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


@torch.no_grad()
def characterize_pair(
    teacher: MnistReluMLP,
    student: MnistReluMLP,
    test_loader: DataLoader,
    device: torch.device,
    reludiff_tests_path: Path,
) -> BehavioralStats:
    teacher.eval()
    student.eval()

    teacher_accuracy = evaluate_accuracy(teacher, test_loader, device)
    student_accuracy = evaluate_accuracy(student, test_loader, device)

    disagreements = 0
    total = 0
    for images, _labels in test_loader:
        images = images.to(device)
        teacher_pred = teacher(images).argmax(dim=1)
        student_pred = student(images).argmax(dim=1)
        disagreements += (teacher_pred != student_pred).sum().item()
        total += images.size(0)
    disagreement_rate = disagreements / total

    pixels, labels, _random_pixels = load_reludiff_mnist_tests(reludiff_tests_path)
    gaps: list[float] = []
    for raw_pixels, label in zip(pixels, labels):
        x = torch.tensor(
            [[pixel / 255.0 for pixel in raw_pixels]],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 28, 28)
        teacher_logit = float(teacher(x)[0, label].item())
        student_logit = float(student(x)[0, label].item())
        gaps.append(abs(teacher_logit - student_logit))

    gaps_sorted = sorted(gaps)
    return BehavioralStats(
        teacher_accuracy=teacher_accuracy,
        student_accuracy=student_accuracy,
        disagreement_rate=disagreement_rate,
        logit_gap_min=gaps_sorted[0],
        logit_gap_median=_percentile(gaps_sorted, 0.5),
        logit_gap_mean=sum(gaps_sorted) / len(gaps_sorted),
        logit_gap_p95=_percentile(gaps_sorted, 0.95),
        logit_gap_max=gaps_sorted[-1],
        teacher_param_count=count_parameters(teacher),
        student_param_count=count_parameters(student),
    )


def suggested_epsilons(stats: BehavioralStats) -> list[float]:
    """Build a small epsilon difficulty curve from observed center-point gaps.

    Starts from the plan's default grid, then adds a few values anchored at the
    empirical median / 95th percentile so the verifier sees both "tight" and
    "loose" regimes relative to the trained pair.
    """
    base = [0.01, 0.05, 0.1, 0.5, 1.0]
    anchored = [
        round(stats.logit_gap_median, 4),
        round(stats.logit_gap_p95, 4),
        round(max(stats.logit_gap_max, stats.logit_gap_p95), 4),
    ]
    # Keep only positive, distinct values; drop anchors that collapse onto base.
    merged = sorted({value for value in base + anchored if value > 0})
    return merged


def save_checkpoint(
    path: Path,
    model: MnistReluMLP,
    *,
    role: str,
    extras: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "role": role,
            "model_state_dict": model.state_dict(),
            "hidden_sizes": model.hidden_sizes,
            "num_classes": model.num_classes,
            **extras,
        },
        path,
    )


def export_pair(
    output_dir: Path,
    teacher: MnistReluMLP,
    student: MnistReluMLP,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_network = torch_mlp_to_network(teacher)
    student_network = torch_mlp_to_network(student)
    write_nnet_from_scratch(teacher_network, output_dir / "teacher.nnet")
    write_nnet_from_scratch(student_network, output_dir / "student.nnet")
    save_checkpoint(
        output_dir / "teacher.pt",
        teacher,
        role="teacher",
        extras={"test_accuracy": metadata["teacher_accuracy"]},
    )
    save_checkpoint(
        output_dir / "student.pt",
        student,
        role="student",
        extras={"test_accuracy": metadata["student_accuracy"]},
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen MNIST teacher and distill a smaller student with "
            "Hinton-style KD, then export .nnet files for verification."
        )
    )
    parser.add_argument(
        "--pair-id",
        type=str,
        default="kd_1",
        help="Output directory name / preset id (default: kd_1).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(PAIR_PRESETS),
        default=None,
        help="Optional architecture/hyperparameter preset. Defaults to --pair-id "
        "when that id is a known preset.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/distillation/mnist"),
        help="Parent directory for pair folders.",
    )
    parser.add_argument(
        "--reludiff-tests",
        type=Path,
        default=Path("data/reludiff_mnist/mnist_tests.h"),
        help="ReluDiff 100-image fixture used for logit-gap characterization.",
    )
    parser.add_argument("--teacher-hidden", type=parse_hidden_sizes, default=None)
    parser.add_argument("--student-hidden", type=parse_hidden_sizes, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--teacher-epochs", type=int, default=10)
    parser.add_argument("--student-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--train-subset-size",
        type=int,
        default=None,
        help="Optional MNIST training subset size (default: full 60k).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing pair directory.",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    preset_name = args.preset
    if preset_name is None and args.pair_id in PAIR_PRESETS:
        preset_name = args.pair_id
    preset = PAIR_PRESETS.get(preset_name, {}) if preset_name else {}

    teacher_hidden = args.teacher_hidden or preset.get("teacher_hidden")
    student_hidden = args.student_hidden or preset.get("student_hidden")
    temperature = args.temperature if args.temperature is not None else preset.get(
        "temperature", 2.0
    )
    alpha = args.alpha if args.alpha is not None else preset.get("alpha", 0.5)

    if teacher_hidden is None or student_hidden is None:
        raise SystemExit(
            "teacher/student hidden sizes are required via --teacher-hidden / "
            "--student-hidden or a known --pair-id/--preset"
        )
    if not 0.0 <= alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1]")
    if temperature <= 0:
        raise SystemExit("--temperature must be positive")

    return {
        "preset": preset_name,
        "teacher_hidden": list(teacher_hidden),
        "student_hidden": list(student_hidden),
        "temperature": float(temperature),
        "alpha": float(alpha),
        "description": preset.get("description", ""),
    }


def main() -> None:
    args = parse_args()
    if args.teacher_epochs < 1 or args.student_epochs < 1:
        raise SystemExit("epoch counts must be at least 1")

    config = resolve_config(args)
    output_dir = args.output_root / args.pair_id
    if output_dir.exists() and not args.force:
        raise SystemExit(
            f"output directory already exists: {output_dir}. Pass --force to overwrite."
        )
    if not args.reludiff_tests.exists():
        raise SystemExit(
            f"missing ReluDiff tests file: {args.reludiff_tests}. "
            "Needed for logit-gap characterization on the 100 benchmark centers."
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_subset_size=args.train_subset_size,
        seed=args.seed,
    )

    teacher = MnistReluMLP(config["teacher_hidden"]).to(device)
    student = MnistReluMLP(config["student_hidden"]).to(device)

    print("Knowledge distillation MNIST pair")
    print("=" * 40)
    print(f"pair_id: {args.pair_id}")
    print(f"device: {device}")
    print(f"teacher: 784-{'-'.join(map(str, config['teacher_hidden']))}-10")
    print(f"student: 784-{'-'.join(map(str, config['student_hidden']))}-10")
    print(f"temperature: {config['temperature']}, alpha: {config['alpha']}")
    print(f"seed: {args.seed}")
    print(f"output: {output_dir}")

    print("\n[1/4] Training teacher (CE)...")
    train_teacher(
        teacher,
        train_loader,
        device,
        epochs=args.teacher_epochs,
        lr=args.lr,
    )
    freeze_model(teacher)

    print("\n[2/4] Distilling student (Hinton KD)...")
    train_student_kd(
        student,
        teacher,
        train_loader,
        device,
        epochs=args.student_epochs,
        lr=args.lr,
        temperature=config["temperature"],
        alpha=config["alpha"],
    )
    freeze_model(student)

    print("\n[3/4] Characterizing teacher/student pair...")
    stats = characterize_pair(
        teacher,
        student,
        test_loader,
        device,
        args.reludiff_tests,
    )
    epsilons = suggested_epsilons(stats)
    print(
        f"  Acc_T={stats.teacher_accuracy:.4%}, Acc_S={stats.student_accuracy:.4%}"
    )
    print(f"  disagreement={stats.disagreement_rate:.4%}")
    print(
        "  |zT[c]-zS[c]| on ReluDiff centers: "
        f"min={stats.logit_gap_min:.4f}, median={stats.logit_gap_median:.4f}, "
        f"mean={stats.logit_gap_mean:.4f}, p95={stats.logit_gap_p95:.4f}, "
        f"max={stats.logit_gap_max:.4f}"
    )
    print(f"  suggested epsilons: {epsilons}")
    print(
        f"  params: teacher={stats.teacher_param_count}, "
        f"student={stats.student_param_count}"
    )

    teacher_arch = network_architecture(torch_mlp_to_network(teacher))
    student_arch = network_architecture(torch_mlp_to_network(student))
    metadata = {
        "pair_id": args.pair_id,
        "preset": config["preset"],
        "description": config["description"],
        "teacher_arch": teacher_arch,
        "student_arch": student_arch,
        "temperature": config["temperature"],
        "alpha": config["alpha"],
        "seed": args.seed,
        "teacher_epochs": args.teacher_epochs,
        "student_epochs": args.student_epochs,
        "train_subset_size": args.train_subset_size,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "distillation": "hinton_response_kd",
        "property": "output_equivalence_logits",
        "property_formula": "|z_T[c] - z_S[c]| <= epsilon",
        "teacher_accuracy": stats.teacher_accuracy,
        "student_accuracy": stats.student_accuracy,
        "disagreement_rate": stats.disagreement_rate,
        "logit_gap_on_reludiff_centers": {
            "min": stats.logit_gap_min,
            "median": stats.logit_gap_median,
            "mean": stats.logit_gap_mean,
            "p95": stats.logit_gap_p95,
            "max": stats.logit_gap_max,
        },
        "suggested_epsilons": epsilons,
        "teacher_param_count": stats.teacher_param_count,
        "student_param_count": stats.student_param_count,
    }

    print("\n[4/4] Exporting networks...")
    export_pair(output_dir, teacher, student, metadata)
    print(f"wrote {output_dir / 'teacher.nnet'}")
    print(f"wrote {output_dir / 'student.nnet'}")
    print(f"wrote {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
