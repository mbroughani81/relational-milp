from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SmallMnistReluNet(nn.Module):
    """Small fully-connected ReLU network for MILP-friendly MNIST verification."""

    def __init__(
        self, hidden_sizes: list[int] | None = None, hidden_size: int | None = None
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [hidden_size if hidden_size is not None else 64]

        self.flatten = nn.Flatten()

        modules: list[nn.Module] = []
        input_size = 28 * 28
        for size in hidden_sizes:
            modules.append(nn.Linear(input_size, size))
            modules.append(nn.ReLU())
            input_size = size
        modules.append(nn.Linear(input_size, 10))

        self.hidden_sizes = hidden_sizes
        self.layers = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.flatten(x))


def get_data_loaders(
    data_dir: Path, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()

    train_set = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1000,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def save_checkpoint(
    path: Path,
    model: SmallMnistReluNet,
    hidden_sizes: list[int],
    test_accuracy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_shape": (1, 28, 28),
            "hidden_size": hidden_sizes[0] if len(hidden_sizes) == 1 else None,
            "hidden_sizes": hidden_sizes,
            "num_classes": 10,
            "test_accuracy": test_accuracy,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> SmallMnistReluNet:
    checkpoint = torch.load(path, map_location=device)
    hidden_sizes = checkpoint.get("hidden_sizes")
    if hidden_sizes is None:
        hidden_sizes = [int(checkpoint.get("hidden_size", 64))]
    model = SmallMnistReluNet(hidden_sizes=[int(size) for size in hidden_sizes]).to(
        device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def parse_hidden_sizes(value: str) -> list[int]:
    hidden_sizes = [int(size) for size in value.split(",") if size.strip()]
    if not hidden_sizes:
        raise argparse.ArgumentTypeError("must include at least one hidden layer size")
    if any(size < 1 for size in hidden_sizes):
        raise argparse.ArgumentTypeError("hidden layer sizes must be positive")
    return hidden_sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or load a small ReLU MNIST network for Phase 2 verification."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/mnist_relu.pt"))
    parser.add_argument(
        "--hidden-sizes",
        type=parse_hidden_sizes,
        default=None,
        help="Comma-separated hidden layer sizes, e.g. 16 or 32,32.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Single hidden layer size. Prefer --hidden-sizes for new runs.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even when the checkpoint already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    if args.hidden_size is not None and args.hidden_size < 1:
        raise SystemExit("--hidden-size must be positive")
    if args.hidden_sizes is None:
        args.hidden_sizes = [args.hidden_size if args.hidden_size is not None else 64]

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    if args.checkpoint.exists() and not args.force_train:
        model = load_checkpoint(args.checkpoint, device)
        test_loss, test_accuracy = evaluate(model, test_loader, device)
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"test loss: {test_loss:.4f}")
        print(f"test accuracy: {test_accuracy:.4%}")
        return

    model = SmallMnistReluNet(hidden_sizes=args.hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print("Small MNIST ReLU network")
    print("=" * 36)
    print(f"device: {device}")
    print(f"hidden sizes: {args.hidden_sizes}")
    print(f"checkpoint: {args.checkpoint}")

    final_test_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        test_loss, final_test_accuracy = evaluate(model, test_loader, device)

        print(
            f"epoch {epoch:02d}: "
            f"train loss={train_loss:.4f}, train acc={train_accuracy:.4%}, "
            f"test loss={test_loss:.4f}, test acc={final_test_accuracy:.4%}"
        )

    save_checkpoint(args.checkpoint, model, args.hidden_sizes, final_test_accuracy)
    print(f"saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
