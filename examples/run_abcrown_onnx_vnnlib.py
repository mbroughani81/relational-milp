from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import torch
from torch import nn

ABCROWN_DIR = Path(
    "/home/mbroughani81/Documents/research/nn-equivalence/alpha-beta-CROWN"
)
WORK_DIR = Path("artifacts/abcrown_toy")
INPUT_CENTER = [0.5, -0.2]
EPSILONS = [0.1, 0.2, 0.3, 0.4, 0.5]
TIMEOUT = 60.0
BATCH_SIZE = 2048
ONNX_OPSET = 18


class ToyReluNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        with torch.no_grad():
            self.net[0].weight.copy_(
                torch.tensor(
                    [
                        [1.0, -1.0],
                        [-0.5, 1.0],
                        [0.75, 0.25],
                    ]
                )
            )
            self.net[0].bias.copy_(torch.tensor([0.0, 0.1, -0.2]))
            self.net[2].weight.copy_(
                torch.tensor(
                    [
                        [1.2, -0.7, 0.5],
                        [-0.4, 1.0, -0.8],
                    ]
                )
            )
            self.net[2].bias.copy_(torch.tensor([0.05, -0.1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def write_vnnlib(
    path: Path,
    epsilon: float,
) -> None:
    lines: list[str] = []
    for i in range(2):
        lines.append(f"(declare-const X_{i} Real)")
    lines.append("")
    for i in range(2):
        lines.append(f"(declare-const Y_{i} Real)")
    lines.append("")
    lines.append("(assert (or")
    lines.append("  (and")
    for i, center in enumerate(INPUT_CENTER):
        lines.append(f"    (>= X_{i} {center - epsilon:.17g})")
        lines.append(f"    (<= X_{i} {center + epsilon:.17g})")
    lines.append("    (<= Y_0 Y_1)")

    lines.append("  )")
    lines.append("))")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_instances_csv(path: Path, vnnlib_paths: list[Path]) -> None:
    path.write_text(
        "".join(f"{vnnlib_path.name}\n" for vnnlib_path in vnnlib_paths),
        encoding="utf-8",
    )


def write_epsilon_manifest(
    path: Path,
    epsilons: list[float],
    vnnlib_paths: list[Path],
) -> None:
    lines = ["index,epsilon,vnnlib\n"]
    for index, (epsilon, vnnlib_path) in enumerate(zip(epsilons, vnnlib_paths)):
        lines.append(f"{index},{epsilon:.17g},{vnnlib_path.name}\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_config(
    path: Path,
    onnx_path: Path,
    root_path: Path,
    csv_path: Path,
    results_path: Path,
    batch_size: int,
    timeout: float,
) -> None:
    path.write_text(
        "\n".join(
            [
                "general:",
                "  device: cpu",
                f"  root_path: {root_path}",
                f"  csv_name: {csv_path.name}",
                f"  results_file: {results_path}",
                "model:",
                f"  onnx_path: {onnx_path}",
                "  input_shape: [-1, 2]",
                "data:",
                "  num_outputs: 2",
                "solver:",
                f"  batch_size: {batch_size}",
                "bab:",
                f"  timeout: {timeout:.17g}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def export_model(model: nn.Module, onnx_path: Path, opset: int) -> None:
    dummy_input = torch.zeros(1, 2)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
    )


def print_epsilon_summary(output: str) -> None:
    results: dict[int, str] = {}
    current_index: int | None = None

    for line in output.splitlines():
        index_match = re.search(r"idx:\s*(\d+)", line)
        if index_match:
            current_index = int(index_match.group(1))
            continue

        result_match = re.search(r"Result:\s*([a-zA-Z0-9_-]+)", line)
        if result_match and current_index is not None:
            results[current_index] = result_match.group(1)

    print()
    print("Epsilon verification summary")
    print("=" * 36)
    for index, epsilon in enumerate(EPSILONS):
        status = results.get(index, "missing")
        passed = status in {"safe", "safe-incomplete", "unsat"}
        outcome = "PASS" if passed else "FAIL"
        print(f"epsilon={epsilon:g}: {outcome} ({status})")


def run_abcrown(abcrown_dir: Path, config_path: Path) -> int:
    python_path = abcrown_dir / ".venv" / "bin" / "python"
    abcrown_path = abcrown_dir / "complete_verifier" / "abcrown.py"
    if not python_path.exists():
        raise FileNotFoundError(f"alpha-beta-CROWN venv Python not found: {python_path}")
    if not abcrown_path.exists():
        raise FileNotFoundError(f"abcrown.py not found: {abcrown_path}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(abcrown_dir)
    command = [str(python_path), str(abcrown_path), "--config", str(config_path)]
    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=abcrown_dir / "complete_verifier",
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")

    print_epsilon_summary(completed.stdout)
    return completed.returncode


def main() -> None:
    if any(epsilon <= 0 for epsilon in EPSILONS):
        raise SystemExit("all EPSILONS must be positive")

    work_dir = WORK_DIR.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    model = ToyReluNet().eval()
    checkpoint_path = work_dir / "toy_relu_model.pt"
    onnx_path = work_dir / "toy_relu_model.onnx"
    csv_path = work_dir / "instances.csv"
    manifest_path = work_dir / "epsilon_manifest.csv"
    config_path = work_dir / "abcrown_config.yaml"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    export_model(model, onnx_path, ONNX_OPSET)

    vnnlib_paths: list[Path] = []
    for epsilon in EPSILONS:
        suffix = str(epsilon).replace(".", "p")
        vnnlib_path = work_dir / f"property_eps_{suffix}.vnnlib"
        write_vnnlib(vnnlib_path, epsilon=epsilon)
        vnnlib_paths.append(vnnlib_path)
    write_instances_csv(csv_path, vnnlib_paths)
    write_epsilon_manifest(manifest_path, EPSILONS, vnnlib_paths)

    write_config(
        config_path,
        onnx_path=onnx_path,
        root_path=work_dir,
        csv_path=csv_path,
        results_path=work_dir / "abcrown_results.txt",
        batch_size=BATCH_SIZE,
        timeout=TIMEOUT,
    )

    print(f"PyTorch checkpoint: {checkpoint_path}", flush=True)
    print(f"ONNX model:         {onnx_path}", flush=True)
    print(f"VNNLIB properties:  {csv_path}", flush=True)
    print(f"Epsilon manifest:   {manifest_path}", flush=True)
    print(f"Config:             {config_path}", flush=True)
    print("Epsilon by abcrown index:", flush=True)
    for index, epsilon in enumerate(EPSILONS):
        print(f"  idx {index}: epsilon={epsilon:.17g}", flush=True)

    raise SystemExit(run_abcrown(ABCROWN_DIR.resolve(), config_path))


if __name__ == "__main__":
    main()
