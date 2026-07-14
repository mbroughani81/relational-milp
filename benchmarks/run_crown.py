from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import hashlib
import importlib
import io
import json
import re
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from benchmarks.common import (
    Benchmark,
    BenchmarkResult,
    BenchmarkStatus,
    BenchmarkSuite,
    validate_benchmark,
)
from nn_equivalence.nn_types import NeuralNetwork
ARTIFACT_ROOT = Path("artifacts/abcrown_benchmarks")
ONNX_OPSET = 18
BATCH_SIZE = 2048
ConfigValue = str | int | float | bool | dict[str, "ConfigValue"]
ConfigDict = dict[str, ConfigValue]


ABCROWN_PROFILES: dict[str, ConfigDict] = {
    # Full alpha-beta-CROWN pipeline: alpha-CROWN first, then beta-CROWN BaB.
    "default": {
    },

    # Incomplete baselines. They can prove safety, but return unknown when
    # their relaxation is not strong enough.
    "crown_only": {
        "general": {
            "complete_verifier": "skip",
            "enable_incomplete_verification": True,
        },
        "solver": {
            "bound_prop_method": "crown",
        },
    },
    "alpha_only": {
        "general": {
            "complete_verifier": "skip",
            "enable_incomplete_verification": True,
        },
        "solver": {
            "bound_prop_method": "alpha-crown",
            "alpha-crown": {
                "iteration": 100,
                "lr_alpha": 0.1,
            },
        },
    },

    # Complete beta-CROWN variants with different optimization budgets.
    "beta_fast": {
        "solver": {
            "alpha-crown": {
                "iteration": 50,
            },
            "beta-crown": {
                "iteration": 10,
            },
        },
        "bab": {
            "branching": {
                "method": "kfsb",
                "candidates": 1,
            },
        },
    },
    "beta_strong": {
        "solver": {
            "alpha-crown": {
                "iteration": 200,
                "lr_decay": 0.99,
            },
            "beta-crown": {
                "iteration": 100,
                "lr_decay": 0.99,
            },
        },
        "bab": {
            "branching": {
                "method": "kfsb",
                "candidates": 5,
            },
        },
    },
    "bab_no_incomplete": {
        "general": {
            "enable_incomplete_verification": False,
        },
    },
    # Branching-heuristic comparison. Keep every other setting unchanged.
    "branch_babsr": {
        "bab": {
            "branching": {
                "method": "babsr",
            },
        },
    },
    "branch_kfsb": {
        "bab": {
            "branching": {
                "method": "kfsb",
                "candidates": 3,
            },
        },
    },
    "branch_fsb": {
        "bab": {
            "branching": {
                "method": "fsb",
                "candidates": 3,
            },
        },
    },

    # Counterexample search before formal verification.
    "attack_heavy": {
        "attack": {
            "pgd_order": "before",
            "pgd_steps": 200,
            "pgd_restarts": 100,
        },
    },

    # Branch over the input box instead of unstable ReLU phases.
    "input_split": {
        "solver": {
            "bound_prop_method": "crown",
        },
        "bab": {
            "branching": {
                "method": "sb",
                "input_split": {
                    "enable": True,
                },
            },
        },
    },

    # Exact MIP baseline for small networks. Requires Gurobi (or change to SCIP).
    "mip_small": {
        "general": {
            "complete_verifier": "mip",
            "enable_incomplete_verification": False,
        },
        "solver": {
            "mip": {
                "mip_solver": "gurobi",
                "formulation": "mip",
                "parallel_solvers": 4,
                "solver_threads": 1,
            },
        },
    },

    # Hybrid: MIP tightens intermediate bounds, then beta-CROWN BaB runs.
    "bab_refine": {
        "general": {
            "complete_verifier": "bab-refine",
        },
        "solver": {
            "mip": {
                "mip_solver": "gurobi",
                "parallel_solvers": 4,
                "solver_threads": 1,
                "refine_neuron_timeout": 5,
                "refine_neuron_time_percentage": 0.5,
            },
        },
    },
}



class ReluNetwork(nn.Module):
    def __init__(self, network: NeuralNetwork) -> None:
        super().__init__()
        modules: list[nn.Module] = []

        for layer_index, (weights, bias) in enumerate(network):
            linear = nn.Linear(len(weights[0]), len(bias))
            with torch.no_grad():
                linear.weight.copy_(torch.tensor(weights, dtype=torch.float32))
                linear.bias.copy_(torch.tensor(bias, dtype=torch.float32))
            modules.append(linear)

            if layer_index != len(network) - 1:
                modules.append(nn.ReLU())

        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DifferenceNetwork(nn.Module):
    def __init__(self, nn1: NeuralNetwork, nn2: NeuralNetwork) -> None:
        super().__init__()
        self.nn1 = ReluNetwork(nn1)
        self.nn2 = ReluNetwork(nn2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn1(x) - self.nn2(x)


def load_suite(name: str) -> BenchmarkSuite:
    module = importlib.import_module(f"benchmarks.{name}")
    return module.load_suite()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an NN equivalence benchmark suite with alpha-beta-CROWN."
    )
    parser.add_argument("--suite", default="sample")
    parser.add_argument(
        "--profile",
        default="beta_strong",
        choices=sorted(ABCROWN_PROFILES),
        help="Named alpha-beta-CROWN configuration profile.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available alpha-beta-CROWN profiles and exit.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "Override every benchmark.timeout_sec before generating CROWN configs. "
            "Without this, each suite benchmark controls its own bab.timeout."
        ),
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def model_key(benchmark: Benchmark) -> str:
    payload = json.dumps([benchmark.nn1, benchmark.nn2], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def export_model_once(
    benchmark: Benchmark,
    model_dir: Path,
    exported_models: dict[str, Path],
) -> Path:
    key = model_key(benchmark)
    if key in exported_models:
        return exported_models[key]

    onnx_path = model_dir / f"diff_{key}.onnx"
    input_dim = len(benchmark.input_region.lower_bounds)
    model = DifferenceNetwork(benchmark.nn1, benchmark.nn2).eval()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        torch.onnx.export(
            model,
            torch.zeros(1, input_dim),
            onnx_path,
            input_names=["input"],
            output_names=["diff"],
            opset_version=ONNX_OPSET,
        )
    exported_models[key] = onnx_path
    return onnx_path


def write_vnnlib(path: Path, benchmark: Benchmark) -> None:
    output_dim = len(benchmark.nn1[-1][1])
    lines: list[str] = []

    for i in range(len(benchmark.input_region.lower_bounds)):
        lines.append(f"(declare-const X_{i} Real)")
    lines.append("")
    for i in range(output_dim):
        lines.append(f"(declare-const Y_{i} Real)")
    lines.append("")
    lines.append("(assert (or")

    for output_index in range(output_dim):
        lines.append("  (and")
        for input_index, (lower, upper) in enumerate(benchmark.input_region.bounds()):
            lines.append(f"    (>= X_{input_index} {lower:.17g})")
            lines.append(f"    (<= X_{input_index} {upper:.17g})")
        lines.append(f"    (>= Y_{output_index} {benchmark.epsilon:.17g})")
        lines.append("  )")

        lines.append("  (and")
        for input_index, (lower, upper) in enumerate(benchmark.input_region.bounds()):
            lines.append(f"    (>= X_{input_index} {lower:.17g})")
            lines.append(f"    (<= X_{input_index} {upper:.17g})")
        lines.append(f"    (<= Y_{output_index} {-benchmark.epsilon:.17g})")
        lines.append("  )")

    lines.append("))")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_instances_csv(
    path: Path,
    rows: list[tuple[Path, Path, float]],
    root_dir: Path,
) -> None:
    lines = []
    for onnx_path, vnnlib_path, timeout in rows:
        lines.append(
            f"{onnx_path.relative_to(root_dir)},"
            f"{vnnlib_path.relative_to(root_dir)},"
            f"{timeout:.17g}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def write_manifest(path: Path, benchmarks: list[Benchmark], rows: list[tuple[Path, Path, float]], root_dir: Path) -> None:
    lines = ["index,benchmark_id,epsilon,expected_status,onnx,vnnlib,timeout\n"]
    for index, (benchmark, (onnx_path, vnnlib_path, timeout)) in enumerate(
        zip(benchmarks, rows)
    ):
        expected = benchmark.expected_status or ""
        lines.append(
            f"{index},{benchmark.benchmark_id},{benchmark.epsilon:.17g},"
            f"{expected},{onnx_path.relative_to(root_dir)},"
            f"{vnnlib_path.relative_to(root_dir)},{timeout:.17g}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def suite_output_dim(benchmarks: list[Benchmark]) -> int:
    output_dims = {len(benchmark.nn1[-1][1]) for benchmark in benchmarks}
    if len(output_dims) != 1:
        raise ValueError(
            "alpha-beta-CROWN config requires one data.num_outputs value; "
            f"got output dimensions {sorted(output_dims)}"
        )
    return output_dims.pop()


def merge_config(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    merged: ConfigDict = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = merge_config(existing, value)
        else:
            merged[key] = value
    return merged


def yaml_scalar(value: ConfigValue) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def config_to_yaml(config: ConfigDict, indent: int = 0) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(config_to_yaml(value, indent + 2))
        else:
            lines.append(f"{prefix}{key}: {yaml_scalar(value)}")
    return lines


def write_config(
    path: Path,
    work_dir: Path,
    results_path: Path,
    output_dim: int,
    profile: str,
    timeout_sec: float,
) -> None:
    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive")

    base_config: ConfigDict = {
        "general": {
            "device": "cpu",
            "complete_verifier": "bab",
            "root_path": str(work_dir),
            "csv_name": "instances.csv",
            "results_file": str(results_path)
        },
        "data": {
            "num_outputs": output_dim,
        },
        "solver": {
            "batch_size": BATCH_SIZE,
        },
        "bab": {
            "timeout": timeout_sec
        }
    }
    config = merge_config(base_config, ABCROWN_PROFILES[profile])
    config.setdefault("bab", {})["timeout"] = timeout_sec
    print("config => ", file=sys.stderr)
    print(config, file=sys.stderr)
    path.write_text("\n".join(config_to_yaml(config)) + "\n", encoding="utf-8")


def run_abcrown(
    config_path: Path,
    benchmarks: list[Benchmark],
    rows: list[tuple[Path, Path, float]],
    profile: str,
) -> tuple[int, str, dict[int, tuple[str, float]]]:
    captured = io.StringIO()
    results: dict[int, tuple[str, float]] = {}
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            from abcrown import ABCrownSolver, ConfigBuilder, IOConstraints

            if len(rows) != len(benchmarks):
                raise ValueError(
                    f"artifact row count ({len(rows)}) does not match benchmark "
                    f"count ({len(benchmarks)})"
                )
            output_dim = suite_output_dim(benchmarks)
            work_dir = config_path.parent
            for index, (benchmark, (onnx_path, vnnlib_path, _)) in enumerate(
                zip(benchmarks, rows)
            ):
                instance_config_path = work_dir / f"abcrown_config_instance_{index}.yaml"
                instance_results_path = work_dir / f"abcrown_results_{index}.txt"
                write_config(
                    instance_config_path,
                    work_dir,
                    instance_results_path,
                    output_dim,
                    profile,
                    benchmark.timeout_sec,
                )
                config = ConfigBuilder.from_yaml(str(instance_config_path)).to_dict()
                solve_result = ABCrownSolver(
                    str(onnx_path),
                    constraint=IOConstraints(vnnlib_path=str(vnnlib_path)),
                    config=config,
                    name=f"{benchmark.suite_name}/{benchmark.benchmark_id}",
                ).verify()
                instance_results_path.write_text(
                    f"{index},{solve_result.status},{solve_result.success},"
                    f"{float(solve_result.stats.get('elapsed') or 0.0):.6f}\n",
                    encoding="utf-8",
                )
                runtime = float(solve_result.stats.get("elapsed") or 0.0)
                results[index] = (solve_result.status, runtime)
    except Exception:
        output = captured.getvalue() + traceback.format_exc()
        return 1, output, {}

    return 0, captured.getvalue(), results


def benchmark_status_from_abcrown(status: str | None) -> BenchmarkStatus:
    if status in {"safe", "safe-incomplete", "unsat", "verified"}:
        return "unsat"
    if status in {"unsafe-pgd", "unsafe-bab", "sat", "falsified"}:
        return "sat"
    if status == "timeout":
        return "timeout"
    return "unknown"


def format_expected(result: BenchmarkResult) -> str:
    if result.expected_status is None:
        return ""
    matched = "yes" if result.matched_expected else "no"
    return f"{result.expected_status}:{matched}"


def print_results(results: list[BenchmarkResult], abcrown_statuses: list[str]) -> None:
    print("benchmark_id,status,abcrown_status,expected,runtime_sec,epsilon")
    for result, abcrown_status in zip(results, abcrown_statuses):
        print(
            f"{result.benchmark_id},"
            f"{result.status},"
            f"{abcrown_status},"
            f"{format_expected(result)},"
            f"{result.runtime_sec:.6f},"
            f"{result.epsilon:.17g}"
        )


def prepare_artifacts(
    suite: BenchmarkSuite,
    profile: str,
) -> tuple[Path, Path, list[Benchmark], list[tuple[Path, Path, float]]]:
    work_dir = (ARTIFACT_ROOT / suite.name / profile).resolve()
    model_dir = work_dir / "models"
    spec_dir = work_dir / "vnnlib"
    model_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[Path, Path, float]] = []
    exported_models: dict[str, Path] = {}

    for benchmark in suite.benchmarks:
        validate_benchmark(benchmark)
        onnx_path = export_model_once(benchmark, model_dir, exported_models)
        vnnlib_path = spec_dir / f"{safe_name(benchmark.benchmark_id)}.vnnlib"
        write_vnnlib(vnnlib_path, benchmark)
        rows.append((onnx_path, vnnlib_path, benchmark.timeout_sec))

    instances_path = work_dir / "instances.csv"
    manifest_path = work_dir / "benchmark_manifest.csv"
    config_path = work_dir / "abcrown_config.yaml"
    write_instances_csv(instances_path, rows, work_dir)
    write_manifest(manifest_path, suite.benchmarks, rows, work_dir)
    write_config(
        config_path,
        work_dir,
        work_dir / "abcrown_results.txt",
        suite_output_dim(suite.benchmarks),
        profile,
        max(benchmark.timeout_sec for benchmark in suite.benchmarks),
    )
    return work_dir, config_path, suite.benchmarks, rows


def apply_timeout_override(
    suite: BenchmarkSuite,
    timeout_sec: float | None,
) -> BenchmarkSuite:
    if timeout_sec is None:
        return suite
    if timeout_sec <= 0:
        raise ValueError("--timeout must be positive")
    return BenchmarkSuite(
        name=suite.name,
        benchmarks=[
            replace(benchmark, timeout_sec=timeout_sec)
            for benchmark in suite.benchmarks
        ],
    )


def build_results(
    benchmarks: list[Benchmark],
    abcrown_result_by_index: dict[int, tuple[str, float]],
) -> tuple[list[BenchmarkResult], list[str]]:
    results: list[BenchmarkResult] = []
    abcrown_statuses: list[str] = []

    for index, benchmark in enumerate(benchmarks):
        abcrown_status, runtime_sec = abcrown_result_by_index.get(
            index, ("missing", 0.0)
        )
        status = benchmark_status_from_abcrown(abcrown_status)
        abcrown_statuses.append(abcrown_status)
        input_dim = len(benchmark.input_region.lower_bounds)
        output_dim = len(benchmark.nn1[-1][1])
        num_relu = sum(len(bias) for _, bias in benchmark.nn1[:-1]) + sum(
            len(bias) for _, bias in benchmark.nn2[:-1]
        )
        results.append(
            BenchmarkResult(
                benchmark_id=benchmark.benchmark_id,
                suite_name=benchmark.suite_name,
                status=status,
                runtime_sec=runtime_sec,
                epsilon=benchmark.epsilon,
                input_dim=input_dim,
                output_dim=output_dim,
                num_layers=len(benchmark.nn1),
                num_relu=num_relu,
                num_active_relu=0,
                num_inactive_relu=0,
                num_unstable_relu=0,
                num_vars=0,
                num_binary_vars=0,
                num_constraints=0,
                max_output_diff=None,
                counterexample=None,
                expected_status=benchmark.expected_status,
            )
        )

    return results, abcrown_statuses


def main() -> None:
    args = parse_args()
    if args.list_profiles:
        print("profile")
        for profile in sorted(ABCROWN_PROFILES):
            print(profile)
        return

    suite = apply_timeout_override(load_suite(args.suite), args.timeout)
    work_dir, config_path, benchmarks, rows = prepare_artifacts(suite, args.profile)
    returncode, output, abcrown_result_by_index = run_abcrown(
        config_path, benchmarks, rows, args.profile
    )
    if returncode != 0:
        print(output, end="")

    results, abcrown_statuses = build_results(benchmarks, abcrown_result_by_index)

    print(file=sys.stderr)
    print(f"artifacts: {work_dir}", file=sys.stderr)
    print(f"profile: {args.profile}", file=sys.stderr)
    print_results(results, abcrown_statuses)
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
