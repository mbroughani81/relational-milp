from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import hashlib
import importlib
import io
import json
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from benchmarks.common import (
    Instance,
    InstanceResult,
    InstanceStatus,
    InstanceSuite,
    SuiteOptions,
    parse_suite_options,
    validate_instance,
)
from nn_equivalence.nn_types import NeuralNetwork
ARTIFACT_ROOT = Path("artifacts/abcrown_instances")
BATCH_SIZE = 512
ConfigValue = str | int | float | bool | dict[str, "ConfigValue"]
ConfigDict = dict[str, ConfigValue]


ABCROWN_COMMON_CONFIG: ConfigDict = {
    "general": {
        "complete_verifier": "bab",
        "device": "cpu",
        "seed": 100,
        "deterministic": True,
    },
    "solver": {
        "batch_size": BATCH_SIZE,
        "alpha-crown": {
            "iteration": 100,
            "lr_alpha": 0.1,
        },
        "beta-crown": {
            "iteration": 50,
            "lr_alpha": 0.01,
            "lr_beta": 0.05,
            "lr_decay": 0.98,
        },
    },
    "bab": {
        "timeout": 600,
    },
    "attack": {
        "pgd_order": "before",
        "pgd_steps": 100,
        "pgd_restarts": 30,
    },
}


ABCROWN_PROFILES: dict[str, ConfigDict] = {
    # Standard beta-CROWN BaB: ReLU splitting with the balanced kFSB heuristic.
    "relu-kfsb": {
        "solver": {
            "bound_prop_method": "alpha-crown",
        },
        "bab": {
            "branching": {
                "method": "kfsb",
                "candidates": 3,
                "reduceop": "min",
            },
        },
    },
    # Strong-branching beta-CROWN: spend more work per ReLU split decision.
    "relu-fsb": {
        "solver": {
            "bound_prop_method": "alpha-crown",
        },
        "bab": {
            "branching": {
                "method": "fsb",
                "candidates": 10,
                "reduceop": "min",
            },
        },
    },
    # Input-space BaB: split the shared input region instead of ReLU phases.
    "input-split": {
        "solver": {
            "bound_prop_method": "alpha-crown",
            "init_bound_prop_method": "alpha-crown",
        },
        "bab": {
            "branching": {
                "method": "sb",
                "input_split": {
                    "enable": True,
                    "split_partitions": 2,
                },
            },
        },
    },
    # MIP-refined beta-CROWN BaB: tighten intermediate bounds before BaB.
    "mip-refined-bab": {
        "general": {
            "complete_verifier": "bab-refine",
        },
        "solver": {
            "mip": {
                "mip_solver": "gurobi",
                "parallel_solvers": 8,
                "solver_threads": 1,
                "refine_neuron_timeout": 15,
                "refine_neuron_time_percentage": 0.5,
            },
        },
        "bab": {
            "branching": {
                "method": "kfsb",
                "candidates": 3,
            },
        },
    },
    # Direct complete MIP: delegate the exact piecewise-linear search to Gurobi.
    "direct-mip": {
        "general": {
            "complete_verifier": "mip",
        },
        "solver": {
            "mip": {
                "mip_solver": "gurobi",
                "formulation": "mip",
                "parallel_solvers": 1,
                "solver_threads": 8,
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


class SafetyMarginNetwork(nn.Module):
    def __init__(self, nn1: NeuralNetwork, nn2: NeuralNetwork, epsilon: float) -> None:
        super().__init__()
        self.diff = DifferenceNetwork(nn1, nn2)
        output_dim = len(nn1[-1][1])
        self.margin = nn.Linear(output_dim, 2 * output_dim)
        with torch.no_grad():
            self.margin.weight.zero_()
            self.margin.bias.fill_(epsilon)
            for output_index in range(output_dim):
                self.margin.weight[output_index, output_index] = -1.0
                self.margin.weight[output_dim + output_index, output_index] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.margin(self.diff(x))


def load_suite(name: str, suite_options: SuiteOptions) -> InstanceSuite:
    module = importlib.import_module(f"benchmarks.{name}")
    return module.load_suite(suite_options)


def apply_timeout_override(
    suite: InstanceSuite,
    timeout_sec: float | None,
) -> InstanceSuite:
    if timeout_sec is None:
        return suite
    if timeout_sec <= 0:
        raise ValueError("--timeout must be positive")
    return replace(
        suite,
        instances=[
            replace(instance, timeout_sec=timeout_sec)
            for instance in suite.instances
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an NN equivalence instance suite with alpha-beta-CROWN."
    )
    parser.add_argument("--suite", default="sample")
    parser.add_argument(
        "--suite-options",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Suite-specific option. Repeat for multiple options; values may "
            "contain commas, e.g. --suite-options modes=global,three_pixel."
        ),
    )
    parser.add_argument(
        "--profile",
        default="relu-kfsb",
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
            "Override every instance.timeout_sec before running CROWN. "
            "Without this, each suite instance controls its own bab.timeout."
        ),
    )
    return parser.parse_args()


def model_key(instance: Instance) -> str:
    payload = json.dumps([instance.nn1, instance.nn2, instance.epsilon], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def model_for_instance(
    instance: Instance,
    models: dict[str, SafetyMarginNetwork],
) -> SafetyMarginNetwork:
    key = model_key(instance)
    if key not in models:
        models[key] = SafetyMarginNetwork(
            instance.nn1,
            instance.nn2,
            instance.epsilon,
        ).eval()
    return models[key]


def write_manifest(path: Path, instances: list[Instance]) -> None:
    lines = ["index,instance_id,epsilon,expected_status,timeout\n"]
    for index, instance in enumerate(instances):
        expected = instance.expected_status or ""
        lines.append(
            f"{index},{instance.instance_id},{instance.epsilon:.17g},"
            f"{expected},{instance.timeout_sec:.17g}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def suite_margin_output_dim(instances: list[Instance]) -> int:
    output_dims = {len(instance.nn1[-1][1]) for instance in instances}
    if len(output_dims) != 1:
        raise ValueError(
            "alpha-beta-CROWN config requires one data.num_outputs value; "
            f"got output dimensions {sorted(output_dims)}"
        )
    return 2 * output_dims.pop()


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
    results_path: Path,
    output_dim: int,
    profile: str,
    timeout_sec: float,
) -> None:
    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive")

    base_config: ConfigDict = {
        "general": {"results_file": str(results_path)},
        "data": {
            "num_outputs": output_dim,
        },
        "solver": {
            "batch_size": BATCH_SIZE,
        },
        "bab": {"timeout": timeout_sec},
    }
    config = merge_config(base_config, ABCROWN_COMMON_CONFIG)
    config = merge_config(config, ABCROWN_PROFILES[profile])
    config.setdefault("bab", {})["timeout"] = timeout_sec
    print("config => ")
    print(config)
    path.write_text("\n".join(config_to_yaml(config)) + "\n", encoding="utf-8")


def make_constraints(
    instance: Instance,
    input_vars_fn,
    output_vars_fn,
    constraints_cls,
    *,
    include_output_constraint: bool,
):
    input_dim = len(instance.input_region.lower_bounds)
    output_dim = 2 * len(instance.nn1[-1][1])
    x = input_vars_fn(input_dim)
    y = output_vars_fn(output_dim)
    lower = torch.tensor(instance.input_region.lower_bounds, dtype=torch.float32)
    upper = torch.tensor(instance.input_region.upper_bounds, dtype=torch.float32)

    input_constraint = (x >= lower) & (x <= upper)
    if not include_output_constraint:
        return (
            x,
            y,
            constraints_cls(
                input_vars=x,
                input_constraint=input_constraint,
            ),
        )

    output_constraint = None
    for output_index in range(output_dim):
        positive_margin = y[output_index] > 0
        output_constraint = (
            positive_margin
            if output_constraint is None
            else output_constraint & positive_margin
        )

    return (
        x,
        y,
        constraints_cls(
            input_vars=x,
            output_vars=y,
            input_constraint=input_constraint,
            output_constraint=output_constraint,
        ),
    )


def bounds_prove_equivalent(bounds_result) -> bool:
    if not bounds_result.success:
        return False
    lower = bounds_result.lower.detach().cpu()
    return bool(torch.all(lower > 0))


def run_abcrown(
    config_path: Path,
    instances: list[Instance],
    profile: str,
) -> tuple[int, str, dict[int, tuple[str, float]]]:
    captured = io.StringIO()
    results: dict[int, tuple[str, float]] = {}
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            from abcrown import (
                ABCrownSolver,
                ConfigBuilder,
                IOConstraints,
                input_vars,
                output_vars,
            )

            output_dim = suite_margin_output_dim(instances)
            work_dir = config_path.parent
            models: dict[str, SafetyMarginNetwork] = {}
            for index, instance in enumerate(instances):
                instance_config_path = work_dir / f"abcrown_config_instance_{index}.yaml"
                instance_results_path = work_dir / f"abcrown_results_{index}.txt"
                write_config(
                    instance_config_path,
                    instance_results_path,
                    output_dim,
                    profile,
                    instance.timeout_sec,
                )
                config = ConfigBuilder.from_yaml(str(instance_config_path)).to_dict()
                start_time = time.perf_counter()
                try:
                    x, y, constraints = make_constraints(
                        instance,
                        input_vars,
                        output_vars,
                        IOConstraints,
                        include_output_constraint=True,
                    )
                    model = model_for_instance(instance, models)
                    solver = ABCrownSolver(
                        model,
                        x,
                        y,
                        config=config,
                        name=f"{instance.suite_name}/{instance.instance_id}",
                    )
                    solve_result = solver.verify(constraints=constraints)
                    if solve_result.status == "unknown":
                        x, y, input_constraints = make_constraints(
                            instance,
                            input_vars,
                            output_vars,
                            IOConstraints,
                            include_output_constraint=False,
                        )
                        bounds_result = ABCrownSolver(
                            model,
                            x,
                            y,
                            config=config,
                            name=f"{instance.suite_name}/{instance.instance_id}/bounds",
                        ).compute_bounds(
                            constraints=input_constraints,
                            objective=[y[i] for i in range(output_dim)],
                        )
                        if bounds_prove_equivalent(bounds_result):
                            solve_result.status = "verified-by-bounds"
                            solve_result.success = True
                            solve_result.stats["elapsed"] = (
                                time.perf_counter() - start_time
                            )
                except Exception:
                    runtime = time.perf_counter() - start_time
                    traceback_path = instance_results_path.with_suffix(".traceback.txt")
                    traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
                    instance_results_path.write_text(
                        f"{index},error,False,{runtime:.6f}\n",
                        encoding="utf-8",
                    )
                    results[index] = ("error", runtime)
                    print(traceback.format_exc(), end="")
                    continue

                runtime = float(solve_result.stats.get("elapsed") or 0.0)
                instance_results_path.write_text(
                    f"{index},{solve_result.status},{solve_result.success},"
                    f"{runtime:.6f}\n",
                    encoding="utf-8",
                )
                results[index] = (solve_result.status, runtime)
    except Exception:
        output = captured.getvalue() + traceback.format_exc()
        return 1, output, {}

    return 0, captured.getvalue(), results


def instance_status_from_abcrown(status: str | None) -> InstanceStatus:
    if status in {
        "safe",
        "safe-incomplete",
        "unsat",
        "verified",
        "verified-by-bounds",
    }:
        return "unsat"
    if status in {"unsafe-pgd", "unsafe-bab", "sat", "falsified"}:
        return "sat"
    if status == "timeout":
        return "timeout"
    return "unknown"


def format_expected(result: InstanceResult) -> str:
    if result.expected_status is None:
        return ""
    matched = "yes" if result.matched_expected else "no"
    return f"{result.expected_status}:{matched}"


def print_results(results: list[InstanceResult], abcrown_statuses: list[str]) -> None:
    print("instance_id,status,abcrown_status,expected,runtime_sec,epsilon")
    for result, abcrown_status in zip(results, abcrown_statuses):
        print(
            f"{result.instance_id},"
            f"{result.status},"
            f"{abcrown_status},"
            f"{format_expected(result)},"
            f"{result.runtime_sec:.6f},"
            f"{result.epsilon:.17g}"
        )


def prepare_artifacts(
    suite: InstanceSuite,
    profile: str,
) -> tuple[Path, list[Instance]]:
    work_dir = (ARTIFACT_ROOT / suite.name / profile).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    for instance in suite.instances:
        validate_instance(instance)

    manifest_path = work_dir / "instance_manifest.csv"
    config_path = work_dir / "abcrown_config.yaml"
    write_manifest(manifest_path, suite.instances)
    write_config(
        config_path,
        work_dir / "abcrown_results.txt",
        suite_margin_output_dim(suite.instances),
        profile,
        max(instance.timeout_sec for instance in suite.instances),
    )
    return config_path, suite.instances


def build_results(
    instances: list[Instance],
    abcrown_result_by_index: dict[int, tuple[str, float]],
) -> tuple[list[InstanceResult], list[str]]:
    results: list[InstanceResult] = []
    abcrown_statuses: list[str] = []

    for index, instance in enumerate(instances):
        abcrown_status, runtime_sec = abcrown_result_by_index.get(
            index, ("missing", 0.0)
        )
        status = instance_status_from_abcrown(abcrown_status)
        abcrown_statuses.append(abcrown_status)
        results.append(
            InstanceResult(
                instance_id=instance.instance_id,
                suite_name=instance.suite_name,
                status=status,
                runtime_sec=runtime_sec,
                epsilon=instance.epsilon,
                expected_status=instance.expected_status,
            )
        )

    return results, abcrown_statuses


def main() -> None:
    args = parse_args()
    if args.list_profiles:
        for profile in sorted(ABCROWN_PROFILES):
            print(profile)
        return

    suite = apply_timeout_override(
        load_suite(args.suite, parse_suite_options(args.suite_options)),
        args.timeout,
    )
    config_path, instances = prepare_artifacts(suite, args.profile)
    returncode, output, abcrown_result_by_index = run_abcrown(
        config_path, instances, args.profile
    )
    if output:
        print(output, end="")

    results, abcrown_statuses = build_results(instances, abcrown_result_by_index)
    print_results(results, abcrown_statuses)
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
