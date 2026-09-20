"""Run an NN-equivalence suite through the ReluDiff/NeuroDiff C verifiers.

Both tools are the ``delta_network_test`` executable built from the NeuroDiff
ASE-2020 artifact (which compiles pure ReluDiff or full NeuroDiff depending on
its build flags). They share one CLI:

    ./delta_network_test PROPERTY NNET1 NNET2 EPSILON [OPTIONS]

For the ReluDiff/NeuroDiff MNIST benchmark, ``PROPERTY = 400 + sample_index``
selects the built-in test image, its correct class, and (with ``-t``) the three
perturbed pixels. This runner therefore only serializes the *second* network of
each pair to a temporary ``.nnet`` (the first is passed straight from disk), then
maps our suite's ``global``/``three_pixel`` modes onto the ASE tool's ``-p``/``-x``
flags. It parses
the tool's stderr (``adv found`` / ``No adv!``) and emits the same CSV schema as
``run_crown`` so the four verifiers produce comparable rows.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.common import (
    Instance,
    InstanceResult,
    InstanceStatus,
    InstanceSuite,
    SolveStats,
    SuiteOptions,
    format_expected,
    parse_suite_options,
    print_progress,
    validate_instance,
)
from nn_equivalence.paths import runtime_path
from nn_equivalence.reludiff_nnet import network_architecture, write_nnet_layers
MNIST_PROPERTY_BASE = 400
# The mnist_reludiff three_pixel mode opens exactly the first three random_pixels
# (see _three_pixel_region), matching the tool's `-x 3` pixel experiment. This
# only holds while runtime/data/reludiff_mnist/mnist_tests.h carries the same
# random_pixels table as the compiled ASE-2020 artifact; the ReluDiff ICSE
# table differs and silently makes the verifier families check different
# regions.
THREE_PIXEL_COUNT = 3
SUPPORTED_SUITES = frozenset({"mnist_reludiff", "distillation"})
DEFAULT_SUITE = "mnist_reludiff"


def build_command(
    binary: Path | str,
    property_id: int,
    nnet1_path: Path | str,
    nnet2_path: Path | str,
    epsilon: float,
    mode: str,
    perturb: float,
) -> list[str]:
    """Assemble the delta_network_test invocation for one instance."""
    command = [
        str(binary),
        str(property_id),
        str(nnet1_path),
        str(nnet2_path),
        f"{epsilon:.17g}",
    ]
    if mode == "three_pixel":
        # -x N runs the pixel experiment: all pixels fixed, the first N built-in
        # random_pixels opened to [0, 1] (perturb is ignored), matching the
        # suite's _three_pixel_region.
        command += ["-x", str(THREE_PIXEL_COUNT)]
    elif mode == "global":
        # Global exp (mnistPixExp defaults to 0): each pixel is +/- perturb/255.
        command += ["-p", f"{perturb:.17g}"]
    else:
        raise ValueError(f"unsupported input mode for diff verifier: {mode!r}")
    return command


def classify_output(
    stdout: str,
    stderr: str,
    timed_out: bool,
) -> tuple[InstanceStatus, str]:
    """Map the tool's console output to (harness status, raw tool status).

    ``No adv!`` means the property was verified (no counterexample within
    epsilon), i.e. the equivalence property holds -> ``unsat``. ``adv found``
    means a concrete counterexample exists -> ``sat``.
    """
    if timed_out:
        return "timeout", "timeout"
    combined = f"{stdout}\n{stderr}"
    if "adv found" in combined:
        return "sat", "adv_found"
    if "No adv!" in combined:
        return "unsat", "no_adv"
    return "unknown", "no_conclusion"


def parse_tool_stats(stderr: str) -> dict[str, float | int]:
    """Extract the tool's self-reported time and split count from stderr."""
    stats: dict[str, float | int] = {}
    time_match = re.search(r"time:\s*([0-9]+(?:\.[0-9]+)?)", stderr)
    if time_match is not None:
        stats["tool_time_sec"] = float(time_match.group(1))
    splits_match = re.search(r"numSplits:\s*([0-9]+)", stderr)
    if splits_match is not None:
        stats["num_splits"] = int(splits_match.group(1))
    return stats


def property_id_for(instance: Instance) -> int:
    sample_index = instance.metadata.get("sample_index")
    if not isinstance(sample_index, int):
        raise ValueError(
            f"instance {instance.instance_id} is missing an integer 'sample_index' "
            "metadata entry required to map onto a diff-verifier property id"
        )
    if not 0 <= sample_index <= 99:
        raise ValueError(
            f"sample_index {sample_index} is outside the MNIST property range 0-99"
        )
    return MNIST_PROPERTY_BASE + sample_index


def load_suite(name: str, suite_options: SuiteOptions) -> InstanceSuite:
    module = importlib.import_module(f"benchmarks.{name}")
    return module.load_suite(suite_options)


def resolve_binary(args: argparse.Namespace) -> Path:
    candidate = args.binary or os.environ.get("DIFFVERIFIER_BINARY")
    if not candidate:
        raise SystemExit(
            "no diff-verifier binary given. Compile delta_network_test from the "
            "NeuroDiff ASE-2020 artifact (pure ReluDiff or full NeuroDiff), then "
            "pass --binary PATH or set DIFFVERIFIER_BINARY."
        )
    binary = Path(candidate).expanduser()
    if not binary.exists():
        raise SystemExit(f"diff-verifier binary not found: {binary}")
    if not os.access(binary, os.X_OK):
        raise SystemExit(f"diff-verifier binary is not executable: {binary}")
    return binary


def nnet2_path_for(
    instance: Instance,
    data_dir: Path,
    work_dir: Path,
    cache: dict[str, Path],
) -> tuple[Path, Path]:
    """Return (nnet1_source_path, nnet2_serialized_path) for an instance.

    ReluDiff/NeuroDiff require identical architectures. For ``mnist_reludiff``
    the second network is a same-arch transform of the base ``.nnet``. For
    Tier A ``distillation`` pairs the teacher ``.nnet`` is the header source and
    the student weights are rewritten next to it under ``runtime/data/``.
    """
    if instance.suite_name == "distillation":
        return _distillation_nnet_paths(instance, cache)

    network = instance.metadata.get("network")
    if not isinstance(network, str):
        raise ValueError(
            f"instance {instance.instance_id} is missing a 'network' metadata entry"
        )
    nnet1_path = data_dir / f"{network}.nnet"
    if not nnet1_path.exists():
        raise FileNotFoundError(
            f"base network .nnet not found: {nnet1_path}. Run "
            "`python3 scripts/download_mnist_reludiff_nnets.py` first."
        )
    if network not in cache:
        nnet2_path = work_dir / f"{network}__nn2.nnet"
        write_nnet_layers(nnet1_path, instance.nn2, nnet2_path)
        cache[network] = nnet2_path
    return nnet1_path, cache[network]


def _distillation_nnet_paths(
    instance: Instance,
    cache: dict[str, Path],
) -> tuple[Path, Path]:
    pair_id = instance.metadata.get("pair_id")
    if not isinstance(pair_id, str):
        raise ValueError(
            f"instance {instance.instance_id} is missing a 'pair_id' metadata entry"
        )

    tier = instance.metadata.get("tier")
    same_architecture = instance.metadata.get("same_architecture")
    is_same_arch = (
        same_architecture in {1, True}
        or network_architecture(instance.nn1) == network_architecture(instance.nn2)
    )
    if tier == "B" or not is_same_arch:
        raise ValueError(
            f"ReluDiff/NeuroDiff cannot run Tier B / diff-arch distillation pair "
            f"{pair_id!r} (architectures differ). Use Tier A same-arch pairs "
            f"(e.g. pairs=kd_a1) or Relational-MILP / ab-CROWN for Tier B."
        )

    nnet1_meta = instance.metadata.get("nnet1_path")
    if isinstance(nnet1_meta, str):
        nnet1_path = Path(nnet1_meta)
    else:
        nnet1_path = runtime_path("data/distillation/mnist", pair_id, "teacher.nnet")
    if not nnet1_path.exists():
        raise FileNotFoundError(
            f"distillation teacher .nnet not found: {nnet1_path}. Train with "
            f"`python -m training.distill_mnist --pair-id {pair_id}`."
        )

    if pair_id not in cache:
        # Write the diffverifier-ready student (teacher header + student
        # weights) next to the teacher under runtime/data/, not into runtime/artifacts/.
        nnet2_path = nnet1_path.parent / f"{pair_id}__student.nnet"
        write_nnet_layers(nnet1_path, instance.nn2, nnet2_path)
        cache[pair_id] = nnet2_path
    return nnet1_path, cache[pair_id]


def run_instance(
    instance: Instance,
    binary: Path,
    data_dir: Path,
    work_dir: Path,
    cache: dict[str, Path],
    verbose: bool,
) -> InstanceResult:
    nnet1_path, nnet2_path = nnet2_path_for(instance, data_dir, work_dir, cache)
    mode = str(instance.metadata.get("input_mode", "global"))
    # For global mode metadata["perturb"] is the numeric strength; for three_pixel
    # it holds the perturbed pixel ids (unused by the tool's -x path).
    perturb = float(instance.metadata["perturb"]) if mode == "global" else 0.0
    command = build_command(
        binary,
        property_id_for(instance),
        nnet1_path,
        nnet2_path,
        instance.epsilon,
        mode,
        perturb,
    )

    start_time = time.perf_counter()
    timed_out = False
    stdout = ""
    stderr = ""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=instance.timeout_sec,
            check=False,
        )
        stdout, stderr = completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as expired:
        timed_out = True
        stdout = expired.stdout or ""
        stderr = expired.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
    runtime_sec = time.perf_counter() - start_time

    if verbose:
        print(f"$ {' '.join(command)}", file=sys.stderr)
        print(stderr, file=sys.stderr)

    status, tool_status = classify_output(stdout, stderr, timed_out)
    tool_stats = parse_tool_stats(stderr)
    details: list[tuple[str, str | int | float]] = [("diff_status", tool_status)]
    for key, value in tool_stats.items():
        details.append((key, value))

    return InstanceResult(
        instance_id=instance.instance_id,
        suite_name=instance.suite_name,
        status=status,
        runtime_sec=runtime_sec,
        epsilon=instance.epsilon,
        expected_status=instance.expected_status,
        stats=[SolveStats(name="diff", details=details)],
    )


def _detail_value(result: InstanceResult, key: str) -> str:
    for solve_stats in result.stats:
        for detail_key, value in solve_stats.details:
            if detail_key == key:
                return str(value)
    return ""


def results_csv(results: list[InstanceResult]) -> str:
    lines = [
        "instance_id,status,diff_status,expected,runtime_sec,epsilon,"
        "num_splits,tool_time_sec"
    ]
    for result in results:
        lines.append(
            f"{result.instance_id},"
            f"{result.status},"
            f"{_detail_value(result, 'diff_status')},"
            f"{format_expected(result)},"
            f"{result.runtime_sec:.6f},"
            f"{result.epsilon:.17g},"
            f"{_detail_value(result, 'num_splits')},"
            f"{_detail_value(result, 'tool_time_sec')}"
        )
    return "\n".join(lines) + "\n"


def write_results_csv(path: Path, results: list[InstanceResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results_csv(results), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an NN-equivalence suite with ReluDiff/NeuroDiff."
    )
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument(
        "--suite-options",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Suite-specific option; repeatable. Values may contain commas.",
    )
    parser.add_argument(
        "--tool",
        default="neurodiff",
        choices=["reludiff", "neurodiff"],
        help="Label recorded for the run; the actual algorithm is fixed at "
        "compile time in the binary passed via --binary.",
    )
    parser.add_argument(
        "--binary",
        default=None,
        help="Path to the compiled delta_network_test executable. Falls back to "
        "the DIFFVERIFIER_BINARY environment variable.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Base network directory for mnist_reludiff "
        "(default: $RUNTIME_DIR/data/reludiff_mnist). Ignored for distillation, which uses "
        "each instance's teacher .nnet path.",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.suite not in SUPPORTED_SUITES:
        raise SystemExit(
            f"run_diffverifier supports suites {sorted(SUPPORTED_SUITES)}; "
            f"got {args.suite!r}. Distillation Tier B (diff-arch) is unsupported "
            "by ReluDiff/NeuroDiff — use Tier A same-arch pairs."
        )
    binary = resolve_binary(args)
    data_dir = args.data_dir or runtime_path("data/reludiff_mnist")
    try:
        suite_options = parse_suite_options(args.suite_options)
        suite = load_suite(args.suite, suite_options)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(error)
        raise SystemExit(2) from error

    for instance in suite.instances:
        validate_instance(instance)

    # Only mnist_reludiff serializes its second network into a work dir; the
    # distillation suite writes next to the teacher under runtime/data/ instead.
    work_dir = (runtime_path("artifacts/diffverifier") / suite.name / args.tool).resolve()
    if suite.name != "distillation":
        work_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[str, Path] = {}
    results: list[InstanceResult] = []
    total = len(suite.instances)
    for index, instance in enumerate(suite.instances, start=1):
        result = run_instance(
            instance, binary, data_dir, work_dir, cache, args.verbose
        )
        results.append(result)
        print_progress(
            index,
            total,
            result,
            extra_fields={"diff_status": _detail_value(result, "diff_status")},
        )

    print(results_csv(results), end="")
    if args.csv is not None:
        write_results_csv(args.csv, results)


if __name__ == "__main__":
    main()
