"""Knowledge-distillation teacher/student equivalence benchmark suite.

Loads frozen teacher/student ``.nnet`` pairs from ``$RUNTIME_DIR/data/distillation/mnist/<pair>/``
and builds the same 100 ReluDiff MNIST centers used by ``pruning_mnist``.

Property A (v1): ``|z_T(x)_c - z_S(x)_c| <= epsilon`` for the labeled class ``c``.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from benchmarks.common import (
    AbstractPolytope,
    Hyperrectangle,
    Instance,
    InstanceSuite,
    SuiteOptions,
)
from nn_equivalence.nn_types import NeuralNetwork
from nn_equivalence.paths import runtime_path
from nn_equivalence.reludiff_nnet import load_nnet_layers, load_reludiff_mnist_tests

DEFAULT_SUITE_OPTIONS: SuiteOptions = {
    "pairs": "",
    "tiers": "",
    "modes": "three_pixel",
    "epsilon": "0.1",
    "radius": "3.0",
    "timeout": "30",
    # Empty -> resolved lazily from $RUNTIME_DIR in load_suite (see below).
    "data_dir": "",
    "tests_path": "",
}


def _normalized_options(suite_options: SuiteOptions | None) -> SuiteOptions:
    options: SuiteOptions = dict(DEFAULT_SUITE_OPTIONS)
    for key, value in (suite_options or {}).items():
        normalized_key = key.strip().lower().replace("-", "_")
        options[normalized_key] = value

    allowed_options = {
        "pairs",
        "tiers",
        "modes",
        "limit",
        "timeout",
        "epsilon",
        "radius",
        "data_dir",
        "tests_path",
    }
    unknown_options = set(options) - allowed_options
    if unknown_options:
        raise ValueError(
            f"unknown distillation_mnist suite options: {sorted(unknown_options)}"
        )
    return options


def _option_tuple(options: SuiteOptions, name: str) -> tuple[str, ...]:
    value = options.get(name)
    if not value:
        return tuple()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _limit(options: SuiteOptions) -> int | None:
    value = options.get("limit")
    if not value:
        return None
    limit = int(value)
    if limit < 1 or limit > 100:
        raise ValueError("distillation_mnist limit must be between 1 and 100")
    return limit


def _timeout(options: SuiteOptions) -> float:
    return float(options["timeout"])


def _epsilon(options: SuiteOptions) -> float:
    return float(options["epsilon"])


def _radius(options: SuiteOptions) -> float:
    return float(options["radius"])


def _global_region(raw_pixels: list[float], radius: float) -> AbstractPolytope:
    return Hyperrectangle(
        low=[max((pixel - radius) / 255.0, 0.0) for pixel in raw_pixels],
        high=[min((pixel + radius) / 255.0, 1.0) for pixel in raw_pixels],
    )


def _three_pixel_region(raw_pixels: list[float], pixel_ids: list[int]) -> AbstractPolytope:
    lower_bounds = [pixel / 255.0 for pixel in raw_pixels]
    upper_bounds = [pixel / 255.0 for pixel in raw_pixels]
    for pixel_id in pixel_ids[:3]:
        lower_bounds[pixel_id] = 0.0
        upper_bounds[pixel_id] = 1.0
    return Hyperrectangle(low=lower_bounds, high=upper_bounds)


def _load_pair(pair_dir: Path) -> tuple[NeuralNetwork, NeuralNetwork, dict]:
    teacher_path = pair_dir / "teacher.nnet"
    student_path = pair_dir / "student.nnet"
    metadata_path = pair_dir / "metadata.json"
    missing = [
        path
        for path in (teacher_path, student_path, metadata_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "missing distillation pair files: "
            + ", ".join(str(path) for path in missing)
            + ". Train with `python -m training.distill_mnist --pair-id "
            f"{pair_dir.name}`."
        )

    teacher = load_nnet_layers(teacher_path)
    student = load_nnet_layers(student_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return teacher, student, metadata


def _pair_metadata_fields(metadata: dict) -> dict[str, str | int | float]:
    fields: dict[str, str | int | float] = {}
    for key in (
        "tier",
        "temperature",
        "alpha",
        "seed",
        "teacher_accuracy",
        "student_accuracy",
        "disagreement_rate",
        "teacher_param_count",
        "student_param_count",
    ):
        if key in metadata and isinstance(metadata[key], (str, int, float)):
            fields[key] = metadata[key]

    teacher_arch = metadata.get("teacher_arch")
    student_arch = metadata.get("student_arch")
    if isinstance(teacher_arch, list):
        fields["teacher_arch"] = "-".join(str(size) for size in teacher_arch)
    if isinstance(student_arch, list):
        fields["student_arch"] = "-".join(str(size) for size in student_arch)

    same_architecture = metadata.get("same_architecture")
    if isinstance(same_architecture, bool):
        fields["same_architecture"] = int(same_architecture)
    elif isinstance(teacher_arch, list) and isinstance(student_arch, list):
        fields["same_architecture"] = int(teacher_arch == student_arch)

    compatible = metadata.get("compatible_with_reludiff_neurodiff")
    if isinstance(compatible, bool):
        fields["compatible_with_reludiff_neurodiff"] = int(compatible)
    elif "same_architecture" in fields:
        fields["compatible_with_reludiff_neurodiff"] = fields["same_architecture"]

    return fields


def _infer_tier(metadata: dict, teacher: NeuralNetwork, student: NeuralNetwork) -> str:
    tier = metadata.get("tier")
    if tier in {"A", "B"}:
        return tier
    from nn_equivalence.reludiff_nnet import network_architecture

    return (
        "A"
        if network_architecture(teacher) == network_architecture(student)
        else "B"
    )


def _resolve_pair_ids(options: SuiteOptions, data_dir: Path) -> tuple[str, ...]:
    pair_ids = list(_option_tuple(options, "pairs"))
    tier_filter = set(_option_tuple(options, "tiers"))
    unknown_tiers = tier_filter - {"A", "B"}
    if unknown_tiers:
        raise ValueError(f"unknown distillation tiers: {sorted(unknown_tiers)}")

    if not pair_ids and not tier_filter:
        raise ValueError("distillation_mnist suite requires pairs= and/or tiers=")

    if not pair_ids:
        # Discover pair directories under data_dir.
        if not data_dir.is_dir():
            raise FileNotFoundError(f"distillation data directory not found: {data_dir}")
        pair_ids = sorted(
            path.name
            for path in data_dir.iterdir()
            if path.is_dir() and (path / "metadata.json").exists()
        )

    if not tier_filter:
        return tuple(pair_ids)

    selected: list[str] = []
    for pair_id in pair_ids:
        pair_dir = data_dir / pair_id
        metadata_path = pair_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"missing metadata for pair {pair_id}: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        tier = metadata.get("tier")
        if tier not in {"A", "B"}:
            teacher_arch = metadata.get("teacher_arch")
            student_arch = metadata.get("student_arch")
            tier = (
                "A"
                if isinstance(teacher_arch, list)
                and isinstance(student_arch, list)
                and teacher_arch == student_arch
                else "B"
            )
        if tier in tier_filter:
            selected.append(pair_id)
    return tuple(selected)


def load_suite(suite_options: SuiteOptions | None = None) -> InstanceSuite:
    suite_name = "distillation_mnist"
    options = _normalized_options(suite_options)
    print(f"{suite_name} suite options: {options}", file=sys.stderr)

    data_dir = Path(options["data_dir"]) if options["data_dir"] else runtime_path(
        "data/distillation/mnist"
    )
    pair_ids = _resolve_pair_ids(options, data_dir)
    if not pair_ids:
        raise ValueError(
            "distillation_mnist suite selected no pairs (check pairs=/tiers=)"
        )

    modes = _option_tuple(options, "modes")
    unknown_modes = set(modes) - {"global", "three_pixel"}
    if unknown_modes:
        raise ValueError(f"unknown distillation_mnist modes: {sorted(unknown_modes)}")
    if not modes:
        raise ValueError("distillation_mnist suite requires at least one mode")

    tests_path = Path(options["tests_path"]) if options["tests_path"] else runtime_path(
        "data/reludiff_mnist/mnist_tests.h"
    )
    if not tests_path.exists():
        raise FileNotFoundError(
            f"missing ReluDiff MNIST tests at {tests_path}; needed for the "
            "shared 100-image / 3-pixel fixtures"
        )

    mnist_tests, labels, random_pixels = load_reludiff_mnist_tests(tests_path)
    limit = _limit(options)
    sample_indices = range(100 if limit is None else limit)
    timeout_sec = _timeout(options)
    epsilon = _epsilon(options)
    radius = _radius(options)

    instances: list[Instance] = []
    for pair_id in pair_ids:
        pair_dir = data_dir / pair_id
        teacher, student, metadata = _load_pair(pair_dir)
        tier = _infer_tier(metadata, teacher, student)
        metadata = {**metadata, "tier": tier}
        pair_fields = _pair_metadata_fields(metadata)
        teacher_nnet = str(pair_dir / "teacher.nnet")

        for mode in modes:
            for sample_index in sample_indices:
                raw_pixels = mnist_tests[sample_index]
                if mode == "global":
                    input_region = _global_region(raw_pixels, radius)
                    radius_metadata: str | float = radius
                else:
                    input_region = _three_pixel_region(
                        raw_pixels, random_pixels[sample_index]
                    )
                    radius_metadata = ",".join(
                        str(pixel_id) for pixel_id in random_pixels[sample_index][:3]
                    )

                instances.append(
                    Instance(
                        instance_id=f"{pair_id}_{mode}_{sample_index}",
                        suite_name=suite_name,
                        nn1=teacher,
                        nn2=student,
                        input_region=input_region,
                        epsilon=epsilon,
                        output_index=labels[sample_index],
                        expected_status=None,
                        timeout_sec=timeout_sec,
                        metadata={
                            "pair_id": pair_id,
                            "tier": tier,
                            "sample_index": sample_index,
                            "correct_class": labels[sample_index],
                            "output_index": labels[sample_index],
                            "input_mode": mode,
                            "radius": radius_metadata,
                            "property": "output_equivalence_logits",
                            # Used by run_diffverifier for Tier A same-arch pairs.
                            "nnet1_path": teacher_nnet,
                            **pair_fields,
                        },
                    )
                )

    return InstanceSuite(name=suite_name, instances=instances)
