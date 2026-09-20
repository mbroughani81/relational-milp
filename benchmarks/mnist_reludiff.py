from __future__ import annotations

from pathlib import Path
import sys

from benchmarks.common import (
    AbstractPolytope,
    Hyperrectangle,
    Instance,
    InstanceSuite,
    SuiteOptions,
)
from nn_equivalence.reludiff_nnet import (
    MNIST_RELUDIFF_NETWORKS,
    load_nnet_layers,
    load_reludiff_mnist_tests,
    prune_network_unstructured,
    quantize_network_float16,
    validate_mnist_reludiff_network,
)
from nn_equivalence.nn_types import NeuralNetwork
from nn_equivalence.paths import runtime_path

DEFAULT_SUITE_OPTIONS: SuiteOptions = {
    "epsilon": "1.0",
    "perturb": "3.0",
    "timeout": "5",
    "perturbation": "quantize",
    "sparsity": "0.5",
}


def _normalized_options(suite_options: SuiteOptions | None) -> SuiteOptions:
    options: SuiteOptions = dict(DEFAULT_SUITE_OPTIONS)
    for key, value in (suite_options or {}).items():
        normalized_key = key.strip().lower().replace("-", "_")
        options[normalized_key] = value

    allowed_options = {
        "networks",
        "modes",
        "limit",
        "timeout",
        "epsilon",
        "perturb",
        "perturbation",
        "sparsity",
    }
    unknown_options = set(options) - allowed_options
    if unknown_options:
        raise ValueError(
            f"unknown mnist_reludiff suite options: {sorted(unknown_options)}"
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
        raise ValueError("mnist_reludiff limit must be between 1 and 100")
    return limit


def _timeout(options: SuiteOptions) -> float:
    return float(options["timeout"])


def _epsilon(options: SuiteOptions) -> float:
    return float(options["epsilon"])


def _perturb(options: SuiteOptions) -> float:
    return float(options["perturb"])


def _perturbation(options: SuiteOptions) -> str:
    value = options["perturbation"].strip().lower()
    if value not in {"quantize", "prune"}:
        raise ValueError(
            "mnist_reludiff perturbation must be 'quantize' or 'prune', "
            f"got {options['perturbation']!r}"
        )
    return value


def _sparsity(options: SuiteOptions) -> float:
    sparsity = float(options["sparsity"])
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("mnist_reludiff sparsity must be in [0.0, 1.0)")
    return sparsity


def _validate_network_names(network_names: tuple[str, ...]) -> None:
    for network_name in network_names:
        if network_name not in MNIST_RELUDIFF_NETWORKS:
            raise ValueError(f"unknown ReluDiff MNIST network: {network_name}")


def _global_region(raw_pixels: list[float], perturb: float) -> AbstractPolytope:
    return Hyperrectangle(
        low=[max((pixel - perturb) / 255.0, 0.0) for pixel in raw_pixels],
        high=[min((pixel + perturb) / 255.0, 1.0) for pixel in raw_pixels],
    )


def _three_pixel_region(raw_pixels: list[float], pixel_ids: list[int]) -> AbstractPolytope:
    lower_bounds = [pixel / 255.0 for pixel in raw_pixels]
    upper_bounds = [pixel / 255.0 for pixel in raw_pixels]
    for pixel_id in pixel_ids[:3]:
        lower_bounds[pixel_id] = 0.0
        upper_bounds[pixel_id] = 1.0
    return Hyperrectangle(low=lower_bounds, high=upper_bounds)


def _make_second_network(
    original: NeuralNetwork,
    perturbation: str,
    sparsity: float,
) -> NeuralNetwork:
    if perturbation == "prune":
        return prune_network_unstructured(original, sparsity)
    return quantize_network_float16(original)


def _load_network_pairs(
    data_dir: Path,
    network_names: tuple[str, ...],
    perturbation: str,
    sparsity: float,
) -> dict[str, tuple[NeuralNetwork, NeuralNetwork]]:
    pairs: dict[str, tuple[NeuralNetwork, NeuralNetwork]] = {}
    for network_name in network_names:
        network_path = data_dir / f"{network_name}.nnet"
        original = load_nnet_layers(network_path)
        validate_mnist_reludiff_network(
            network_name,
            original,
            source_path=network_path,
        )
        second = _make_second_network(original, perturbation, sparsity)
        pairs[network_name] = (original, second)
    return pairs


def _require_data(data_dir: Path, network_names: tuple[str, ...]) -> None:
    required = [data_dir / "mnist_tests.h"]
    required.extend(data_dir / f"{name}.nnet" for name in network_names)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing ReluDiff MNIST instance data; run "
            "`python3 scripts/download_mnist_reludiff_nnets.py` first. Missing: "
            + ", ".join(str(path) for path in missing)
        )


def load_suite(suite_options: SuiteOptions | None = None) -> InstanceSuite:
    suite_name = "mnist_reludiff"
    options = _normalized_options(suite_options)
    print(f"{suite_name} suite options: {options}", file=sys.stderr)
    data_dir = runtime_path("data/reludiff_mnist")
    network_names = _option_tuple(options, "networks")
    _validate_network_names(network_names)
    _require_data(data_dir, network_names)

    mnist_tests, labels, random_pixels = load_reludiff_mnist_tests(
        data_dir / "mnist_tests.h"
    )

    modes = _option_tuple(options, "modes")
    unknown_modes = set(modes) - {"global", "three_pixel"}
    if unknown_modes:
        raise ValueError(f"unknown mnist_reludiff modes: {sorted(unknown_modes)}")

    limit = _limit(options)
    sample_indices = range(100 if limit is None else limit)
    timeout_sec = _timeout(options)
    epsilon = _epsilon(options)
    perturb = _perturb(options)
    perturbation = _perturbation(options)
    sparsity = _sparsity(options)
    network_pairs = _load_network_pairs(
        data_dir, network_names, perturbation, sparsity
    )

    if perturbation == "prune":
        transform_metadata = {"perturbation": "prune", "sparsity": sparsity}
    else:
        transform_metadata = {
            "perturbation": "quantize",
            "quantization": "float32_to_float16",
        }

    instances: list[Instance] = []
    for network_name, (original, second) in network_pairs.items():
        for mode in modes:
            for sample_index in sample_indices:
                raw_pixels = mnist_tests[sample_index]
                if mode == "global":
                    input_region = _global_region(raw_pixels, perturb)
                    perturb_metadata = perturb
                else:
                    input_region = _three_pixel_region(
                        raw_pixels, random_pixels[sample_index]
                    )
                    perturb_metadata = ",".join(
                        str(pixel_id) for pixel_id in random_pixels[sample_index][:3]
                    )

                instances.append(
                    Instance(
                        instance_id=f"{network_name}_{mode}_{sample_index}",
                        suite_name=suite_name,
                        nn1=original,
                        nn2=second,
                        input_region=input_region,
                        epsilon=epsilon,
                        output_index=labels[sample_index],
                        expected_status=None,
                        timeout_sec=timeout_sec,
                        metadata={
                            "network": network_name,
                            "sample_index": sample_index,
                            "correct_class": labels[sample_index],
                            "output_index": labels[sample_index],
                            "input_mode": mode,
                            "perturb": perturb_metadata,
                            **transform_metadata,
                        },
                    )
                )

    return InstanceSuite(name=suite_name, instances=instances)
