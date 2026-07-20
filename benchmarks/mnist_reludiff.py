from __future__ import annotations

from pathlib import Path
import sys

from benchmarks.common import Instance, InstanceSuite, InputRegion, SuiteOptions
from nn_equivalence.reludiff_nnet import (
    MNIST_RELUDIFF_NETWORKS,
    load_nnet_layers,
    load_reludiff_mnist_tests,
    quantize_network_float16,
)
from nn_equivalence.nn_types import NeuralNetwork

DEFAULT_SUITE_OPTIONS: SuiteOptions = {
    "epsilon": "1.0",
    "perturb": "3.0",
    "timeout": "5",
}


def _normalized_options(suite_options: SuiteOptions | None) -> SuiteOptions:
    options: SuiteOptions = dict(DEFAULT_SUITE_OPTIONS)
    for key, value in (suite_options or {}).items():
        normalized_key = key.strip().lower().replace("-", "_")
        options[normalized_key] = value

    allowed_options = {"networks", "modes", "limit", "timeout", "epsilon", "perturb"}
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


def _validate_network_names(network_names: tuple[str, ...]) -> None:
    for network_name in network_names:
        if network_name not in MNIST_RELUDIFF_NETWORKS:
            raise ValueError(f"unknown ReluDiff MNIST network: {network_name}")


def _global_region(raw_pixels: list[float], perturb: float) -> InputRegion:
    return InputRegion(
        lower_bounds=[max((pixel - perturb) / 255.0, 0.0) for pixel in raw_pixels],
        upper_bounds=[min((pixel + perturb) / 255.0, 1.0) for pixel in raw_pixels],
    )


def _three_pixel_region(raw_pixels: list[float], pixel_ids: list[int]) -> InputRegion:
    lower_bounds = [pixel / 255.0 for pixel in raw_pixels]
    upper_bounds = [pixel / 255.0 for pixel in raw_pixels]
    for pixel_id in pixel_ids[:3]:
        lower_bounds[pixel_id] = 0.0
        upper_bounds[pixel_id] = 1.0
    return InputRegion(lower_bounds=lower_bounds, upper_bounds=upper_bounds)


def _load_network_pairs(
    data_dir: Path,
    network_names: tuple[str, ...],
) -> dict[str, tuple[NeuralNetwork, NeuralNetwork]]:
    pairs: dict[str, tuple[NeuralNetwork, NeuralNetwork]] = {}
    for network_name in network_names:
        original = load_nnet_layers(data_dir / f"{network_name}.nnet")
        pairs[network_name] = (original, quantize_network_float16(original))
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
    data_dir = Path("data/reludiff_mnist")
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
    network_pairs = _load_network_pairs(data_dir, network_names)

    instances: list[Instance] = []
    for network_name, (original, quantized) in network_pairs.items():
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
                        nn2=quantized,
                        input_region=input_region,
                        epsilon=epsilon,
                        expected_status=None,
                        timeout_sec=timeout_sec,
                        metadata={
                            "network": network_name,
                            "sample_index": sample_index,
                            "correct_class": labels[sample_index],
                            "input_mode": mode,
                            "perturb": perturb_metadata,
                            "quantization": "float32_to_float16",
                        },
                    )
                )

    return InstanceSuite(name=suite_name, instances=instances)
