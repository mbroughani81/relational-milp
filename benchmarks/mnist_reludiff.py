from __future__ import annotations

import os
from pathlib import Path

from benchmarks.common import Benchmark, BenchmarkSuite, InputRegion
from nn_equivalence.reludiff_nnet import (
    MNIST_RELUDIFF_NETWORKS,
    load_nnet_layers,
    load_reludiff_mnist_tests,
    quantize_network_float16,
)
from nn_equivalence.nn_types import NeuralNetwork

SUITE_NAME = "mnist_reludiff"
DEFAULT_EPSILON = 1.0
DEFAULT_PERTURB = 3.0
DEFAULT_TIMEOUT_SEC = 300

def get_env_tuple(name: str) -> tuple[str, ...]:
    value = os.environ.get(name)
    if not value:
        return tuple()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _limit() -> int | None:
    value = os.environ.get("MNIST_RELUDIFF_LIMIT")
    if not value:
        return None
    limit = int(value)
    if limit < 1 or limit > 100:
        raise ValueError("MNIST_RELUDIFF_LIMIT must be between 1 and 100")
    return limit


def _timeout() -> float:
    return float(os.environ.get("MNIST_RELUDIFF_TIMEOUT", DEFAULT_TIMEOUT_SEC))


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


def _load_network_pairs(data_dir: Path) -> dict[str, tuple[NeuralNetwork, NeuralNetwork]]:
    pairs: dict[str, tuple[NeuralNetwork, NeuralNetwork]] = {}
    for network_name in get_env_tuple("MNIST_RELUDIFF_NETWORKS"):
        if network_name not in MNIST_RELUDIFF_NETWORKS:
            raise ValueError(f"unknown ReluDiff MNIST network: {network_name}")
        original = load_nnet_layers(data_dir / f"{network_name}.nnet")
        pairs[network_name] = (original, quantize_network_float16(original))
    return pairs


def _require_data(data_dir: Path) -> None:
    required = [data_dir / "mnist_tests.h"]
    required.extend(data_dir / f"{name}.nnet" for name in MNIST_RELUDIFF_NETWORKS)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing ReluDiff MNIST benchmark data; run "
            "`python3 scripts/download_mnist_reludiff_nnets.py` first. Missing: "
            + ", ".join(str(path) for path in missing)
        )


def load_suite() -> BenchmarkSuite:
    data_dir = Path("data/reludiff_mnist")
    _require_data(data_dir)

    mnist_tests, labels, random_pixels = load_reludiff_mnist_tests(
        data_dir / "mnist_tests.h"
    )

    modes = get_env_tuple("MNIST_RELUDIFF_MODES")
    unknown_modes = set(modes) - {"global", "three_pixel"}
    if unknown_modes:
        raise ValueError(f"unknown MNIST_RELUDIFF_MODES entries: {sorted(unknown_modes)}")

    limit = _limit()
    sample_indices = range(100 if limit is None else limit)
    timeout_sec = _timeout()
    network_pairs = _load_network_pairs(data_dir)

    benchmarks: list[Benchmark] = []
    for network_name, (original, quantized) in network_pairs.items():
        for mode in modes:
            for sample_index in sample_indices:
                raw_pixels = mnist_tests[sample_index]
                if mode == "global":
                    input_region = _global_region(raw_pixels, DEFAULT_PERTURB)
                    perturb_metadata = DEFAULT_PERTURB
                else:
                    input_region = _three_pixel_region(
                        raw_pixels, random_pixels[sample_index]
                    )
                    perturb_metadata = ",".join(
                        str(pixel_id) for pixel_id in random_pixels[sample_index][:3]
                    )

                benchmarks.append(
                    Benchmark(
                        benchmark_id=f"{network_name}_{mode}_{sample_index}",
                        suite_name=SUITE_NAME,
                        nn1=original,
                        nn2=quantized,
                        input_region=input_region,
                        epsilon=DEFAULT_EPSILON,
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

    return BenchmarkSuite(name=SUITE_NAME, benchmarks=benchmarks)
