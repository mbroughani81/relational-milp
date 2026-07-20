from __future__ import annotations

import random

from benchmarks.common import Instance, InstanceSuite, InputRegion, SuiteOptions
from nn_equivalence.nn_types import LinearLayer, NeuralNetwork


def make_random_network(
    architecture: list[int],
    seed: int,
    weight_scale: float = 0.7,
    bias_scale: float = 0.2,
) -> NeuralNetwork:
    rng = random.Random(seed)
    layers: list[LinearLayer] = []

    for input_size, output_size in zip(architecture, architecture[1:]):
        weights = [
            [rng.uniform(-weight_scale, weight_scale) for _ in range(input_size)]
            for _ in range(output_size)
        ]
        bias = [rng.uniform(-bias_scale, bias_scale) for _ in range(output_size)]
        layers.append((weights, bias))

    return layers


def perturb_network(
    nn: NeuralNetwork,
    seed: int,
    noise_scale: float = 0.03,
) -> NeuralNetwork:
    rng = random.Random(seed)
    perturbed: list[LinearLayer] = []

    for weights, bias in nn:
        perturbed_weights = [
            [weight + rng.uniform(-noise_scale, noise_scale) for weight in row]
            for row in weights
        ]
        perturbed_bias = [
            bias_value + rng.uniform(-noise_scale, noise_scale) for bias_value in bias
        ]
        perturbed.append((perturbed_weights, perturbed_bias))

    return perturbed


def architecture_id(architecture: list[int]) -> str:
    return "_".join(str(size) for size in architecture)


def epsilon_id(epsilon: float) -> str:
    return str(epsilon).replace(".", "p")


def load_suite(suite_options: SuiteOptions | None = None) -> InstanceSuite:
    del suite_options
    region = InputRegion(
        lower_bounds=[-1.0, -1.0],
        upper_bounds=[1.0, 1.0],
    )
    instances: list[Instance] = []
    architectures = [
        [2, 10, 10, 2],
        # [2, 20, 20, 2],
        # [2, 40, 40, 2],
        # [2, 60, 60, 2],
        # [2, 100, 100, 2],
    ]
    identical_epsilons = []
    noisy_epsilons = [0.01, 0.1, 1, 10]

    for case_index, architecture in enumerate(architectures, start=1):
        base = make_random_network(architecture, seed=100 + case_index)
        noisy = perturb_network(base, seed=200 + case_index)
        arch_id = architecture_id(architecture)

        for epsilon in identical_epsilons:
            instances.append(
                Instance(
                    instance_id=(
                        f"synthetic_identical_{arch_id}_eps_{epsilon_id(epsilon)}"
                    ),
                    suite_name="synthetic",
                    nn1=base,
                    nn2=base,
                    input_region=region,
                    epsilon=epsilon,
                    expected_status="unsat",
                    timeout_sec=10.0,
                    metadata={"pair_type": "identical", "architecture": arch_id},
                )
            )

        for epsilon in noisy_epsilons:
            instances.append(
                Instance(
                    instance_id=(
                        f"synthetic_noisy_{arch_id}_eps_{epsilon_id(epsilon)}"
                    ),
                    suite_name="synthetic",
                    nn1=base,
                    nn2=noisy,
                    input_region=region,
                    epsilon=epsilon,
                    expected_status=None,
                    timeout_sec=10.0,
                    metadata={"pair_type": "noisy", "architecture": arch_id},
                )
            )

    return InstanceSuite(name="synthetic", instances=instances)
