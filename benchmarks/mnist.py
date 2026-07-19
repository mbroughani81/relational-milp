from __future__ import annotations

from pathlib import Path

from torchvision import datasets, transforms

from benchmarks.common import Instance, InstanceSuite, InputRegion
from nn_equivalence.nn_loader import load_linear_layers


def load_mnist_sample(data_dir: Path, sample_index: int) -> tuple[list[float], int]:
    dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    image, label = dataset[sample_index]
    return image.flatten().tolist(), int(label)


def input_region_from_pixels(pixels: list[float], radius: float) -> InputRegion:
    return InputRegion(
        lower_bounds=[max(0.0, pixel - radius) for pixel in pixels],
        upper_bounds=[min(1.0, pixel + radius) for pixel in pixels],
    )


def load_suite() -> InstanceSuite:
    data_dir = Path("data")
    instances: list[Instance] = []
    input_radius = 0.03
    output_epsilons = [0.01, 10, 20, 100, 1000]
    timeout_sec = 10.0
    model_pairs = [
        # (
        #     "32_32",
        #     Path("models/nn_equivalence/smoky_32_32_seed0.pt"),
        #     Path("models/nn_equivalence/smoky_32_32_seed1.pt"),
        # ),
        (
            "64_64",
            Path("models/nn_equivalence/smoky_64_64_seed0.pt"),
            Path("models/nn_equivalence/smoky_64_64_seed1.pt"),
        ),
    ]

    for model_name, first_model, second_model in model_pairs:
        nn1 = load_linear_layers(first_model)
        nn2 = load_linear_layers(second_model)

        for sample_index in [0, 1, 2]:
            pixels, label = load_mnist_sample(data_dir, sample_index)
            input_region = input_region_from_pixels(pixels, input_radius)

            for output_epsilon in output_epsilons:
                epsilon_id = str(output_epsilon).replace(".", "p")
                instances.append(
                    Instance(
                        instance_id=(
                            f"mnist_{model_name}_sample_{sample_index:03d}"
                            f"_eps_{epsilon_id}"
                        ),
                        suite_name="mnist",
                        nn1=nn1,
                        nn2=nn2,
                        input_region=input_region,
                        epsilon=output_epsilon,
                        expected_status=None,
                        timeout_sec=timeout_sec,
                        metadata={
                            "model_name": model_name,
                            "sample_index": sample_index,
                            "true_label": label,
                            "input_radius": input_radius,
                            "first_model": str(first_model),
                            "second_model": str(second_model),
                        },
                    )
                )

    return InstanceSuite(name="mnist", instances=instances)
