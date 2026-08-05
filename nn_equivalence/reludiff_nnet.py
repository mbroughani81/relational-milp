from __future__ import annotations

import re
import struct
from pathlib import Path

from nn_equivalence.nn_types import LinearLayer, NeuralNetwork

MNIST_RELUDIFF_NETWORKS = (
    "mnist_relu_2_512",
    "mnist_relu_3_100",
    "mnist_relu_4_1024",
)

MNIST_RELUDIFF_ARCHITECTURES: dict[str, list[int]] = {
    "mnist_relu_2_512": [784, 512, 512, 10],
    "mnist_relu_3_100": [784, 100, 100, 10, 10],
    "mnist_relu_4_1024": [784, 1024, 1024, 1024, 10, 10],
}


def _csv_values(line: str) -> list[str]:
    return [value for value in line.strip(" ,\n").split(",") if value]


def _next_data_line(file) -> str:
    for line in file:
        stripped = line.strip()
        if stripped and not stripped.startswith("//"):
            return line
    raise ValueError("unexpected end of .nnet file")


def read_nnet_architecture(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as file:
        header_values = [int(value) for value in _csv_values(_next_data_line(file))]
        if len(header_values) != 4:
            raise ValueError(f"invalid .nnet header in {path}")
        num_layers, input_size, output_size, _ = header_values
        layer_sizes = [int(value) for value in _csv_values(_next_data_line(file))]

    if len(layer_sizes) != num_layers + 1:
        raise ValueError(
            f"invalid .nnet architecture in {path}: expected {num_layers + 1} "
            f"layer sizes, found {len(layer_sizes)}"
        )
    if layer_sizes[0] != input_size:
        raise ValueError(
            f"invalid .nnet input size in {path}: header={input_size}, "
            f"architecture={layer_sizes[0]}"
        )
    if layer_sizes[-1] != output_size:
        raise ValueError(
            f"invalid .nnet output size in {path}: header={output_size}, "
            f"architecture={layer_sizes[-1]}"
        )
    return layer_sizes


def network_architecture(network: NeuralNetwork) -> list[int]:
    if not network:
        raise ValueError("neural network must have at least one layer")
    first_weights, _ = network[0]
    if not first_weights or not first_weights[0]:
        raise ValueError("first network layer must be non-empty")
    return [len(first_weights[0]), *(len(bias) for _, bias in network)]


def validate_mnist_reludiff_network(
    network_name: str,
    network: NeuralNetwork,
    source_path: Path | None = None,
) -> None:
    try:
        expected = MNIST_RELUDIFF_ARCHITECTURES[network_name]
    except KeyError as error:
        raise ValueError(f"unknown ReluDiff MNIST network: {network_name}") from error

    actual = network_architecture(network)
    if actual != expected:
        source = f" loaded from {source_path}" if source_path is not None else ""
        raise ValueError(
            f"{network_name}{source} has architecture {actual}, but the ReluDiff "
            f"benchmark requires {expected}. Delete the faulty data directory and run "
            "`python3 scripts/download_mnist_reludiff_nnets.py --force`."
        )


def load_nnet_layers(path: Path) -> NeuralNetwork:
    expected_architecture = read_nnet_architecture(path)

    with path.open("r", encoding="utf-8") as file:
        header_values = [int(value) for value in _csv_values(_next_data_line(file))]
        num_layers = header_values[0]
        layer_sizes = [int(value) for value in _csv_values(_next_data_line(file))]

        for _ in range(5):
            _next_data_line(file)

        layers: list[LinearLayer] = []
        for layer_index in range(num_layers):
            input_size = layer_sizes[layer_index]
            output_size = layer_sizes[layer_index + 1]
            weights: list[list[float]] = []
            for _ in range(output_size):
                row = [float(value) for value in _csv_values(_next_data_line(file))]
                if len(row) != input_size:
                    raise ValueError(
                        f"invalid weight row in {path}, layer {layer_index + 1}: "
                        f"expected {input_size} values, found {len(row)}"
                    )
                weights.append(row)

            bias: list[float] = []
            for _ in range(output_size):
                values = _csv_values(_next_data_line(file))
                if len(values) != 1:
                    raise ValueError(
                        f"invalid bias row in {path}, layer {layer_index + 1}"
                    )
                bias.append(float(values[0]))
            layers.append((weights, bias))

    actual_architecture = network_architecture(layers)
    if actual_architecture != expected_architecture:
        raise ValueError(
            f"loaded architecture mismatch in {path}: header={expected_architecture}, "
            f"parameters={actual_architecture}"
        )
    return layers


def quantize_network_float16(network: NeuralNetwork) -> NeuralNetwork:
    def quantize(value: float) -> float:
        return float(struct.unpack("e", struct.pack("e", float(value)))[0])

    quantized: list[LinearLayer] = []
    for weights, bias in network:
        quantized_weights = [
            [quantize(value) for value in row]
            for row in weights
        ]
        quantized_bias = [quantize(value) for value in bias]
        quantized.append((quantized_weights, quantized_bias))
    return quantized


def _extract_initializer(text: str, name: str) -> str:
    marker = re.search(rf"\b{name}\b[^=]*=", text)
    if marker is None:
        raise ValueError(f"could not find {name} initializer")

    start = text.find("{", marker.end())
    if start == -1:
        raise ValueError(f"could not find opening brace for {name}")

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index]

    raise ValueError(f"could not find closing brace for {name}")


def _parse_nested_int_rows(initializer: str) -> list[list[int]]:
    rows = re.findall(r"\{([^{}]*)\}", initializer)
    return [
        [int(value) for value in re.findall(r"-?\d+", row)]
        for row in rows
    ]


def load_reludiff_mnist_tests(
    path: Path,
) -> tuple[list[list[float]], list[int], list[list[int]]]:
    text = path.read_text(encoding="utf-8")

    mnist_rows = _parse_nested_int_rows(_extract_initializer(text, "mnist_test"))
    if len(mnist_rows) != 100 or any(len(row) < 784 for row in mnist_rows):
        raise ValueError("expected mnist_test to contain 100 rows of 784 pixels")
    pixels = [[float(value) for value in row[:784]] for row in mnist_rows]

    correct_class = [
        int(value)
        for value in re.findall(r"-?\d+", _extract_initializer(text, "correct_class"))
    ]
    if len(correct_class) != 100:
        raise ValueError("expected correct_class to contain 100 labels")
    if any(label < 0 or label >= 10 for label in correct_class):
        raise ValueError("correct_class contains a label outside 0-9")

    random_pixels = _parse_nested_int_rows(_extract_initializer(text, "random_pixels"))
    if len(random_pixels) != 100 or any(len(row) < 3 for row in random_pixels):
        raise ValueError("expected random_pixels to contain 100 rows of pixel ids")
    if any(pixel < 0 or pixel >= 784 for row in random_pixels for pixel in row[:3]):
        raise ValueError("random_pixels contains a pixel id outside 0-783")

    return pixels, correct_class, random_pixels
