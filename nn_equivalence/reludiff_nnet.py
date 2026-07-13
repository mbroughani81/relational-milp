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


def _csv_values(line: str) -> list[str]:
    return [value for value in line.strip(" ,\n").split(",") if value]


def load_nnet_layers(path: Path) -> NeuralNetwork:
    with path.open("r", encoding="utf-8") as file:
        first = file.readline()
        while first.startswith("/"):
            first = file.readline()

        num_layers, _, _, _ = [int(value) for value in _csv_values(first)]
        layer_sizes = [int(value) for value in _csv_values(file.readline())]

        for _ in range(5):
            file.readline()

        layers: list[LinearLayer] = []
        for layer_index in range(num_layers):
            output_size = layer_sizes[layer_index + 1]
            weights = [
                [float(value) for value in _csv_values(file.readline())]
                for _ in range(output_size)
            ]
            bias = [float(_csv_values(file.readline())[0]) for _ in range(output_size)]
            layers.append((weights, bias))

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

    random_pixels = _parse_nested_int_rows(_extract_initializer(text, "random_pixels"))
    if len(random_pixels) != 100 or any(len(row) < 3 for row in random_pixels):
        raise ValueError("expected random_pixels to contain 100 rows of pixel ids")

    return pixels, correct_class, random_pixels
