"""Fundamental type aliases shared across the harness.

A ``NeuralNetwork`` is a list of ``(weights, bias)`` layers, output-major
(``weights[out][in]``), with a ReLU after every layer except the last.
"""

Vector = list[float]
Matrix = list[list[float]]
Bounds = list[tuple[float, float]]
LinearLayer = tuple[Matrix, Vector]
NeuralNetwork = list[LinearLayer]
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]
