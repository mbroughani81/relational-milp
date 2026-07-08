import gurobipy as gp
from gurobipy import GRB

from typing import Sequence
Vector = list[float]
Matrix = list[list[float]]

def relu(value: float) -> float:
    return max(0.0, value)

def affine_layer(weights: Matrix, bias: Vector, inputs: Sequence[float]) -> Vector:
    return [
        sum(weight * input_value for weight, input_value in zip(row, inputs)) + bias_value
        for row, bias_value in zip(weights, bias)
    ]

def relu_layer(values: Sequence[float]) -> Vector:
    return [relu(value) for value in values]

def predict(scores: Sequence[float]) -> int:
    return max(range(len(scores)), key=lambda i: scores[i])

def print_vector(name: str, values: Sequence[float]) -> None:
    formatted = ", ".join(f"{value:.6f}" for value in values)
    print(f"{name}: [{formatted}]")

def main() -> None:
    x: Vector = [0.5, -0.2]
    w1: Matrix = [
        [1.0, -1.0],
        [-0.5, 1],
        [0.75, 0.25]
    ]
    b1: Vector = [0.0, 0.1, -0.2]
    w2: Matrix = [
        [1.2, -0.7, 0.5],
        [-0.4, 1.0, -0.8]
    ]
    b2: Vector = [0.05, -0.1]
    z1 = affine_layer(w1, b1, x)
    a1 = relu_layer(z1)
    z2 = affine_layer(w1, b2, a1)

    predicted_class = predict(z2)
    print("Tiny hand-written ReLU network")
    print("=" * 36)

    print()
    print("Input")
    print_vector("x", x)

    print()
    print("Layer 1: z1 = W1 x + b1")
    print_vector("z1", z1)

    print()
    print("Layer 1 activation: a1 = ReLU(z1)")
    print_vector("a1", a1)

    print()
    print("Output layer: z2 = W2 a1 + b2")
    print_vector("z2 / class scores", z2)

    print()
    print(f"predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
