from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gurobipy as gp
import torch
from gurobipy import GRB
from torchvision import datasets, transforms

Vector = list[float]
Matrix = list[list[float]]
Bounds = list[tuple[float, float]]
LinearLayer = tuple[Matrix, Vector]


def load_weights(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_linear_layers(path: Path) -> list[LinearLayer]:
    if path.suffix.lower() == ".json":
        weights = load_weights(path)
        layers: list[LinearLayer] = []
        layer_index = 1
        while f"W{layer_index}" in weights and f"b{layer_index}" in weights:
            layers.append((weights[f"W{layer_index}"], weights[f"b{layer_index}"]))
            layer_index += 1
        if layers:
            return layers
        return [(weights["W1"], weights["b1"]), (weights["W2"], weights["b2"])]

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    linear_indices = sorted(
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("layers.") and key.endswith(".weight")
    )

    return [
        (
            state_dict[f"layers.{index}.weight"].detach().cpu().tolist(),
            state_dict[f"layers.{index}.bias"].detach().cpu().tolist(),
        )
        for index in linear_indices
    ]


def load_mnist_sample(data_dir: Path, sample_index: int) -> tuple[Vector, int]:
    dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    image, label = dataset[sample_index]
    return image.flatten().tolist(), int(label)


def affine_values(weights: Matrix, bias: Vector, inputs: Vector) -> Vector:
    return [
        sum(weight * value for weight, value in zip(row, inputs)) + bias_value
        for row, bias_value in zip(weights, bias)
    ]


def relu_values(values: Vector) -> Vector:
    return [max(0.0, value) for value in values]


def predict(scores: Vector) -> int:
    return max(range(len(scores)), key=lambda i: scores[i])


def input_bounds_from_pixels(pixels: Vector, epsilon: float) -> Bounds:
    return [
        (max(0.0, pixel - epsilon), min(1.0, pixel + epsilon))
        for pixel in pixels
    ]


def affine_bounds(weights: Matrix, bias: Vector, input_bounds: Bounds) -> Bounds:
    output_bounds: Bounds = []

    for row, bias_value in zip(weights, bias):
        lower = bias_value
        upper = bias_value

        for weight, (input_lower, input_upper) in zip(row, input_bounds):
            if weight >= 0:
                lower += weight * input_lower
                upper += weight * input_upper
            else:
                lower += weight * input_upper
                upper += weight * input_lower

        output_bounds.append((lower, upper))

    return output_bounds


def relu_bounds(z_bounds: Bounds) -> Bounds:
    return [(max(0.0, lower), max(0.0, upper)) for lower, upper in z_bounds]


def forward_values(layers: list[LinearLayer], pixels: Vector) -> Vector:
    values = pixels
    for weights, bias in layers[:-1]:
        values = relu_values(affine_values(weights, bias, values))

    output_weights, output_bias = layers[-1]
    return affine_values(output_weights, output_bias, values)


def hidden_pre_activation_bounds(
    layers: list[LinearLayer], input_bounds: Bounds
) -> tuple[list[Bounds], Bounds]:
    current_bounds = input_bounds
    hidden_bounds: list[Bounds] = []

    for weights, bias in layers[:-1]:
        z_bounds = affine_bounds(weights, bias, current_bounds)
        hidden_bounds.append(z_bounds)
        current_bounds = relu_bounds(z_bounds)

    return hidden_bounds, current_bounds


def add_affine_constraints(
    model: gp.Model,
    output_vars: list[gp.Var],
    weights: Matrix,
    input_vars: list[gp.Var],
    bias: Vector,
    layer_name: str,
) -> None:
    for i, output_var in enumerate(output_vars):
        model.addConstr(
            output_var
            == gp.quicksum(
                weights[i][j] * input_vars[j] for j in range(len(input_vars))
            )
            + bias[i],
            name=f"{layer_name}_{i}",
        )


def add_relu_big_m_constraints(
    model: gp.Model,
    z_vars: list[gp.Var],
    a_vars: list[gp.Var],
    z_bounds: Bounds,
    layer_name: str,
) -> list[gp.Var]:
    deltas: list[gp.Var] = []

    for i, (z, a) in enumerate(zip(z_vars, a_vars)):
        lower, upper = z_bounds[i]

        delta = model.addVar(vtype=GRB.BINARY, name=f"{layer_name}_delta_{i}")
        deltas.append(delta)

        model.addConstr(a >= z, name=f"{layer_name}_relu_{i}_a_ge_z")
        model.addConstr(a >= 0, name=f"{layer_name}_relu_{i}_a_ge_0")
        model.addConstr(
            a <= z - lower * (1 - delta),
            name=f"{layer_name}_relu_{i}_a_le_z_minus_L_inactive",
        )
        model.addConstr(
            a <= upper * delta,
            name=f"{layer_name}_relu_{i}_a_le_U_active",
        )

    return deltas


def build_target_model(
    pixels: Vector,
    epsilon: float,
    layers: list[LinearLayer],
    predicted_class: int,
    target_class: int,
    time_limit: float,
) -> gp.Model:
    input_bounds = input_bounds_from_pixels(pixels, epsilon)
    hidden_bounds, final_activation_bounds = hidden_pre_activation_bounds(
        layers, input_bounds
    )

    model = gp.Model(f"mnist_target_{target_class}")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit

    x = [
        model.addVar(lb=lower, ub=upper, name=f"x_{i}")
        for i, (lower, upper) in enumerate(input_bounds)
    ]

    current_vars = x
    deltas: list[gp.Var] = []
    for layer_index, ((weights, bias), z_bounds) in enumerate(
        zip(layers[:-1], hidden_bounds), start=1
    ):
        z_vars = [
            model.addVar(lb=lower, ub=upper, name=f"z{layer_index}_{i}")
            for i, (lower, upper) in enumerate(z_bounds)
        ]
        a_vars = [
            model.addVar(lb=0.0, ub=max(0.0, upper), name=f"a{layer_index}_{i}")
            for i, (_, upper) in enumerate(z_bounds)
        ]

        add_affine_constraints(
            model,
            z_vars,
            weights,
            current_vars,
            bias,
            f"hidden_{layer_index}_affine",
        )
        deltas.extend(
            add_relu_big_m_constraints(
                model, z_vars, a_vars, z_bounds, f"hidden_{layer_index}"
            )
        )
        current_vars = a_vars

    output_weights, output_bias = layers[-1]
    z2 = [
        model.addVar(lb=-GRB.INFINITY, name=f"z_out_{i}")
        for i in range(len(output_bias))
    ]

    add_affine_constraints(
        model, z2, output_weights, current_vars, output_bias, "output_affine"
    )

    model.addConstr(
        z2[target_class] >= z2[predicted_class],
        name=f"adversarial_target_{target_class}",
    )

    model.setObjective(0.0, GRB.MINIMIZE)
    return model


def status_name(status: int) -> str:
    names = {
        GRB.OPTIMAL: "optimal",
        GRB.INFEASIBLE: "infeasible",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INF_OR_UNBD: "inf_or_unbd",
        GRB.UNBOUNDED: "unbounded",
    }
    return names.get(status, f"status_{status}")


def verify_sample(
    pixels: Vector,
    label: int,
    layers: list[LinearLayer],
    epsilon: float,
    time_limit: float,
) -> None:
    scores = forward_values(layers, pixels)
    predicted_class = predict(scores)
    total_hidden_relus = sum(len(bias) for _, bias in layers[:-1])

    print("MNIST DNN robustness verification")
    print("=" * 36)
    print(f"architecture: 784 -> {' -> '.join(str(len(bias)) for _, bias in layers)}")
    print(f"true label: {label}")
    print(f"predicted class: {predicted_class}")
    print(f"hidden ReLUs: {total_hidden_relus}")
    print(f"epsilon: {epsilon}")
    print(f"time limit per target: {time_limit}s")
    print()

    verified_robust = True

    for target_class in range(len(scores)):
        # skip is target is same as predicted
        if target_class == predicted_class:
            continue

        model = build_target_model(
            pixels=pixels,
            epsilon=epsilon,
            layers=layers,
            predicted_class=predicted_class,
            target_class=target_class,
            time_limit=time_limit,
        )
        model.optimize()

        print(f"target class: {target_class}")
        print(f"status: {status_name(model.Status)}")
        print(f"solve time: {model.Runtime:.4f}s")
        print(f"variables: {model.NumVars}")
        print(f"binary variables: {model.NumBinVars}")
        print(f"constraints: {model.NumConstrs}")

        if model.Status == GRB.OPTIMAL:
            verified_robust = False
            print("adversarial example exists: yes")
        elif model.Status == GRB.INFEASIBLE:
            print("adversarial example exists: no")
        else:
            verified_robust = False
            print("verification inconclusive")

        print()

    print(f"verified robust: {'yes' if verified_robust else 'no'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify local robustness of the exported MNIST ReLU DNN."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/mnist_relu.pt"),
        help="Checkpoint .pt or exported weights .json.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--time-limit", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epsilon < 0:
        raise SystemExit("--epsilon must be non-negative")

    layers = load_linear_layers(args.model)
    pixels, label = load_mnist_sample(args.data_dir, args.sample_index)

    verify_sample(
        pixels=pixels,
        label=label,
        layers=layers,
        epsilon=args.epsilon,
        time_limit=args.time_limit,
    )


if __name__ == "__main__":
    main()
