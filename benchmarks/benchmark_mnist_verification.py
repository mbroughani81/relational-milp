from __future__ import annotations

import argparse
from pathlib import Path

from gurobipy import GRB

from verify_mnist_dnn import (
    build_target_model,
    forward_values,
    load_linear_layers,
    load_mnist_sample,
    predict,
    status_name,
)


def parse_float_list(value: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("must include at least one value")
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("epsilon values must be non-negative")
    return values


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("must include at least one value")
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("sample indices must be non-negative")
    return values


def architecture_name(layers: list[tuple[list[list[float]], list[float]]]) -> str:
    return "784-" + "-".join(str(len(bias)) for _, bias in layers)


def verify_case(
    model_path: Path,
    sample_index: int,
    epsilon: float,
    data_dir: Path,
    time_limit: float,
) -> tuple[str, int, int, float, str, float, int, int, int]:
    layers = load_linear_layers(model_path)
    pixels, label = load_mnist_sample(data_dir, sample_index)
    scores = forward_values(layers, pixels)
    predicted_class = predict(scores)

    total_solve_time = 0.0
    result = "robust"
    num_vars = 0
    num_binary = 0
    num_constraints = 0

    for target_class in range(len(scores)):
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

        total_solve_time += model.Runtime
        num_vars = model.NumVars
        num_binary = model.NumBinVars
        num_constraints = model.NumConstrs

        if model.Status == GRB.OPTIMAL:
            result = "adversarial"
            break
        if model.Status != GRB.INFEASIBLE:
            result = status_name(model.Status)

    return (
        architecture_name(layers),
        sample_index,
        label,
        epsilon,
        result,
        total_solve_time,
        num_vars,
        num_binary,
        num_constraints,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MNIST DNN verification time across checkpoints."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--models",
        type=Path,
        nargs="+",
        default=[Path("models/mnist_relu.pt")],
        help="Checkpoint .pt files or exported weights .json files.",
    )
    parser.add_argument("--sample-indices", type=parse_int_list, default=[0])
    parser.add_argument("--epsilons", type=parse_float_list, default=[0.03])
    parser.add_argument("--time-limit", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("model,architecture,sample,true_label,epsilon,result,total_solve_time,vars,binaries,constraints")
    for model_path in args.models:
        for sample_index in args.sample_indices:
            for epsilon in args.epsilons:
                (
                    architecture,
                    sample,
                    label,
                    eps,
                    result,
                    solve_time,
                    num_vars,
                    num_binary,
                    num_constraints,
                ) = verify_case(
                    model_path=model_path,
                    sample_index=sample_index,
                    epsilon=epsilon,
                    data_dir=args.data_dir,
                    time_limit=args.time_limit,
                )
                print(
                    f"{model_path},{architecture},{sample},{label},{eps},"
                    f"{result},{solve_time:.6f},{num_vars},{num_binary},{num_constraints}"
                )


if __name__ == "__main__":
    main()
