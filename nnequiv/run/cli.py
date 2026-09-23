"""The single ``nnequiv`` command line.

    nnequiv run --verifier milp --suite pruning_mnist \\
        -o networks=mnist_relu_3_100 -o modes=global -o limit=3 \\
        --verifier-opt bound_tightening=abcrown --csv out.csv

Shared flags live here once; anything backend-specific goes through repeatable
``--verifier-opt KEY=VALUE`` and is parsed by the chosen verifier's
``from_options``. The ``sweep`` subcommand is added in a later phase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nnequiv.report import build_rows, print_progress, rows_to_csv, write_csv
from nnequiv.run.runner import run_suite
from nnequiv.suites import parse_suite_options
from nnequiv.sweep.experiments import EXPERIMENT_NAMES, get_experiment
from nnequiv.verifiers import available


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verifier", required=True, choices=available())
    parser.add_argument("--suite", required=True)
    parser.add_argument(
        "-o",
        "--suite-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="suite_options",
        help="Suite option; repeatable. Values may contain commas.",
    )
    parser.add_argument(
        "--verifier-opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="verifier_options",
        help="Verifier-specific option; repeatable.",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")


def _run(args: argparse.Namespace) -> int:
    try:
        suite_options = parse_suite_options(args.suite_options)
        verifier_options = parse_suite_options(args.verifier_options)
        instances, results = run_suite(
            args.verifier,
            args.suite,
            suite_options,
            verifier_options,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(error)
        return 2

    total = len(results)
    for index, (instance, result) in enumerate(zip(instances, results), start=1):
        print_progress(index, total, args.verifier, instance, result)

    rows = build_rows(args.verifier, instances, results)
    print(rows_to_csv(rows), end="")
    if args.csv is not None:
        write_csv(args.csv, rows)
    return 0


def _sweep(args: argparse.Namespace) -> int:
    experiment = get_experiment(args.experiment)
    if args.dry_run:
        print(json.dumps(experiment.build_plan(), indent=2))
        return 0
    return experiment.run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nnequiv",
        description="Neural-network equivalence verifier-comparison harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run", help="run a verifier over an equivalence suite"
    )
    _add_run_arguments(run_parser)

    sweep_parser = subparsers.add_parser(
        "sweep", help="run an experiment grid (or print its plan with --dry-run)"
    )
    sweep_parser.add_argument("experiment", choices=list(EXPERIMENT_NAMES))
    sweep_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan as a JSON array of {result, command} objects and exit",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        raise SystemExit(_run(args))
    if args.command == "sweep":
        raise SystemExit(_sweep(args))
    parser.error(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
