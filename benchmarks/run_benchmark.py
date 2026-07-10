from __future__ import annotations

import argparse
import importlib

from benchmarks.common import BenchmarkResult, BenchmarkSuite, run_benchmark


def load_suite(name: str) -> BenchmarkSuite:
    module = importlib.import_module(f"benchmarks.{name}")
    return module.load_suite()


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def format_expected(result: BenchmarkResult) -> str:
    if result.expected_status is None:
        return ""
    matched = "yes" if result.matched_expected else "no"
    return f"{result.expected_status}:{matched}"


def print_results(results: list[BenchmarkResult]) -> None:
    print(
        "benchmark_id,status,expected,runtime_sec,vars,binaries,"
        "constraints,num_relu,max_output_diff"
    )
    for result in results:
        print(
            f"{result.benchmark_id},"
            f"{result.status},"
            f"{format_expected(result)},"
            f"{result.runtime_sec:.6f},"
            f"{result.num_vars},"
            f"{result.num_binary_vars},"
            f"{result.num_constraints},"
            f"{result.num_relu},"
            f"{format_float(result.max_output_diff)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an NN equivalence benchmark suite.")
    parser.add_argument("--suite", default="sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = load_suite(args.suite)
    results = [run_benchmark(benchmark) for benchmark in suite.benchmarks]
    print_results(results)


if __name__ == "__main__":
    main()
