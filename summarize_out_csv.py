from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize runtime and verifier statuses from a benchmark CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=Path("out.csv"),
        type=Path,
        help="Benchmark CSV path. Defaults to out.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total_runtime = 0.0
    row_count = 0
    verified_count = 0
    unknown_count = 0

    with args.csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row_count += 1
            total_runtime += float(row["runtime_sec"])

            abcrown_status = row["abcrown_status"].strip().lower()
            if abcrown_status == "verified":
                verified_count += 1
            elif abcrown_status == "unknown":
                unknown_count += 1

    average_runtime = total_runtime / row_count if row_count else 0.0

    print(f"sum_runtime_sec: {total_runtime:.6f}")
    print(f"average_runtime_sec: {average_runtime:.6f}")
    print(f"verified_count: {verified_count}")
    print(f"unknown_count: {unknown_count}")


if __name__ == "__main__":
    main()
