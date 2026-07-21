from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize runtime and verifier statuses from an instance CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=Path("out.csv"),
        type=Path,
        help="Instance CSV path. Defaults to out.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total_runtime = 0.0
    row_count = 0
    status_counts: Counter[str] = Counter()
    abcrown_status_counts: Counter[str] = Counter()

    with args.csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        if "runtime_sec" not in fieldnames:
            raise SystemExit(f"{args.csv_path} is missing required column: runtime_sec")
        if "status" not in fieldnames:
            raise SystemExit(f"{args.csv_path} is missing required column: status")

        has_abcrown_status = "abcrown_status" in fieldnames
        for row in reader:
            row_count += 1
            total_runtime += float(row["runtime_sec"])
            status_counts[row["status"].strip().lower()] += 1

            if has_abcrown_status:
                abcrown_status_counts[row["abcrown_status"].strip().lower()] += 1

    average_runtime = total_runtime / row_count if row_count else 0.0

    print(f"sum_runtime_sec: {total_runtime:.6f}")
    print(f"average_runtime_sec: {average_runtime:.6f}")
    print(f"row_count: {row_count}")
    for status, count in sorted(status_counts.items()):
        print(f"status_{status}_count: {count}")

    if abcrown_status_counts:
        for status, count in sorted(abcrown_status_counts.items()):
            print(f"abcrown_status_{status}_count: {count}")


if __name__ == "__main__":
    main()
