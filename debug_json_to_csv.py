#!/usr/bin/env python3
"""Convert relational-MILP debug JSON into a flat CSV table.

The script supports both:

1. Current instance-level logs containing a ``directions`` object.
2. Older direction-level logs containing a ``binary_variables`` object.

By default, it keeps the fields that are most useful for evaluating encoding and
solver difficulty: identifiers/status, timings, binary-variable counts,
before/after-presolve model sizes, and CPLEX progress.
"""

from __future__ import annotations

import argparse
import csv
from fnmatch import fnmatchcase
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


DEFAULT_COLUMN_PATTERNS = [
    "instance_id",
    "suite_name",
    "status",
    "expected",
    "runtime_sec",
    "epsilon",
    "phase_timings_sec.measured_total_sec",
    "direction",
    "timings_sec.*",
    "total.*",
    "before_presolve.*",
    "after_presolve.*",
    "cplex_progress.*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten a relational-MILP debug JSON file into CSV. "
            "One CSV row is produced per instance and solve direction."
        )
    )
    parser.add_argument("json_path", type=Path, help="Input debug JSON/JSONL file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path. Defaults to <input-name>.csv.",
    )
    parser.add_argument(
        "--all-columns",
        action="store_true",
        help="Start with every available flattened column instead of the defaults.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PATTERN",
        help=(
            "Start with only matching columns. May be repeated or comma-separated. "
            "Shell-style wildcards are supported, e.g. 'timings_sec.*'."
        ),
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="PATTERN",
        help=(
            "Add matching columns to the current selection. May be repeated or "
            "comma-separated."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help=(
            "Remove matching columns from the selection. May be repeated or "
            "comma-separated."
        ),
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="Print all available flattened columns and exit without writing CSV.",
    )
    parser.add_argument(
        "--float-digits",
        type=int,
        default=6,
        metavar="N",
        help=(
            "Maximum number of digits after the decimal point for floating-point "
            "values. Trailing zeros are removed. Default: 6."
        ),
    )
    args = parser.parse_args()
    if args.float_digits < 0:
        parser.error("--float-digits must be zero or greater.")
    return args


def load_json_records(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"Could not read {path}: {error}") from error

    if not text.strip():
        raise SystemExit(f"{path} is empty.")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to JSON Lines so the converter also works with logs that
        # contain one JSON object per line.
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise SystemExit(
                    f"JSONL line {line_number} must contain an object, "
                    f"not {type(value).__name__}."
                )
            records.append(value)
        if not records:
            raise SystemExit(f"No JSON objects found in {path}.")
        return records

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        bad_indices = [
            index for index, value in enumerate(payload) if not isinstance(value, dict)
        ]
        if bad_indices:
            raise SystemExit(
                "The top-level JSON list must contain only objects; invalid indices: "
                + ", ".join(map(str, bad_indices[:10]))
            )
        return payload

    raise SystemExit(
        f"Top-level JSON must be an object or list, not {type(payload).__name__}."
    )


def normalize_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize supported log schemas to one object per solve direction."""

    normalized: list[dict[str, Any]] = []

    for record in records:
        directions = record.get("directions")
        if isinstance(directions, dict):
            # Current schema: one instance object containing one object per
            # direction. Duplicate the instance-level metadata into each row.
            instance_fields = {
                key: value for key, value in record.items() if key != "directions"
            }

            if directions:
                for direction_name, direction_payload in directions.items():
                    if not isinstance(direction_payload, dict):
                        continue
                    row: dict[str, Any] = {}

                    # Keep the most readable instance-level fields first.
                    for key in (
                        "type",
                        "instance_id",
                        "suite_name",
                        "status",
                        "expected",
                        "runtime_sec",
                        "epsilon",
                        "phase_timings_sec",
                    ):
                        if key in instance_fields:
                            row[key] = instance_fields[key]

                    # Preserve any future instance-level fields not listed above.
                    for key, value in instance_fields.items():
                        if key not in row:
                            row[key] = value

                    row["direction"] = direction_payload.get(
                        "direction", direction_name
                    )
                    for key, value in direction_payload.items():
                        if key != "direction":
                            row[key] = value
                    normalized.append(row)
            else:
                row = dict(instance_fields)
                row.setdefault("direction", None)
                normalized.append(row)
            continue

        # Older schema: one record per direction, with binary statistics grouped
        # under ``binary_variables``. Promote the groups so both schemas expose
        # the same CSV column names.
        row = dict(record)
        binary_variables = row.pop("binary_variables", None)
        if isinstance(binary_variables, dict):
            for group_name, group_value in binary_variables.items():
                row.setdefault(group_name, group_value)
        normalized.append(row)

    return normalized


def flatten_object(
    value: Any,
    prefix: str = "",
    output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if output is None:
        output = {}

    if isinstance(value, dict):
        if not value and prefix:
            output[prefix] = None
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_object(child, child_prefix, output)
    elif isinstance(value, list):
        # Lists are kept in a single CSV cell rather than creating a varying
        # number of columns.
        output[prefix] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        output[prefix] = value

    return output


def split_patterns(values: Iterable[str]) -> list[str]:
    patterns: list[str] = []
    for value in values:
        patterns.extend(part.strip() for part in value.split(",") if part.strip())
    return patterns


def matching_columns(pattern: str, available: list[str]) -> list[str]:
    return [column for column in available if fnmatchcase(column, pattern)]


def resolve_columns(
    available: list[str],
    *,
    all_columns: bool,
    only_patterns: list[str],
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> list[str]:
    if all_columns and only_patterns:
        raise SystemExit("--all-columns and --only cannot be used together.")

    unmatched: list[str] = []

    def resolve(
        patterns: Iterable[str], *, warn_if_unmatched: bool = True
    ) -> list[str]:
        result: list[str] = []
        for pattern in patterns:
            matches = matching_columns(pattern, available)
            if not matches and warn_if_unmatched:
                unmatched.append(pattern)
            for column in matches:
                if column not in result:
                    result.append(column)
        return result

    if only_patterns:
        selected = resolve(only_patterns)
    elif all_columns:
        selected = list(available)
    else:
        # A field may legitimately be absent in some runs (for example, CPLEX
        # progress on an immediately solved instance), so missing default fields
        # are not warnings.
        selected = resolve(DEFAULT_COLUMN_PATTERNS, warn_if_unmatched=False)

    for column in resolve(include_patterns):
        if column not in selected:
            selected.append(column)

    excluded = set(resolve(exclude_patterns))
    selected = [column for column in selected if column not in excluded]

    if unmatched:
        print(
            "Warning: these patterns matched no columns: "
            + ", ".join(repr(pattern) for pattern in unmatched),
            file=sys.stderr,
        )

    if not selected:
        raise SystemExit("The final column selection is empty.")

    return selected


def collect_available_columns(rows: Iterable[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for column in row:
            if column not in seen:
                seen.add(column)
                columns.append(column)
    return columns


def csv_value(value: Any, float_digits: int) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        formatted = f"{value:.{float_digits}f}"
        if float_digits > 0:
            formatted = formatted.rstrip("0").rstrip(".")
        # Avoid emitting "-0" after rounding a very small negative value.
        return "0" if formatted == "-0" else formatted
    return value


def main() -> None:
    args = parse_args()
    records = load_json_records(args.json_path)
    normalized = normalize_records(records)
    flattened_rows = [flatten_object(record) for record in normalized]

    if not flattened_rows:
        raise SystemExit(f"No records found in {args.json_path}.")

    available = collect_available_columns(flattened_rows)
    if args.list_columns:
        default_columns = set(
            resolve_columns(
                available,
                all_columns=False,
                only_patterns=[],
                include_patterns=[],
                exclude_patterns=[],
            )
        )
        for column in available:
            marker = "*" if column in default_columns else " "
            print(f"{marker} {column}")
        print("\n* = selected by default", file=sys.stderr)
        return

    columns = resolve_columns(
        available,
        all_columns=args.all_columns,
        only_patterns=split_patterns(args.only),
        include_patterns=split_patterns(args.include),
        exclude_patterns=split_patterns(args.exclude),
    )

    output_path = args.output or args.json_path.with_suffix(".csv")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in flattened_rows:
                writer.writerow(
                    {
                        column: csv_value(row.get(column), args.float_digits)
                        for column in columns
                    }
                )
    except OSError as error:
        raise SystemExit(f"Could not write {output_path}: {error}") from error

    print(
        f"Wrote {len(flattened_rows)} rows and {len(columns)} columns to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
