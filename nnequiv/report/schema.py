"""One canonical, long-format result schema for every verifier.

A row has a fixed core (identity + status + timing) plus per-verifier ``extra.*``
columns flattened from each result's ``SolveStats.details`` and namespaced by
verifier, so results from different backends concatenate cleanly and analysis
reads a single schema. ``extra`` columns are the union across the rows; a row
that lacks one leaves it blank.
"""

from __future__ import annotations

from dataclasses import dataclass

from nnequiv.core import Instance, InstanceResult

CORE_COLUMNS: tuple[str, ...] = (
    "instance_id",
    "suite",
    "verifier",
    "property",
    "status",
    "expected",
    "matched",
    "runtime_sec",
    "epsilon",
)


@dataclass(frozen=True)
class ReportRow:
    instance_id: str
    suite: str
    verifier: str
    property: str
    status: str
    expected: str
    matched: str
    runtime_sec: float
    epsilon: float
    extra: dict[str, str]


def _format_value(value: str | int | float) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _extras(verifier: str, result: InstanceResult) -> dict[str, str]:
    """Flatten every ``SolveStats.details`` entry into namespaced extra columns."""
    extras: dict[str, str] = {}
    for stats in result.stats:
        for key, value in stats.details:
            extras[f"extra.{verifier}.{key}"] = _format_value(value)
    return extras


def build_row(verifier: str, instance: Instance, result: InstanceResult) -> ReportRow:
    if result.expected_status is None:
        expected, matched = "", ""
    else:
        expected = result.expected_status
        matched = "yes" if result.matched_expected else "no"
    return ReportRow(
        instance_id=result.instance_id,
        suite=result.suite_name,
        verifier=verifier,
        property=instance.property_kind,
        status=result.status,
        expected=expected,
        matched=matched,
        runtime_sec=result.runtime_sec,
        epsilon=result.epsilon,
        extra=_extras(verifier, result),
    )


def build_rows(
    verifier: str,
    instances: list[Instance],
    results: list[InstanceResult],
) -> list[ReportRow]:
    return [
        build_row(verifier, instance, result)
        for instance, result in zip(instances, results)
    ]


def _csv_field(value: str) -> str:
    if any(character in value for character in (",", '"', "\n")):
        escaped = value.replace('"', '""')
        return f'"{escaped}"'
    return value


def rows_to_csv(rows: list[ReportRow]) -> str:
    extra_columns = sorted({key for row in rows for key in row.extra})
    header = list(CORE_COLUMNS) + extra_columns
    lines = [",".join(header)]
    for row in rows:
        core = [
            row.instance_id,
            row.suite,
            row.verifier,
            row.property,
            row.status,
            row.expected,
            row.matched,
            f"{row.runtime_sec:.6f}",
            f"{row.epsilon:.17g}",
        ]
        extras = [row.extra.get(column, "") for column in extra_columns]
        lines.append(",".join(_csv_field(field) for field in (*core, *extras)))
    return "\n".join(lines) + "\n"


def write_csv(path, rows: list[ReportRow]) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows_to_csv(rows), encoding="utf-8")
