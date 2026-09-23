"""Instance selection: the ``limit`` / ``ids`` knobs, applied after suite load.

Deduplicates the ``extract_selection_from_suite_options`` + ``parse_instance_ids``
+ ``filter_instances`` logic that ``run_pyomo`` and ``run_crown`` each carried a
copy of. ``limit`` and ``ids`` are pulled out of the suite-option bag before the
suite is loaded, then applied to the generated instance list.
"""

from __future__ import annotations

from nnequiv.core import Instance
from nnequiv.suites import SuiteOptions


def _parse_ids(raw_ids: str) -> list[str]:
    return [
        instance_id.strip()
        for instance_id in raw_ids.split(",")
        if instance_id.strip()
    ]


def extract_selection(
    suite_options: SuiteOptions,
) -> tuple[SuiteOptions, int | None, list[str]]:
    """Split the suite-option bag into (remaining options, limit, ids)."""
    options = dict(suite_options)
    limit = options.pop("limit", None)
    ids = options.pop("ids", None)
    if limit is not None and ids is not None:
        raise ValueError("suite options 'limit' and 'ids' cannot be combined")
    if limit is not None:
        return options, int(limit), []
    if ids is not None:
        return options, None, _parse_ids(ids)
    return options, None, []


def select_instances(
    instances: list[Instance],
    limit: int | None,
    ids: list[str],
) -> list[Instance]:
    if limit is not None:
        if limit < 1:
            raise ValueError("suite option 'limit' must be at least 1")
        return instances[:limit]
    if not ids:
        raise ValueError("either suite option 'limit' or 'ids' must be provided")

    selected_ids = set(ids)
    selected = [
        instance for instance in instances if instance.instance_id in selected_ids
    ]
    missing = selected_ids - {instance.instance_id for instance in selected}
    if missing:
        raise ValueError(f"unknown instance ids: {sorted(missing)}")
    return selected
