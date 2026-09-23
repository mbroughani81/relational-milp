"""Suite loading and suite-option parsing.

``load_suite(name, options)`` resolves an instance-generator suite by name. The
suite modules currently live in ``benchmarks.suites``; this loader imports them
lazily, so this package stays import-light and a later phase can relocate the
suite modules under here without changing callers. ``parse_suite_options`` /
``SuiteOptions`` moved here from ``benchmarks.common`` (which now re-exports them).
"""

from __future__ import annotations

import importlib
import json

from nnequiv.core import InstanceSuite

SuiteOptions = dict[str, str]


def load_suite(name: str, options: SuiteOptions | None = None) -> InstanceSuite:
    module = importlib.import_module(f"nnequiv.suites.{name}")
    return module.load_suite(options or {})


def parse_suite_options(raw_options: list[str] | None) -> SuiteOptions:
    options: SuiteOptions = {}
    for raw_option in raw_options or []:
        raw_option = raw_option.strip()
        if not raw_option:
            continue
        if raw_option.startswith("{"):
            parsed = json.loads(raw_option)
            if not isinstance(parsed, dict):
                raise ValueError("--suite-options JSON value must be an object")
            options.update({str(key): str(value) for key, value in parsed.items()})
            continue

        for part in raw_option.split(";"):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError("--suite-options entries must be KEY=VALUE pairs")
            key, value = part.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError("--suite-options keys must be non-empty")
            options[key] = value.strip()

    return options


__all__ = ["SuiteOptions", "load_suite", "parse_suite_options"]
