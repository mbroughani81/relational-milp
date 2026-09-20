"""Resolve the runtime base directory from the ``RUNTIME_DIR`` environment var.

All generated/downloaded content (data fixtures, third_party checkouts, build
artifacts) lives under a single base directory chosen at run time via the
``RUNTIME_DIR`` environment variable. It is *required*: helpers here fail fast
with a clear message when it is unset, so a misconfigured run stops immediately
instead of silently reading/writing the wrong place.

Resolution is lazy (only when a path is actually needed), so importing modules
that reference runtime paths — and ``--help`` — works without the env set.
"""

from __future__ import annotations

import os
from pathlib import Path

RUNTIME_ENV_VAR = "RUNTIME_DIR"


def runtime_dir() -> Path:
    """Return the runtime base dir, or exit if ``RUNTIME_DIR`` is unset/empty."""
    value = os.environ.get(RUNTIME_ENV_VAR, "").strip()
    if not value:
        raise SystemExit(
            f"{RUNTIME_ENV_VAR} is not set. Export it to the base directory for "
            f"generated/downloaded content before running, e.g.:\n"
            f'    export {RUNTIME_ENV_VAR}="$PWD/runtime"'
        )
    return Path(value)


def runtime_path(*parts: str) -> Path:
    """Join ``parts`` onto the runtime base dir (e.g. ``runtime_path("data")``)."""
    return runtime_dir().joinpath(*parts)
