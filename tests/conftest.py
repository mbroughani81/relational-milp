"""Shared pytest configuration.

The runtime path helpers require ``RUNTIME_DIR`` to be set. For tests we default
it to ``<repo>/runtime`` when the caller hasn't exported it, so the suite runs
without every invocation having to set it. A real export still wins.
"""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("RUNTIME_DIR", str(_REPO_ROOT / "runtime"))
