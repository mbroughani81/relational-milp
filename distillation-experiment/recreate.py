#!/usr/bin/env python3
"""Thin shim: the distillation sweep now lives in
``nnequiv.sweep.experiments.distillation``.

Kept so ``distillation-experiment/recreate.py [--dry-run]`` — and
``scripts/worker.py``, which consumes its ``--dry-run`` JSON — keep working
exactly as before. The equivalent new entry point is ``nnequiv sweep
distillation``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nnequiv.sweep.experiments.distillation import *  # noqa: E402,F401,F403
from nnequiv.sweep.experiments.distillation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
