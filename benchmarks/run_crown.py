"""Backward-compat shim: the alpha-beta-CROWN runner moved to
:mod:`nnequiv.verifiers.crown.runner`.

``python -m benchmarks.run_crown`` keeps the old flags and CSV schema so the
sweeps and fleet are unaffected; the logic now lives under nnequiv.
"""

from nnequiv.verifiers.crown.runner import *  # noqa: F401,F403
from nnequiv.verifiers.crown.runner import main  # noqa: F401

if __name__ == "__main__":
    main()
