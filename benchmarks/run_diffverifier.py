"""Backward-compat shim: the ReluDiff/NeuroDiff runner moved to
:mod:`nnequiv.verifiers.diff.runner`.

``python -m benchmarks.run_diffverifier`` keeps the old flags and CSV schema so
the sweeps and fleet are unaffected; the logic now lives under nnequiv.
"""

from nnequiv.verifiers.diff.runner import *  # noqa: F401,F403
from nnequiv.verifiers.diff.runner import _distillation_nnet_paths, main  # noqa: F401

if __name__ == "__main__":
    main()
