"""Backward-compat shim: the MILP encoder moved to
:mod:`nnequiv.verifiers.milp.encoder`.

Kept so existing ``import nn_equivalence.encoder_pyomo`` sites (tests, the
distillation_mnist8 suite, scripts) keep working; import from the new location
in new code.
"""

from nnequiv.verifiers.milp.encoder import *  # noqa: F401,F403
