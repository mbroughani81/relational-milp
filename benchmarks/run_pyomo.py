"""Backward-compat shim: the MILP runner moved to
:mod:`nnequiv.verifiers.milp.runner`.

``python -m benchmarks.run_pyomo`` keeps the old flags and the old CSV schema so
the sweeps and fleet are unaffected; the solve logic now lives under nnequiv.
Re-exports ``compute_bounds`` (imported by the encoder-property tests) and
``run_instance``.
"""

from nnequiv.verifiers.milp.runner import (  # noqa: F401
    compute_bounds,
    main,
    run_instance,
)

if __name__ == "__main__":
    main()
