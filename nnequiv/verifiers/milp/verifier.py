"""Relational-MILP verifier (CPLEX via Pyomo).

Phase 2 wrapper: delegates each instance to the existing
``benchmarks.run_pyomo.run_instance`` (which validates, computes bounds, encodes
both directions / the top-1 model, solves, and validates the witness). Phase 3
will move that logic into this package and reduce ``run_pyomo`` to a CLI shim.
The import is lazy so ``import nnequiv.verifiers`` stays torch/pyomo-free.
"""

from __future__ import annotations

from nnequiv.core import Instance, InstanceResult
from nnequiv.verifiers.base import Verifier, VerifierOptions, parse_bool
from nnequiv.verifiers.registry import register


class MilpVerifier(Verifier):
    name = "milp"
    supported_properties = frozenset({"logit_class", "linf", "top1"})

    def __init__(
        self,
        *,
        solver: str = "cplex",
        bound_tightening: str = "interval",
        fix_stable_relu_binaries: bool = True,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        self.solver = solver
        self.bound_tightening = bound_tightening
        self.fix_stable_relu_binaries = fix_stable_relu_binaries
        self.verbose = verbose
        self.debug = debug

    def verify_suite(self, instances: list[Instance]) -> list[InstanceResult]:
        from benchmarks.run_pyomo import run_instance

        return [
            run_instance(
                instance,
                self.solver,
                self.bound_tightening,
                verbose=self.verbose,
                debug=self.debug,
                fix_stable_relu_binaries=self.fix_stable_relu_binaries,
            )
            for instance in instances
        ]

    @classmethod
    def from_options(cls, options: VerifierOptions) -> "MilpVerifier":
        opts = dict(options)
        return cls(
            solver=opts.get("solver", "cplex"),
            bound_tightening=opts.get(
                "bound_tightening", opts.get("bound-tightening", "interval")
            ),
            fix_stable_relu_binaries=parse_bool(
                opts.get("fix_stable_relu_binaries", "true")
            ),
            verbose=parse_bool(opts.get("verbose", "false")),
            debug=parse_bool(opts.get("debug", "false")),
        )


register(MilpVerifier.name, MilpVerifier.from_options)
