"""alpha-beta-CROWN verifier.

Phase 2 wrapper: CROWN is batch-native (it caches torch models across a run), so
``verify_suite`` delegates to the existing ``benchmarks.run_crown`` batch
machinery in one call. It only expresses a single-output safety margin, so it
supports ``logit_class`` only. The imports are lazy so ``import
nnequiv.verifiers`` does not pull in torch.
"""

from __future__ import annotations

from nnequiv.core import Instance, InstanceResult, InstanceSuite
from nnequiv.verifiers.base import Verifier, VerifierOptions, parse_bool
from nnequiv.verifiers.registry import register


class CrownVerifier(Verifier):
    name = "crown"
    supported_properties = frozenset({"logit_class"})

    def __init__(self, *, profile: str = "relu-kfsb", verbose: bool = False) -> None:
        self.profile = profile
        self.verbose = verbose

    def verify_suite(self, instances: list[Instance]) -> list[InstanceResult]:
        if not instances:
            return []
        from benchmarks.run_crown import build_results, prepare_artifacts, run_abcrown

        suite = InstanceSuite(name=instances[0].suite_name, instances=list(instances))
        config_path, prepared = prepare_artifacts(suite, self.profile)
        _returncode, _output, results_by_index = run_abcrown(
            config_path, prepared, self.profile, None
        )
        results, _statuses = build_results(prepared, results_by_index)
        return results

    @classmethod
    def from_options(cls, options: VerifierOptions) -> "CrownVerifier":
        opts = dict(options)
        return cls(
            profile=opts.get("profile", "relu-kfsb"),
            verbose=parse_bool(opts.get("verbose", "false")),
        )


register(CrownVerifier.name, CrownVerifier.from_options)
