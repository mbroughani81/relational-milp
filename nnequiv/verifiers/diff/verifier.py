"""ReluDiff / NeuroDiff verifier (the ``delta_network_test`` C tools).

Phase 2 wrapper: replicates the small per-run setup that
``benchmarks.run_diffverifier.main`` does (binary resolution, data/work dirs, the
shared ``.nnet`` serialization cache) around the existing per-instance
``run_instance``, without touching that module's internals. The C tools verify a
fixed per-class property, so this supports ``logit_class`` only. Imports are lazy
so ``import nnequiv.verifiers`` pulls in no heavy modules.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nnequiv.core import Instance, InstanceResult, validate_instance
from nnequiv.verifiers.base import Verifier, VerifierOptions, parse_bool
from nnequiv.verifiers.registry import register


class DiffVerifier(Verifier):
    name = "diff"
    supported_properties = frozenset({"logit_class"})

    def __init__(
        self,
        *,
        tool: str = "neurodiff",
        binary: str | None = None,
        data_dir: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.tool = tool
        self.binary = binary
        self.data_dir = Path(data_dir) if data_dir else None
        self.verbose = verbose

    def verify_suite(self, instances: list[Instance]) -> list[InstanceResult]:
        if not instances:
            return []
        from nn_equivalence.paths import runtime_path
        from nnequiv.verifiers.diff.runner import (
            SUPPORTED_SUITES,
            resolve_binary,
            run_instance,
        )

        suite_name = instances[0].suite_name
        if suite_name not in SUPPORTED_SUITES:
            raise ValueError(
                f"diff verifier supports suites {sorted(SUPPORTED_SUITES)}; "
                f"got {suite_name!r}"
            )
        for instance in instances:
            validate_instance(instance)

        binary = resolve_binary(SimpleNamespace(binary=self.binary))
        data_dir = self.data_dir or runtime_path("data/reludiff_mnist")
        work_dir = (
            runtime_path("artifacts/diffverifier") / suite_name / self.tool
        ).resolve()
        # Only pruning_mnist serializes its second network into work_dir; the
        # distillation_mnist suite writes next to the teacher under runtime/data/.
        if suite_name != "distillation_mnist":
            work_dir.mkdir(parents=True, exist_ok=True)

        cache: dict[str, Path] = {}
        return [
            run_instance(instance, binary, data_dir, work_dir, cache, self.verbose)
            for instance in instances
        ]

    @classmethod
    def from_options(cls, options: VerifierOptions) -> "DiffVerifier":
        opts = dict(options)
        return cls(
            tool=opts.get("tool", "neurodiff"),
            binary=opts.get("binary"),
            data_dir=opts.get("data_dir") or opts.get("data-dir"),
            verbose=parse_bool(opts.get("verbose", "false")),
        )


register(DiffVerifier.name, DiffVerifier.from_options)
